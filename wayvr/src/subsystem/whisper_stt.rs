use std::{
    fmt,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::whisper_pw_capture;

const WHISPER_SAMPLE_RATE: usize = 16_000;
const MAX_DURATION: Duration = Duration::from_secs(30);
const UNLOAD_AFTER: Duration = Duration::from_mins(5);
/// Upper bound for opening the mic stream. Bounds the UI-thread wait in
/// `ptt_start` so a stalled/suspended capture device can't freeze the overlay.
const READY_TIMEOUT: Duration = Duration::from_secs(3);
/// Peak amplitude (linear, 0..1) below which a window is treated as silence and
/// not decoded. Whisper hallucinates confident phrases on silence, so gating it
/// keeps garbage out of the panel. ~0.02 ≈ -34 dBFS, above room noise.
const SILENCE_PEAK: f32 = 0.02;
/// If no new audio arrives for this long the recognizer treats the utterance as
/// finished and runs the final decode. This makes the result appear right after
/// PTT release even when the capture thread can't drop its sender promptly
/// (a stalled/blocking mic.next()), instead of hanging until the deadline.
const RECOGNIZER_IDLE: Duration = Duration::from_millis(350);

#[derive(Clone, Debug)]
pub struct WhisperSttConfig {
    pub model_path: PathBuf,
    pub language: Option<String>,

    pub initial_prompt: Option<String>,
    pub n_threads: i32,

    /// ignore extremely short accidental taps
    pub min_audio_ms: u64,

    /// PipeWire source node to capture from (e.g. `"wivrn.source"`, the headset
    /// mic). `None` connects to the default source.
    pub capture_target: Option<String>,

    pub use_gpu: bool,
    pub gpu_device: i32,
    pub flash_attn: bool,
}

impl WhisperSttConfig {
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        let n_threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(4) as i32);

        Self {
            model_path: model_path.as_ref().to_path_buf(),
            // Fork patch: read whisper language from WAYVR_WHISPER_LANG env var
            // (e.g. "tr" for Turkish). Unset/empty => None => auto-detect (upstream default).
            language: std::env::var("WAYVR_WHISPER_LANG").ok().filter(|s| !s.is_empty()),
            // Optional dictation/domain context via WAYVR_WHISPER_PROMPT. A short
            // prompt in the target language steers short/noisy audio toward
            // dictation instead of training-data hallucinations. Language-neutral
            // by default (unset => None); set it per deployment.
            initial_prompt: std::env::var("WAYVR_WHISPER_PROMPT")
                .ok()
                .filter(|s| !s.is_empty()),
            n_threads,
            min_audio_ms: 250,
            // Fork patch: capture straight from this PipeWire source (e.g.
            // "wivrn.source", the headset mic) via WAYVR_WHISPER_SOURCE, instead
            // of the default source.
            capture_target: std::env::var("WAYVR_WHISPER_SOURCE")
                .ok()
                .filter(|s| !s.is_empty()),
            use_gpu: true,
            gpu_device: 0,
            flash_attn: false,
        }
    }
}

#[derive(Debug)]
pub enum WhisperSttError {
    ModelLoad(String),
    Whisper(String),
    CaptureInit(String),
    ThreadSpawn(String),
    AlreadyRecording,
    NotRecording,
}

impl fmt::Display for WhisperSttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelLoad(e) => write!(f, "failed to load whisper model: {e}"),
            Self::Whisper(e) => write!(f, "whisper error: {e}"),
            Self::CaptureInit(e) => write!(f, "failed to initialize capture: {e}"),
            Self::ThreadSpawn(e) => write!(f, "failed to spawn thread: {e}"),
            Self::AlreadyRecording => write!(f, "PTT is already active"),
            Self::NotRecording => write!(f, "PTT is not active"),
        }
    }
}

impl std::error::Error for WhisperSttError {}

struct CaptureSession {
    pw_capture: whisper_pw_capture::PwCapture,
    recognizer_thread: Option<JoinHandle<()>>,
    deadline: Instant,
}

pub struct WhisperStt {
    id: usize,
    config: WhisperSttConfig,
    ctx: Arc<WhisperContext>,

    active: Option<CaptureSession>,
    // Stopped threads are reaped lazily off the UI thread (see `reap_finished_threads`).
    // Joining a capture/recognizer synchronously could block the caller — and the UI
    // thread that drives it — if the mic stalls or a decode is still running.
    finished_recognizers: Vec<JoinHandle<()>>,
    finished_captures: Vec<JoinHandle<()>>,

    // Receiver for the CURRENT session only. A fresh channel is created per
    // ptt_start; a detached/stale recognizer from a previous session holds the
    // old Sender, so its late final decode lands in a dropped channel instead of
    // corrupting the new session's panel.
    completed_rx: Option<mpsc::Receiver<Result<String, String>>>,

    last_error: Option<String>,
    unload_at: Instant,
}

pub enum PttProgress {
    VuVolume(f32),
    SentSamples(u32),
    ProcessedSamples(u32),
}

impl WhisperStt {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, WhisperSttError> {
        Self::init(WhisperSttConfig::new(model_path))
    }

    pub fn init(config: WhisperSttConfig) -> Result<Self, WhisperSttError> {
        let ctx_params = WhisperContextParameters {
            use_gpu: config.use_gpu,
            gpu_device: config.gpu_device,
            flash_attn: config.flash_attn,
            ..Default::default()
        };

        let ctx = WhisperContext::new_with_params(&config.model_path, ctx_params)
            .map_err(|e| WhisperSttError::ModelLoad(e.to_string()))?;

        static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        log::debug!(target: "whisper", "[2] CREATE WhisperStt #{id} (model loaded)");

        Ok(Self {
            id,
            config,
            ctx: Arc::new(ctx),
            active: None,
            finished_recognizers: Vec::new(),
            finished_captures: Vec::new(),
            completed_rx: None,
            last_error: None,
            unload_at: Instant::now() + UNLOAD_AFTER,
        })
    }

    /// starts a fresh capture stream and a transcription worker
    pub fn ptt_start(&mut self) -> Result<mpsc::Receiver<PttProgress>, WhisperSttError> {
        log::debug!(target: "whisper", "[3] ptt_start #{}", self.id);
        self.unload_at = Instant::now() + UNLOAD_AFTER;
        self.reap_finished_threads();

        if self.active.is_some() {
            log::debug!(target: "whisper", "[3!] ptt_start #{} REJECTED: already recording", self.id);
            return Err(WhisperSttError::AlreadyRecording);
        }

        let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
        let (progress_tx, progress_rx) = mpsc::channel::<PttProgress>();
        // Fresh transcription channel per session: this recognizer is the only
        // producer for this receiver. A stale recognizer from an earlier session
        // keeps its own (now orphaned) sender and cannot reach this session.
        let (completed_tx, completed_rx) = mpsc::channel::<Result<String, String>>();

        let recognizer_thread = spawn_recognizer_thread(
            self.id,
            Arc::clone(&self.ctx),
            self.config.clone(),
            audio_rx,
            completed_tx,
            progress_tx,
        )?;

        // Native PipeWire capture connected straight to the source node (headset
        // mic), delivering 16 kHz mono f32. It stops instantly by quitting its own
        // loop — no blocking mic.next() the stop path can't interrupt.
        let target = self.config.capture_target.clone();
        let pw_capture =
            whisper_pw_capture::start(target, WHISPER_SAMPLE_RATE as u32, audio_tx, ready_tx)
                .map_err(WhisperSttError::CaptureInit)?;

        // Bounded wait for the stream to connect (runs on the UI thread).
        match ready_rx.recv_timeout(READY_TIMEOUT) {
            Ok(Ok(())) => {
                // Switch the panel to listen on this session's channel. The old
                // receiver (if any) drops here, discarding any stale in-flight text.
                self.completed_rx = Some(completed_rx);
                self.active = Some(CaptureSession {
                    pw_capture,
                    recognizer_thread: Some(recognizer_thread),
                    deadline: Instant::now() + MAX_DURATION,
                });

                Ok(progress_rx)
            }
            result => {
                // Init failed or timed out. Dropping the capture stops it instantly
                // (which drops audio_tx, so the recognizer exits); reap the worker.
                drop(pw_capture);
                self.finished_recognizers.push(recognizer_thread);

                let msg = match result {
                    Ok(Err(e)) => e,
                    Err(_) => "microphone did not start within timeout".to_string(),
                    Ok(Ok(())) => unreachable!(),
                };
                log::debug!(target: "whisper", "[3!] #{} capture init failed: {msg}", self.id);
                Err(WhisperSttError::CaptureInit(msg))
            }
        }
    }

    fn stop_active_capture(&mut self) -> Result<(), WhisperSttError> {
        let Some(mut session) = self.active.take() else {
            log::debug!(target: "whisper", "[8!] stop_active_capture #{}: no active session", self.id);
            return Err(WhisperSttError::NotRecording);
        };
        log::debug!(target: "whisper", "[8] stop_active_capture #{} (stop capture + reaper)", self.id);

        // Stop the capture: quits its loop and joins its thread in ~1ms (verified),
        // then drops audio_tx so the recognizer's recv() returns Err and it runs
        // the final decode. The recognizer is handed to the lazy reaper — never
        // joined on the UI thread, since a final decode may still be running.
        session.pw_capture.stop();
        if let Some(recognizer_thread) = session.recognizer_thread.take() {
            self.finished_recognizers.push(recognizer_thread);
        }

        Ok(())
    }

    fn drain_completed_transcriptions(&mut self) -> Option<String> {
        let Some(rx) = self.completed_rx.as_ref() else {
            return None;
        };

        let mut latest = None;
        let mut count = 0;
        let mut error = None;

        while let Ok(result) = rx.try_recv() {
            count += 1;
            log::debug!(target: "whisper", "[12-recv] #{} drained msg: {result:?}", self.id);
            match result {
                Ok(text) => {
                    let text = normalize_transcript(text);
                    if !text.is_empty() {
                        latest = Some(text);
                    }
                }
                Err(e) => {
                    error = Some(e);
                }
            }
        }

        if let Some(e) = error {
            self.last_error = Some(e);
        }
        if count > 0 {
            log::debug!(target: "whisper", "[12-drain] #{} drained {count} msg(s), latest={latest:?}", self.id);
        }

        latest
    }

    /// stops the pw stream & finalizes recognition asynchronously
    /// poll `take_transcription()` from your main loop to receive transcription
    pub fn ptt_end(&mut self) -> Result<(), WhisperSttError> {
        self.unload_at = Instant::now() + UNLOAD_AFTER;
        self.stop_active_capture()
    }

    pub fn take_transcription(&mut self) -> Option<String> {
        self.reap_finished_threads();

        let latest = self.drain_completed_transcriptions();

        if let Some(text) = &latest {
            log::debug!(target: "whisper", "[12] #{} take_transcription -> Some ({} chars)", self.id, text.len());
            self.unload_at = Instant::now() + UNLOAD_AFTER;
            return latest;
        }

        // been recording for too long, force send a stop signal
        if self
            .active
            .as_ref()
            .is_some_and(|session| Instant::now() >= session.deadline)
            && let Err(e) = self.stop_active_capture()
        {
            self.last_error = Some(e.to_string());
        }

        None
    }

    pub fn should_unload(&self) -> bool {
        self.unload_at < Instant::now()
    }

    pub fn id(&self) -> usize {
        self.id
    }

    /// Join any capture/recognizer threads that have already finished. Only
    /// finished handles are joined, so this never blocks the caller.
    fn reap_finished_threads(&mut self) {
        Self::reap(&mut self.finished_recognizers);
        Self::reap(&mut self.finished_captures);
    }

    fn reap(handles: &mut Vec<JoinHandle<()>>) {
        let mut i = 0;
        while i < handles.len() {
            if handles[i].is_finished() {
                let _ = handles.swap_remove(i).join();
            } else {
                i += 1;
            }
        }
    }
}

impl Drop for WhisperStt {
    fn drop(&mut self) {
        // Signal any active capture to stop. Finished workers are joined; any
        // still running (a stalled mic, an in-flight decode) are detached rather
        // than joined, so dropping/closing the panel can never freeze the UI.
        if self.active.is_some() {
            let _ = self.ptt_end();
        }
        self.reap_finished_threads();
    }
}

fn spawn_recognizer_thread(
    id: usize,
    ctx: Arc<WhisperContext>,
    config: WhisperSttConfig,
    audio_rx: mpsc::Receiver<Vec<f32>>,
    completed_tx: mpsc::Sender<Result<String, String>>,
    progress_tx: mpsc::Sender<PttProgress>,
) -> Result<JoinHandle<()>, WhisperSttError> {
    thread::Builder::new()
        .name("whisper-stt-recognizer".to_string())
        .spawn(move || {
            recognizer_thread(id, ctx, config, audio_rx, completed_tx, progress_tx);
        })
        .map_err(|e| WhisperSttError::ThreadSpawn(e.to_string()))
}

fn recognizer_thread(
    id: usize,
    ctx: Arc<WhisperContext>,
    config: WhisperSttConfig,
    audio_rx: mpsc::Receiver<Vec<f32>>,
    completed_tx: mpsc::Sender<Result<String, String>>,
    progress_tx: mpsc::Sender<PttProgress>,
) {
    log::debug!(target: "whisper", "[5] recognizer #{id} started");
    let min_samples = ms_to_samples(config.min_audio_ms);

    let mut audio = Vec::<f32>::new();
    let latest_partial = String::new();
    let mut last_voice: Option<Instant> = None;
    let mut voiced_samples = 0usize;
    loop {
        // Break (and run the final decode) when the capture closes the channel or
        // no samples arrive for a while.
        let chunk = match audio_rx.recv_timeout(RECOGNIZER_IDLE) {
            Ok(chunk) => chunk,
            Err(_) => break,
        };
        if chunk.is_empty() {
            continue;
        }

        // The mic is kept warm, so the stream never actually goes quiet — it just
        // delivers silence. The reliable end-of-utterance signal is therefore
        // "voice, then RECOGNIZER_IDLE of silence", not "no samples".
        let chunk_peak = chunk.iter().fold(0f32, |m, &s| m.max(s.abs()));
        if chunk_peak >= SILENCE_PEAK {
            last_voice = Some(Instant::now());
            voiced_samples += chunk.len();
        }

        audio.extend_from_slice(&chunk);

        // No interim decoding: a full whisper decode every stride hogged this loop
        // so the end-of-utterance break never ran. Just accumulate + track voice
        // here, then decode ONCE below. Cheap loop -> the break fires promptly.
        // Feed the live VU meter straight from the captured chunk.
        let _ = progress_tx.send(PttProgress::VuVolume(chunk_peak));
        let _ = progress_tx.send(PttProgress::SentSamples(audio.len() as u32));
        let _ = progress_tx.send(PttProgress::ProcessedSamples(audio.len() as u32));
        if let Some(voiced_at) = last_voice
            && voiced_at.elapsed() >= RECOGNIZER_IDLE
        {
            break;
        }
    }

    // Capture stopped (PTT released or deadline hit). Run one final decode over
    // the whole utterance for best quality; it supersedes the streamed partials.
    log::debug!(
        target: "whisper",
        "[9] #{id} recognizer loop EXIT (audio {:.2}s, min {:.2}s)",
        audio.len() as f32 / WHISPER_SAMPLE_RATE as f32,
        min_samples as f32 / WHISPER_SAMPLE_RATE as f32
    );
    if audio.len() < min_samples {
        log::debug!(target: "whisper", "[10-empty] #{id} audio too short -> send empty");
        let _ = completed_tx.send(Ok(String::new()));
        return;
    }

    // Require a real amount of actual speech, not just a noise blip. Without this
    // whisper confidently hallucinates a phrase ("Teşekkürler", "Altyazı M.K.")
    // over a near-silent buffer.
    if voiced_samples < min_samples {
        log::debug!(
            target: "whisper",
            "[10-novoice] #{id} only {:.2}s of voice -> send empty",
            voiced_samples as f32 / WHISPER_SAMPLE_RATE as f32
        );
        let _ = completed_tx.send(Ok(String::new()));
        return;
    }

    match transcribe_audio(&ctx, &config, &audio) {
        Ok(text) => {
            log::debug!(target: "whisper", "[10] #{id} FINAL decode -> completed_tx: {text:?}");
            let _ = completed_tx.send(Ok(text));
        }
        Err(e) if !latest_partial.trim().is_empty() => {
            // Prefer a recent partial over losing the utterance completely.
            log::debug!(target: "whisper", "[10-partial] #{id} final failed, send latest partial: {e}");
            let _ = completed_tx.send(Ok(latest_partial));
            let _ = completed_tx.send(Err(e.to_string()));
        }
        Err(e) => {
            log::debug!(target: "whisper", "[10-err] #{id} final decode error: {e}");
            let _ = completed_tx.send(Err(e.to_string()));
        }
    }
}

fn transcribe_audio(
    ctx: &WhisperContext,
    config: &WhisperSttConfig,
    audio: &[f32],
) -> Result<String, WhisperSttError> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    params.set_n_threads(config.n_threads);
    params.set_language(config.language.as_deref());
    params.set_no_timestamps(true);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Anti-hallucination: greedy/deterministic decode, drop blanks, reject
    // segments the model flags as non-speech, and suppress non-speech tokens
    // (curbs "Altyazı M.K." / "Teşekkürler" on silence).
    params.set_temperature(0.0);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_no_speech_thold(0.6);
    // Short independent PTT command -> one segment, no carry-over context. Also
    // trims decode time.
    params.set_single_segment(true);
    params.set_no_context(true);

    if let Some(prompt) = config.initial_prompt.as_deref() {
        params.set_initial_prompt(prompt);
    }

    let mut state = ctx
        .create_state()
        .map_err(|e| WhisperSttError::Whisper(e.to_string()))?;

    state
        .full(params, audio)
        .map_err(|e| WhisperSttError::Whisper(e.to_string()))?;

    let text = state
        .as_iter()
        .map(|segment| segment.to_string())
        .collect::<String>();

    Ok(normalize_transcript(text))
}

const fn ms_to_samples(ms: u64) -> usize {
    ((ms as usize) * WHISPER_SAMPLE_RATE) / 1000
}

fn normalize_transcript(text: String) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ms_to_samples_converts_at_16khz() {
        assert_eq!(ms_to_samples(0), 0);
        assert_eq!(ms_to_samples(1000), 16_000);
        assert_eq!(ms_to_samples(250), 4_000); // min_audio_ms default
        assert_eq!(ms_to_samples(400), 6_400);
    }

    #[test]
    fn normalize_transcript_collapses_whitespace() {
        assert_eq!(
            normalize_transcript("  Merhaba   ben  Ali. ".to_string()),
            "Merhaba ben Ali."
        );
        assert_eq!(normalize_transcript("\n\ttek\n".to_string()), "tek");
        assert_eq!(normalize_transcript(String::new()), "");
        assert_eq!(normalize_transcript("   ".to_string()), "");
    }
}
