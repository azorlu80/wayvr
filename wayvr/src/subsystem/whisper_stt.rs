use std::{
    fmt,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use wlx_common::audio::rodio::{
    Source,
    microphone::{MicrophoneBuilder, available_inputs},
};

const WHISPER_SAMPLE_RATE: usize = 16_000;
const MAX_DURATION: Duration = Duration::from_secs(30);
const UNLOAD_AFTER: Duration = Duration::from_mins(5);
/// Upper bound for opening the mic stream. Bounds the UI-thread wait in
/// `ptt_start` so a stalled/suspended capture device can't freeze the overlay.
const READY_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Clone, Debug)]
pub struct WhisperSttConfig {
    pub model_path: PathBuf,
    pub language: Option<String>,

    pub initial_prompt: Option<String>,
    pub n_threads: i32,

    /// lower values reduce release-time lag but cost more CPU/GPU
    pub partial_decode_interval_ms: u64,

    /// ignore extremely short accidental taps
    pub min_audio_ms: u64,

    /// force a specific recording device; see `rodio::microphone::available_inputs()`
    pub rodio_input_device_name: Option<String>,

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
            initial_prompt: None,
            n_threads,
            partial_decode_interval_ms: 700,
            min_audio_ms: 250,
            rodio_input_device_name: None,
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
    Rodio(String),
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
            Self::Rodio(e) => write!(f, "rodio error: {e}"),
            Self::CaptureInit(e) => write!(f, "failed to initialize capture: {e}"),
            Self::ThreadSpawn(e) => write!(f, "failed to spawn thread: {e}"),
            Self::AlreadyRecording => write!(f, "PTT is already active"),
            Self::NotRecording => write!(f, "PTT is not active"),
        }
    }
}

impl std::error::Error for WhisperSttError {}

struct StopCapture;

struct CaptureSession {
    stop_tx: mpsc::Sender<StopCapture>,
    capture_thread: Option<JoinHandle<()>>,
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
        let (stop_tx, stop_rx) = mpsc::channel::<StopCapture>();
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
            progress_tx.clone(),
        )?;

        let input_device_name = self.config.rodio_input_device_name.clone();

        let capture_thread = thread::Builder::new()
            .name("whisper-stt-rodio-capture".to_string())
            .spawn(move || {
                rodio_capture_thread(audio_tx, stop_rx, input_device_name, ready_tx, progress_tx);
            })
            .map_err(|e| WhisperSttError::ThreadSpawn(e.to_string()))?;

        // Wait for the capture thread to report the stream is open, but bounded:
        // opening a suspended/stalled PipeWire source can hang, and this runs on
        // the UI thread, so an unbounded recv() would freeze the whole overlay.
        match ready_rx.recv_timeout(READY_TIMEOUT) {
            Ok(Ok(())) => {
                // Switch the panel to listen on this session's channel. The old
                // receiver (if any) drops here, discarding any stale in-flight text.
                self.completed_rx = Some(completed_rx);
                self.active = Some(CaptureSession {
                    stop_tx,
                    capture_thread: Some(capture_thread),
                    recognizer_thread: Some(recognizer_thread),
                    deadline: Instant::now() + MAX_DURATION,
                });

                Ok(progress_rx)
            }
            result => {
                // Init failed or timed out. Signal stop and hand the threads to the
                // reaper — never join here, a stalled mic open would block the UI.
                let _ = stop_tx.send(StopCapture);
                self.finished_captures.push(capture_thread);
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
        log::debug!(target: "whisper", "[8] stop_active_capture #{} (signal + reaper)", self.id);

        // Signal both workers to wind down, then hand them to the lazy reaper.
        // We must not join here: this runs on the UI thread (button release,
        // panel close, Drop), and joining a stalled mic capture or an in-flight
        // final decode would freeze the whole overlay.
        let _ = session.stop_tx.send(StopCapture);

        if let Some(capture_thread) = session.capture_thread.take() {
            self.finished_captures.push(capture_thread);
        }
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
    let partial_stride_samples =
        ms_to_samples(config.partial_decode_interval_ms).max(WHISPER_SAMPLE_RATE / 4);
    let min_samples = ms_to_samples(config.min_audio_ms);

    let mut audio = Vec::<f32>::new();
    let mut last_decoded_len = 0usize;
    let mut latest_partial = String::new();
    let mut processed_samples = 0;

    while let Ok(chunk) = audio_rx.recv() {
        if chunk.is_empty() {
            continue;
        }

        audio.extend_from_slice(&chunk);

        let enough_new_audio =
            audio.len().saturating_sub(last_decoded_len) >= partial_stride_samples;

        if audio.len() >= min_samples && enough_new_audio {
            let _ = progress_tx.send(PttProgress::ProcessedSamples(processed_samples));
            processed_samples += (audio.len() - last_decoded_len) as u32;
            if let Ok(text) = transcribe_audio(&ctx, &config, &audio) {
                // FORK: stream each partial so take_transcription() gets live text
                // without depending on a clean PTT stop -> loop-exit -> final decode.
                // That chain is fragile (mic.next() can block, the loop may not exit,
                // fast taps yield audio.len=0). Streaming partials makes the panel fill
                // word-by-word as the user speaks and survives a hung final decode.
                if !text.trim().is_empty() {
                    log::debug!(target: "whisper", "[6] #{id} stream partial ({} chars) -> completed_tx", text.len());
                    let _ = completed_tx.send(Ok(text.clone()));
                }
                latest_partial = text;
                last_decoded_len = audio.len();
            } else {
                // do not fail the session on a speculative decode
                // the final decode after PTT end gets reported
            }
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

    match transcribe_audio(&ctx, &config, &audio) {
        Ok(text) => {
            log::debug!(target: "whisper", "[10] #{id} FINAL decode ({} chars) -> completed_tx", text.len());
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

fn rodio_capture_thread(
    audio_tx: mpsc::Sender<Vec<f32>>,
    stop_rx: mpsc::Receiver<StopCapture>,
    input_device_name: Option<String>,
    ready_tx: mpsc::Sender<Result<(), String>>,
    progress_tx: mpsc::Sender<PttProgress>,
) {
    let mut ready_tx = Some(ready_tx);

    let result = run_rodio_capture(
        audio_tx,
        stop_rx,
        input_device_name,
        &mut ready_tx,
        progress_tx,
    );

    if let Err(e) = result
        && let Some(ready_tx) = ready_tx.take()
    {
        let _ = ready_tx.send(Err(e.to_string()));
    }
}

fn run_rodio_capture(
    audio_tx: mpsc::Sender<Vec<f32>>,
    stop_rx: mpsc::Receiver<StopCapture>,
    input_device_name: Option<String>,
    ready_tx: &mut Option<mpsc::Sender<Result<(), String>>>,
    progress_tx: mpsc::Sender<PttProgress>,
) -> Result<(), WhisperSttError> {
    let builder = MicrophoneBuilder::new();

    let builder = if let Some(input_device_name) = input_device_name {
        let inputs = available_inputs().map_err(|e| WhisperSttError::Rodio(e.to_string()))?;
        let input_device_name_lower = input_device_name.to_lowercase();

        let input = inputs
            .into_iter()
            .find(|input| {
                input
                    .to_string()
                    .to_lowercase()
                    .contains(&input_device_name_lower)
            })
            .ok_or_else(|| {
                WhisperSttError::Rodio(format!(
                    "no rodio input device matched {input_device_name:?}"
                ))
            })?;

        builder
            .device(input)
            .map_err(|e| WhisperSttError::Rodio(e.to_string()))?
    } else {
        builder
            .default_device()
            .map_err(|e| WhisperSttError::Rodio(e.to_string()))?
    };

    let builder = builder
        .default_config()
        .map_err(|e| WhisperSttError::Rodio(e.to_string()))?
        .prefer_channel_counts([
            1.try_into().expect("not zero"),
            2.try_into().expect("not zero"),
        ])
        .prefer_sample_rates([
            16_000.try_into().expect("not zero"),
            32_000.try_into().expect("not zero"),
            48_000.try_into().expect("not zero"),
        ])
        .prefer_buffer_sizes(512..);

    let mut mic = builder
        .open_stream()
        .map_err(|e| WhisperSttError::Rodio(e.to_string()))?;

    let channels = mic.channels().get() as usize;
    let input_rate = mic.sample_rate().get() as usize;
    log::debug!(target: "whisper", "[4] mic capture opened (channels={channels} rate={input_rate})");

    if let Some(ready_tx) = ready_tx.take() {
        let _ = ready_tx.send(Ok(()));
    }

    let mut resampler = StreamingResampler::default();
    let mut interleaved = Vec::new();

    // ~20 ms of input frames; whisper still receives 16 kHz mono chunks
    let chunk_input_samples = ((input_rate / 50).max(1)) * channels.max(1);

    let mut sent_samples: u32 = 0;

    'capture: loop {
        if stop_rx.try_recv().is_ok() {
            break;
        }

        interleaved.clear();

        while interleaved.len() < chunk_input_samples {
            if stop_rx.try_recv().is_ok() {
                break 'capture;
            }

            let Some(sample) = mic.next() else {
                return Err(WhisperSttError::Rodio(
                    "microphone stream ended unexpectedly".to_string(),
                ));
            };

            // Rodio's default sample type is f32. This cast also keeps the code
            // compiling if the crate is built with rodio's `64bit` feature.
            interleaved.push(sample);
        }

        let resampled_vec = resampler.push_interleaved_mono_16k(&interleaved, channels, input_rate);

        if !resampled_vec.is_empty() {
            let mut loudest_sample: f32 = 0.0;
            for sample in &resampled_vec {
                loudest_sample = loudest_sample.max(*sample);
            }

            let _ = progress_tx.send(PttProgress::VuVolume(loudest_sample));
            let _ = progress_tx.send(PttProgress::SentSamples(sent_samples));
            sent_samples += resampled_vec.len() as u32;

            if audio_tx.send(resampled_vec).is_err() {
                break;
            }
        }
    }

    Ok(())
}

#[derive(Default)]
struct StreamingResampler {
    pending: Vec<f32>,
    position: f64,
    input_rate: usize,
}

impl StreamingResampler {
    fn push_interleaved_mono_16k(
        &mut self,
        samples: &[f32],
        channels: usize,
        input_rate: usize,
    ) -> Vec<f32> {
        if channels == 0 || input_rate == 0 {
            return Vec::new();
        }

        if self.input_rate != input_rate {
            self.pending.clear();
            self.position = 0.0;
            self.input_rate = input_rate;
        }

        let frames = samples.len() / channels;
        if frames == 0 {
            return Vec::new();
        }

        let mut mono = Vec::with_capacity(frames);

        for frame in 0..frames {
            let frame_start = frame * channels;
            let mut sum = 0.0f32;

            for ch in 0..channels {
                sum += samples[frame_start + ch];
            }

            mono.push(sum / channels as f32);
        }

        self.pending.extend_from_slice(&mono);

        let step = input_rate as f64 / WHISPER_SAMPLE_RATE as f64;
        let mut out = Vec::with_capacity(
            ((self.pending.len() as f64 - self.position) / step).max(0.0) as usize,
        );

        #[allow(clippy::while_float)]
        while self.position + 1.0 < self.pending.len() as f64 {
            let i = self.position.floor() as usize;
            let frac = (self.position - i as f64) as f32;

            let a = self.pending[i];
            let b = self.pending[i + 1];

            out.push(a + (b - a) * frac);

            self.position += step;
        }

        let drop_count = self.position.floor() as usize;
        if drop_count > 0 {
            self.pending.drain(..drop_count);
            self.position -= drop_count as f64;
        }

        out
    }
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
        assert_eq!(ms_to_samples(250), 4_000); // min_audio default
        assert_eq!(ms_to_samples(700), 11_200); // partial stride default
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

    #[test]
    fn resampler_rejects_degenerate_input() {
        let mut r = StreamingResampler::default();
        assert!(r.push_interleaved_mono_16k(&[0.1, 0.2], 0, 16_000).is_empty());
        assert!(r.push_interleaved_mono_16k(&[0.1, 0.2], 1, 0).is_empty());
    }

    #[test]
    fn resampler_passthrough_16khz_mono() {
        let mut r = StreamingResampler::default();
        let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let out = r.push_interleaved_mono_16k(&input, 1, 16_000);
        // step == 1.0 at 16 kHz mono -> ~one output sample per input sample.
        assert!((out.len() as i64 - 1000).abs() <= 3, "got {}", out.len());
        assert!((out[0] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn resampler_averages_stereo_to_mono() {
        let mut r = StreamingResampler::default();
        // 4 stereo frames, L=1.0 R=3.0 -> each mono sample should be 2.0.
        let input = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        let out = r.push_interleaved_mono_16k(&input, 2, 16_000);
        assert!(!out.is_empty());
        for s in out {
            assert!((s - 2.0).abs() < 1e-3, "expected mono avg 2.0, got {s}");
        }
    }

    #[test]
    fn resampler_downsamples_48k_to_16k() {
        let mut r = StreamingResampler::default();
        let input: Vec<f32> = (0..480).map(|i| (i % 2) as f32).collect(); // 480 mono @ 48 kHz
        let out = r.push_interleaved_mono_16k(&input, 1, 48_000);
        // 48 kHz -> 16 kHz is a 1/3 ratio, so expect ~160 samples.
        assert!((out.len() as i64 - 160).abs() <= 3, "got {}", out.len());
    }

    #[test]
    fn resampler_resets_state_on_rate_change() {
        let mut r = StreamingResampler::default();
        let _ = r.push_interleaved_mono_16k(&[0.5; 100], 1, 44_100);
        // A new input rate must clear stale pending samples, not blend across rates.
        let out = r.push_interleaved_mono_16k(&[0.0; 480], 1, 48_000);
        assert!((out.len() as i64 - 160).abs() <= 3, "got {}", out.len());
    }
}
