//! Native PipeWire capture for the Whisper microphone.
//!
//! Replaces the rodio/cpal/ALSA path, which routed through the flaky
//! `default` -> pulse -> pipewire chain and, worse, blocked in `mic.next()` so
//! the recognizer could never be stopped cleanly on PTT release.
//!
//! This connects a PipeWire capture stream directly to a source node (by name,
//! e.g. `wivrn.source`, or the default source when no target is given), asks the
//! graph for F32 mono at the requested rate (PipeWire inserts the converter), and
//! delivers samples over an `mpsc` channel. Stopping just quits this stream's own
//! main loop from any thread — no blocking call to interrupt.

use std::io::Cursor;
use std::sync::mpsc;
use std::thread;

use pipewire as pw;
use pw::properties::properties;
use pw::spa;
use spa::param::audio::{AudioFormat, AudioInfoRaw};
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::{ParamType, format_utils};
use spa::pod::serialize::PodSerializer;
use spa::pod::{Object, Pod, Value};
use spa::utils::{Direction, SpaTypes};

/// Owns the capture thread. Stopping (or dropping) quits the PipeWire loop and
/// joins the thread — never blocks on the audio device itself.
pub struct PwCapture {
    stop_tx: pw::channel::Sender<()>,
    thread: Option<thread::JoinHandle<()>>,
}

impl PwCapture {
    pub fn stop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PwCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

struct UserData {
    format: AudioInfoRaw,
    audio_tx: mpsc::Sender<Vec<f32>>,
}

/// Start capturing from `target` (a PipeWire node name; `None` = default source).
/// Samples arrive as F32 mono at `rate` on `audio_tx`. `ready_tx` gets `Ok(())`
/// once the stream is connected, or `Err` if setup fails.
pub fn start(
    target: Option<String>,
    rate: u32,
    audio_tx: mpsc::Sender<Vec<f32>>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<PwCapture, String> {
    let (stop_tx, stop_rx) = pw::channel::channel::<()>();

    let thread = thread::Builder::new()
        .name("whisper-pw-capture".to_string())
        .spawn(move || {
            if let Err(e) = run(target, rate, audio_tx, &ready_tx, stop_rx) {
                let _ = ready_tx.send(Err(e));
            }
        })
        .map_err(|e| e.to_string())?;

    Ok(PwCapture {
        stop_tx,
        thread: Some(thread),
    })
}

fn run(
    target: Option<String>,
    rate: u32,
    audio_tx: mpsc::Sender<Vec<f32>>,
    ready_tx: &mpsc::Sender<Result<(), String>>,
    stop_rx: pw::channel::Receiver<()>,
) -> Result<(), String> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(|e| e.to_string())?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(|e| e.to_string())?;
    let core = context.connect_rc(None).map_err(|e| e.to_string())?;

    // Stop from any thread: send () -> quit the loop -> run() returns -> the
    // stream drops and disconnects. No blocking device call to interrupt.
    let _stop = stop_rx.attach(mainloop.loop_(), {
        let mainloop = mainloop.clone();
        move |_| mainloop.quit()
    });

    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
        *pw::keys::NODE_NAME => "whisper-capture",
    };
    if let Some(target) = target {
        props.insert(*pw::keys::TARGET_OBJECT, target);
    }

    let data = UserData {
        format: AudioInfoRaw::default(),
        audio_tx,
    };

    let stream =
        pw::stream::StreamBox::new(&core, "whisper-capture", props).map_err(|e| e.to_string())?;

    let _listener = stream
        .add_local_listener_with_user_data(data)
        .param_changed(|_, ud, id, param| {
            let Some(param) = param else {
                return;
            };
            if id != ParamType::Format.as_raw() {
                return;
            }
            let Ok((media_type, media_subtype)) = format_utils::parse_format(param) else {
                return;
            };
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }
            let _ = ud.format.parse(param);
        })
        .process(|stream, ud| {
            while let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    continue;
                }
                let data = &mut datas[0];
                let size = data.chunk().size() as usize;
                let Some(bytes) = data.data() else {
                    continue;
                };
                let n = size.min(bytes.len());
                let mut samples = Vec::with_capacity(n / 4);
                for frame in bytes[..n].chunks_exact(4) {
                    samples.push(f32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]));
                }
                if !samples.is_empty() {
                    let _ = ud.audio_tx.send(samples);
                }
            }
        })
        .register()
        .map_err(|e| e.to_string())?;

    // Ask for F32 mono at the whisper rate; the graph inserts a converter so we
    // never resample by hand.
    let mut audio_info = AudioInfoRaw::new();
    audio_info.set_format(AudioFormat::F32LE);
    audio_info.set_rate(rate);
    audio_info.set_channels(1);
    let obj = Object {
        type_: SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values: Vec<u8> = PodSerializer::serialize(Cursor::new(Vec::new()), &Value::Object(obj))
        .map_err(|e| e.to_string())?
        .0
        .into_inner();
    let mut params = [Pod::from_bytes(&values).ok_or("invalid format pod")?];

    stream
        .connect(
            Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .map_err(|e| e.to_string())?;

    let _ = ready_tx.send(Ok(()));
    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Headless smoke test: capture from a live PipeWire source, verify samples
    /// flow at ~the requested rate and that stop() returns promptly (never blocks
    /// on the device). Needs a running PipeWire graph with a capture source.
    ///   WAYVR_PW_TARGET=wivrn.source cargo test --release --all-features \
    ///     --bin wayvr -- --ignored --nocapture pw_capture
    #[test]
    #[ignore = "needs a running PipeWire + a capture source"]
    fn pw_capture_streams_and_stops_cleanly() {
        let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
        let rate = 16_000u32;

        let target = std::env::var("WAYVR_PW_TARGET").ok().filter(|s| !s.is_empty());
        let mut cap = start(target, rate, audio_tx, ready_tx).expect("spawn capture thread");

        let ready = ready_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("no ready signal within 3s");
        assert!(ready.is_ok(), "capture failed to start: {ready:?}");

        let mut total = 0usize;
        let deadline = Instant::now() + Duration::from_millis(1500);
        while Instant::now() < deadline {
            if let Ok(chunk) = audio_rx.recv_timeout(Duration::from_millis(300)) {
                total += chunk.len();
            }
        }

        let before_stop = Instant::now();
        cap.stop();
        let stop_took = before_stop.elapsed();
        assert!(
            stop_took < Duration::from_secs(1),
            "stop() blocked for {stop_took:?} (should be instant)"
        );

        let expected = (rate as f32 * 1.5) as usize;
        assert!(
            total > expected / 3,
            "too few samples: {total} (expected ~{expected}); is the source live?"
        );
        eprintln!("OK: captured {total} f32 samples in ~1.5s at {rate} Hz, stop took {stop_took:?}");
    }
}
