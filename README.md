# WayVR — Whisper Voice Dictation

> **Fork of [wayvr-org/wayvr](https://github.com/wayvr-org/wayvr).** This README
> covers the Whisper voice-dictation work in this fork; for the base overlay
> (installation, general usage, features) see **[UPSTREAM-README.md](UPSTREAM-README.md)**.

Reliable, low-latency **push-to-talk speech-to-text inside VR** for the
[WayVR](https://github.com/wayvr-org/wayvr) overlay on Linux. Hold a button,
speak, release — the recognized text lands in the panel and can be pasted into
the focused application. Built for hands-on-keyboard-free *voice coding* in a
Wayland/X11 desktop streamed to a standalone headset over
[WiVRn](https://github.com/WiVRn/WiVRn).

This fork rebuilds the audio-capture and decode pipeline that upstream shipped as
experimental, turning it into a production-grade, language-neutral component.

---

## Highlights

- **Deterministic push-to-talk lifecycle.** Press starts capture; release ends
  the utterance and triggers exactly one transcription. No background decode loop
  racing the capture.
- **Native PipeWire capture.** Connects a PipeWire stream directly to a named
  source node (the headset mic) and stops in ~1.5 ms by quitting its own loop —
  no blocking device read to interrupt, and no fragile `ALSA default → pulse →
  pipewire` routing.
- **One decode per utterance.** Audio is buffered while the button is held and
  decoded once on release (or on a short trailing silence via lightweight VAD).
  Eliminates the panel flicker and GPU thrash of streaming interim decodes.
- **Hallucination-resistant.** A voice-activity gate plus tuned Whisper
  parameters keep the model from emitting confident stock phrases over silence.
- **Language-neutral by default.** Language, capture source, and dictation prompt
  are all environment-driven; nothing locale-specific is hard-coded.
- **UI-thread safe.** Capture/recognizer threads are never joined on the render
  thread, so a stalled mic or an in-flight decode can never freeze the overlay.

---

## Quick start

Build WayVR with the `whisper` feature (pulls in `pipewire` and
`whisper-rs`/Vulkan):

```sh
cargo build --release --all-features
```

Point WayVR at the headset microphone and your language. With WiVRn, set the
environment in `~/.config/wivrn/config.json`:

```json
{
  "application": [
    "env",
    "WAYVR_WHISPER_LANG=tr",
    "WAYVR_WHISPER_SOURCE=wivrn.source",
    "WAYVR_WHISPER_PROMPT=Yazılım geliştirici Türkçe sesli komut veriyor. Claude, terminal, kod, commit, refactor.",
    "/path/to/target/release/wayvr"
  ]
}
```

Then in VR: open the Whisper panel, **hold** the transcribe button while
speaking, **release**, and use **Paste-and-Go** to send the text to the focused
window.

> The Whisper model (`ggml-large-v3-turbo`) is loaded from WayVR's data
> directory, as upstream.

---

## Configuration reference

All configuration is via environment variables — safe defaults, no code changes.

| Variable | Purpose | Default |
|---|---|---|
| `WAYVR_WHISPER_LANG` | Whisper decode language. | unset → auto-detect |
| `WAYVR_WHISPER_SOURCE` | PipeWire source node to capture from. Set to the headset mic. | unset → default source |
| `WAYVR_WHISPER_GPU` | Whisper/ggml Vulkan device index. On a laptop the discrete GPU is often `1`; the default `0` may be the slow integrated GPU. | `0` |
| `WAYVR_WHISPER_PROMPT` | Optional dictation/domain context in the target language; steers short or noisy audio toward dictation and away from training-data hallucinations. | unset → none |
| `RUST_LOG=whisper=debug` | Emit the numbered pipeline trace (`[1]…[13]`) for diagnostics. | off |

---

## Architecture

```
 PTT press ──▶ ptt_start ──▶ PipeWire capture ──▶ audio (16 kHz mono f32)
                                   │                     │
                                   ▼                     ▼
                            (headset mic)         recognizer thread
                                                    accumulate
 PTT release ─▶ stop_active_capture ─▶ stop stream ─▶ end of utterance
                                                         │
                                                    one Whisper decode
                                                         │
                                                    per-session channel
                                                         │
                                                   render tick ─▶ panel ─▶ paste
```

- **`subsystem/whisper_pw_capture.rs`** — the PipeWire capture backend. Owns a
  main loop on a dedicated thread; a `pipewire::channel` from any thread quits it
  for an instant, allocation-free stop. Requests F32 mono at 16 kHz so the graph
  does the resampling.
- **`subsystem/whisper_stt.rs`** — session lifecycle and the recognizer. Each PTT
  session gets a fresh transcription channel (a late decode from a previous
  session cannot leak into the next), threads are reaped lazily off the UI thread,
  and `ptt_start` waits for stream-open with a bounded timeout.
- **Anti-hallucination** — a minimum voiced-duration gate before decoding, plus
  `temperature 0`, `suppress_blank`, `suppress_nst`, `no_speech_thold`,
  `single_segment`, and `no_context`.
- **UI** — the panel clears on each press; the VU meter/progress reset on release,
  not on a partial; Paste-and-Go uses Ctrl+Shift+V (terminals ignore plain
  Ctrl+V).

---

## Testing

Pure-logic unit tests run headless:

```sh
cargo test --release --all-features --bin wayvr
```

An end-to-end capture smoke test verifies that samples flow from a live source at
the requested rate and that stop returns instantly (needs a running PipeWire graph
and a live source):

```sh
WAYVR_PW_TARGET=wivrn.source \
  cargo test --release --all-features --bin wayvr -- --ignored --nocapture pw_capture
```

---

## Upstreaming

The pipeline is language-neutral and self-contained; the fork changes are on
`patch/whisper-lang-env` as focused, individually reviewable commits, intended to
be portable back to `wayvr-org/wayvr`.
