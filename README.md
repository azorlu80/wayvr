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

### Runtime prerequisites

WayVR links `libopenvr_api.so` at load time even on the OpenXR path, so install
the `openvr` package (Arch: `pacman -S openvr`) — otherwise the process aborts
before `main()` with a bare `error while loading shared libraries`, invisible to
WayVR's own logging. The Whisper model (`ggml-large-v3-turbo`) is loaded from
WayVR's data directory, as upstream.

### Launch from the WiVRn app launcher (recommended)

WayVR is an OpenXR **overlay** — it runs *alongside* whatever else is streaming,
so launch it on demand from the WiVRn *Applications* launcher rather than
auto-starting it. Install a desktop entry that carries the Whisper environment
and is tagged with the `X-WiVRn-VR` category so WiVRn lists it:

```ini
# ~/.local/share/applications/wayvr.desktop
[Desktop Entry]
Type=Application
Name=WayVR
Exec=env WAYVR_WHISPER_LANG=tr WAYVR_WHISPER_SOURCE=wivrn.source WAYVR_WHISPER_GPU=1 /path/to/target/release/wayvr
Icon=wayvr
Categories=Utility;X-WiVRn-VR;
```

Leave the WiVRn config's `application` **unset** so connecting drops you into the
launcher instead of force-starting one app. Copy `wayvr/wayvr.{png,svg}` into
`~/.local/share/icons/hicolor/{128x128,scalable}/apps/` so the entry shows the
WayVR icon.

> **Why on-demand, not auto-launch?** The Whisper model stays resident on the
> GPU while WayVR runs. On a single-GPU laptop that same GPU is encoding the
> WiVRn video stream, so keeping WayVR up during a pure gaming session steals
> encoder headroom and surfaces as periodic frame glitches. Launch WayVR when you
> want to dictate; close it when you don't.

Then in VR: open the Whisper panel, **hold** the transcribe button while
speaking, **release**, and use **Paste-and-Go** to send the text to the focused
window — which also clears the panel, so the same utterance is never pasted
twice.

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
- **UI** — the panel clears on each press *and* after Paste-and-Go (so a sent
  utterance can't be pasted twice); the VU meter/progress reset on release, not
  on a partial; Paste-and-Go uses Ctrl+Shift+V (terminals ignore plain Ctrl+V).

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
