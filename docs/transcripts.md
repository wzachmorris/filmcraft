# Transcripts and text-based editing

FilmCraft's Text panel ▸ **Transcript** tab shows the dialogue of the open sequence as text. Select
words to mark In/Out, then extract or lift them; remove filler words and long pauses in one step;
turn the transcript into captions. Every action is an engine command (`transcript.*`), so the CLI,
the control channel and MCP agents can do the same.

## Model

- A **transcript** belongs to a media item (`Project::transcripts`, saved in the `.fcproj` since
  schema v9). It lists **words** with media-time bounds (`Tick`s), an optional speaker index and a
  confidence, plus the speaker names and the language. Because the times are media time, the
  transcript stays valid however the clip is trimmed, moved, sped up or reused.
- The **sequence transcript** is derived, never stored (`filmcraft_edit::transcript::sequence_words`):
  audio tracks are read top first; a word is heard through the first enabled clip whose range
  covers the word's midpoint (duplicates of the same dialogue on lower tracks read once); disabled,
  reversed and frame-hold clips contribute nothing.
- Speaker names come from the clip transcripts, so renaming "Speaker 1" in every transcript renames
  it across the sequence.

## Commands

| Command | What it does |
|---|---|
| `transcript.generate` | Transcribe media items (`items`, else the Project selection, else the media of the sequence's audio clips). From a menu or the Text panel with no parameters it opens the Transcribe dialog (Sequence ▸ Transcribe Sequence…), so the language is chosen each time. Params: `model` (default `whisper-base`), `language` (`auto` = detect), `diarize`, `maxSpeakers`. One undo step. |
| `transcript.set` | Store a transcript you bring (JSON: `language`, `speakers`, `words` with `text`/`start`/`end`/`speaker`); it is sorted and made well formed. |
| `transcript.delete` | Remove transcripts. |
| `transcript.inspect` | The sequence transcript: words (index, text, sequence times, speaker, clip), paragraphs, speakers, the word at the playhead. |
| `transcript.search` | Word-index ranges matching a phrase (case and punctuation ignored; the last word may be a prefix). |
| `transcript.select` | Mark In/Out around words `from..=to` (frame-snapped outward) and move the playhead there. |
| `transcript.extract` / `transcript.lift` | Extract (ripple) or lift the words' frames on the targeted tracks. |
| `transcript.renameSpeaker` | Rename a speaker by name (every transcript) or by index in one `item`. |
| `transcript.removeFillers` | Ripple-delete filler words (`fillers`, default um/uh/erm/…; phrases such as "you know" allowed). |
| `transcript.removePauses` | Ripple-delete pauses longer than `minSeconds`, keeping `keepSeconds` of air on both sides. |
| `transcript.createCaptions` | Lay the words out as captions on a new caption track (`maxChars`, `lines`, `minSeconds`, `maxSeconds`, `gapFrames`). `clips: [id]` or `selection: true` takes only the words heard while those clips play; `track: "C1" \| id \| "top"` puts them on an existing track instead (the first one with `"top"`, made when there is none), replacing what they land on. By default (`fit: true`) each caption holds as many characters as fit on one line of the frame in the track's style, so a bigger font gives fewer words per caption and a smaller one more, and each caption lasts as long as its words are spoken; `maxChars` / `lines` override that. |
| `transcript.createCaptionsForSelection` | The selected clips' words as captions onto the top caption track: Sequence ▸ Transcript, the clip's right-click menu, or the Text panel's "Captions for selected clips" button. Re-caption a part you changed without touching the rest. |
| `transcript.models` / `transcript.downloadModel` | List the speech models (size, licence, installed) / download one. |

## Speech recognition

Recognition goes through the `Transcriber` trait (`crates/speech`). The built-in recogniser is
OpenAI's Whisper, run in pure Rust on [candle](https://github.com/huggingface/candle) on the CPU,
with timestamp decoding, language detection and word times from cross-attention alignment (see the
`filmcraft_speech::whisper` module docs). Speakers are labelled by clustering per-chunk MFCC
statistics (`filmcraft_speech::diarize`); no model is involved.

Both are **optional features**, off by default and never built for the web:

- `whisper` (on `filmcraft-speech`, `filmcraft-engine`, and the `filmcraft` / `filmcraft-cli`
  apps, where it also enables downloads): candle inference.
- `download` (`speech-download` on the engine): HTTPS downloads with rustls + RustCrypto and the
  operating system's certificate verifier.

`language: "auto"` (the dialog's default) always detects; the Settings ▸ Media Analysis default
language applies only when no language is passed at all. A finished transcript is announced in
the status bar (a toast: words, speakers, language) since the dialog closes without a trace.

Without `whisper`, and with no recogniser installed, `transcript.generate` and Transcribe Sequence
are disabled, with "speech-to-text is not available in this build" as the reason (`describe`,
`command_list {"enabled_only": true}` and the menus show it); with Automatically transcribe clips
on, `file.import` reports the same reason as a `transcription: …` entry in its `errors`.
Without `speech-download`, `transcript.downloadModel` is disabled the same way. Transcripts can
still be imported with `transcript.set` and edited with every other command. Hosts and tests can
install any recogniser in `Session::transcriber`, which enables transcription in any build.

### Your own recogniser (`FILMCRAFT_TRANSCRIBER`)

Any command on the machine that writes Whisper-style JSON can be the recogniser, in every build
(no feature needed): set `FILMCRAFT_TRANSCRIBER` to its command line before starting FilmCraft or
`filmcraft-cli`, and `transcript.generate` / Transcribe Sequence run it instead of the built-in
model. Typical use: a GPU recogniser with a larger model than the CPU catalogue offers.

```sh
# Apple silicon: mlx-whisper on the GPU with large-v3-turbo
FILMCRAFT_TRANSCRIBER="mlx_whisper --model mlx-community/whisper-large-v3-turbo" filmcraft
```

- **Whisper CLI mode** (no placeholders in the value): FilmCraft appends the audio path and
  `--output-format json --output-dir <dir> --output-name transcript --word-timestamps True
  --verbose False`, plus `--language <code>` when a language is chosen. These are the flags of
  `mlx_whisper` and of OpenAI's `whisper` CLI.
- **Template mode** (any argument contains `{audio}` or `{outdir}`): only `{audio}`, `{outdir}`
  and `{language}` are substituted; an argument with `{language}` (and a `--flag` right before it)
  is dropped when no language is chosen. For tools with other flags:
  `FILMCRAFT_TRANSCRIBER="my-asr --in {audio} --json {outdir}/transcript.json --lang {language}"`.

FilmCraft writes the clip's audio as mono 16 kHz 16-bit WAV in a private temp directory, runs the
command, reads `<dir>/transcript.json` (else the first `.json` there) and deletes the directory.
It understands `language` and `segments[].words[].{word|text, start, end, probability}` (seconds);
a segment without word times gets its text spread evenly over its span. Every number is clamped
to the audio. The words then go through the same voice-activity tightening and speaker labelling
as the built-in recogniser, and the transcript's `source` is `<program>:<model>` (for example
`mlx_whisper:mlx-community/whisper-large-v3-turbo`). Like the built-in recogniser the command runs
inside the command (the window waits); the Transcribe dialog's cancel kills the process. A
failure reports the exit code and the tail of the program's stderr, as does a clean exit that
wrote no transcript (Whisper's CLI skips a file it cannot read, for instance when `ffmpeg` is not
found). The command runs with the program's own directory, `~/.local/bin`, `/opt/homebrew/bin` and
`/usr/local/bin` added to `PATH`, so an app started from the Dock or Finder, whose `PATH` is bare,
still finds `ffmpeg` beside the recogniser. (`filmcraft_speech::external`;
`Session::new` installs it, so hosts that set `Session::transcriber` themselves still win.)

### Models

Weights are **never** bundled or committed. They are downloaded on request into
`<data dir>/models/<id>/` (see `filmcraft_engine::autosave::default_data_dir`), each file pinned to a
revision of OpenAI's Hugging Face repositories and checked against its SHA-256:

| Id | Languages | Licence |
|---|---|---|
| `whisper-tiny` | multilingual | MIT (OpenAI); HF conversion Apache-2.0 |
| `whisper-base` (default) | multilingual | MIT (OpenAI); HF conversion Apache-2.0 |
| `whisper-small` | multilingual | MIT (OpenAI); HF conversion Apache-2.0 |

### Testing

Unit and engine tests use a fake `Transcriber` (`FixedTranscriber`), so CI needs no model. The
end-to-end test `crates/speech/tests/whisper_model.rs` (feature `whisper`) runs only when weights are
in `target/models/<id>/` (or `$FILMCRAFT_MODELS_DIR`) and speech samples (mono 16 kHz f32 with a
reference `.txt`) are in `target/fixtures/speech/`; it reports the word error rate and otherwise
prints SKIPPED. Measured on 2026-10-01: `whisper-tiny`, English, 12.2 % WER over 797 words of
the local speech samples (LibriSpeech read speech and dialogue clips) (about 9.5 minutes for the run in a release build on an
Apple-silicon laptop CPU).

## Limits

- Transcription runs synchronously inside the command (no background job or progress bar yet).
- Track items that refer to a subclip are looked up by the subclip's id, so a transcript made for
  the parent media is not shown through subclip clips yet.
