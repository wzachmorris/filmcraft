//! An external speech recogniser: any command on this machine that turns a WAV file into
//! Whisper-style JSON (`segments[].words[].{word, start, end, probability}`), such as
//! `mlx_whisper` (Apple-silicon GPU), `whisper` (OpenAI's reference CLI) or `whisper-ctranslate2`.
//!
//! FilmCraft's own recogniser (feature `whisper`) runs small models on the CPU. People who already
//! have a faster or larger model installed can point FilmCraft at it with one environment
//! variable and keep the whole Text panel ▸ Transcript workflow:
//!
//! ```text
//! FILMCRAFT_TRANSCRIBER="mlx_whisper --model mlx-community/whisper-large-v3-turbo"
//! ```
//!
//! The value is a command line split on whitespace. Two modes:
//!
//! - **Whisper CLI mode** (no placeholders): the audio path and
//!   `--output-format json --output-dir <dir> --output-name transcript --word-timestamps True
//!   --verbose False` are appended, plus `--language <code>` when the user chose one. This is the
//!   flag set of `mlx_whisper` and OpenAI's `whisper`.
//! - **Template mode** (any argument contains `{audio}` or `{outdir}`): the placeholders
//!   `{audio}`, `{outdir}` and `{language}` are substituted and nothing else is added. An argument
//!   that contains `{language}` is dropped (together with a `--flag` right before it) when no
//!   language was chosen. Use this for any tool with other flags.
//!
//! In both modes the recogniser reads `<outdir>/transcript.json`, else the first `.json` file in
//! the directory. Word times are seconds from the start of the audio. Segments without word
//! times get their text spread evenly over the segment. The result goes through the same
//! voice-activity tightening and speaker labelling as the built-in recogniser.
//!
//! The command runs synchronously inside `transcript.generate`, like the built-in recogniser;
//! the progress callback is polled while it runs, so cancelling kills the process. The program's
//! stderr is kept for the error message when it fails, or when it exits cleanly without writing
//! a transcript (Whisper's CLI does that when, say, it cannot find `ffmpeg`). An app launched
//! from the Dock or Finder has a bare `PATH`, so the child gets the program's own directory,
//! `~/.local/bin`, `/opt/homebrew/bin` and `/usr/local/bin` appended to it ([`child_path`]).
//! Nothing here is bundled, downloaded or shipped: the command is the user's own, named by the
//! user.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use filmcraft_project::{Transcript, Word};
use serde::Deserialize;

use crate::{Options, ProgressFn, SAMPLE_RATE, SpeechError, TICKS_PER_SAMPLE, Transcriber, sample_tick, seconds_tick};

/// The environment variable naming the external command.
pub const ENV: &str = "FILMCRAFT_TRANSCRIBER";
/// Placeholder for the WAV file's path (template mode).
pub const AUDIO: &str = "{audio}";
/// Placeholder for the output directory (template mode).
pub const OUTDIR: &str = "{outdir}";
/// Placeholder for the ISO 639-1 language code (template mode; dropped when none was chosen).
pub const LANGUAGE: &str = "{language}";

/// Longest stderr tail kept for an error message.
const STDERR_TAIL: usize = 2000;
/// How often the running process is checked (and the progress callback asked whether to go on).
const POLL: Duration = Duration::from_millis(100);

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A recogniser that runs an external command (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalTranscriber {
    /// The program (a name on `PATH` or a path).
    pub program: String,
    /// Its arguments, before FilmCraft's own.
    pub args: Vec<String>,
    id: String,
}

impl ExternalTranscriber {
    /// Parse a command line (split on whitespace). `None` when it is blank.
    pub fn parse(spec: &str) -> Option<Self> {
        let mut parts = spec.split_whitespace().map(str::to_string);
        let program = parts.next()?;
        let args: Vec<String> = parts.collect();
        let id = derive_id(&program, &args);
        Some(Self { program, args, id })
    }

    /// The recogniser named by [`ENV`], if the variable is set and not blank.
    pub fn from_env() -> Option<Self> {
        std::env::var(ENV).ok().and_then(|s| Self::parse(&s))
    }

    /// Whether any argument carries a placeholder (template mode).
    fn is_template(&self) -> bool {
        self.args.iter().any(|a| a.contains(AUDIO) || a.contains(OUTDIR))
    }

    /// The full argument list for one run.
    fn command_line(&self, audio: &Path, outdir: &Path, language: Option<&str>) -> Vec<String> {
        let audio_s = audio.to_string_lossy();
        let outdir_s = outdir.to_string_lossy();
        if self.is_template() {
            let mut out: Vec<String> = Vec::with_capacity(self.args.len());
            for a in &self.args {
                if a.contains(LANGUAGE) {
                    match language {
                        Some(l) => out.push(a.replace(LANGUAGE, l).replace(AUDIO, &audio_s).replace(OUTDIR, &outdir_s)),
                        None => {
                            // `--language {language}` without a language: drop the flag too.
                            if out.last().is_some_and(|p| p.starts_with('-')) {
                                out.pop();
                            }
                        }
                    }
                } else {
                    out.push(a.replace(AUDIO, &audio_s).replace(OUTDIR, &outdir_s));
                }
            }
            return out;
        }
        let mut out = self.args.clone();
        out.push(audio_s.into_owned());
        for s in ["--output-format", "json", "--output-dir"] {
            out.push(s.into());
        }
        out.push(outdir_s.into_owned());
        for s in ["--output-name", "transcript", "--word-timestamps", "True", "--verbose", "False"] {
            out.push(s.into());
        }
        if let Some(l) = language {
            out.push("--language".into());
            out.push(l.into());
        }
        out
    }
}

/// `program:model` when the arguments name a model (`--model X` or `--model=X`), else the
/// program's file name.
fn derive_id(program: &str, args: &[String]) -> String {
    let name = Path::new(program).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| program.to_string());
    let mut model = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--model" || a == "-m" {
            model = it.next().cloned();
            break;
        }
        if let Some(m) = a.strip_prefix("--model=") {
            model = Some(m.to_string());
            break;
        }
    }
    match model {
        Some(m) if !m.is_empty() => format!("{name}:{m}"),
        _ => name,
    }
}

impl Transcriber for ExternalTranscriber {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn transcribe(&self, audio: &[f32], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        if !progress(0.0, "Preparing audio") {
            return Err(SpeechError::Cancelled);
        }
        let dir = temp_dir()?;
        let result = self.run_in(&dir, audio, opts, progress);
        // Best effort: the directory holds only our WAV and the tool's output.
        let _ = std::fs::remove_dir_all(&dir);
        result
    }
}

impl ExternalTranscriber {
    fn run_in(&self, dir: &Path, audio: &[f32], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        let wav = dir.join("audio.wav");
        write_wav(&wav, audio)?;
        let outdir = dir.join("out");
        std::fs::create_dir_all(&outdir)?;
        let argv = self.command_line(&wav, &outdir, opts.language.as_deref());
        let stderr_path = dir.join("stderr.txt");
        let stderr = std::fs::File::create(&stderr_path)?;
        let mut child = Command::new(&self.program)
            .args(&argv)
            .env("PATH", child_path(&self.program, std::env::var_os("PATH").as_deref(), std::env::var_os("HOME").as_deref()))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()
            .map_err(|e| SpeechError::Io(format!("cannot run `{}`: {e}", self.program)))?;

        let status_label = format!("Running {}", self.id);
        let started = Instant::now();
        // Rough progress: the fraction rises towards 0.9 on a time scale of a tenth of the audio
        // (plus a couple of seconds for the model to load), which suits GPU recognisers.
        let tau = (audio.len() as f32 / SAMPLE_RATE as f32) * 0.1 + 2.0;
        let status = loop {
            if let Some(st) = child.try_wait()? {
                break st;
            }
            let f = 0.05 + 0.85 * (1.0 - (-started.elapsed().as_secs_f32() / tau).exp());
            if !progress(f.min(0.9), &status_label) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(SpeechError::Cancelled);
            }
            std::thread::sleep(POLL);
        };
        if !status.success() {
            let tail = stderr_tail(&stderr_path);
            let code = status.code().map(|c| c.to_string()).unwrap_or_else(|| "a signal".into());
            return Err(SpeechError::Model(format!(
                "`{}` exited with {code}{}",
                self.program,
                if tail.is_empty() { String::new() } else { format!(": {tail}") }
            )));
        }
        if !progress(0.92, "Reading transcript") {
            return Err(SpeechError::Cancelled);
        }
        let json_path = find_output(&outdir).ok_or_else(|| {
            let tail = stderr_tail(&stderr_path);
            SpeechError::Model(format!(
                "`{}` wrote no .json file to {}{}",
                self.program,
                outdir.display(),
                if tail.is_empty() { String::new() } else { format!(": {tail}") }
            ))
        })?;
        let json = std::fs::read_to_string(&json_path)?;
        let mut t = parse_output(&json, audio.len(), &self.id)?;
        crate::vad::tighten_words(audio, &mut t.words);
        if opts.diarize && !t.words.is_empty() {
            if !progress(0.96, "Labelling speakers") {
                return Err(SpeechError::Cancelled);
            }
            let p = crate::diarize::Params { max_speakers: opts.max_speakers, ..Default::default() };
            crate::diarize::diarize(audio, &mut t, &p);
        }
        progress(1.0, "Done");
        Ok(t)
    }
}

/// The `PATH` for the child: the current one, then the program's own directory (when it was
/// named by path), `~/.local/bin` (pipx, uv), `/opt/homebrew/bin` and `/usr/local/bin` (Homebrew's
/// `ffmpeg`, which Whisper CLIs call to read audio), each once and only when it exists. GUI apps
/// on macOS start with `/usr/bin:/bin:/usr/sbin:/sbin` and nothing of the user's.
pub fn child_path(program: &str, current: Option<&std::ffi::OsStr>, home: Option<&std::ffi::OsStr>) -> std::ffi::OsString {
    let mut dirs: Vec<PathBuf> = current.map(std::env::split_paths).into_iter().flatten().collect();
    let mut add = |d: PathBuf| {
        if d.is_dir() && !dirs.contains(&d) {
            dirs.push(d);
        }
    };
    if let Some(parent) = Path::new(program).parent().filter(|p| !p.as_os_str().is_empty()) {
        add(parent.to_path_buf());
    }
    if let Some(h) = home {
        add(Path::new(h).join(".local").join("bin"));
    }
    add(PathBuf::from("/opt/homebrew/bin"));
    add(PathBuf::from("/usr/local/bin"));
    std::env::join_paths(dirs).unwrap_or_else(|_| current.map(|c| c.to_os_string()).unwrap_or_default())
}

/// A fresh, private directory under the system temp dir.
fn temp_dir() -> Result<PathBuf, SpeechError> {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("filmcraft-speech-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Write mono 16 kHz samples as 16-bit PCM WAV.
fn write_wav(path: &Path, audio: &[f32]) -> Result<(), SpeechError> {
    let data_len = u32::try_from(audio.len().saturating_mul(2)).map_err(|_| SpeechError::Io("audio too long for a WAV file".into()))?;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&data_len.saturating_add(36).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // chunk size
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&SAMPLE_RATE.to_le_bytes())?;
    f.write_all(&(SAMPLE_RATE * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?; // block align
    f.write_all(&16u16.to_le_bytes())?; // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for &s in audio {
        let v = if s.is_finite() { (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16 } else { 0 };
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

/// `transcript.json` in `dir`, else the first `.json` file there.
fn find_output(dir: &Path) -> Option<PathBuf> {
    let preferred = dir.join("transcript.json");
    if preferred.is_file() {
        return Some(preferred);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("json")))
        .collect();
    files.sort();
    files.into_iter().next()
}

/// The last [`STDERR_TAIL`] characters of the file, whitespace collapsed.
fn stderr_tail(path: &Path) -> String {
    let s = std::fs::read_to_string(path).unwrap_or_default();
    let s: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let start = s.char_indices().rev().nth(STDERR_TAIL).map(|(i, _)| i).unwrap_or(0);
    s.get(start..).unwrap_or("").to_string()
}

#[derive(Deserialize)]
struct Output {
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    segments: Vec<Segment>,
}

#[derive(Deserialize)]
struct Segment {
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    end: Option<f64>,
    #[serde(default)]
    text: String,
    #[serde(default)]
    words: Option<Vec<OutWord>>,
}

#[derive(Deserialize)]
struct OutWord {
    #[serde(default, alias = "text")]
    word: String,
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    end: Option<f64>,
    #[serde(default)]
    probability: Option<f64>,
}

/// Whisper-style JSON → a transcript whose words are in order, non-overlapping and within
/// `audio_len` samples. Every number is treated as hostile.
pub fn parse_output(json: &str, audio_len: usize, source: &str) -> Result<Transcript, SpeechError> {
    let out: Output = serde_json::from_str(json).map_err(|e| SpeechError::Model(format!("unreadable transcript JSON: {e}")))?;
    let end_tick = sample_tick(i64::try_from(audio_len).unwrap_or(i64::MAX / 2));
    // Seconds past the audio (plus a little slack) are clamped before they become ticks, so an
    // absurd number can neither overflow nor allocate.
    let max_secs = audio_len as f64 / SAMPLE_RATE as f64 + 1.0;
    let tick = |s: f64| seconds_tick(s.min(max_secs));
    let mut words: Vec<Word> = Vec::new();
    for seg in out.segments {
        match seg.words {
            Some(ws) if !ws.is_empty() => {
                for w in ws {
                    let text = w.word.trim();
                    if text.is_empty() {
                        continue;
                    }
                    let (Some(start), Some(end)) = (secs(w.start), secs(w.end)) else { continue };
                    let confidence = w.probability.filter(|p| p.is_finite()).map(|p| p.clamp(0.0, 1.0) as f32).unwrap_or(1.0);
                    words.push(Word { text: text.to_string(), start: tick(start), end: tick(end), speaker: None, confidence });
                }
            }
            _ => {
                // No word times: spread the segment's words evenly over its span.
                let (Some(start), Some(end)) = (secs(seg.start), secs(seg.end)) else { continue };
                let parts: Vec<&str> = seg.text.split_whitespace().collect();
                if parts.is_empty() || end <= start {
                    continue;
                }
                let (start, end) = (start.min(max_secs), end.min(max_secs));
                if end <= start {
                    continue;
                }
                let step = (end - start) / parts.len() as f64;
                for (i, p) in parts.iter().enumerate() {
                    let a = start + step * i as f64;
                    words.push(Word { text: (*p).to_string(), start: tick(a), end: tick(a + step), speaker: None, confidence: 1.0 });
                }
            }
        }
    }
    // Order, keep inside the audio, make every word non-empty and non-overlapping.
    words.sort_by_key(|w| w.start);
    let mut prev_end = filmcraft_time::Tick(0);
    for w in &mut words {
        if w.start < prev_end {
            w.start = prev_end;
        }
        if w.end <= w.start {
            w.end = filmcraft_time::Tick(w.start.0.saturating_add(TICKS_PER_SAMPLE));
        }
        if w.end > end_tick {
            w.end = end_tick;
        }
        prev_end = w.end;
    }
    words.retain(|w| w.start < end_tick && w.end > w.start);
    let language = out.language.map(|l| l.trim().to_lowercase()).filter(|l| !l.is_empty()).unwrap_or_else(|| "en".into());
    Ok(Transcript { language, source: source.to_string(), speakers: Vec::new(), words })
}

/// A finite, non-negative number of seconds.
fn secs(v: Option<f64>) -> Option<f64> {
    v.filter(|s| s.is_finite()).map(|s| s.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two segments of real `mlx_whisper --word-timestamps True` output (Spanish), trimmed.
    const MLX: &str = r#"{"text": " Zach, ¿por qué estás colgando la ropa?", "language": "es", "segments": [
      {"id": 0, "seek": 0, "start": 0.0, "end": 1.2, "text": " Zach, ¿por qué", "tokens": [1, 2], "avg_logprob": -0.3,
       "words": [{"word": " Zach,", "start": 0.0, "end": 0.28, "probability": 0.2156},
                 {"word": " ¿por", "start": 0.48, "end": 0.62, "probability": 0.89},
                 {"word": " qué", "start": 0.62, "end": 0.74, "probability": 0.97}]},
      {"id": 1, "seek": 0, "start": 1.2, "end": 2.0, "text": " estás colgando", "tokens": [3],
       "words": [{"word": " estás", "start": 0.74, "end": 0.92, "probability": 0.96},
                 {"word": " colgando", "start": 0.92, "end": 1.4, "probability": 0.99}]}]}"#;

    #[test]
    fn parses_a_command_line_and_derives_the_id() {
        let t = ExternalTranscriber::parse("  mlx_whisper   --model mlx-community/whisper-large-v3-turbo ").unwrap();
        assert_eq!(t.program, "mlx_whisper");
        assert_eq!(t.args, vec!["--model", "mlx-community/whisper-large-v3-turbo"]);
        assert_eq!(t.id(), "mlx_whisper:mlx-community/whisper-large-v3-turbo");
        assert_eq!(ExternalTranscriber::parse("/opt/bin/whisper --model=small").unwrap().id(), "whisper:small");
        assert_eq!(ExternalTranscriber::parse("whisper").unwrap().id(), "whisper");
        assert!(ExternalTranscriber::parse("   ").is_none());
    }

    #[test]
    fn whisper_cli_mode_appends_the_standard_flags() {
        let t = ExternalTranscriber::parse("mlx_whisper --model m").unwrap();
        let argv = t.command_line(Path::new("/tmp/a.wav"), Path::new("/tmp/out"), Some("es"));
        assert_eq!(
            argv,
            vec![
                "--model",
                "m",
                "/tmp/a.wav",
                "--output-format",
                "json",
                "--output-dir",
                "/tmp/out",
                "--output-name",
                "transcript",
                "--word-timestamps",
                "True",
                "--verbose",
                "False",
                "--language",
                "es"
            ]
        );
        let argv = t.command_line(Path::new("/tmp/a.wav"), Path::new("/tmp/out"), None);
        assert!(!argv.iter().any(|a| a == "--language"));
    }

    #[test]
    fn template_mode_substitutes_and_drops_a_missing_language() {
        let t = ExternalTranscriber::parse("tool -i {audio} -o {outdir}/transcript.json --lang {language} --fast").unwrap();
        let argv = t.command_line(Path::new("a.wav"), Path::new("out"), Some("pt"));
        assert_eq!(argv, vec!["-i", "a.wav", "-o", "out/transcript.json", "--lang", "pt", "--fast"]);
        let argv = t.command_line(Path::new("a.wav"), Path::new("out"), None);
        assert_eq!(argv, vec!["-i", "a.wav", "-o", "out/transcript.json", "--fast"]);
    }

    #[test]
    fn parses_mlx_whisper_output() {
        let t = parse_output(MLX, SAMPLE_RATE as usize * 10, "mlx").unwrap();
        assert_eq!(t.language, "es");
        assert_eq!(t.source, "mlx");
        let texts: Vec<&str> = t.words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["Zach,", "¿por", "qué", "estás", "colgando"]);
        assert_eq!(t.words[0].start, seconds_tick(0.0));
        assert_eq!(t.words[0].end, seconds_tick(0.28));
        assert!((t.words[0].confidence - 0.2156).abs() < 1e-4);
        t.check().unwrap();
    }

    #[test]
    fn hostile_numbers_never_escape() {
        let json = r#"{"segments": [
          {"start": 0, "end": 1, "text": "a b", "words": [
             {"word": "late", "start": 5.0, "end": 4.0},
             {"word": "neg", "start": -3.0, "end": -1.0},
             {"word": "huge", "start": 1e300, "end": 1e301},
             {"word": "  ", "start": 0.1, "end": 0.2},
             {"word": "ok", "start": 0.5, "end": 0.6, "probability": 7.0},
             {"word": "nostart", "end": 0.6}]},
          {"start": 2.0, "end": 3.0, "text": "spread these words"},
          {"start": 4.0, "end": 3.0, "text": "inverted segment"},
          {"text": "no times at all"}]}"#;
        let t = parse_output(json, SAMPLE_RATE as usize * 6, "x").unwrap();
        t.check().unwrap();
        let texts: Vec<&str> = t.words.iter().map(|w| w.text.as_str()).collect();
        // "huge" starts past the audio and is dropped; the rest are ordered and non-overlapping.
        assert_eq!(texts, vec!["neg", "ok", "spread", "these", "words", "late"]);
        assert!(t.words.iter().all(|w| w.end > w.start && w.confidence <= 1.0));
        assert_eq!(t.language, "en");
        assert!(parse_output("not json", 100, "x").is_err());
        assert!(parse_output("42", 100, "x").is_err());
        assert_eq!(parse_output("{}", 100, "x").unwrap().words.len(), 0);
    }

    #[test]
    fn writes_a_readable_wav() {
        let dir = temp_dir().unwrap();
        let p = dir.join("a.wav");
        write_wav(&p, &[0.0, 0.5, -0.5, 1.5, f32::NAN]).unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(&b[8..16], b"WAVEfmt ");
        assert_eq!(b.len(), 44 + 10);
        assert_eq!(i16::from_le_bytes([b[46], b[47]]), 16384); // 0.5
        assert_eq!(i16::from_le_bytes([b[50], b[51]]), i16::MAX); // clamped
        assert_eq!(i16::from_le_bytes([b[52], b[53]]), 0); // NaN
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    fn script(body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir().unwrap();
        let p = dir.join("tool.sh");
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    #[cfg(unix)]
    #[test]
    fn runs_a_command_and_reads_its_json() {
        let fixture = temp_dir().unwrap().join("fixture.json");
        std::fs::write(&fixture, MLX).unwrap();
        // Whisper CLI mode: the tool sees the audio path first and `--output-dir <dir>` later.
        let tool = script(&format!(
            "test -s \"$1\" || exit 3\nwhile [ $# -gt 0 ]; do if [ \"$1\" = --output-dir ]; then cp {} \"$2/transcript.json\"; fi; shift; done",
            fixture.display()
        ));
        let t = ExternalTranscriber::parse(&tool.to_string_lossy()).unwrap();
        let audio = vec![0.0f32; SAMPLE_RATE as usize * 2];
        let mut log = Vec::new();
        let tr = t
            .transcribe(&audio, &Options { diarize: false, ..Default::default() }, &mut |f, s| {
                log.push((f, s.to_string()));
                true
            })
            .unwrap();
        assert_eq!(tr.words.len(), 5);
        assert_eq!(tr.language, "es");
        assert_eq!(log.last().map(|l| l.0), Some(1.0));
        assert!(log.iter().any(|l| l.1.starts_with("Running tool.sh")));
    }

    #[cfg(unix)]
    #[test]
    fn a_failing_command_is_an_error_with_its_stderr() {
        let tool = script("echo 'model not found' >&2; exit 7");
        let t = ExternalTranscriber::parse(&tool.to_string_lossy()).unwrap();
        let err = t.transcribe(&[0.0; 1600], &Options::default(), &mut |_, _| true).unwrap_err();
        assert!(matches!(err, SpeechError::Model(ref m) if m.contains("exited with 7") && m.contains("model not found")), "{err}");
        let err = ExternalTranscriber::parse("/definitely/not/here").unwrap().transcribe(&[0.0; 1600], &Options::default(), &mut |_, _| true).unwrap_err();
        assert!(matches!(err, SpeechError::Io(_)), "{err}");
        // Whisper's CLI prints the traceback and exits 0 when it skips a file: that tail is the
        // only clue, so it rides along.
        let tool = script("echo 'FileNotFoundError: ffmpeg' >&2; exit 0");
        let err = ExternalTranscriber::parse(&tool.to_string_lossy()).unwrap().transcribe(&[0.0; 1600], &Options::default(), &mut |_, _| true).unwrap_err();
        assert!(matches!(err, SpeechError::Model(ref m) if m.contains("no .json") && m.contains("FileNotFoundError: ffmpeg")), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn the_child_sees_the_users_tool_directories_on_path() {
        // A program named by path finds its siblings, and `~/.local/bin` is appended; a bare
        // GUI PATH stays in front, nothing is listed twice and missing directories are skipped.
        let home = temp_dir().unwrap();
        let local = home.join(".local").join("bin");
        std::fs::create_dir_all(&local).unwrap();
        let tool = script("exit 0");
        let tool_dir = tool.parent().unwrap().to_path_buf();
        let bare = std::ffi::OsString::from("/usr/bin:/bin");
        let p = child_path(&tool.to_string_lossy(), Some(&bare), Some(home.as_os_str()));
        let dirs: Vec<PathBuf> = std::env::split_paths(&p).collect();
        assert_eq!(dirs.first(), Some(&PathBuf::from("/usr/bin")));
        assert!(dirs.contains(&tool_dir), "{dirs:?}");
        assert!(dirs.contains(&local), "{dirs:?}");
        assert!(!dirs.contains(&PathBuf::from("/definitely/not/here")));
        let again = child_path(&tool.to_string_lossy(), Some(&p), Some(home.as_os_str()));
        assert_eq!(std::env::split_paths(&again).count(), dirs.len(), "nothing is added twice");
        // a bare name on PATH adds no program directory; no PATH at all still yields the extras
        let p = child_path("mlx_whisper", None, Some(home.as_os_str()));
        let dirs: Vec<PathBuf> = std::env::split_paths(&p).collect();
        assert!(dirs.contains(&local) && !dirs.iter().any(|d| d.as_os_str().is_empty()), "{dirs:?}");
        let _ = std::fs::remove_dir_all(home);
    }

    #[cfg(unix)]
    #[test]
    fn cancelling_kills_the_command() {
        let tool = script("sleep 30");
        let t = ExternalTranscriber::parse(&tool.to_string_lossy()).unwrap();
        let started = Instant::now();
        let mut calls = 0;
        let err = t
            .transcribe(&[0.0; 1600], &Options::default(), &mut |_, _| {
                calls += 1;
                calls < 3
            })
            .unwrap_err();
        assert_eq!(err, SpeechError::Cancelled);
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
