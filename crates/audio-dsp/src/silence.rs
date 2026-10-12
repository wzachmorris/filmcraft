//! Silence (pause) detection from the level envelope of a recording: no model, no network, one
//! deterministic measurement of how loud the sound is, so the same recording always gives the
//! same cuts.
//!
//! The signal is cut into short windows (10 ms by default) and the RMS level of each window, in
//! dB, is the **envelope**. A window is *silent* when its level sits below a threshold and *sound*
//! when it rises above the threshold plus a hysteresis band, so a level that hovers around the
//! threshold does not chatter. Runs of silent windows become pauses; a pause has to last
//! `min_silence` to count, a blip of sound shorter than `min_sound` inside a pause (a click, a lip
//! smack) does not end it, and every pause is shrunk by `pad` at both ends so the decay of the
//! word before it and the attack of the word after it are never cut.
//!
//! The threshold is chosen from the recording itself when none is given: the quietest tenth of
//! the windows is the **noise floor**, the loudest tenth is the **speech level**, and the
//! threshold sits 25 dB under the speech level but never closer than 6 dB to the floor (and when
//! the two are within 10 dB of each other there is no pause to find). That
//! follows the recording's own dynamics (a studio voice and a phone recording on a street get
//! different thresholds) instead of one fixed number.

/// How to find the pauses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SilenceOptions {
    /// Envelope window, milliseconds (the time resolution of every edge).
    pub window_ms: f64,
    /// Level under which a window is silent, dBFS; `None` picks one from the recording.
    pub threshold_db: Option<f64>,
    /// A window has to rise this far above the threshold to end a pause.
    pub hysteresis_db: f64,
    /// Shortest pause reported, milliseconds.
    pub min_silence_ms: f64,
    /// Sound shorter than this inside a pause does not end it, milliseconds.
    pub min_sound_ms: f64,
    /// Air left on both sides of every pause, milliseconds.
    pub pad_ms: f64,
}

impl Default for SilenceOptions {
    fn default() -> Self {
        Self { window_ms: 10.0, threshold_db: None, hysteresis_db: 4.0, min_silence_ms: 300.0, min_sound_ms: 60.0, pad_ms: 40.0 }
    }
}

/// One pause, in sample frames of the analysed signal (`end` exclusive).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SilenceSpan {
    pub start: usize,
    pub end: usize,
}

impl SilenceSpan {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What the detector found and the levels it decided with.
#[derive(Clone, Debug, PartialEq)]
pub struct SilenceReport {
    pub spans: Vec<SilenceSpan>,
    /// The threshold used, dBFS.
    pub threshold_db: f64,
    /// Level of the quietest tenth of the windows, dBFS.
    pub noise_floor_db: f64,
    /// Level of the loudest tenth of the windows, dBFS.
    pub speech_db: f64,
    /// Envelope windows analysed.
    pub windows: usize,
    /// Sample frames per window.
    pub window_frames: usize,
}

/// Level of a window with no energy at all.
pub const FLOOR_DB: f32 = -120.0;

/// Builds the level envelope of a stream fed in blocks of any size: planar channels in, one RMS
/// level in dB per window out. Windows never straddle blocks differently depending on how the
/// stream is cut, so the result is block-invariant.
#[derive(Clone, Debug)]
pub struct Envelope {
    window: usize,
    sum_sq: f64,
    count: usize,
    levels: Vec<f32>,
}

impl Envelope {
    /// `window_frames` sample frames per window (at least 1).
    pub fn new(window_frames: usize) -> Self {
        Self { window: window_frames.max(1), sum_sq: 0.0, count: 0, levels: Vec::new() }
    }

    pub fn window_frames(&self) -> usize {
        self.window
    }

    /// Feed a block (one slice per channel; channels shorter than the first are read as far as
    /// they go). Each window's level is the RMS over all channels and frames in it.
    pub fn push(&mut self, channels: &[&[f32]]) {
        let frames = channels.first().map_or(0, |c| c.len());
        let n_ch = channels.len().max(1) as f64;
        for i in 0..frames {
            let mut e = 0.0f64;
            for c in channels {
                if let Some(x) = c.get(i) {
                    e += f64::from(*x) * f64::from(*x);
                }
            }
            self.sum_sq += e / n_ch;
            self.count += 1;
            if self.count >= self.window {
                self.flush();
            }
        }
    }

    fn flush(&mut self) {
        if self.count > 0 {
            let ms = self.sum_sq / self.count as f64;
            let db = if ms > 1e-12 { (10.0 * ms.log10()) as f32 } else { FLOOR_DB };
            self.levels.push(db.max(FLOOR_DB));
        }
        self.sum_sq = 0.0;
        self.count = 0;
    }

    /// The levels so far, including a last partial window.
    pub fn finish(mut self) -> Vec<f32> {
        self.flush();
        self.levels
    }
}

/// Level envelope of a whole signal (see [`Envelope`]).
pub fn envelope_db(channels: &[&[f32]], window_frames: usize) -> Vec<f32> {
    let mut e = Envelope::new(window_frames);
    e.push(channels);
    e.finish()
}

fn percentile(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return FLOOR_DB;
    }
    let pos = (q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted.get(pos.min(sorted.len() - 1)).copied().unwrap_or(FLOOR_DB)
}

/// Noise floor, speech level and the threshold chosen from an envelope (see the module docs).
pub fn auto_threshold(levels: &[f32]) -> (f64, f64, f64) {
    let mut sorted: Vec<f32> = levels.iter().copied().filter(|x| x.is_finite()).collect();
    sorted.sort_by(f32::total_cmp);
    let floor = f64::from(percentile(&sorted, 0.10));
    let speech = f64::from(percentile(&sorted, 0.90));
    // a recording with no quiet stretch at all (a tone, a loud crowd) has nothing to find: a
    // threshold under the floor reports nothing instead of calling everything silence
    let thr = if speech - floor < 10.0 { floor - 1.0 } else { (speech - 25.0).max(floor + 6.0) };
    (floor, speech, thr)
}

/// Find the pauses in an envelope made with `window_frames`-frame windows at `sample_rate`.
pub fn detect_envelope(levels: &[f32], window_frames: usize, sample_rate: u32, opts: &SilenceOptions) -> SilenceReport {
    let window = window_frames.max(1);
    let sr = f64::from(sample_rate.max(1));
    let to_windows = |ms: f64| -> usize {
        let frames = (ms.max(0.0) / 1000.0 * sr).round();
        if frames.is_finite() { (frames as usize).div_ceil(window) } else { 0 }
    };
    let (floor, speech, auto) = auto_threshold(levels);
    let thr = opts.threshold_db.filter(|t| t.is_finite()).unwrap_or(auto);
    let hi = thr + opts.hysteresis_db.max(0.0);
    let min_silence = to_windows(opts.min_silence_ms).max(1);
    let min_sound = to_windows(opts.min_sound_ms);
    let pad_frames = {
        let f = (opts.pad_ms.max(0.0) / 1000.0 * sr).round();
        if f.is_finite() { f as usize } else { 0 }
    };

    // 1. classify every window with hysteresis
    let mut silent = Vec::with_capacity(levels.len());
    let mut in_silence = levels.first().is_some_and(|l| f64::from(*l) < thr);
    for &l in levels {
        let l = f64::from(l);
        if in_silence {
            if l > hi {
                in_silence = false;
            }
        } else if l < thr {
            in_silence = true;
        }
        silent.push(in_silence);
    }

    // 2. runs of silent windows; a short burst of sound between two runs is swallowed
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < silent.len() {
        if silent.get(i).copied().unwrap_or(false) {
            let start = i;
            while i < silent.len() && silent.get(i).copied().unwrap_or(false) {
                i += 1;
            }
            match runs.last_mut() {
                Some(last) if start.saturating_sub(last.1) <= min_sound && min_sound > 0 => last.1 = i,
                _ => runs.push((start, i)),
            }
        } else {
            i += 1;
        }
    }

    // 3. long enough, shrunk by the pad
    let spans = runs
        .into_iter()
        .filter(|(a, b)| b.saturating_sub(*a) >= min_silence)
        .filter_map(|(a, b)| {
            let start = a.saturating_mul(window).saturating_add(pad_frames);
            let end = b.saturating_mul(window).saturating_sub(pad_frames);
            (end > start).then_some(SilenceSpan { start, end })
        })
        .collect();
    SilenceReport { spans, threshold_db: thr, noise_floor_db: floor, speech_db: speech, windows: levels.len(), window_frames: window }
}

/// Find the pauses in a signal (planar channels at `sample_rate`).
pub fn detect(channels: &[&[f32]], sample_rate: u32, opts: &SilenceOptions) -> SilenceReport {
    let window = ((opts.window_ms.max(0.1) / 1000.0) * f64::from(sample_rate.max(1))).round().max(1.0) as usize;
    let levels = envelope_db(channels, window);
    detect_envelope(&levels, window, sample_rate, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    /// `secs` of a tone at `db`, or digital silence when `db` is None.
    fn piece(secs: f64, db: Option<f64>) -> Vec<f32> {
        let n = (secs * f64::from(SR)) as usize;
        match db {
            None => vec![0.0; n],
            Some(db) => {
                let amp = 10f64.powf(db / 20.0) * std::f64::consts::SQRT_2; // RMS = db
                (0..n).map(|i| (amp * (i as f64 * 2.0 * std::f64::consts::PI * 220.0 / f64::from(SR)).sin()) as f32).collect()
            }
        }
    }

    fn signal(parts: &[(f64, Option<f64>)]) -> Vec<f32> {
        parts.iter().flat_map(|(s, d)| piece(*s, *d)).collect()
    }

    fn secs(frames: usize) -> f64 {
        frames as f64 / f64::from(SR)
    }

    #[test]
    fn finds_a_pause_between_two_words_with_the_pad_kept() {
        let x = signal(&[(1.0, Some(-20.0)), (0.8, None), (1.0, Some(-20.0))]);
        let r = detect(&[&x], SR, &SilenceOptions::default());
        assert_eq!(r.spans.len(), 1, "{r:?}");
        let s = r.spans[0];
        assert!((secs(s.start) - 1.04).abs() < 0.015, "start {}", secs(s.start));
        assert!((secs(s.end) - 1.76).abs() < 0.015, "end {}", secs(s.end));
        assert!(r.threshold_db < -20.0 && r.threshold_db > -100.0, "{r:?}");
    }

    #[test]
    fn short_pauses_and_the_ends_are_not_reported() {
        let x = signal(&[(0.5, Some(-20.0)), (0.2, None), (0.5, Some(-20.0))]);
        let r = detect(&[&x], SR, &SilenceOptions::default());
        assert!(r.spans.is_empty(), "{r:?}");
        let r = detect(&[&x], SR, &SilenceOptions { min_silence_ms: 150.0, ..Default::default() });
        assert_eq!(r.spans.len(), 1, "{r:?}");
    }

    #[test]
    fn a_click_inside_a_pause_does_not_split_it() {
        let x = signal(&[(1.0, Some(-20.0)), (0.4, None), (0.03, Some(-20.0)), (0.4, None), (1.0, Some(-20.0))]);
        let r = detect(&[&x], SR, &SilenceOptions::default());
        assert_eq!(r.spans.len(), 1, "{r:?}");
        assert!(secs(r.spans[0].len()) > 0.7, "{r:?}");
        // a real word (200 ms) does split it
        let x = signal(&[(1.0, Some(-20.0)), (0.4, None), (0.2, Some(-20.0)), (0.4, None), (1.0, Some(-20.0))]);
        let r = detect(&[&x], SR, &SilenceOptions::default());
        assert_eq!(r.spans.len(), 2, "{r:?}");
    }

    #[test]
    fn the_threshold_follows_the_recording() {
        // room tone at -50 dB between words at -20: the pause is still found
        let x = signal(&[(1.0, Some(-20.0)), (0.8, Some(-50.0)), (1.0, Some(-20.0))]);
        let r = detect(&[&x], SR, &SilenceOptions::default());
        assert_eq!(r.spans.len(), 1, "{r:?}");
        assert!((r.noise_floor_db + 50.0).abs() < 1.0 && (r.speech_db + 20.0).abs() < 1.0, "{r:?}");
        // a fixed threshold under the room tone finds nothing
        let r = detect(&[&x], SR, &SilenceOptions { threshold_db: Some(-60.0), ..Default::default() });
        assert!(r.spans.is_empty(), "{r:?}");
        // and a recording with no pause at all reports none
        let x = signal(&[(3.0, Some(-20.0))]);
        assert!(detect(&[&x], SR, &SilenceOptions::default()).spans.is_empty());
    }

    #[test]
    fn the_envelope_is_block_invariant() {
        let x = signal(&[(0.5, Some(-20.0)), (0.5, None), (0.5, Some(-30.0))]);
        let whole = envelope_db(&[&x], 480);
        let mut e = Envelope::new(480);
        for chunk in x.chunks(1234) {
            e.push(&[chunk]);
        }
        assert_eq!(e.finish(), whole);
        // stereo with one dead channel sits 3 dB under the mono level
        let zero = vec![0.0f32; x.len()];
        let st = envelope_db(&[&x, &zero], 480);
        assert!((st[10] - (whole[10] - 3.01)).abs() < 0.05, "{} vs {}", st[10], whole[10]);
    }

    #[test]
    fn hostile_options_do_not_panic() {
        let x = signal(&[(0.3, Some(-20.0)), (0.5, None), (0.3, Some(-20.0))]);
        for o in [
            SilenceOptions { window_ms: 0.0, ..Default::default() },
            SilenceOptions { window_ms: f64::NAN, min_silence_ms: f64::INFINITY, ..Default::default() },
            SilenceOptions { pad_ms: 1e12, ..Default::default() },
            SilenceOptions { threshold_db: Some(f64::NAN), hysteresis_db: -5.0, ..Default::default() },
            SilenceOptions { min_sound_ms: -1.0, ..Default::default() },
        ] {
            let _ = detect(&[&x], SR, &o);
            let _ = detect(&[&x], 0, &o);
        }
        let _ = detect(&[], SR, &SilenceOptions::default());
        let _ = detect_envelope(&[], 0, 0, &SilenceOptions::default());
    }
}
