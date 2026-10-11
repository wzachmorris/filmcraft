//! Time stretching that keeps the pitch: WSOLA (waveform-similarity overlap-add). The source is
//! cut into overlapping Hann-windowed grains (24 ms, half overlapping) taken `speed` times as far
//! apart as they are laid down, each slid by up to ±9 ms to continue the previous grain's
//! waveform. Speech at 2× reads as fast talking, not as a chipmunk; at ½× it is slow talking.
//!
//! [`Stretcher`] pulls its source on demand, so the sequence mixer (a clip with Maintain Audio
//! Pitch) and the shuttle (the whole mix played faster) share it. Reads of the source stay
//! contiguous; a jump in the output position (a seek) restarts the grain chain there.

/// Grain length in seconds: short enough to follow speech, long enough to hold two periods of a
/// low voice.
const GRAIN_SECONDS: f64 = 0.024;
/// How far (as a fraction of the grain) a grain may slide to line up with the previous one.
const SEARCH_FRACTION: f64 = 0.375;
/// Coarse step of the alignment search; the best coarse offset is then refined by ±(step − 1).
const SEARCH_STEP: i64 = 4;
/// Speeds outside this range play at 1.
pub const SPEED_RANGE: (f64, f64) = (1.0 / 64.0, 64.0);
/// Most channels a pull is believed about.
const MAX_CHANNELS: usize = 64;

/// `speed` clamped to [`SPEED_RANGE`]; anything else (NaN, ∞) is 1.
pub fn clamp_speed(speed: f64) -> f64 {
    if speed.is_finite() { speed.clamp(SPEED_RANGE.0, SPEED_RANGE.1) } else { 1.0 }
}

/// One stretched stream. Output sample `out_origin` is where source sample `src_origin` lands;
/// output sample `o` draws on source `src_origin + (o - out_origin) × speed`.
pub struct Stretcher {
    speed: f64,
    out_origin: i64,
    src_origin: i64,
    grain: usize,
    hop: i64,
    tol: i64,
    window: Vec<f32>,
    /// Channels of the source (0 until the first pull answered).
    ch: usize,
    /// Source samples read so far and still needed, planar; `src_base` is the index of the first,
    /// `next` the source sample the next contiguous read starts at.
    src: Vec<Vec<f32>>,
    src_base: i64,
    next: i64,
    /// The next grain to lay down: it starts at output sample `out_origin + k × hop`.
    k: i64,
    /// Where the previous grain was actually taken from.
    prev: Option<i64>,
    /// Output laid down so far, planar, from output sample `acc_base`.
    acc: Vec<Vec<f32>>,
    acc_base: i64,
    /// The output position the next `process` should bring.
    expect: i64,
    template: Vec<f32>,
    mono: Vec<f32>,
}

impl Stretcher {
    /// A stream at `speed` (clamped to [`SPEED_RANGE`]) for a source at `sample_rate` (0 is taken
    /// as 48 kHz); output sample `out_origin` maps to source sample `src_origin`.
    pub fn new(speed: f64, sample_rate: u32, out_origin: i64, src_origin: i64) -> Self {
        let speed = clamp_speed(speed);
        let sr = if sample_rate == 0 { 48_000 } else { sample_rate };
        // even grain length so two half-overlapping periodic Hann windows sum to exactly one
        let grain = (((sr as f64 * GRAIN_SECONDS) / 2.0).round().max(2.0) as usize) * 2;
        let hop = (grain / 2) as i64;
        let tol = ((grain as f64 * SEARCH_FRACTION) as i64).max(SEARCH_STEP);
        let window = (0..grain).map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / grain as f64).cos() as f32).collect();
        Self {
            speed,
            out_origin,
            src_origin,
            grain,
            hop,
            tol,
            window,
            ch: 0,
            src: Vec::new(),
            src_base: src_origin,
            next: src_origin,
            k: 0,
            prev: None,
            acc: Vec::new(),
            acc_base: 0,
            expect: out_origin,
            template: Vec::new(),
            mono: Vec::new(),
        }
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// `(out_origin, src_origin)` as given to [`Stretcher::new`].
    pub fn origins(&self) -> (i64, i64) {
        (self.out_origin, self.src_origin)
    }

    /// Whether the stream is a plain pass-through (speed exactly 1).
    pub fn is_identity(&self) -> bool {
        (self.speed - 1.0).abs() < 1e-9
    }

    /// Output samples `out .. out + n`, planar (as many channels as the source answers with; none
    /// until it has). `pull(pos, len)` returns `len` source samples from `pos`, planar (shorter or
    /// missing channels read as silence).
    pub fn process<P>(&mut self, out: i64, n: usize, mut pull: P) -> Vec<Vec<f32>>
    where
        P: FnMut(i64, usize) -> Vec<Vec<f32>>,
    {
        if n == 0 {
            return vec![Vec::new(); self.ch];
        }
        if self.is_identity() {
            let mut v = pull(self.src_origin.saturating_add(out.saturating_sub(self.out_origin)), n);
            v.truncate(MAX_CHANNELS);
            for c in &mut v {
                c.resize(n, 0.0);
            }
            return v;
        }
        let o = out.saturating_sub(self.out_origin);
        let acc_len = self.acc.first().map_or(0, Vec::len) as i64;
        let acc_end = self.acc_base.saturating_add(acc_len);
        let fresh = self.prev.is_none();
        if fresh || out != self.expect && (o < self.acc_base || o > acc_end.saturating_add(self.grain as i64)) {
            // the start, or a seek: start over at `o`, one grain early so the first samples
            // handed out get both their halves
            for c in &mut self.src {
                c.clear();
            }
            for c in &mut self.acc {
                c.clear();
            }
            self.prev = None;
            self.k = o.div_euclid(self.hop).saturating_sub(1);
            self.acc_base = self.k.saturating_mul(self.hop);
            let s = self.source_of(self.k);
            self.src_base = s;
            self.next = s;
        } else if o > self.acc_base {
            // the previous block (or a short gap the device played as silence): drop what is done
            let drop = ((o - self.acc_base) as usize).min(acc_len as usize);
            for c in &mut self.acc {
                c.drain(..drop.min(c.len()));
            }
            self.acc_base = o;
        }
        self.expect = out.saturating_add(n as i64);
        let end = o.saturating_add(n as i64);
        // every sample before `k × hop` is final once grain k − 1 is down (counted, so saturation
        // near the i64 ceiling cannot spin)
        let mut budget = (n as i64 / self.hop).saturating_add(3);
        while self.k.saturating_mul(self.hop) < end && budget > 0 {
            self.lay_grain(&mut pull);
            budget -= 1;
        }
        let from = (o - self.acc_base).max(0) as usize;
        let out: Vec<Vec<f32>> = self.acc.iter().map(|c| (0..n).map(|i| c.get(from + i).copied().unwrap_or(0.0)).collect()).collect();
        for c in &mut self.acc {
            let drop = (from + n).min(c.len());
            c.drain(..drop);
        }
        self.acc_base = end;
        // source samples no later grain or template can reach are done with
        let keep_from = self.source_of(self.k).saturating_sub(self.tol).min(self.prev.map_or(i64::MAX, |p| p.saturating_add(self.hop)));
        if keep_from > self.src_base {
            let drop = (keep_from - self.src_base) as usize;
            for c in &mut self.src {
                c.drain(..drop.min(c.len()));
            }
            self.src_base = keep_from;
        }
        out
    }

    /// The nominal source sample grain `k` is taken from (never before `src_origin`).
    fn source_of(&self, k: i64) -> i64 {
        let off = (k as f64 * self.hop as f64 * self.speed).round().clamp(i64::MIN as f64, i64::MAX as f64) as i64;
        self.src_origin.saturating_add(off).max(self.src_origin)
    }

    /// Make sure source samples `lo..hi` are in `src`, reading forward from `next`.
    fn ensure<P: FnMut(i64, usize) -> Vec<Vec<f32>>>(&mut self, pull: &mut P, lo: i64, hi: i64) {
        if lo < self.src_base {
            // only right after a seek that landed just ahead of the alignment window: read back once
            let count = (self.src_base - lo) as usize;
            let head = self.read(pull, lo, count);
            if self.ch > 0 {
                for (c, h) in self.src.iter_mut().zip(head) {
                    let mut v = h;
                    v.append(c);
                    *c = v;
                }
                self.src_base = lo;
            }
        }
        if hi > self.next {
            let count = (hi - self.next) as usize;
            let fresh = self.read(pull, self.next, count);
            if self.ch > 0 {
                if self.src.len() != self.ch {
                    self.src = vec![Vec::new(); self.ch];
                }
                for (c, f) in self.src.iter_mut().zip(fresh) {
                    c.extend(f);
                }
                self.next = hi;
            }
        }
    }

    /// `count` source samples from `pos`, exactly `ch` channels of `count` each; the first answer
    /// fixes the channel count.
    fn read<P: FnMut(i64, usize) -> Vec<Vec<f32>>>(&mut self, pull: &mut P, pos: i64, count: usize) -> Vec<Vec<f32>> {
        let mut v = pull(pos, count);
        v.truncate(MAX_CHANNELS);
        if self.ch == 0 {
            self.ch = v.len();
            self.acc = vec![Vec::new(); self.ch];
            self.src = vec![Vec::new(); self.ch];
        }
        v.resize(self.ch, Vec::new());
        for c in &mut v {
            c.resize(count, 0.0);
        }
        v
    }

    /// Mono (channel sum) of source samples `pos..pos + len` into `out`.
    fn mono_of(&self, pos: i64, len: usize, out: &mut Vec<f32>) {
        out.clear();
        let first = (pos - self.src_base).max(0) as usize;
        for f in 0..len {
            out.push(self.src.iter().map(|c| c.get(first + f).copied().unwrap_or(0.0)).sum());
        }
    }

    /// Windowed grain `k` added into `acc` at output sample `k × hop`, slid to continue the
    /// previous grain's waveform.
    fn lay_grain<P: FnMut(i64, usize) -> Vec<Vec<f32>>>(&mut self, pull: &mut P) {
        let k = self.k;
        self.k = k.saturating_add(1);
        let nominal = self.source_of(k);
        let g = self.grain as i64;
        let pos = match self.prev {
            None => {
                self.ensure(pull, nominal, nominal.saturating_add(g));
                nominal
            }
            Some(prev) => {
                let lo = nominal.saturating_sub(self.tol).max(self.src_origin);
                let hi = nominal.saturating_add(self.tol);
                self.ensure(pull, lo.min(prev.saturating_add(self.hop)), hi.saturating_add(g).max(prev.saturating_add(g)));
                // the previous grain's natural continuation is the template
                let mut template = std::mem::take(&mut self.template);
                self.mono_of(prev.saturating_add(self.hop), self.hop as usize, &mut template);
                let mut best = (f32::MIN, nominal);
                let score_at = |st: &mut Self, p: i64, best: &mut (f32, i64)| {
                    let mut mono = std::mem::take(&mut st.mono);
                    st.mono_of(p, st.hop as usize, &mut mono);
                    let (mut dot, mut energy) = (0.0f32, 0.0f32);
                    for (a, b) in template.iter().zip(&mono) {
                        dot += a * b;
                        energy += b * b;
                    }
                    let s = dot / (energy + 1e-6).sqrt();
                    if s > best.0 || (s == best.0 && (p - nominal).abs() < (best.1 - nominal).abs()) {
                        *best = (s, p);
                    }
                    st.mono = mono;
                };
                let span = hi.saturating_sub(lo).max(0) as usize;
                for d in (0..=span).step_by(SEARCH_STEP as usize) {
                    score_at(self, lo.saturating_add(d as i64), &mut best);
                }
                let coarse = best.1;
                for d in (1 - SEARCH_STEP)..SEARCH_STEP {
                    let p = coarse.saturating_add(d);
                    if d != 0 && p >= lo && p <= hi {
                        score_at(self, p, &mut best);
                    }
                }
                self.template = template;
                best.1
            }
        };
        self.prev = Some(pos);
        if self.ch == 0 {
            return;
        }
        let out_at = k.saturating_mul(self.hop);
        let need = (out_at.saturating_add(g) - self.acc_base).max(0) as usize;
        let src_first = (pos - self.src_base).max(0) as usize;
        let acc_first = (out_at - self.acc_base).max(0) as usize;
        for (sc, ac) in self.src.iter().zip(&mut self.acc) {
            if ac.len() < need {
                ac.resize(need, 0.0);
            }
            for (i, w) in self.window.iter().enumerate() {
                let s = sc.get(src_first + i).copied().unwrap_or(0.0);
                if let Some(o) = ac.get_mut(acc_first + i) {
                    *o += s * w;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Stretcher;

    const SR: u32 = 48_000;

    /// A stereo tone of `hz` on both channels.
    fn tone(hz: f64) -> impl FnMut(i64, usize) -> Vec<Vec<f32>> {
        move |pos: i64, n: usize| {
            let c: Vec<f32> = (0..n).map(|i| (std::f64::consts::TAU * hz * (pos + i as i64) as f64 / SR as f64).sin() as f32).collect();
            vec![c.clone(), c]
        }
    }

    /// Channel 0 is the source sample index: what comes out says which samples were read.
    fn ramp() -> impl FnMut(i64, usize) -> Vec<Vec<f32>> {
        |pos: i64, n: usize| vec![(0..n).map(|i| (pos + i as i64) as f32).collect()]
    }

    fn zero_crossings(v: &[f32]) -> usize {
        v.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count()
    }

    #[test]
    fn the_pitch_stays_and_the_source_runs_at_speed() {
        for speed in [2.0, 0.5, 1.15] {
            let mut s = Stretcher::new(speed, SR, 1000, 5000);
            let mut out = Vec::new();
            let mut furthest = 0i64;
            let mut src = tone(440.0);
            for b in 0..(SR as i64 / 480) {
                let v = s.process(1000 + b * 480, 480, |pos, n| {
                    furthest = furthest.max(pos + n as i64);
                    src(pos, n)
                });
                assert_eq!(v.len(), 2);
                out.extend_from_slice(&v[0]);
            }
            let left = &out[4800..];
            let zc = zero_crossings(left) as f64 / (left.len() as f64 / SR as f64);
            assert!((zc - 880.0).abs() < 40.0, "speed {speed}: {zc} crossings/s, expected ~880 (resampled would be {})", 880.0 * speed);
            let peak = left.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(peak > 0.85 && peak <= 1.01, "speed {speed}: peak {peak}");
            let read = furthest - 5000;
            assert!((read as f64 - SR as f64 * speed).abs() < 3000.0, "speed {speed}: read {read} source samples for one second of output");
        }
    }

    #[test]
    fn a_steady_level_comes_out_steady_from_the_first_sample() {
        let mut s = Stretcher::new(2.0, SR, 0, 0);
        let dc = |_: i64, n: usize| vec![vec![0.5; n]; 2];
        let a = s.process(0, 2000, dc);
        let b = s.process(2000, 2000, dc);
        for (i, v) in a[1].iter().chain(&b[1]).enumerate() {
            assert!((v - 0.5).abs() < 1e-3, "sample {i} = {v}");
        }
    }

    #[test]
    fn a_seek_restarts_where_it_lands_and_a_short_gap_is_bridged() {
        let mut s = Stretcher::new(2.0, SR, 0, 0);
        s.process(0, 4800, ramp());
        // 9600 output samples skipped: the source is ~19200 further on than it was
        let c = s.process(4800 + 9600, 480, ramp());
        let want = (4800.0 + 9600.0) * 2.0;
        assert!((c[0][0] - want).abs() < 2500.0, "got {} expected ~{want}", c[0][0]);
        let d = s.process(4800 + 9600 + 480 + 100, 480, ramp());
        assert!((d[0][0] - (want + 1160.0)).abs() < 2500.0, "got {} expected ~{}", d[0][0], want + 1160.0);
        // backwards: a fresh chain there
        let e = s.process(960, 480, ramp());
        assert!((e[0][0] - 1920.0).abs() < 2500.0, "got {}", e[0][0]);
    }

    #[test]
    fn identity_and_hostile_input_are_harmless() {
        let mut s = Stretcher::new(1.0, 0, 7, 100);
        assert_eq!(s.process(7, 3, ramp()), vec![vec![100.0, 101.0, 102.0]], "1× maps the origins and passes through");
        let s = Stretcher::new(f64::NAN, SR, 0, 0);
        assert!(s.is_identity(), "NaN plays at 1×");
        let mut s = Stretcher::new(1.0e9, SR, 0, 0);
        assert_eq!(s.speed(), 64.0, "clamped");
        assert!(s.process(0, 0, ramp()).iter().all(Vec::is_empty));
        let mut s = Stretcher::new(2.0, SR, 0, 0);
        let none = |_: i64, _: usize| Vec::new();
        assert!(s.process(0, 480, none).is_empty(), "a source with no channels gives no channels");
        let v = s.process(480, 480, ramp());
        assert_eq!(v.len(), 1, "then the channel count is learnt");
        let mut s = Stretcher::new(64.0, SR, i64::MAX - 10, i64::MAX - 10);
        let _ = s.process(i64::MAX - 10, 480, ramp());
        let _ = s.process(i64::MIN + 10, 480, ramp());
        // a source that answers with more channels than it did first: the extra are ignored
        let mut s = Stretcher::new(2.0, SR, 0, 0);
        let _ = s.process(0, 480, ramp());
        let v = s.process(480, 480, tone(100.0));
        assert_eq!(v.len(), 1);
    }
}
