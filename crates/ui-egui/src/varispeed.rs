//! Varispeed playback: a mixer played faster or slower than real time for the J/K/L shuttle.
//! Two flavours: [`resample`] lets the pitch follow the speed (a tape machine), [`stretch`]
//! keeps the pitch (speech at 2x stays intelligible, as in CapCut and DaVinci). Pure sample
//! arithmetic, so both run on the web build too.

/// Most channels a device fill is asked for.
const MAX_CHANNELS: usize = 64;

/// Speeds outside this range play at 1.
const SPEED_RANGE: (f64, f64) = (1.0 / 64.0, 64.0);

fn clamp_speed(speed: f64) -> f64 {
    if speed.is_finite() { speed.clamp(SPEED_RANGE.0, SPEED_RANGE.1) } else { 1.0 }
}

/// Play `mix` at `speed` times real time, forward: each output frame is read `speed` source
/// frames further on, linearly interpolated between the two nearest, so the pitch follows the
/// speed as it does in Premiere's shuttle. Reads of the source stay contiguous (frames read for
/// one block but not passed yet are kept for the next), and a jump in the output cursor (frames
/// the device played as silence) skips `speed` times as many source frames. `speed` is clamped
/// to 1/64..64; anything else plays at 1.
pub fn resample<M>(mut mix: M, speed: f64, start: i64) -> impl FnMut(i64, &mut [f32], usize) + Send + 'static
where
    M: FnMut(i64, &mut [f32], usize) + Send + 'static,
{
    let speed = clamp_speed(speed);
    // `next`: the next source frame to read; `carry`: frames read but not passed (interleaved);
    // `frac`: how far into carry's first frame the next output frame falls; `expect`: the output
    // cursor the next call should bring
    let mut next = start;
    let mut carry: Vec<f32> = Vec::new();
    let mut frac = 0.0f64;
    let mut expect = start;
    let mut tmp: Vec<f32> = Vec::new();
    move |cursor: i64, buf: &mut [f32], ch: usize| {
        let ch = ch.clamp(1, MAX_CHANNELS);
        let n = buf.len() / ch;
        if n == 0 {
            return;
        }
        let jump = cursor.saturating_sub(expect);
        if jump > 0 {
            let pos = frac + jump as f64 * speed;
            let whole = pos.floor();
            frac = pos - whole;
            let whole = whole.min(i64::MAX as f64) as i64;
            let carried = (carry.len() / ch) as i64;
            if whole >= carried {
                next = next.saturating_add(whole - carried);
                carry.clear();
            } else {
                carry.drain(..(whole as usize).saturating_mul(ch));
            }
        }
        expect = cursor.saturating_add(n as i64);
        // frames needed from carry's first: up to the one after the last output position
        let last = frac + (n as f64 - 1.0) * speed;
        let need = (last.floor().max(0.0) as usize).saturating_add(2);
        let have = carry.len() / ch;
        tmp.clear();
        tmp.extend_from_slice(&carry);
        if need > have {
            let fresh = need - have;
            let at = tmp.len();
            tmp.resize(need.saturating_mul(ch), 0.0);
            if let Some(dst) = tmp.get_mut(at..) {
                mix(next, dst, ch);
            }
            next = next.saturating_add(fresh as i64);
        }
        for i in 0..n {
            let p = frac + i as f64 * speed;
            let j = p.floor().max(0.0) as usize;
            let f = (p - j as f64).clamp(0.0, 1.0) as f32;
            for c in 0..ch {
                let a = tmp.get(j.saturating_mul(ch).saturating_add(c)).copied().unwrap_or(0.0);
                let b = tmp.get((j + 1).saturating_mul(ch).saturating_add(c)).copied().unwrap_or(a);
                if let Some(o) = buf.get_mut(i * ch + c) {
                    *o = a + (b - a) * f;
                }
            }
        }
        // the frames before the next output position are done with; the rest carry over
        let adv = frac + n as f64 * speed;
        let whole = adv.floor();
        frac = adv - whole;
        let whole = whole.max(0.0).min(i64::MAX as f64) as usize;
        let total = tmp.len() / ch;
        carry.clear();
        if whole >= total {
            next = next.saturating_add((whole - total) as i64);
        } else {
            carry.extend_from_slice(tmp.get(whole.saturating_mul(ch)..).unwrap_or_default());
        }
    }
}

/// Grain length in seconds for [`stretch`]: short enough to follow speech, long enough to hold
/// two periods of a low voice.
const GRAIN_SECONDS: f64 = 0.024;
/// How far (as a fraction of the grain) a grain may slide to line up with the previous one.
const SEARCH_FRACTION: f64 = 0.375;
/// Coarse step of the alignment search; the best coarse offset is then refined by ±(step - 1).
const SEARCH_STEP: i64 = 4;

/// Play `mix` at `speed` times real time, forward, with the pitch kept: the source is cut into
/// overlapping Hann-windowed grains (24 ms, half overlapping) taken `speed` times as far apart
/// as they are laid down, each slid by up to ±9 ms to continue the previous grain's waveform
/// (WSOLA). Speech at 2x reads as fast talking, not as a chipmunk. Reads of the source stay
/// contiguous; a jump in the output cursor skips `speed` times as many source frames. `speed`
/// is clamped to 1/64..64; anything else, or exactly 1, passes the mix through. `sample_rate` 0
/// is taken as 48 kHz.
pub fn stretch<M>(mut mix: M, speed: f64, start: i64, sample_rate: u32) -> impl FnMut(i64, &mut [f32], usize) + Send + 'static
where
    M: FnMut(i64, &mut [f32], usize) + Send + 'static,
{
    let speed = clamp_speed(speed);
    let sr = if sample_rate == 0 { 48_000 } else { sample_rate };
    // even grain length so two half-overlapping periodic Hann windows sum to exactly one
    let grain = (((sr as f64 * GRAIN_SECONDS) / 2.0).round().max(2.0) as usize) * 2;
    let hop = grain / 2;
    let tol = ((grain as f64 * SEARCH_FRACTION) as i64).max(SEARCH_STEP);
    let window: Vec<f32> = (0..grain).map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / grain as f64).cos() as f32).collect();
    let mut st = Stretch {
        speed,
        start,
        grain,
        hop: hop as i64,
        tol,
        window,
        ch: 0,
        src: Vec::new(),
        src_base: start,
        next: start,
        k: 0,
        prev: None,
        acc: Vec::new(),
        acc_base: 0,
        expect: start,
        template: Vec::new(),
        mono: Vec::new(),
    };
    let pass_through = (speed - 1.0).abs() < 1e-9;
    move |cursor: i64, buf: &mut [f32], ch: usize| {
        if pass_through {
            mix(cursor, buf, ch);
            return;
        }
        let ch = ch.clamp(1, MAX_CHANNELS);
        let n = buf.len() / ch;
        if n == 0 {
            return;
        }
        st.fill(&mut mix, cursor, buf, ch, n);
    }
}

/// The state of one [`stretch`] playback. Output frame indices are relative to `start`.
struct Stretch {
    speed: f64,
    start: i64,
    grain: usize,
    hop: i64,
    tol: i64,
    window: Vec<f32>,
    ch: usize,
    /// Source frames read so far and still needed, interleaved, `src_base` being the index of
    /// the first; `next` is the source frame the next contiguous read starts at.
    src: Vec<f32>,
    src_base: i64,
    next: i64,
    /// The next grain to lay down: it starts at output frame `k * hop`.
    k: i64,
    /// Where the previous grain was actually taken from.
    prev: Option<i64>,
    /// Output frames laid down so far, interleaved, from output frame `acc_base`.
    acc: Vec<f32>,
    acc_base: i64,
    /// The output cursor the next call should bring.
    expect: i64,
    template: Vec<f32>,
    mono: Vec<f32>,
}

impl Stretch {
    fn fill<M: FnMut(i64, &mut [f32], usize)>(&mut self, mix: &mut M, cursor: i64, buf: &mut [f32], ch: usize, n: usize) {
        let o = cursor.saturating_sub(self.start);
        let acc_end = self.acc_base.saturating_add((self.acc.len() / ch) as i64);
        if ch != self.ch || o < self.acc_base || o > acc_end.saturating_add(self.grain as i64) || cursor != self.expect && self.acc.is_empty() {
            // a new channel count, a cursor going back or a long jump: start over at `o`, one
            // grain early so the first frames handed out get both their halves
            self.ch = ch;
            self.src.clear();
            self.acc.clear();
            self.prev = None;
            self.k = o.div_euclid(self.hop).saturating_sub(1);
            self.acc_base = self.k.saturating_mul(self.hop);
            // the source read restarts where this output position maps to
            let s = self.source_of(self.k);
            self.src_base = s;
            self.next = s;
        } else if o > self.acc_base {
            // a short jump (or just the previous block): drop what was played or skipped
            let drop = ((o - self.acc_base) as usize).saturating_mul(ch).min(self.acc.len());
            self.acc.drain(..drop);
            self.acc_base = o;
        }
        self.expect = cursor.saturating_add(n as i64);
        let end = o.saturating_add(n as i64);
        // every frame before `k * hop` is final once grain k - 1 is down (at most n / hop + 2
        // grains, whatever saturation does to the positions)
        let mut budget = (n as i64 / self.hop).saturating_add(3);
        while self.k.saturating_mul(self.hop) < end && budget > 0 {
            self.lay_grain(mix, ch);
            budget -= 1;
        }
        let from = ((o - self.acc_base).max(0) as usize).saturating_mul(ch);
        for (i, v) in buf.iter_mut().enumerate() {
            *v = self.acc.get(from + i).copied().unwrap_or(0.0);
        }
        let drop = (from + buf.len()).min(self.acc.len());
        self.acc.drain(..drop);
        self.acc_base = end;
        // source frames no later grain or template can reach are done with
        let keep_from = self.source_of(self.k).saturating_sub(self.tol).min(self.prev.map_or(i64::MAX, |p| p.saturating_add(self.hop)));
        if keep_from > self.src_base {
            let drop = ((keep_from - self.src_base) as usize).saturating_mul(ch).min(self.src.len());
            self.src.drain(..drop);
            self.src_base = keep_from;
        }
    }

    /// The nominal source frame grain `k` is taken from.
    fn source_of(&self, k: i64) -> i64 {
        let off = (k as f64 * self.hop as f64 * self.speed).round();
        self.start.saturating_add(off.clamp(i64::MIN as f64, i64::MAX as f64) as i64).max(self.start)
    }

    /// Make sure source frames `lo..hi` are in `src`, reading forward from `next`.
    fn ensure<M: FnMut(i64, &mut [f32], usize)>(&mut self, mix: &mut M, lo: i64, hi: i64, ch: usize) {
        if lo < self.src_base {
            // only after a long jump landed just ahead of the alignment window: read back once
            let count = (self.src_base - lo) as usize;
            let mut head = vec![0.0f32; count.saturating_mul(ch)];
            mix(lo, &mut head, ch);
            head.extend_from_slice(&self.src);
            self.src = head;
            self.src_base = lo;
        }
        if hi > self.next {
            let count = (hi - self.next) as usize;
            let at = self.src.len();
            self.src.resize(at.saturating_add(count.saturating_mul(ch)), 0.0);
            if let Some(dst) = self.src.get_mut(at..) {
                mix(self.next, dst, ch);
            }
            self.next = hi;
        }
    }

    /// Mono (channel sum) of source frames `pos..pos + len` into `out`.
    fn mono_of(&self, pos: i64, len: usize, ch: usize, out: &mut Vec<f32>) {
        out.clear();
        let first = (pos - self.src_base).max(0) as usize;
        for f in 0..len {
            let at = (first + f).saturating_mul(ch);
            let s: f32 = self.src.get(at..at + ch).map(|fr| fr.iter().sum()).unwrap_or(0.0);
            out.push(s);
        }
    }

    /// Windowed grain `k` added into `acc` at output frame `k * hop`, slid to continue the
    /// previous grain's waveform.
    fn lay_grain<M: FnMut(i64, &mut [f32], usize)>(&mut self, mix: &mut M, ch: usize) {
        let k = self.k;
        self.k = k.saturating_add(1);
        let nominal = self.source_of(k);
        let g = self.grain as i64;
        let pos = match self.prev {
            None => {
                self.ensure(mix, nominal, nominal.saturating_add(g), ch);
                nominal
            }
            Some(prev) => {
                let lo = nominal.saturating_sub(self.tol).max(self.start);
                let hi = nominal.saturating_add(self.tol);
                self.ensure(mix, lo.min(prev.saturating_add(self.hop)), hi.saturating_add(g).max(prev.saturating_add(g)), ch);
                // the previous grain's natural continuation is the template
                let mut template = std::mem::take(&mut self.template);
                self.mono_of(prev.saturating_add(self.hop), self.hop as usize, ch, &mut template);
                let mut best = (f32::MIN, nominal);
                let score_at = |st: &mut Self, p: i64, best: &mut (f32, i64)| {
                    let mut mono = std::mem::take(&mut st.mono);
                    st.mono_of(p, st.hop as usize, ch, &mut mono);
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
                // counted, not compared: near the i64 ceiling a saturating step never passes `hi`
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
        let out_at = k.saturating_mul(self.hop);
        let need = ((out_at.saturating_add(g) - self.acc_base).max(0) as usize).saturating_mul(ch);
        if self.acc.len() < need {
            self.acc.resize(need, 0.0);
        }
        let src_first = (pos - self.src_base).max(0) as usize;
        let acc_first = (out_at - self.acc_base).max(0) as usize;
        for i in 0..self.grain {
            let w = self.window.get(i).copied().unwrap_or(0.0);
            for c in 0..ch {
                let s = self.src.get((src_first + i).saturating_mul(ch) + c).copied().unwrap_or(0.0);
                if let Some(o) = self.acc.get_mut((acc_first + i).saturating_mul(ch) + c) {
                    *o += s * w;
                }
            }
        }
    }
}

#[cfg(test)]
mod varispeed_tests {
    use super::{resample, stretch};

    /// A mixer whose channel-0 sample is the source frame index and channel 1 its negative:
    /// what comes out says exactly which source frames were read.
    fn ramp() -> impl FnMut(i64, &mut [f32], usize) + Send + 'static {
        |cursor: i64, buf: &mut [f32], ch: usize| {
            for (i, fr) in buf.chunks_mut(ch).enumerate() {
                let v = (cursor + i as i64) as f32;
                fr[0] = v;
                if let Some(x) = fr.get_mut(1) {
                    *x = -v;
                }
            }
        }
    }

    fn take(f: &mut impl FnMut(i64, &mut [f32], usize), cursor: i64, n: usize, ch: usize) -> Vec<f32> {
        let mut buf = vec![0.0; n * ch];
        f(cursor, &mut buf, ch);
        buf
    }

    #[test]
    fn double_speed_reads_every_other_frame_and_stays_contiguous_across_blocks() {
        let mut v = resample(ramp(), 2.0, 100);
        let a = take(&mut v, 100, 4, 2);
        assert_eq!(a, vec![100.0, -100.0, 102.0, -102.0, 104.0, -104.0, 106.0, -106.0]);
        let b = take(&mut v, 104, 4, 2);
        assert_eq!(b.iter().step_by(2).copied().collect::<Vec<_>>(), vec![108.0, 110.0, 112.0, 114.0], "{b:?}");
    }

    #[test]
    fn half_speed_interpolates_between_frames() {
        let mut v = resample(ramp(), 0.5, 0);
        let a = take(&mut v, 0, 6, 1);
        assert_eq!(a, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        let b = take(&mut v, 6, 4, 1);
        assert_eq!(b, vec![3.0, 3.5, 4.0, 4.5]);
    }

    #[test]
    fn fractional_speed_keeps_its_phase_and_a_cursor_jump_skips_source() {
        let mut v = resample(ramp(), 1.5, 0);
        let a = take(&mut v, 0, 3, 1);
        assert_eq!(a, vec![0.0, 1.5, 3.0]);
        let b = take(&mut v, 3, 3, 1);
        assert_eq!(b, vec![4.5, 6.0, 7.5]);
        // the device played 10 frames of silence: 15 source frames are gone
        let c = take(&mut v, 3 + 3 + 10, 2, 1);
        assert_eq!(c, vec![9.0 + 15.0, 10.5 + 15.0]);
    }

    #[test]
    fn eight_times_skips_whole_blocks_and_hostile_speeds_fall_back() {
        let mut v = resample(ramp(), 8.0, 0);
        assert_eq!(take(&mut v, 0, 3, 1), vec![0.0, 8.0, 16.0]);
        assert_eq!(take(&mut v, 3, 2, 1), vec![24.0, 32.0]);
        let mut v = resample(ramp(), f64::NAN, 5);
        assert_eq!(take(&mut v, 5, 2, 1), vec![5.0, 6.0], "NaN plays at 1x");
        let mut v = resample(ramp(), 1.0e9, 0);
        assert_eq!(take(&mut v, 0, 2, 1), vec![0.0, 64.0], "clamped to 64x");
        let mut v = resample(ramp(), 2.0, 0);
        take(&mut v, 0, 0, 1);
        assert_eq!(take(&mut v, 0, 1, 1), vec![0.0], "an empty buffer is a no-op");
    }

    const SR: u32 = 48_000;

    /// A stereo tone of `hz` on both channels.
    fn tone(hz: f64) -> impl FnMut(i64, &mut [f32], usize) + Send + 'static {
        move |cursor: i64, buf: &mut [f32], ch: usize| {
            for (i, fr) in buf.chunks_mut(ch).enumerate() {
                let t = (cursor + i as i64) as f64 / SR as f64;
                let v = (std::f64::consts::TAU * hz * t).sin() as f32;
                fr.fill(v);
            }
        }
    }

    /// A mixer that records the furthest source frame read; its sound is a ramp.
    fn watched(seen: std::sync::Arc<std::sync::atomic::AtomicI64>) -> impl FnMut(i64, &mut [f32], usize) + Send + 'static {
        let mut r = ramp();
        move |cursor: i64, buf: &mut [f32], ch: usize| {
            r(cursor, buf, ch);
            let end = cursor + (buf.len() / ch) as i64;
            seen.fetch_max(end, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn zero_crossings(v: &[f32]) -> usize {
        v.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count()
    }

    #[test]
    fn stretching_keeps_the_pitch_and_consumes_the_source_at_speed() {
        for speed in [2.0, 0.5, 1.5] {
            let seen = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0));
            let mut s = stretch(tone(440.0), speed, 0, SR);
            // one second of output in device-sized blocks; the tone stays 440 Hz (880 crossings)
            let mut out = Vec::new();
            for b in 0..(SR as i64 / 480) {
                out.extend(take(&mut s, b * 480, 480, 2));
            }
            let left: Vec<f32> = out.iter().step_by(2).skip(4800).copied().collect();
            let zc = zero_crossings(&left) as f64 / (left.len() as f64 / SR as f64);
            assert!((zc - 880.0).abs() < 40.0, "speed {speed}: {zc} crossings/s, expected ~880 (resampled would be {})", 880.0 * speed);
            let peak = left.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(peak > 0.85 && peak <= 1.01, "speed {speed}: peak {peak}");
            // the source advances `speed` frames per output frame
            let mut w = stretch(watched(seen.clone()), speed, 1000, SR);
            for b in 0..(SR as i64 / 480) {
                take(&mut w, 1000 + b * 480, 480, 1);
            }
            let read = seen.load(std::sync::atomic::Ordering::Relaxed) - 1000;
            let want = SR as f64 * speed;
            assert!((read as f64 - want).abs() < 3000.0, "speed {speed}: read {read} source frames for {} output, expected ~{want}", SR);
        }
    }

    #[test]
    fn a_steady_level_comes_out_steady() {
        // the half-overlapping windows sum to one, from the very first frame handed out
        let dc = |_: i64, buf: &mut [f32], _: usize| buf.fill(0.5);
        let mut s = stretch(dc, 2.0, 0, SR);
        let a = take(&mut s, 0, 2000, 2);
        let b = take(&mut s, 2000, 2000, 2);
        for (i, v) in a.iter().chain(&b).enumerate() {
            assert!((v - 0.5).abs() < 1e-3, "frame {} = {v}", i / 2);
        }
    }

    #[test]
    fn a_cursor_jump_skips_source_and_hostile_input_is_harmless() {
        let mut s = stretch(ramp(), 2.0, 0, SR);
        take(&mut s, 0, 4800, 1);
        // 9600 output frames of silence went by: the source is ~19200 further on than it was
        let c = take(&mut s, 4800 + 9600, 480, 1);
        let want = (4800.0 + 9600.0) * 2.0;
        assert!((c[0] - want).abs() < 2500.0, "got {} expected ~{want}", c[0]);
        // and a short gap is bridged without restarting
        let d = take(&mut s, 4800 + 9600 + 480 + 100, 480, 1);
        assert!((d[0] - (want + 1160.0)).abs() < 2500.0, "got {} expected ~{}", d[0], want + 1160.0);
        let mut s = stretch(ramp(), f64::NAN, 7, SR);
        assert_eq!(take(&mut s, 7, 3, 1), vec![7.0, 8.0, 9.0], "NaN passes the mix through");
        let mut s = stretch(ramp(), 1.0, 7, 0);
        assert_eq!(take(&mut s, 7, 2, 2), vec![7.0, -7.0, 8.0, -8.0], "1x passes through, 0 Hz is fine");
        let mut s = stretch(ramp(), 2.0, 0, SR);
        take(&mut s, 0, 0, 1);
        take(&mut s, 0, 0, 0);
        assert_eq!(take(&mut s, 0, 1, 1).len(), 1);
        let mut s = stretch(ramp(), 64.0, i64::MAX - 10, SR);
        let _ = take(&mut s, i64::MAX - 10, 480, 2);
    }
}
