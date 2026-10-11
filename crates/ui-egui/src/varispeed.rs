//! Varispeed playback: a mixer played faster or slower than real time for the J/K/L shuttle.
//! Two flavours: [`resample`] lets the pitch follow the speed (a tape machine), [`stretch`]
//! keeps the pitch (speech at 2x stays intelligible, as in CapCut and DaVinci; Settings ▸ Audio ▸
//! Maintain pitch while shuttling picks). Pure sample arithmetic, so both run on the web build too.

use filmcraft_audio_dsp::stretch::{Stretcher, clamp_speed};

/// Most channels a device fill is asked for.
const MAX_CHANNELS: usize = 64;

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

/// Play `mix` at `speed` times real time, forward, with the pitch kept
/// ([`filmcraft_audio_dsp::stretch::Stretcher`]): speech at 2x reads as fast talking, not as a
/// chipmunk. Reads of the source stay contiguous; a jump in the output cursor skips `speed` times
/// as many source frames. `speed` is clamped to 1/64..64; anything else, or exactly 1, passes the
/// mix through. `sample_rate` 0 is taken as 48 kHz.
pub fn stretch<M>(mut mix: M, speed: f64, start: i64, sample_rate: u32) -> impl FnMut(i64, &mut [f32], usize) + Send + 'static
where
    M: FnMut(i64, &mut [f32], usize) + Send + 'static,
{
    let mut st = Stretcher::new(speed, sample_rate, start, start);
    let pass_through = st.is_identity();
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
        let planar = st.process(cursor, n, |pos, len| {
            let mut inter = vec![0.0f32; len.saturating_mul(ch)];
            mix(pos, &mut inter, ch);
            (0..ch).map(|c| inter.iter().skip(c).step_by(ch).copied().collect()).collect()
        });
        for i in 0..n {
            for c in 0..ch {
                if let Some(o) = buf.get_mut(i * ch + c) {
                    *o = planar.get(c).and_then(|v| v.get(i)).copied().unwrap_or(0.0);
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
