//! Varispeed playback: a mixer played faster or slower than real time, the pitch following, for
//! the J/K/L shuttle. Pure sample arithmetic, so it runs on the web build too.

/// Most channels a device fill is asked for.
const MAX_CHANNELS: usize = 64;

/// Play `mix` at `speed` times real time, forward: each output frame is read `speed` source
/// frames further on, linearly interpolated between the two nearest, so the pitch follows the
/// speed as it does in Premiere's shuttle. Reads of the source stay contiguous (frames read for
/// one block but not passed yet are kept for the next), and a jump in the output cursor (frames
/// the device played as silence) skips `speed` times as many source frames. `speed` is clamped
/// to 1/64..64; anything else plays at 1.
pub fn varispeed<M>(mut mix: M, speed: f64, start: i64) -> impl FnMut(i64, &mut [f32], usize) + Send + 'static
where
    M: FnMut(i64, &mut [f32], usize) + Send + 'static,
{
    let speed = if speed.is_finite() { speed.clamp(1.0 / 64.0, 64.0) } else { 1.0 };
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

#[cfg(test)]
mod varispeed_tests {
    use super::varispeed;

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
        let mut v = varispeed(ramp(), 2.0, 100);
        let a = take(&mut v, 100, 4, 2);
        assert_eq!(a, vec![100.0, -100.0, 102.0, -102.0, 104.0, -104.0, 106.0, -106.0]);
        let b = take(&mut v, 104, 4, 2);
        assert_eq!(b.iter().step_by(2).copied().collect::<Vec<_>>(), vec![108.0, 110.0, 112.0, 114.0], "{b:?}");
    }

    #[test]
    fn half_speed_interpolates_between_frames() {
        let mut v = varispeed(ramp(), 0.5, 0);
        let a = take(&mut v, 0, 6, 1);
        assert_eq!(a, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        let b = take(&mut v, 6, 4, 1);
        assert_eq!(b, vec![3.0, 3.5, 4.0, 4.5]);
    }

    #[test]
    fn fractional_speed_keeps_its_phase_and_a_cursor_jump_skips_source() {
        let mut v = varispeed(ramp(), 1.5, 0);
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
        let mut v = varispeed(ramp(), 8.0, 0);
        assert_eq!(take(&mut v, 0, 3, 1), vec![0.0, 8.0, 16.0]);
        assert_eq!(take(&mut v, 3, 2, 1), vec![24.0, 32.0]);
        let mut v = varispeed(ramp(), f64::NAN, 5);
        assert_eq!(take(&mut v, 5, 2, 1), vec![5.0, 6.0], "NaN plays at 1x");
        let mut v = varispeed(ramp(), 1.0e9, 0);
        assert_eq!(take(&mut v, 0, 2, 1), vec![0.0, 64.0], "clamped to 64x");
        let mut v = varispeed(ramp(), 2.0, 0);
        take(&mut v, 0, 0, 1);
        assert_eq!(take(&mut v, 0, 1, 1), vec![0.0], "an empty buffer is a no-op");
    }
}
