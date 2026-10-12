//! Clip ▸ Audio Options ▸ Detect Silence / Remove Silence: pauses found from the sound itself.
//!
//! Both commands measure the level of the selected audio clips (or every clip on `track`) with
//! [`filmcraft_audio_dsp::silence`]: a short-window RMS envelope, a threshold taken from the
//! recording's own noise floor and speech level unless `thresholdDb` fixes one, hysteresis so
//! the edges don't chatter, and a pad of air kept on both sides of every pause. No model, no
//! network, the same answer every time.
//!
//! - `clip.detectSilence` reports the pauses (sequence time) and, by default, marks each one with
//!   a rose range marker named by its length, so they can be checked on the timeline first.
//! - `clip.removeSilence` ripple-deletes the inside of every pause, keeping `keepSeconds` of air
//!   in the middle of each one, in one undo step. A 0.9 s breath becomes a 0.12 s beat; a 0.2 s
//!   gap between words is left alone (`minSeconds`). `scope: "sequence"` (default) closes the
//!   time on every unlocked track (the picture, the sound, sync-locked captions), like Remove
//!   Pauses; `scope: "track"` tightens only the analysed clips' own tracks (and sync-locked
//!   captions), for a voice track that will be cut to afterwards.

use filmcraft_audio_dsp::silence::{Envelope, SilenceOptions, auto_threshold, detect_envelope};
use filmcraft_project::{ClipId, Label, Marker, MarkerId, MarkerKind};
use filmcraft_time::{TICKS_PER_SECOND, Tick, TimeRange};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p, f64_p, has_seq, str_p};
use crate::{Result, Session};

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "clip.detectSilence",
            label: "Detect Silence…",
            menu: &["Clip", "Audio Options"],
            shortcut: None,
            params: r#"{"clips":[id]?,"track":"A1"|id?,"thresholdDb":f64? (auto),"minSeconds":f64=0.3,"padSeconds":f64=0.04,"windowMs":f64=10,"markers":bool=true,"color":label=Rose}"#,
            enabled: has_audio,
            run: detect,
            journal: true,
        },
        CommandSpec {
            id: "clip.removeSilence",
            label: "Remove Silence…",
            menu: &["Clip", "Audio Options"],
            shortcut: None,
            params: r#"{"clips":[id]?,"track":"A1"|id?,"thresholdDb":f64? (auto),"minSeconds":f64=0.3,"keepSeconds":f64=0.12,"padSeconds":f64=0.04,"windowMs":f64=10,"scope":"sequence|track"=sequence}"#,
            enabled: has_audio,
            run: remove,
            journal: true,
        },
    ]
}

fn has_audio(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    let q = s.active_sequence().ok_or("no sequence is open")?;
    let any = s.state.selection.iter().any(|c| q.audio_tracks.iter().any(|t| t.items.iter().any(|i| i.id == *c)));
    if any || q.audio_tracks.iter().any(|t| !t.items.is_empty()) { Ok(()) } else { Err("select an audio clip in the timeline".into()) }
}

/// One pause, in sequence time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pause {
    pub clip: ClipId,
    pub range: TimeRange,
}

/// What an analysis found.
#[derive(Clone, Debug, PartialEq)]
pub struct Analysis {
    pub pauses: Vec<Pause>,
    pub clips: usize,
    pub threshold_db: f64,
    pub noise_floor_db: f64,
    pub speech_db: f64,
}

fn options(id: &str, p: &Value) -> Result<SilenceOptions> {
    let mut o = SilenceOptions::default();
    if let Some(t) = f64_p(p, "thresholdDb") {
        if !t.is_finite() || !(-120.0..=0.0).contains(&t) {
            return Err(bad(id, "`thresholdDb` must be between -120 and 0"));
        }
        o.threshold_db = Some(t);
    }
    if let Some(m) = f64_p(p, "minSeconds") {
        if !m.is_finite() || m <= 0.0 || m > 3600.0 {
            return Err(bad(id, "`minSeconds` must be a positive number of seconds"));
        }
        o.min_silence_ms = m * 1000.0;
    }
    if let Some(x) = f64_p(p, "padSeconds") {
        if !x.is_finite() || !(0.0..=10.0).contains(&x) {
            return Err(bad(id, "`padSeconds` must be between 0 and 10"));
        }
        o.pad_ms = x * 1000.0;
    }
    if let Some(w) = f64_p(p, "windowMs") {
        if !w.is_finite() || !(1.0..=1000.0).contains(&w) {
            return Err(bad(id, "`windowMs` must be between 1 and 1000"));
        }
        o.window_ms = w;
    }
    Ok(o)
}

/// The audio clips to analyse: `clips`, else the clips on `track`, else the selected audio clips,
/// else every clip on every unlocked audio track.
fn targets(s: &Session, id: &str, p: &Value) -> Result<Vec<ClipId>> {
    let q = s.active_sequence().ok_or(crate::EngineError::NoSequence)?;
    let on_audio = |c: ClipId| q.audio_tracks.iter().any(|t| t.items.iter().any(|i| i.id == c));
    if let Some(a) = p.get("clips").and_then(Value::as_array) {
        let ids: Vec<ClipId> = a.iter().filter_map(|v| v.as_u64().map(ClipId)).filter(|c| on_audio(*c)).collect();
        if ids.is_empty() {
            return Err(bad(id, "`clips` names no audio clip of the sequence"));
        }
        return Ok(ids);
    }
    if let Some(t) = str_p(p, "track") {
        let tr = crate::commands::track_p(s, p, "track", id)?.ok_or_else(|| bad(id, format!("no audio track `{t}`")))?;
        let tr = q.audio_tracks.iter().find(|x| x.id == tr).ok_or_else(|| bad(id, format!("`{t}` is not an audio track")))?;
        return Ok(tr.items.iter().map(|i| i.id).collect());
    }
    let sel: Vec<ClipId> = s.state.selection.iter().copied().filter(|c| on_audio(*c)).collect();
    if !sel.is_empty() {
        return Ok(sel);
    }
    Ok(q.audio_tracks.iter().filter(|t| !t.locked).flat_map(|t| t.items.iter().map(|i| i.id)).collect())
}

/// Measure the clips and find their pauses in sequence time.
pub fn analyse(s: &Session, clips: &[ClipId], opts: &SilenceOptions) -> Result<Analysis> {
    let seq = s.active_sequence().ok_or(crate::EngineError::NoSequence)?;
    let sr = seq.settings.sample_rate.clamp(1, 384_000);
    let window = ((opts.window_ms.clamp(1.0, 1000.0) / 1000.0) * f64::from(sr)).round().max(1.0) as usize;
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    // 1. one envelope per clip. The threshold is picked once over every clip together: a track
    // that has already been cut into short all-speech pieces has no quiet stretch inside any one
    // piece to measure a noise floor from, so a per-clip threshold would be garbage.
    struct Prep {
        clip: ClipId,
        start: Tick,
        end: Tick,
        speed: f64,
        levels: Vec<f32>,
    }
    let mut preps = Vec::new();
    for t in &seq.audio_tracks {
        for it in t.items.iter().filter(|i| clips.contains(&i.id) && i.enabled) {
            let Some(src) = filmcraft_render::SourceProvider::source(&provider, it.item) else { continue };
            if !src.info().has_audio() || it.reverse {
                continue;
            }
            let speed = if it.speed.is_finite() && it.speed.abs() >= 1e-3 { it.speed.abs() } else { 1.0 };
            let s0 = it.source_in.to_units_floor(i64::from(sr));
            let len = ((it.duration.to_units_floor(i64::from(sr)) as f64) * speed).ceil();
            if !len.is_finite() || len <= 0.0 {
                continue;
            }
            let len = len as i64;
            let mut env = Envelope::new(window);
            let mut pos = s0;
            let block = (i64::from(sr)).max(1);
            while pos < s0.saturating_add(len) {
                let n = (s0.saturating_add(len) - pos).min(block).max(1) as usize;
                match src.audio_stream(it.audio_stream, pos, n, sr) {
                    Ok(b) => {
                        let chans: Vec<&[f32]> = b.channels.iter().map(Vec::as_slice).collect();
                        if chans.is_empty() {
                            env.push(&[&vec![0.0f32; n]]);
                        } else {
                            env.push(&chans);
                        }
                    }
                    Err(_) => env.push(&[&vec![0.0f32; n]]),
                }
                pos = pos.saturating_add(n as i64);
            }
            preps.push(Prep { clip: it.id, start: it.start, end: it.start + it.duration, speed, levels: env.finish() });
        }
    }
    let all: Vec<f32> = preps.iter().flat_map(|p| p.levels.iter().copied()).collect();
    let (floor, speech, auto) = auto_threshold(&all);
    let thr = opts.threshold_db.filter(|t| t.is_finite()).unwrap_or(auto);
    let fixed = SilenceOptions { threshold_db: Some(thr), ..*opts };
    // 2. clips that butt against each other on a track are one continuous signal: a pause that
    // straddles a cut (the tail of one clip and the head of the next) is found whole. Joined
    // envelopes are detected together, then every span is mapped back through the clip it lands in.
    preps.sort_by_key(|p| p.start);
    let join_slack = Tick(((window as f64 / f64::from(sr)) * TICKS_PER_SECOND as f64).round() as i64);
    let mut groups: Vec<Vec<&Prep>> = Vec::new();
    for p in &preps {
        match groups.last_mut() {
            Some(g) if g.last().is_some_and(|l| (p.start - l.end).0.abs() <= join_slack.0) => g.push(p),
            _ => groups.push(vec![p]),
        }
    }
    let mut pauses = Vec::new();
    for g in &groups {
        let mut levels = Vec::new();
        let mut offsets = Vec::with_capacity(g.len());
        for p in g {
            offsets.push(levels.len());
            levels.extend_from_slice(&p.levels);
        }
        let r = detect_envelope(&levels, window, sr, &fixed);
        // media frame offset in the joined envelope → the clip it falls in → sequence time
        let locate = |frames: usize| -> (usize, Tick) {
            let w = frames as f64 / window as f64;
            let k = offsets.iter().rposition(|&o| (o as f64) <= w).unwrap_or(0);
            let p = g[k];
            let local = (w - offsets[k] as f64) * window as f64;
            let secs = local / f64::from(sr) / p.speed;
            (k, (p.start + Tick((secs * TICKS_PER_SECOND as f64).round() as i64)).min(p.end))
        };
        for sp in r.spans {
            let (k, a) = locate(sp.start);
            let (_, b) = locate(sp.end);
            if b > a {
                pauses.push(Pause { clip: g[k].clip, range: TimeRange::from_bounds(a, b) });
            }
        }
    }
    pauses.sort_by_key(|p| p.range.start);
    let (thr, floor, speech) = if preps.is_empty() { (f64::NAN, f64::NAN, f64::NAN) } else { (thr, floor, speech) };
    Ok(Analysis { pauses, clips: preps.len(), threshold_db: thr, noise_floor_db: floor, speech_db: speech })
}

fn db(v: f64) -> Value {
    if v.is_finite() { json!(v) } else { Value::Null }
}

fn report(a: &Analysis) -> Value {
    let total: i64 = a.pauses.iter().map(|p| p.range.duration.0).sum();
    json!({
        "clips": a.clips,
        "pauses": a.pauses.iter().map(|p| json!({
            "clip": p.clip.0,
            "start": p.range.start.0,
            "end": p.range.end().0,
            "seconds": p.range.duration.0 as f64 / TICKS_PER_SECOND as f64,
        })).collect::<Vec<_>>(),
        "totalSeconds": total as f64 / TICKS_PER_SECOND as f64,
        "thresholdDb": db(a.threshold_db),
        "noiseFloorDb": db(a.noise_floor_db),
        "speechDb": db(a.speech_db),
    })
}

fn detect(s: &mut Session, p: &Value) -> Result<Value> {
    let id = "clip.detectSilence";
    let opts = options(id, p)?;
    let clips = targets(s, id, p)?;
    let a = analyse(s, &clips, &opts)?;
    let mut out = report(&a);
    if bool_p(p, "markers").unwrap_or(true) && !a.pauses.is_empty() {
        let color = str_p(p, "color").and_then(Label::from_name).unwrap_or(Label::Rose);
        let pauses = a.pauses.clone();
        let ids = s.edit_sequence("Detect Silence", move |q, ctx, _| {
            let mut ids = Vec::with_capacity(pauses.len());
            for pz in &pauses {
                let id = MarkerId(ctx.alloc());
                let secs = pz.range.duration.0 as f64 / TICKS_PER_SECOND as f64;
                q.markers.push(Marker {
                    id,
                    start: pz.range.start,
                    duration: pz.range.duration,
                    name: format!("Silence {secs:.2}s"),
                    comment: String::new(),
                    kind: MarkerKind::Comment,
                    color,
                });
                ids.push(id.0);
            }
            q.markers.sort_by_key(|m| m.start);
            Ok(ids)
        })?;
        out["markers"] = json!(ids);
    }
    Ok(out)
}

fn remove(s: &mut Session, p: &Value) -> Result<Value> {
    let id = "clip.removeSilence";
    let opts = options(id, p)?;
    let keep = f64_p(p, "keepSeconds").unwrap_or(0.12);
    if !keep.is_finite() || !(0.0..=60.0).contains(&keep) {
        return Err(bad(id, "`keepSeconds` must be between 0 and 60"));
    }
    let keep = Tick::from_seconds_f64(keep);
    let track_only = match str_p(p, "scope").unwrap_or("sequence") {
        "sequence" | "all" => false,
        "track" | "tracks" | "clips" => true,
        other => return Err(bad(id, format!("unknown scope `{other}` (sequence, track)"))),
    };
    let clips = targets(s, id, p)?;
    let a = analyse(s, &clips, &opts)?;
    let rate = s.sequence_rate();
    let frame = rate.frame_duration();
    // the inside of each pause, `keep` of air left in the middle, snapped to frames so the picture
    // cuts land on frame edges
    let ranges: Vec<TimeRange> = a
        .pauses
        .iter()
        .filter_map(|pz| {
            let half = Tick(keep.0 / 2);
            let a = rate.snap(pz.range.start + half + frame - Tick(1));
            let b = rate.snap(pz.range.end() - half);
            (b >= a + frame).then(|| TimeRange::from_bounds(a, b))
        })
        .collect();
    let n = ranges.len();
    let mut out = report(&a);
    if n == 0 {
        out["removed"] = json!(0);
        out["removedSeconds"] = json!(0.0);
        return Ok(out);
    }
    let total = if track_only {
        let tracks: Vec<filmcraft_project::TrackId> = {
            let q = s.active_sequence().ok_or(crate::EngineError::NoSequence)?;
            q.audio_tracks.iter().filter(|t| t.items.iter().any(|i| clips.contains(&i.id))).map(|t| t.id).collect()
        };
        s.edit_sequence("Remove Silence", |q, ctx, _| {
            let mut total = Tick::ZERO;
            for r in filmcraft_edit::transcript::merge_ranges(ranges).into_iter().rev() {
                if r.duration > Tick::ZERO {
                    filmcraft_edit::extract_only(q, &tracks, r, ctx);
                    total += r.duration;
                }
            }
            Ok(total)
        })?
    } else {
        s.edit_sequence("Remove Silence", |q, ctx, _| Ok(filmcraft_edit::transcript::ripple_delete_ranges(q, ranges, ctx)))?
    };
    out["removed"] = json!(n);
    out["removedSeconds"] = json!(total.0 as f64 / TICKS_PER_SECOND as f64);
    Ok(out)
}
