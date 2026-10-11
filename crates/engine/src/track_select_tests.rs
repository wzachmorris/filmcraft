//! `timeline.trackSelect` and a drag of that selection: captions come along with the clips, land
//! where the clips do (overwriting what sits there), stop at the sequence start with them and go
//! with them on Delete.

use filmcraft_project::{ClipId, TrackId};
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde_json::json;

use crate::Session;

/// `n` seconds, on a frame of the active sequence (captions and edits snap to frames).
fn secs(s: &Session, n: i64) -> Tick {
    s.sequence_rate().snap_nearest(Tick(TICKS_PER_SECOND.saturating_mul(n)))
}

/// An empty sequence with a caption track; one picture+sound clip at `at` seconds per entry.
/// Returns (V1, picture clips, caption track).
fn stage(s: &mut Session, at: &[i64]) -> (TrackId, Vec<ClipId>, TrackId) {
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.execute("file.newSequence", json!({"name": "Track Select", "video": 2, "audio": 2})).unwrap();
    let item = s.project.items.values().find(|i| i.has_video() && i.has_audio() && i.as_media().is_some()).map(|i| i.id).unwrap();
    let mut clips = Vec::new();
    for sec in at {
        s.execute("timeline.place", json!({"item": item.0, "seconds": *sec as f64})).unwrap();
        let want = s.sequence_rate().snap_nearest(Tick(TICKS_PER_SECOND.saturating_mul(*sec)));
        let q = s.active_sequence().unwrap();
        // `timeline.place` puts the clip at the exact time asked; pick the one nearest the frame
        clips.push(q.video_tracks[0].items.iter().min_by_key(|i| (i.start - want).0.abs()).unwrap().id);
    }
    s.execute("captions.newTrack", json!({"format": "Subtitle"})).unwrap();
    let q = s.active_sequence().unwrap();
    (q.video_tracks[0].id, clips, q.caption_tracks[0].id)
}

fn caption(s: &mut Session, track: TrackId, at: i64, dur: i64, text: &str) -> ClipId {
    let r = s.execute("captions.add", json!({"track": track.0, "text": text, "time": secs(s, at).0, "durationSeconds": dur as f64})).unwrap();
    ClipId(r["caption"].as_u64().unwrap())
}

fn start_of(s: &Session, c: ClipId) -> Tick {
    s.active_sequence().unwrap().find_item(c).map(|(_, i)| i.start).unwrap()
}

fn caption_span(s: &Session, id: ClipId) -> Option<(Tick, Tick)> {
    s.active_sequence().unwrap().find_caption(id).map(|(_, c)| (c.start, c.end()))
}

#[test]
fn track_select_takes_clips_and_captions_from_a_time_on() {
    let mut s = Session::default();
    let (v1, clips, ct) = stage(&mut s, &[0, 20, 40]);
    let early = caption(&mut s, ct, 2, 2, "early");
    let mid = caption(&mut s, ct, 20, 2, "mid");
    let late = caption(&mut s, ct, 41, 2, "late");

    // at the second clip's start (captions snap to frames, placed clips sit where asked)
    let t = start_of(&s, clips[1]);
    let r = s.execute("timeline.trackSelect", json!({"time": t.0})).unwrap();
    assert_eq!(s.state.caption_selection, vec![mid, late], "{r}");
    assert!(s.state.selection.contains(&clips[1]) && s.state.selection.contains(&clips[2]), "{r}");
    assert!(!s.state.selection.contains(&clips[0]), "the clip that ends before the time stays out: {r}");
    assert_eq!(s.state.selection.len(), 4, "picture and sound of two clips: {r}");

    // backward: everything that starts before the time
    s.execute("timeline.trackSelect", json!({"time": t.0, "forward": false})).unwrap();
    assert_eq!(s.state.caption_selection, vec![early]);
    assert!(s.state.selection.contains(&clips[0]) && !s.state.selection.contains(&clips[1]));

    // one track only (Shift-click): no captions, no sound
    s.execute("timeline.trackSelect", json!({"time": t.0, "track": v1.0})).unwrap();
    assert!(s.state.caption_selection.is_empty());
    assert_eq!(s.state.selection, vec![clips[1], clips[2]]);

    // a locked caption track stays out, like a locked media track
    s.execute("captions.setTrack", json!({"track": ct.0, "locked": true})).unwrap();
    s.execute("timeline.trackSelect", json!({"time": t.0})).unwrap();
    assert!(s.state.caption_selection.is_empty());
    assert_eq!(s.state.selection.len(), 4);

    // no time: from the playhead
    s.execute("captions.setTrack", json!({"track": ct.0, "locked": false})).unwrap();
    s.execute("playhead.set", json!({"time": secs(&s, 41).0})).unwrap();
    s.execute("timeline.trackSelect", json!({})).unwrap();
    assert_eq!(s.state.caption_selection, vec![late]);
    assert_eq!(s.state.selection.len(), 2);
}

#[test]
fn a_caption_track_as_the_track_selects_captions_only() {
    let mut s = Session::default();
    let (_, clips, ct) = stage(&mut s, &[0, 20]);
    let early = caption(&mut s, ct, 2, 2, "early");
    let mid = caption(&mut s, ct, 20, 2, "mid");
    let late = caption(&mut s, ct, 23, 2, "late");
    let t = start_of(&s, clips[1]);
    s.execute("timeline.trackSelect", json!({"time": t.0, "track": "C1"})).unwrap();
    assert_eq!(s.state.caption_selection, vec![mid, late]);
    assert!(s.state.selection.is_empty(), "no clips: {:?}", s.state.selection);
    s.execute("timeline.trackSelect", json!({"time": t.0, "track": ct.0, "forward": false})).unwrap();
    assert_eq!(s.state.caption_selection, vec![early]);
    assert!(s.state.selection.is_empty());
    // from the very start: every caption on the track
    s.execute("timeline.trackSelect", json!({"time": 0, "track": "C1"})).unwrap();
    assert_eq!(s.state.caption_selection, vec![early, mid, late]);
    let r = s.execute("timeline.trackSelect", json!({"time": t.0, "track": "C7"}));
    assert!(r.is_err(), "no such track: {r:?}");
}

#[test]
fn dragging_a_track_selection_carries_the_captions() {
    let mut s = Session::default();
    let (v1, clips, ct) = stage(&mut s, &[0, 20]);
    let early = caption(&mut s, ct, 2, 2, "early");
    let mid = caption(&mut s, ct, 20, 2, "mid");
    let late = caption(&mut s, ct, 23, 2, "late");
    // a stale caption in the gap the block is dragged into, as after a manual cut
    let stale = caption(&mut s, ct, 10, 4, "stale");

    s.execute("timeline.trackSelect", json!({"time": secs(&s, 20).0})).unwrap();
    let was = start_of(&s, clips[1]);
    let (mid0, late0, early0, stale0) =
        (caption_span(&s, mid).unwrap(), caption_span(&s, late).unwrap(), caption_span(&s, early).unwrap(), caption_span(&s, stale).unwrap());
    // the drag: the clip from 20 s to 11 s, the captions by the same shift
    let shift = secs(&s, 11) - secs(&s, 20);
    let r = s
        .execute(
            "timeline.move",
            json!({
                "moves": [{"clip": clips[1].0, "track": v1.0, "time": secs(&s, 11).0}],
                "captions": [{"caption": mid.0, "time": (mid0.0 + shift).0}, {"caption": late.0, "time": (late0.0 + shift).0}],
                "linked": true
            }),
        )
        .unwrap();
    assert_eq!(r["captions"], json!([mid.0, late.0]), "{r}");
    let q = s.active_sequence().unwrap();
    assert_eq!(q.find_item(clips[1]).map(|(_, i)| i.start), Some(secs(&s, 11)));
    assert_eq!(caption_span(&s, mid), Some((mid0.0 + shift, mid0.1 + shift)), "moved by the clip's shift");
    assert_eq!(caption_span(&s, late), Some((late0.0 + shift, late0.1 + shift)));
    assert_eq!(caption_span(&s, early), Some(early0), "unselected captions stay");
    // the stale caption (10–14) gave way: `mid` landed inside it, so it is split around `mid`,
    // and `late` landing on its tail cut the right piece back to where `late` starts
    assert_eq!(caption_span(&s, stale), Some((stale0.0, mid0.0 + shift)), "what the block lands on is overwritten");
    let spans: Vec<(Tick, Tick, &str)> = s.active_sequence().unwrap().caption_tracks[0].captions.iter().map(|c| (c.start, c.end(), c.text.as_str())).collect();
    let texts: Vec<&str> = spans.iter().map(|x| x.2).collect();
    assert_eq!(texts, vec!["early", "stale", "mid", "stale", "late"], "{spans:?}");
    assert_eq!((spans[3].0, spans[3].1), (mid0.1 + shift, late0.0 + shift), "{spans:?}");

    // one undo brings everything back
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(caption_span(&s, mid), Some(mid0));
    assert_eq!(caption_span(&s, stale), Some(stale0));
    assert_eq!(start_of(&s, clips[1]), was);
}

#[test]
fn a_caption_across_the_whole_landing_span_is_split_around_it() {
    let mut s = Session::default();
    let (v1, clips, ct) = stage(&mut s, &[0, 20]);
    let long = caption(&mut s, ct, 8, 10, "long");
    let mid = caption(&mut s, ct, 20, 2, "mid");
    let (long0, mid0) = (caption_span(&s, long).unwrap(), caption_span(&s, mid).unwrap());
    let shift = secs(&s, 12) - secs(&s, 20);
    s.execute(
        "timeline.move",
        json!({"moves": [{"clip": clips[1].0, "track": v1.0, "time": secs(&s, 12).0}], "captions": [{"caption": mid.0, "time": (mid0.0 + shift).0}], "linked": true}),
    )
    .unwrap();
    let spans: Vec<(Tick, Tick, String)> = s.active_sequence().unwrap().caption_tracks[0].captions.iter().map(|c| (c.start, c.end(), c.text.clone())).collect();
    let (a, b) = (mid0.0 + shift, mid0.1 + shift);
    assert_eq!(spans, vec![(long0.0, a, "long".into()), (a, b, "mid".into()), (b, long0.1, "long".into())], "{spans:?}");
}

#[test]
fn the_block_stops_at_the_sequence_start_as_one() {
    let mut s = Session::default();
    let (v1, clips, ct) = stage(&mut s, &[10]);
    let c = caption(&mut s, ct, 12, 2, "c");
    let c0 = caption_span(&s, c).unwrap();
    // asked 12 s left, the clip can only go 10: the caption goes 10 too
    let shift = -secs(&s, 12);
    s.execute(
        "timeline.move",
        json!({"moves": [{"clip": clips[0].0, "track": v1.0, "time": (secs(&s, 10) + shift).0}], "captions": [{"caption": c.0, "time": (c0.0 + shift).0}], "linked": true}),
    )
    .unwrap();
    assert_eq!(s.active_sequence().unwrap().find_item(clips[0]).map(|(_, i)| i.start), Some(Tick::ZERO));
    assert_eq!(caption_span(&s, c), Some((c0.0 - secs(&s, 10), c0.1 - secs(&s, 10))));
}

#[test]
fn captions_must_move_as_one_block_and_locked_tracks_refuse() {
    let mut s = Session::default();
    let (v1, clips, ct) = stage(&mut s, &[0, 20]);
    let a = caption(&mut s, ct, 20, 2, "a");
    let b = caption(&mut s, ct, 23, 2, "b");
    let was = start_of(&s, clips[1]);
    let r = s.execute(
        "timeline.move",
        json!({"moves": [{"clip": clips[1].0, "track": v1.0, "time": secs(&s, 10).0}], "captions": [{"caption": a.0, "time": secs(&s, 10).0}, {"caption": b.0, "time": secs(&s, 12).0}]}),
    );
    assert!(r.is_err(), "different shifts: {r:?}");
    assert_eq!(s.active_sequence().unwrap().find_item(clips[1]).map(|(_, i)| i.start), Some(was), "nothing moved");
    let r = s.execute(
        "timeline.move",
        json!({"moves": [{"clip": clips[1].0, "track": v1.0, "time": secs(&s, 10).0}], "captions": [{"caption": 987_654_321u64, "time": 0}]}),
    );
    assert!(r.is_err(), "unknown caption: {r:?}");
    s.execute("captions.setTrack", json!({"track": ct.0, "locked": true})).unwrap();
    let r = s.execute(
        "timeline.move",
        json!({"moves": [{"clip": clips[1].0, "track": v1.0, "time": secs(&s, 10).0}], "captions": [{"caption": a.0, "time": secs(&s, 10).0}]}),
    );
    assert!(r.is_err(), "locked caption track: {r:?}");
    assert_eq!(s.active_sequence().unwrap().find_item(clips[1]).map(|(_, i)| i.start), Some(was), "the whole move is refused");
    // hostile times
    let r = s.execute("timeline.move", json!({"moves": [{"clip": clips[1].0, "track": v1.0, "time": 0}], "captions": [{"caption": a.0, "time": i64::MIN}]}));
    assert!(r.is_err(), "{r:?}");
}

#[test]
fn delete_removes_the_selected_captions_with_the_clips() {
    let mut s = Session::default();
    let (_, clips, ct) = stage(&mut s, &[0, 20]);
    let early = caption(&mut s, ct, 2, 2, "early");
    let mid = caption(&mut s, ct, 20, 2, "mid");
    s.execute("timeline.trackSelect", json!({"time": secs(&s, 20).0})).unwrap();
    s.execute("edit.clear", json!({})).unwrap();
    let q = s.active_sequence().unwrap();
    assert!(q.find_item(clips[1]).is_none() && q.find_item(clips[0]).is_some());
    assert!(caption_span(&s, mid).is_none(), "the selected caption went with the clips");
    assert!(caption_span(&s, early).is_some());
    assert!(s.state.selection.is_empty() && s.state.caption_selection.is_empty());
    s.execute("edit.undo", json!({})).unwrap();
    assert!(caption_span(&s, mid).is_some() && s.active_sequence().unwrap().find_item(clips[1]).is_some(), "one undo for both");
}
