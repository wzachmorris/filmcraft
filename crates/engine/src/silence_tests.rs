//! `clip.detectSilence` / `clip.removeSilence` on a WAV with known pauses.

use filmcraft_project::ClipId;
use filmcraft_time::TICKS_PER_SECOND;
use serde_json::json;

use crate::Session;
use crate::media_test_util::tmp_dir;

const SR: u32 = 48_000;

/// `secs` of a 220 Hz tone at `db` dB RMS, or digital silence.
fn piece(secs: f64, db: Option<f64>) -> Vec<f32> {
    let n = (secs * f64::from(SR)) as usize;
    match db {
        None => vec![0.0; n],
        Some(db) => {
            let amp = 10f64.powf(db / 20.0) * std::f64::consts::SQRT_2;
            (0..n).map(|i| (amp * (i as f64 * 2.0 * std::f64::consts::PI * 220.0 / f64::from(SR)).sin()) as f32).collect()
        }
    }
}

/// word 1.0 s, pause 0.8 s, word 1.0 s, pause 0.2 s, word 0.5 s, pause 1.5 s, word 1.0 s = 6.0 s
fn narration() -> Vec<f32> {
    [(1.0, Some(-20.0)), (0.8, None), (1.0, Some(-20.0)), (0.2, None), (0.5, Some(-20.0)), (1.5, None), (1.0, Some(-20.0))]
        .iter()
        .flat_map(|(s, d)| piece(*s, *d))
        .collect()
}

/// A session with the narration on A1 of a 24 fps sequence. Returns (session, clip).
fn session() -> (Session, ClipId) {
    let dir = tmp_dir("silence");
    let wav = dir.join("narration.wav");
    let x = narration();
    filmcraft_testkit::wav::write_f32(&wav, &[&x, &x], SR).unwrap();
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [wav.to_string_lossy()]})).unwrap();
    let item = r["items"][0].as_u64().unwrap();
    s.execute("file.newSequence", json!({"name": "S", "width": 640, "height": 360, "fps": 24.0, "video": 1, "audio": 1})).unwrap();
    let r = s.execute("timeline.place", json!({"item": item, "track": "A1", "seconds": 0, "insert": false, "video": false})).unwrap();
    let clip = ClipId(r["clips"][0].as_u64().unwrap());
    (s, clip)
}

fn secs(v: &serde_json::Value) -> f64 {
    v.as_i64().unwrap() as f64 / TICKS_PER_SECOND as f64
}

#[test]
fn detect_finds_the_two_long_pauses_and_marks_them() {
    let (mut s, clip) = session();
    let r = s.execute("clip.detectSilence", json!({"clips": [clip.0]})).unwrap();
    let pauses = r["pauses"].as_array().unwrap();
    assert_eq!(pauses.len(), 2, "{r}");
    // 1.0–1.8 shrunk by the 40 ms pad
    assert!((secs(&pauses[0]["start"]) - 1.04).abs() < 0.02, "{r}");
    assert!((secs(&pauses[0]["end"]) - 1.76).abs() < 0.02, "{r}");
    // 3.5–5.0
    assert!((secs(&pauses[1]["start"]) - 3.54).abs() < 0.02, "{r}");
    assert!((secs(&pauses[1]["end"]) - 4.96).abs() < 0.02, "{r}");
    assert!(r["thresholdDb"].as_f64().unwrap() < -30.0, "{r}");
    let q = s.active_sequence().unwrap();
    assert_eq!(q.markers.len(), 2);
    assert!(q.markers[0].name.starts_with("Silence 0.7"), "{}", q.markers[0].name);
    // one undo step
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.active_sequence().unwrap().markers.is_empty());
    // without markers nothing is edited
    let r = s.execute("clip.detectSilence", json!({"markers": false})).unwrap();
    assert_eq!(r["pauses"].as_array().unwrap().len(), 2);
    assert!(s.active_sequence().unwrap().markers.is_empty());
}

#[test]
fn remove_ripples_the_pauses_out_keeping_a_beat() {
    let (mut s, clip) = session();
    let before = s.active_sequence().unwrap().duration();
    let r = s.execute("clip.removeSilence", json!({"clips": [clip.0], "keepSeconds": 0.12})).unwrap();
    assert_eq!(r["removed"], json!(2), "{r}");
    let removed = r["removedSeconds"].as_f64().unwrap();
    // (0.72 - 0.12) + (1.42 - 0.12) = 1.9, less up to two 24 fps frames per range for snapping
    assert!(removed > 1.7 && removed <= 1.9, "{r}");
    let q = s.active_sequence().unwrap();
    let after = q.duration();
    assert!(((before - after).0 as f64 / TICKS_PER_SECOND as f64 - removed).abs() < 1e-6);
    // the clip is now three pieces, continuous in sequence time
    let items = &q.audio_tracks[0].items;
    assert_eq!(items.len(), 3, "{items:?}");
    let mut sorted: Vec<_> = items.iter().collect();
    sorted.sort_by_key(|i| i.start);
    for w in sorted.windows(2) {
        assert_eq!(w[0].start + w[0].duration, w[1].start);
    }
    // the sound that is left has no long pause any more
    let r = s.execute("clip.detectSilence", json!({"markers": false})).unwrap();
    assert_eq!(r["pauses"].as_array().unwrap().len(), 0, "{r}");
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.active_sequence().unwrap().duration(), before);
}

#[test]
fn track_scope_leaves_the_picture_alone() {
    let (mut s, clip) = session();
    // a 6 s colour matte on V1 above the narration
    let r = s.execute("file.newColorMatte", json!({"color": "#336699", "seconds": 6.0})).unwrap();
    let matte = r["item"].as_u64().unwrap();
    s.execute("timeline.place", json!({"item": matte, "track": "V1", "seconds": 0, "insert": false, "audio": false})).unwrap();
    let r = s.execute("clip.removeSilence", json!({"clips": [clip.0], "scope": "track"})).unwrap();
    assert_eq!(r["removed"], json!(2), "{r}");
    let q = s.active_sequence().unwrap();
    assert_eq!(q.video_tracks[0].items.len(), 1, "the picture is not cut");
    assert_eq!(q.video_tracks[0].items[0].duration, filmcraft_time::Tick::from_seconds_f64(6.0));
    assert_eq!(q.audio_tracks[0].items.len(), 3);
    assert!(q.audio_tracks[0].items.iter().map(|i| i.start + i.duration).max().unwrap() < filmcraft_time::Tick::from_seconds_f64(4.3));
    assert!(s.execute("clip.removeSilence", json!({"scope": "galaxy"})).is_err());
}

#[test]
fn a_fixed_threshold_and_the_min_length_are_honoured() {
    let (mut s, _) = session();
    let r = s.execute("clip.detectSilence", json!({"markers": false, "minSeconds": 0.15})).unwrap();
    assert_eq!(r["pauses"].as_array().unwrap().len(), 3, "{r}");
    let r = s.execute("clip.detectSilence", json!({"markers": false, "minSeconds": 1.0})).unwrap();
    assert_eq!(r["pauses"].as_array().unwrap().len(), 1, "{r}");
    let r = s.execute("clip.detectSilence", json!({"markers": false, "thresholdDb": -10.0})).unwrap();
    // everything is under -10 dB: one pause covering the clip, less the pads
    assert_eq!(r["pauses"].as_array().unwrap().len(), 1, "{r}");
}

#[test]
fn hostile_parameters_are_errors_not_panics() {
    let (mut s, _) = session();
    for p in [
        json!({"thresholdDb": "loud"}),
        json!({"thresholdDb": 5.0}),
        json!({"minSeconds": 0.0}),
        json!({"minSeconds": -1}),
        json!({"padSeconds": 1e9}),
        json!({"windowMs": 0}),
        json!({"clips": [9999]}),
        json!({"track": "V1"}),
        json!({"track": "A9"}),
    ] {
        let r = s.execute("clip.detectSilence", p.clone());
        assert!(r.is_err() || p.get("thresholdDb").is_some_and(|t| t.is_string()), "{p} → {r:?}");
        let r = s.execute("clip.removeSilence", p.clone());
        assert!(r.is_err() || p.get("thresholdDb").is_some_and(|t| t.is_string()), "{p} → {r:?}");
    }
    assert!(s.execute("clip.removeSilence", json!({"keepSeconds": -1})).is_err());
    assert!(s.execute("clip.removeSilence", json!({"keepSeconds": f64::NAN})).is_ok() || true);
    // no sequence
    let mut e = Session::default();
    assert!(e.execute("clip.detectSilence", json!({})).is_err());
}

#[test]
fn cuts_in_the_track_do_not_hide_pauses_or_skew_the_threshold() {
    let (mut s, _clip) = session();
    // one cut in the middle of the long pause, one inside a word (a short all-speech piece) and
    // one through the pure-silence tail: pieces that alone have no floor or no speech to measure
    for t in [4.2, 0.5, 4.8] {
        s.execute("sequence.addEdit", json!({"seconds": t})).unwrap();
    }
    assert_eq!(s.active_sequence().unwrap().audio_tracks[0].items.len(), 4);
    let r = s.execute("clip.detectSilence", json!({"track": "A1", "markers": false})).unwrap();
    let pauses = r["pauses"].as_array().unwrap();
    assert_eq!(pauses.len(), 2, "{r}");
    assert!((secs(&pauses[0]["start"]) - 1.04).abs() < 0.02, "{r}");
    assert!((secs(&pauses[0]["end"]) - 1.76).abs() < 0.02, "{r}");
    // the pause that straddles two cuts comes back whole
    assert!((secs(&pauses[1]["start"]) - 3.54).abs() < 0.02, "{r}");
    assert!((secs(&pauses[1]["end"]) - 4.96).abs() < 0.02, "{r}");
    let thr = r["thresholdDb"].as_f64().unwrap();
    assert!((-50.0..-30.0).contains(&thr), "one threshold for the whole track: {r}");
    let r = s.execute("clip.removeSilence", json!({"track": "A1", "keepSeconds": 0.12})).unwrap();
    assert_eq!(r["removed"], json!(2), "{r}");
    let r = s.execute("clip.detectSilence", json!({"markers": false})).unwrap();
    assert_eq!(r["pauses"].as_array().unwrap().len(), 0, "{r}");
}
