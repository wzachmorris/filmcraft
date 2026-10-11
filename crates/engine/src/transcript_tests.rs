use std::sync::Arc;

use serde_json::json;

use crate::{Event, Session};
use filmcraft_project::{ItemId, Transcript, Word};
use filmcraft_speech::FixedTranscriber;
use filmcraft_time::Tick;

/// The demo project with a fake transcriber whose transcript fits the first A1 clip's media:
/// "Hello um world." (Speaker 1), a 1.2 s pause, "Second speaker here." (Speaker 2).
fn session() -> (Session, ItemId, Tick) {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let q = s.active_sequence().unwrap();
    let a = &q.audio_tracks[0].items[0];
    let (item, sin) = (a.item, a.source_in);
    let sec = |x: f64| sin + Tick::from_seconds_f64(x);
    let mut t = Transcript { language: "en".into(), ..Default::default() };
    for (text, a, b, sp) in
        [("Hello", 0.2, 0.5, 0), ("um", 0.6, 0.9, 0), ("world.", 1.0, 1.4, 0), ("Second", 2.6, 3.0, 1), ("speaker", 3.0, 3.5, 1), ("here.", 3.5, 4.0, 1)]
    {
        let mut w = Word::new(text, sec(a), sec(b));
        w.speaker = Some(sp);
        t.words.push(w);
    }
    t.normalize();
    s.transcriber = Some(Arc::new(FixedTranscriber { transcript: t, id: "fixed".into() }));
    let start = a_start(&s);
    (s, item, start)
}

fn a_start(s: &Session) -> Tick {
    s.active_sequence().unwrap().audio_tracks[0].items[0].start
}

fn words(s: &mut Session) -> Vec<String> {
    let r = s.execute("transcript.inspect", json!({})).unwrap();
    r["words"].as_array().unwrap().iter().map(|w| w["text"].as_str().unwrap().to_string()).collect()
}

#[test]
fn generate_inspect_search_and_rename() {
    let (mut s, item, start) = session();
    assert!(s.execute("transcript.select", json!({"from": 0})).is_err(), "disabled before transcribing");
    let r = s.execute("transcript.generate", json!({"items": [item.0]})).unwrap();
    assert_eq!(r["items"][0]["words"], 6, "{r}");
    assert_eq!(r["items"][0]["source"], "fixed");
    assert_eq!(s.project.transcripts[&item].words.len(), 6);
    assert!(
        s.drain_events().iter().any(|e| matches!(e, Event::Toast { message, error: false } if message.starts_with("Transcribed: 6 words, 2 speakers (en)"))),
        "the user is told when the transcript lands"
    );

    let r = s.execute("transcript.inspect", json!({})).unwrap();
    assert_eq!(words(&mut s), ["Hello", "um", "world.", "Second", "speaker", "here."]);
    assert_eq!(r["words"][0]["start"].as_i64().unwrap(), (start + Tick::from_seconds_f64(0.2)).0, "mapped to sequence time");
    assert_eq!(r["paragraphs"].as_array().unwrap().len(), 2, "speaker change splits paragraphs: {}", r["paragraphs"]);
    assert_eq!(r["speakers"], json!(["Speaker 1", "Speaker 2"]));

    let r = s.execute("transcript.search", json!({"query": "second spea"})).unwrap();
    assert_eq!(r["matches"], json!([{"from": 3, "to": 4, "start": r["matches"][0]["start"], "end": r["matches"][0]["end"]}]));

    s.execute("transcript.renameSpeaker", json!({"speaker": "Speaker 2", "name": "Ann"})).unwrap();
    let r = s.execute("transcript.inspect", json!({})).unwrap();
    assert_eq!(r["words"][3]["speaker"], "Ann");
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.project.transcripts[&item].speakers[1].name, "Speaker 2");
    s.execute("transcript.renameSpeaker", json!({"speaker": 0, "item": item.0, "name": "Bo"})).unwrap();
    assert_eq!(s.project.transcripts[&item].speakers[0].name, "Bo");
    assert!(s.execute("transcript.renameSpeaker", json!({"speaker": "Nobody", "name": "X"})).is_err());

    // transcribing is one undo step
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.project.transcripts.is_empty());
}

#[test]
fn select_extract_and_lift_by_words() {
    let (mut s, item, _) = session();
    s.execute("transcript.generate", json!({"items": [item.0]})).unwrap();
    let rate = s.sequence_rate();
    let before = s.active_sequence().unwrap().duration();

    let r = s.execute("transcript.select", json!({"from": 3, "to": 5})).unwrap();
    let (a, b) = (Tick(r["start"].as_i64().unwrap()), Tick(r["end"].as_i64().unwrap()));
    let q = s.active_sequence().unwrap();
    assert_eq!(q.mark_in, Some(a));
    assert_eq!(q.mark_out, Some(b - rate.frame_duration()), "Out is the last frame inside");
    assert_eq!(rate.snap(a), a, "frame aligned");
    assert_eq!(s.playhead(), a);

    // Extract "um": the sequence gets shorter by the word's frames and the word is gone
    let r = s.execute("transcript.extract", json!({"from": 1})).unwrap();
    let cut = Tick(r["end"].as_i64().unwrap()) - Tick(r["start"].as_i64().unwrap());
    assert!(cut > Tick::ZERO);
    assert_eq!(s.active_sequence().unwrap().duration(), before - cut);
    assert!(!words(&mut s).contains(&"um".to_string()), "{:?}", words(&mut s));
    s.active_sequence().unwrap().check().unwrap();
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.active_sequence().unwrap().duration(), before);

    // Lift leaves a gap: same duration, word gone
    s.execute("transcript.lift", json!({"from": 1, "to": 1})).unwrap();
    assert_eq!(s.active_sequence().unwrap().duration(), before);
    assert_eq!(words(&mut s), ["Hello", "world.", "Second", "speaker", "here."]);
    assert!(s.execute("transcript.extract", json!({"from": 99})).is_err());
}

#[test]
fn remove_fillers_pauses_and_create_captions() {
    let (mut s, item, _) = session();
    s.execute("transcript.generate", json!({"items": [item.0]})).unwrap();
    let before = s.active_sequence().unwrap().duration();
    let r = s.execute("transcript.removeFillers", json!({})).unwrap();
    assert_eq!(r["removed"], 1, "{r}");
    assert!(!words(&mut s).contains(&"um".to_string()));
    let r = s.execute("transcript.removePauses", json!({"minSeconds": 1.0, "keepSeconds": 0.1})).unwrap();
    assert_eq!(r["removed"], 1, "{r}");
    let after = s.active_sequence().unwrap().duration();
    assert!(after < before - Tick::from_seconds_f64(1.0), "the pause and the filler are gone");
    assert_eq!(words(&mut s).len(), 5);
    s.active_sequence().unwrap().check().unwrap();

    let r = s.execute("transcript.createCaptions", json!({"maxChars": 32})).unwrap();
    assert_eq!(r["captions"], 2, "one caption per speaker: {r}");
    let q = s.active_sequence().unwrap();
    let tr = &q.caption_tracks[0];
    assert_eq!(tr.captions[0].text, "Hello world.");
    assert_eq!(tr.captions[1].speaker.as_deref(), Some("Speaker 2"));
    q.check().unwrap();
}

#[test]
fn set_delete_and_models() {
    let (mut s, item, _) = session();
    let t = json!({"language": "en", "words": [
        {"text": "b", "start": 2000, "end": 3000, "speaker": 1},
        {"text": "a", "start": 0, "end": 1000},
    ]});
    let r = s.execute("transcript.set", json!({"item": item.0, "transcript": t})).unwrap();
    assert_eq!(r["words"], 2);
    let tr = &s.project.transcripts[&item];
    assert_eq!(tr.words[0].text, "a", "normalized: sorted");
    assert_eq!(tr.speakers.len(), 2, "missing speaker labels added");
    assert_eq!(tr.source, "imported");
    assert!(s.execute("transcript.set", json!({"item": 999_999, "transcript": {}})).is_err());
    assert!(s.execute("transcript.set", json!({"item": item.0, "transcript": {"words": 3}})).is_err());

    // survives save/load
    let bytes = filmcraft_format::encode(&s.project, true);
    let back = filmcraft_format::decode(&bytes).unwrap();
    assert_eq!(back.project.transcripts[&item].words.len(), 2);

    s.execute("transcript.delete", json!({"items": [item.0]})).unwrap();
    assert!(s.project.transcripts.is_empty());
    assert!(s.execute("transcript.delete", json!({})).is_err());

    let m = s.execute("transcript.models", json!({})).unwrap();
    assert_eq!(m["available"], filmcraft_speech::available());
    assert!(m["models"].as_array().unwrap().iter().any(|x| x["id"] == "whisper-base"));
}

#[test]
fn generate_without_a_transcriber() {
    let (mut s, item, _) = session();
    s.transcriber = None;
    let e = s.execute("transcript.generate", json!({"items": [item.0], "model": "nope"})).unwrap_err().to_string();
    // a build without speech-to-text says that first: the command is disabled (#97)
    let why = if filmcraft_speech::available() { "unknown speech model" } else { "not available in this build" };
    assert!(e.contains(why), "{e}");
    if !filmcraft_speech::available() {
        assert!(!s.is_enabled("sequence.transcribe"), "Transcribe Sequence follows transcript.generate");
        let e = s.execute("transcript.generate", json!({"items": [item.0]})).unwrap_err().to_string();
        assert!(e.contains("whisper") && e.contains("not available"), "{e}");
        #[cfg(not(feature = "speech-download"))]
        assert!(s.execute("transcript.downloadModel", json!({})).unwrap_err().to_string().contains("not available"));
    }
    // defaults to the media of the open sequence's audio clips
    s.transcriber = Some(Arc::new(FixedTranscriber { transcript: Transcript::default(), id: "empty".into() }));
    assert!(s.is_enabled("sequence.transcribe"), "an installed recogniser enables Transcribe Sequence");
    let r = s.execute("transcript.generate", json!({})).unwrap();
    assert!(r["items"].as_array().unwrap().len() >= 2, "{r}");
}

/// Captions for the selected clips only, onto the existing top track: what was there under them
/// gives way, the rest of the track stays.
#[test]
fn captions_for_the_selected_clips_replace_only_that_stretch() {
    let (mut s, item, start) = session();
    s.execute("transcript.generate", json!({"items": [item.0]})).unwrap();
    // cut the clip at 2 s: the first speaker before, the second after
    s.execute("playhead.set", json!({"time": (start + Tick::from_seconds_f64(2.0)).0})).unwrap();
    s.execute("sequence.addEditAllTracks", json!({})).unwrap();
    let q = s.active_sequence().unwrap();
    let second = q.audio_tracks[0].items.iter().find(|i| i.start > start).map(|i| i.id).unwrap();
    let a1 = q.audio_tracks[0].items.len();
    assert!(a1 >= 2, "the clip was cut: {a1}");

    // nothing selected: refused; a clip without words: refused
    assert!(s.execute("transcript.createCaptionsForSelection", json!({})).is_err());
    assert!(s.execute("transcript.createCaptions", json!({"clips": [987_654_321u64]})).is_err());

    // no caption track yet: "top" makes one, with the selected clip's words only
    s.execute("timeline.select", json!({"clips": [second.0]})).unwrap();
    let r = s.execute("transcript.createCaptionsForSelection", json!({"maxChars": 32})).unwrap();
    let q = s.active_sequence().unwrap();
    assert_eq!(q.caption_tracks.len(), 1, "{r}");
    let texts: Vec<&str> = q.caption_tracks[0].captions.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(texts, vec!["Second speaker here."], "{r}");
    assert!(q.caption_tracks[0].captions[0].words.len() == 3, "word times kept: {:?}", q.caption_tracks[0].captions[0].words);

    // the whole sequence onto that same track, then spoil the second caption and re-caption the
    // second clip: only that stretch is replaced
    let r = s.execute("transcript.createCaptions", json!({"maxChars": 32, "track": "C1"})).unwrap();
    assert_eq!(r["captions"], 2, "{r}");
    let q = s.active_sequence().unwrap();
    assert_eq!(q.caption_tracks.len(), 1, "onto the existing track, not a new one");
    let (first, stale) = (q.caption_tracks[0].captions[0].id, q.caption_tracks[0].captions[1].id);
    s.execute("captions.setText", json!({"caption": stale.0, "text": "stale"})).unwrap();
    s.execute("timeline.select", json!({"clips": [second.0]})).unwrap();
    s.execute("transcript.createCaptionsForSelection", json!({"maxChars": 32})).unwrap();
    let q = s.active_sequence().unwrap();
    let texts: Vec<&str> = q.caption_tracks[0].captions.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(texts, vec!["Hello um world.", "Second speaker here."], "the stale caption gave way, the first stayed");
    assert_eq!(q.caption_tracks[0].captions[0].id, first);
    assert!(q.find_caption(stale).is_none());
    q.check().unwrap();

    // a locked track refuses
    s.execute("captions.setTrack", json!({"track": "C1", "locked": true})).unwrap();
    assert!(s.execute("transcript.createCaptionsForSelection", json!({})).is_err());
    // one undo per step
    s.execute("captions.setTrack", json!({"track": "C1", "locked": false})).unwrap();
}

/// A recogniser that notes the language it was asked for.
struct Recording(std::sync::Mutex<Vec<Option<String>>>);

impl filmcraft_speech::Transcriber for Recording {
    fn id(&self) -> String {
        "rec".into()
    }
    fn transcribe(
        &self,
        _: &[f32],
        o: &filmcraft_speech::Options,
        _: filmcraft_speech::ProgressFn,
    ) -> std::result::Result<Transcript, filmcraft_speech::SpeechError> {
        if let Ok(mut v) = self.0.lock() {
            v.push(o.language.clone());
        }
        Ok(Transcript { language: o.language.clone().unwrap_or_else(|| "xx".into()), ..Default::default() })
    }
}

/// "auto" in the Transcribe dialog means detect, even when Settings ▸ Media Analysis names a
/// default language (that default applies only when no language is passed at all). Whisper
/// asked for English on Spanish speech translates it, which is how this was found.
#[test]
fn the_dialogs_auto_beats_the_settings_default_language() {
    let (mut s, item, _) = session();
    let rec = Arc::new(Recording(Default::default()));
    s.transcriber = Some(rec.clone());
    let mut prefs = s.prefs.clone();
    prefs.media_analysis.language_auto_detect = false;
    prefs.media_analysis.default_language = "en".into();
    s.set_prefs(prefs).unwrap();
    let asked = |rec: &Recording| rec.0.lock().map(|v| v.clone()).unwrap_or_default();
    s.execute("transcript.generate", json!({"items": [item.0]})).unwrap();
    assert_eq!(asked(&rec), vec![Some("en".to_string())], "no language at all: the Settings default");
    s.execute("transcript.generate", json!({"items": [item.0], "language": "auto"})).unwrap();
    assert_eq!(asked(&rec).last(), Some(&None), "auto detects");
    let before = asked(&rec).len();
    s.execute("sequence.transcribe", json!({"language": "auto", "track": "mix"})).unwrap();
    let after = asked(&rec);
    assert!(after.len() > before && after[before..].iter().all(Option::is_none), "the dialog's auto reaches every clip: {after:?}");
    s.execute("transcript.generate", json!({"items": [item.0], "language": "es"})).unwrap();
    assert_eq!(asked(&rec).last(), Some(&Some("es".to_string())));
    assert_eq!(s.project.transcripts[&item].language, "es");
}

/// Captions fit the track's style by default: a big font gives few words per caption, a small one
/// many; `maxChars` still overrides.
#[test]
fn captions_fit_the_tracks_font_size_by_default() {
    let (mut s, item, _) = session();
    s.execute("transcript.generate", json!({"items": [item.0]})).unwrap();
    s.execute("captions.newTrack", json!({"format": "Subtitle"})).unwrap();
    let count = |s: &mut Session, size: f64| -> (usize, usize) {
        s.execute("captions.setStyle", json!({"track": "C1", "size": size})).unwrap();
        s.execute("transcript.createCaptions", json!({"track": "C1"})).unwrap();
        let q = s.active_sequence().unwrap();
        let c = &q.caption_tracks[0].captions;
        (c.len(), c.iter().map(|x| x.text.chars().count()).max().unwrap_or(0))
    };
    let (n_big, chars_big) = count(&mut s, 400.0);
    let (n_small, chars_small) = count(&mut s, 12.0);
    assert!(n_big > n_small, "a huge font splits the words into more captions: {n_big} vs {n_small}");
    assert!(chars_big < chars_small, "{chars_big} vs {chars_small}");
    assert!(s.active_sequence().unwrap().caption_tracks[0].captions.iter().all(|c| !c.text.contains('\n')), "one line each");
    // an explicit recipe wins over the fit
    s.execute("captions.newTrack", json!({"format": "Subtitle"})).unwrap();
    s.execute("transcript.createCaptions", json!({"track": "C1", "maxChars": 32, "lines": 2})).unwrap();
    assert_eq!(s.active_sequence().unwrap().caption_tracks[0].captions.len(), 2);
}
