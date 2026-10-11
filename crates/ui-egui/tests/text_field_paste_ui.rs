//! Edit ▸ Paste from the native menu bar while a text field has the keyboard (macOS key
//! equivalents): the real `FilmcraftApp` under `egui_kittest`, with the desktop app's command
//! inbox and clipboard-text hook stood in for.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    menu: Sender<String>,
}

impl Driver {
    fn demo() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        let (tx, rx) = channel();
        let (menu, inbox) = channel();
        let mut app = FilmcraftApp::new(session).with_control(rx);
        app.command_inbox = Some(inbox);
        // the system clipboard holds " mundo"
        app.hooks.clipboard_text = Some(Box::new(|| Some(" mundo".into())));
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 1400.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx, menu };
        d.frames(4);
        d
    }
    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &ctx, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }
    fn ok(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                assert_eq!(v["ok"], true, "{method} {params}: {v}");
                return v["result"].clone();
            }
        }
        panic!("no reply to {method} {params}");
    }
    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }
    /// What the native menu bar sends when its item (or key equivalent) fires.
    fn menu(&mut self, id: &str) {
        self.menu.send(id.into()).unwrap();
        self.frames(3);
    }
    fn text(&mut self) -> String {
        self.exec("graphics.list", json!({}))["layers"][0]["text"].as_str().unwrap_or_default().to_string()
    }
}

#[test]
fn edit_paste_from_the_menu_bar_lands_in_the_focused_text_field() {
    let mut d = Driver::demo();
    let r = d.exec("graphics.newText", json!({"text": "Hola"}));
    let clip = r["clip"].as_u64().expect("clip id");
    d.exec("timeline.select", json!({"clips": [clip]}));
    // the Properties panel's Source Text field takes the keyboard
    d.ok("ui.click", json!({"id": "graphics.sourceText"}));
    d.ok("ui.key", json!({"key": "End"}));
    assert!(d.harness.ctx.egui_wants_keyboard_input(), "the field has the keyboard");
    d.menu("edit.paste");
    assert_eq!(d.text(), "Hola mundo", "Cmd+V pasted the clipboard text into the field");
    assert_eq!(d.harness.state().ui.status, "", "no Timeline paste was attempted");
    // Select All + Cut empty the field instead of cutting clips off the Timeline
    d.menu("edit.selectAll");
    d.menu("edit.cut");
    assert_eq!(d.text(), "", "Cmd+A, Cmd+X cut the field's text");
    let clips = |d: &mut Driver| {
        d.exec("sequence.inspect", json!({}))["video"].as_array().map_or(0, |v| v.iter().map(|t| t["items"].as_array().map_or(0, Vec::len)).sum::<usize>())
    };
    let n = clips(&mut d);
    assert!(n >= 3, "the clips are all still there: {n}");
    // Undo in the field is the field's own undo, not the project's
    d.menu("edit.undo");
    assert_eq!(d.text(), "Hola mundo", "Cmd+Z undid the cut inside the field");
    assert_eq!(clips(&mut d), n);
    // without a focused field the same menu item is the Timeline's Paste again
    d.ok("ui.key", json!({"key": "Escape"}));
    d.ok("ui.set", json!({"focused": "Timeline"}));
    assert!(!d.harness.ctx.egui_wants_keyboard_input());
    d.menu("edit.paste");
    assert_eq!(d.text(), "Hola mundo", "nothing more went into the field");
    assert!(d.harness.state().ui.status.contains("clipboard is empty"), "the Timeline handled it: {:?}", d.harness.state().ui.status);
}
