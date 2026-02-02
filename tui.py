"""
Textual TUI for FloorShiftDetector.

Run:
  python tui.py
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Label, Input, Select, Checkbox, Button, Footer, Header
from textual.message import Message

from floorshiftdetector import PipelineConfig, infer_mask_path, run_pipeline


class RunClicked(Message):
    def __init__(self, ref: str, cur: str, mode: str, debug: bool, save: str) -> None:
        super().__init__()
        self.ref = ref
        self.cur = cur
        self.mode = mode
        self.debug = debug
        self.save = save


class FloorTUI(App):
    TITLE = "FloorShiftDetector TUI"
    CSS = """
    Screen { padding: 1; }
    #panel { width: 70; }
    Input { width: 1fr; }
    Select { width: 1fr; }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="panel"):
            yield Label("Reference image path (Reference.JPG)")
            yield Input(value="Images/Bedroom/Reference.JPG", id="ref")
            yield Label("Current image path (any image in same folder)")
            yield Input(value="Images/Bedroom/IMG_6567.JPG", id="cur")
            yield Label("Floor ROI mode (mask auto-uses masks/<Scene>.png)")
            yield Select(
                options=[("detection", "detection"), ("mask", "mask")],
                value="detection",
                id="mode",
            )
            yield Checkbox("Debug windows (grid)", id="debug", value=False)
            yield Label("Save result path (optional, PNG/JPG)")
            yield Input(value="", id="save")
            with Horizontal():
                yield Button("Run", id="run", variant="primary")
                yield Button("Quit", id="quit")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.exit()
            return
        if event.button.id == "run":
            ref = self.query_one("#ref", Input).value.strip()
            cur = self.query_one("#cur", Input).value.strip()
            mode = self.query_one("#mode", Select).value
            debug = self.query_one("#debug", Checkbox).value
            save = self.query_one("#save", Input).value.strip()
            self.post_message(RunClicked(ref, cur, mode, debug, save))

    def on_run_clicked(self, message: RunClicked) -> None:
        cfg = PipelineConfig(show_debug_windows=message.debug, debug_mode="grid")
        cfg.floor_roi_mode = message.mode
        if cfg.enable_floor_roi and cfg.floor_roi_mode == "mask":
            mask_path = infer_mask_path(message.ref)
            if not mask_path:
                self.bell()
                self.notify("No mask found in masks/<Scene>.png for this reference.")
                return
            cfg.floor_mask_override_path = mask_path

        outputs = run_pipeline(message.ref, message.cur, cfg)
        if message.save:
            Path(message.save).parent.mkdir(parents=True, exist_ok=True)
            import cv2
            cv2.imwrite(message.save, outputs["result"])
            self.notify(f"Saved result: {message.save}")

        import cv2
        if not message.debug:
            cv2.imshow("Result", outputs["result"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    FloorTUI().run()
