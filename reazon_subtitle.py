from __future__ import annotations

import ctypes
import os
import queue
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from audio_pipeline import CancelledError
from subtitle_processor import ProcessSummary, process_video
from translator_service import MODELS


VIDEO_TYPES = [
    ("영상 파일", "*.mp4 *.mkv *.webm *.avi *.mov *.m4v *.ts *.mts *.m2ts *.wmv"),
    ("모든 파일", "*.*"),
]


def configure_windows_process() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "ReazonSubtitle.Desktop"
        )
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class ReazonSubtitleApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ReazonSubtitle")
        self.geometry("820x610")
        self.minsize(720, 540)
        self._events: queue.Queue[tuple] = queue.Queue()
        self._cancel_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._closing = False
        self._output_was_edited = False

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.precision_var = tk.StringVar(value="int8")
        self.model_var = tk.StringVar(value=MODELS["mt2"].label)
        self.status_var = tk.StringVar(value="영상을 선택해 주세요.")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._configure_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("맑은 고딕", 18, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")
        style.configure("Accent.TButton", font=("맑은 고딕", 10, "bold"), padding=(12, 8))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=20)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="ReazonSubtitle", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            outer,
            text="영상의 원본 오디오 → FAST VAD → Reazon 일본어 인식 → HyTrans 한국어 SRT",
            style="Hint.TLabel",
        ).pack(anchor=tk.W, pady=(3, 18))

        paths = ttk.LabelFrame(outer, text="파일", padding=12)
        paths.pack(fill=tk.X)
        paths.columnconfigure(1, weight=1)

        ttk.Label(paths, text="영상").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.input_entry = ttk.Entry(paths, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=1, sticky=tk.EW)
        self.browse_input = ttk.Button(paths, text="선택…", command=self._choose_input)
        self.browse_input.grid(row=0, column=2, padx=(8, 0))

        ttk.Label(paths, text="SRT").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.output_entry = ttk.Entry(paths, textvariable=self.output_var)
        self.output_entry.grid(row=1, column=1, sticky=tk.EW, pady=(10, 0))
        self.output_entry.bind("<Key>", lambda _event: self._mark_output_edited())
        self.browse_output = ttk.Button(paths, text="저장 위치…", command=self._choose_output)
        self.browse_output.grid(row=1, column=2, padx=(8, 0), pady=(10, 0))

        options = ttk.LabelFrame(outer, text="처리 설정", padding=12)
        options.pack(fill=tk.X, pady=(14, 0))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Reazon 정밀도").grid(row=0, column=0, sticky=tk.W)
        self.precision_combo = ttk.Combobox(
            options,
            textvariable=self.precision_var,
            values=("int8", "fp32"),
            state="readonly",
            width=12,
        )
        self.precision_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 24))

        ttk.Label(options, text="번역 모델").grid(row=0, column=2, sticky=tk.W)
        self.model_combo = ttk.Combobox(
            options,
            textvariable=self.model_var,
            values=tuple(model.label for model in MODELS.values()),
            state="readonly",
            width=27,
        )
        self.model_combo.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

        ttk.Label(
            options,
            text=(
                "VAD는 MekiAudioCapture FAST 기준 고정: silence 0.25초 · 최대 20초 · "
                "앞/뒤 여백 0.15/0.35초"
            ),
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(10, 0))

        progress_frame = ttk.Frame(outer)
        progress_frame.pack(fill=tk.X, pady=(16, 0))
        self.progress = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress.pack(fill=tk.X)
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor=tk.W, pady=(6, 0))

        log_frame = ttk.LabelFrame(outer, text="처리 기록", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        self.log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            height=12,
            state=tk.DISABLED,
            font=("맑은 고딕", 9),
            relief=tk.FLAT,
            background="#f7f7f7",
        )
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(14, 0))
        self.open_folder_button = ttk.Button(
            buttons,
            text="출력 폴더 열기",
            command=self._open_output_folder,
            state=tk.DISABLED,
        )
        self.open_folder_button.pack(side=tk.LEFT)
        self.cancel_button = ttk.Button(
            buttons,
            text="취소",
            command=self._cancel,
            state=tk.DISABLED,
        )
        self.cancel_button.pack(side=tk.RIGHT)
        self.start_button = ttk.Button(
            buttons,
            text="한국어 SRT 만들기",
            style="Accent.TButton",
            command=self._start,
        )
        self.start_button.pack(side=tk.RIGHT, padx=(0, 8))

    def _mark_output_edited(self) -> None:
        self._output_was_edited = True

    def _choose_input(self) -> None:
        selected = filedialog.askopenfilename(title="영상 파일 선택", filetypes=VIDEO_TYPES)
        if not selected:
            return
        self.input_var.set(selected)
        if not self._output_was_edited or not self.output_var.get().strip():
            source = Path(selected)
            self.output_var.set(str(source.with_suffix(".ko.srt")))
        self.status_var.set("준비되었습니다.")

    def _choose_output(self) -> None:
        current = self.output_var.get().strip()
        selected = filedialog.asksaveasfilename(
            title="SRT 저장 위치",
            initialdir=str(Path(current).parent) if current else None,
            initialfile=Path(current).name if current else "subtitle.ko.srt",
            defaultextension=".srt",
            filetypes=[("SubRip 자막", "*.srt")],
        )
        if selected:
            self.output_var.set(selected)
            self._output_was_edited = True

    def _set_running(self, running: bool) -> None:
        field_state = tk.DISABLED if running else tk.NORMAL
        combo_state = tk.DISABLED if running else "readonly"
        for widget in (
            self.input_entry,
            self.output_entry,
            self.browse_input,
            self.browse_output,
        ):
            widget.configure(state=field_state)
        self.precision_combo.configure(state=combo_state)
        self.model_combo.configure(state=combo_state)
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.cancel_button.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, str(text).rstrip() + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _start(self) -> None:
        input_text = self.input_var.get().strip()
        output_text = self.output_var.get().strip()
        if not input_text or not Path(input_text).is_file():
            messagebox.showerror("ReazonSubtitle", "유효한 영상 파일을 선택해 주세요.", parent=self)
            return
        if not output_text:
            messagebox.showerror("ReazonSubtitle", "SRT 저장 경로를 지정해 주세요.", parent=self)
            return
        output = Path(output_text)
        if output.exists() and not messagebox.askyesno(
            "ReazonSubtitle",
            f"이미 존재하는 파일을 덮어쓸까요?\n\n{output}",
            parent=self,
        ):
            return

        label_to_key = {model.label: key for key, model in MODELS.items()}
        model_key = label_to_key.get(self.model_var.get(), "mt2")
        self._cancel_event.clear()
        self._set_running(True)
        self.open_folder_button.configure(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("작업을 시작합니다.")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

        self._worker = threading.Thread(
            target=self._run_job,
            args=(Path(input_text), output, self.precision_var.get(), model_key),
            name="ReazonSubtitle-Job",
            daemon=False,
        )
        self._worker.start()

    def _run_job(self, input_path: Path, output_path: Path, precision: str, model: str) -> None:
        try:
            summary = process_video(
                input_path,
                output_path,
                precision=precision,
                translation_model=model,
                status=lambda ratio, message: self._events.put(("status", ratio, message)),
                log=lambda text: self._events.put(("log", text)),
                cancel_event=self._cancel_event,
            )
            self._events.put(("done", summary))
        except CancelledError as exc:
            self._events.put(("cancelled", str(exc)))
        except Exception as exc:
            self._events.put(("error", str(exc), traceback.format_exc()))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self._events.get_nowait()
                kind = event[0]
                if kind == "status":
                    self.progress_var.set(max(0, min(100, float(event[1]) * 100)))
                    self.status_var.set(str(event[2]))
                elif kind == "log":
                    self._append_log(str(event[1]))
                elif kind == "done":
                    self._finish_success(event[1])
                elif kind == "cancelled":
                    self._finish_cancelled(str(event[1]))
                elif kind == "error":
                    self._finish_error(str(event[1]), str(event[2]))
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(80, self._poll_events)

    def _finish_success(self, summary: ProcessSummary) -> None:
        self._worker = None
        self._set_running(False)
        self.progress_var.set(100)
        self.status_var.set(
            f"완료: {summary.recognized}개 인식, {summary.translated}개 번역 "
            f"({summary.elapsed:.1f}초)"
        )
        self.open_folder_button.configure(state=tk.NORMAL)
        warning = (
            f"\n번역 실패 {summary.translation_failures}개는 일본어 원문으로 표시했습니다."
            if summary.translation_failures
            else ""
        )
        messagebox.showinfo(
            "ReazonSubtitle",
            f"한국어 SRT를 저장했습니다.\n\n{summary.output_path}{warning}",
            parent=self,
        )
        if self._closing:
            self.destroy()

    def _finish_cancelled(self, message: str) -> None:
        self._worker = None
        self._set_running(False)
        self.status_var.set(message)
        self._append_log(message)
        if self._closing:
            self.destroy()

    def _finish_error(self, message: str, detail: str) -> None:
        self._worker = None
        self._set_running(False)
        self.status_var.set(f"처리 실패: {message}")
        self._append_log(detail)
        messagebox.showerror("ReazonSubtitle", f"처리하지 못했습니다.\n\n{message}", parent=self)
        if self._closing:
            self.destroy()

    def _cancel(self) -> None:
        if self._worker and self._worker.is_alive():
            self._cancel_event.set()
            self.cancel_button.configure(state=tk.DISABLED)
            self.status_var.set("취소하고 임시 파일을 정리하고 있습니다…")

    def _open_output_folder(self) -> None:
        output = Path(self.output_var.get().strip())
        folder = output.parent if output.parent.is_dir() else Path.cwd()
        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]

    def _on_close(self) -> None:
        if self._worker and self._worker.is_alive():
            if not messagebox.askyesno(
                "ReazonSubtitle",
                "진행 중인 작업을 취소하고 종료할까요?",
                parent=self,
            ):
                return
            self._closing = True
            self._cancel()
            self.withdraw()
            return
        self.destroy()


def main() -> int:
    configure_windows_process()
    if "--self-test" in sys.argv[1:]:
        from audio_pipeline import create_reazon_recognizer
        from resource_paths import assets_dir
        from translator_service import BrowserTranslator

        probe = tk.Tk()
        probe.withdraw()
        probe.update_idletasks()
        probe.destroy()
        recognizer = create_reazon_recognizer(assets_dir(), "int8", 1)
        del recognizer
        translator = BrowserTranslator("mt2")
        translator.validate_assets()
        # In a windowed PyInstaller build stdout/stderr are None. Constructing
        # the embedded server config here protects the frozen-only path that
        # previously crashed inside Uvicorn's color formatter.
        config = translator._uvicorn_config()
        if config.log_config is not None:
            raise RuntimeError("Uvicorn console logging must be disabled in the GUI EXE.")
        return 0
    app = ReazonSubtitleApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
