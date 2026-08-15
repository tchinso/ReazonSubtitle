from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import socket
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio_pipeline import CancelledError
from resource_paths import assets_dir, chrome_profile_root


@dataclass(frozen=True)
class TranslationModel:
    key: str
    label: str
    model_id: str
    revision: str
    dtype: str
    prompt: str
    files: dict[str, int]


MODELS: dict[str, TranslationModel] = {
    "mt2": TranslationModel(
        key="mt2",
        label="Hy-MT2 1.8B (권장)",
        model_id="tchinso/Hy-MT2-1.8B-onnx-q4f16",
        revision="6b6a4f12235342ed00ac089159c7192ea40bf6e8",
        dtype="q4f16",
        prompt=(
            "Translate the following text into {target}. Note that you should only "
            "output the translated result without any additional explanation:\n\n{text}"
        ),
        files={
            "chat_template.jinja": 654,
            "config.json": 1_518,
            "generation_config.json": 221,
            "special_tokens_map.json": 488,
            "tokenizer.json": 9_527_287,
            "tokenizer_config.json": 166_491,
            "onnx/model_q4f16.onnx": 1_373_443_906,
        },
    ),
    "mt1.5": TranslationModel(
        key="mt1.5",
        label="HY-MT1.5 1.8B (호환)",
        model_id="onnx-community/HY-MT1.5-1.8B-ONNX",
        revision="2f11819b25de08cecd344735cdfa5136ade41a67",
        dtype="q4",
        prompt=(
            "Translate the following segment into {target}, without additional "
            "explanation.\n\n{text}"
        ),
        files={
            "config.json": 1_639,
            "generation_config.json": 255,
            "tokenizer.json": 8_672_322,
            "tokenizer_config.json": 1_172,
            "onnx/model_q4.onnx": 448_829,
            "onnx/model_q4.onnx_data": 1_405_788_224,
        },
    ),
}


class ModelFileResponse(FileResponse):
    chunk_size = 8 * 1024 * 1024


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def find_chromium() -> Path | None:
    candidates = [
        shutil.which("msedge"),
        shutil.which("msedge.exe"),
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class BrowserTranslator:
    """Run the local HYTrans ONNX model in a private headless Chromium worker."""

    def __init__(self, model_key: str = "mt2") -> None:
        self.model = MODELS.get(model_key, MODELS["mt2"])
        self.port = _reserve_port()
        self._app = FastAPI(title="ReazonSubtitle HYTrans Worker")
        self._server: uvicorn.Server | None = None
        self._server_thread: threading.Thread | None = None
        self._browser: subprocess.Popen | None = None
        self._browser_started_at = 0.0
        self._profile_dir: Path | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._websocket: WebSocket | None = None
        self._pending: dict[str, asyncio.Future[str]] = {}
        self._server_ready = threading.Event()
        self._worker_ready = threading.Event()
        self._state_lock = threading.RLock()
        self._state = "대기 중"
        self._error = ""
        self._device = ""
        self._build_app()

    @property
    def runtime_root(self) -> Path:
        return assets_dir() / "hytrans-runtime"

    @property
    def model_root(self) -> Path:
        return assets_dir().joinpath(*self.model.model_id.split("/"))

    def _set_state(self, state: str, error: str = "") -> None:
        with self._state_lock:
            self._state = str(state)
            self._error = str(error)

    def status(self) -> tuple[str, str]:
        with self._state_lock:
            return self._state, self._error

    def validate_assets(self) -> None:
        runtime_files = (
            "worker.html",
            "worker.js",
            "transformers.min.js",
            "wasm/ort-wasm-simd-threaded.asyncify.mjs",
            "wasm/ort-wasm-simd-threaded.asyncify.wasm",
        )
        for relative in runtime_files:
            path = self.runtime_root.joinpath(*relative.split("/"))
            if not path.is_file():
                raise FileNotFoundError(f"HyTrans 실행 파일이 없습니다: {path}")
        for relative, expected_size in self.model.files.items():
            path = self.model_root.joinpath(*relative.split("/"))
            try:
                valid = path.is_file() and path.stat().st_size == expected_size
            except OSError:
                valid = False
            if not valid:
                raise FileNotFoundError(f"HyTrans 모델 파일이 없거나 손상되었습니다: {path}")
        if find_chromium() is None:
            raise RuntimeError("Microsoft Edge 또는 Google Chrome을 찾을 수 없습니다.")

    def _build_app(self) -> None:
        service = self

        @self._app.on_event("startup")
        async def startup() -> None:
            service._loop = asyncio.get_running_loop()
            service._server_ready.set()

        @self._app.get("/health")
        async def health() -> dict[str, object]:
            state, error = service.status()
            return {
                "ok": True,
                "ready": service._worker_ready.is_set(),
                "state": state,
                "error": error,
                "device": service._device,
            }

        @self._app.get("/config")
        async def config() -> dict[str, object]:
            return {
                "modelKey": service.model.key,
                "modelId": service.model.model_id,
                "revision": service.model.revision,
                "dtype": service.model.dtype,
                "promptTemplate": service.model.prompt,
                "modelMode": "local",
                "modelFiles": dict(service.model.files),
                "source": "Japanese",
                "target": "Korean",
                "maxNewTokens": 2048,
                "hasLocalWasm": True,
                "debugLog": False,
            }

        @self._app.get("/worker.html")
        async def worker_html() -> FileResponse:
            return FileResponse(service.runtime_root / "worker.html")

        @self._app.api_route("/models/{relative_path:path}", methods=["GET", "HEAD"])
        async def local_model(relative_path: str) -> FileResponse:
            root = assets_dir().resolve()
            target = (root / Path(relative_path)).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="model file not found") from exc
            if not target.is_file():
                raise HTTPException(status_code=404, detail="model file not found")
            return ModelFileResponse(target)

        @self._app.post("/client-log")
        async def client_log() -> dict[str, bool]:
            return {"ok": True}

        @self._app.websocket("/ws/worker")
        async def worker_socket(websocket: WebSocket) -> None:
            await websocket.accept()
            if service._websocket is not None:
                try:
                    await service._websocket.close(code=1001)
                except Exception:
                    pass
            service._websocket = websocket
            service._worker_ready.clear()
            service._set_state("번역 모델에 연결했습니다.")
            try:
                while True:
                    message = json.loads(await websocket.receive_text())
                    kind = message.get("type")
                    if kind == "loading":
                        service._set_state(str(message.get("message") or "번역 모델 로드 중"))
                    elif kind == "ready":
                        if (
                            str(message.get("model") or "") != service.model.model_id
                            or str(message.get("dtype") or "") != service.model.dtype
                        ):
                            raise RuntimeError("HyTrans worker 모델 구성이 일치하지 않습니다.")
                        service._device = str(message.get("device") or "")
                        service._set_state(f"번역 모델 준비 완료 ({service._device})")
                        service._worker_ready.set()
                    elif kind in {"result", "error"}:
                        request_id = str(message.get("id") or "")
                        future = service._pending.get(request_id)
                        if future is not None and not future.done():
                            if kind == "result":
                                future.set_result(str(message.get("text") or ""))
                            else:
                                future.set_exception(
                                    RuntimeError(str(message.get("message") or "번역 실패"))
                                )
                    elif kind == "fatal":
                        raise RuntimeError(str(message.get("message") or "번역 worker 오류"))
            except WebSocketDisconnect:
                service._set_state("번역 worker 연결이 종료되었습니다.", "worker disconnected")
            except Exception as exc:
                service._set_state("번역 worker 오류", str(exc))
            finally:
                if service._websocket is websocket:
                    service._websocket = None
                    service._worker_ready.clear()
                    for future in list(service._pending.values()):
                        if not future.done():
                            future.set_exception(RuntimeError("번역 worker 연결이 종료되었습니다."))

        self._app.mount(
            "/assets",
            StaticFiles(directory=str(self.runtime_root), check_dir=False),
            name="assets",
        )

    def start(self) -> None:
        self.validate_assets()
        self._set_state("HyTrans 로컬 서버를 시작하고 있습니다.")
        config = self._uvicorn_config()
        self._server = uvicorn.Server(config)
        self._server_thread = threading.Thread(
            target=self._server.run,
            name="ReazonSubtitle-HYTrans",
            daemon=True,
        )
        self._server_thread.start()
        if not self._server_ready.wait(timeout=15):
            raise RuntimeError("HyTrans 로컬 서버가 시작되지 않았습니다.")

        browser = find_chromium()
        assert browser is not None
        self._profile_dir = chrome_profile_root() / uuid.uuid4().hex
        self._profile_dir.mkdir(parents=True, exist_ok=False)
        worker_url = f"http://127.0.0.1:{self.port}/worker.html"
        command = [
            str(browser),
            "--headless=new",
            "--edge-skip-compat-layer-relaunch",
            f"--user-data-dir={self._profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-background-mode",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
            "--disable-gpu-sandbox",
            "--window-size=800,600",
            worker_url,
        ]
        self._set_state("HyTrans 브라우저 worker를 시작하고 있습니다.")
        self._browser = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_creation_flags(),
        )
        self._browser_started_at = time.monotonic()

    def _uvicorn_config(self) -> uvicorn.Config:
        """Create a server config that is safe in a windowed PyInstaller EXE.

        A ``console=False`` executable intentionally has no stdout/stderr.
        Uvicorn's default color formatter probes ``sys.stderr.isatty()`` while
        Config is constructed and crashes before the local translation server
        starts. The app does not need console logging, so disable that logging
        dictionary entirely instead of attaching a dummy global stream.
        """

        return uvicorn.Config(
            self._app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            access_log=False,
            log_config=None,
            use_colors=False,
        )

    def wait_ready(
        self,
        timeout: float = 600,
        *,
        progress: Callable[[str], None] | None = None,
        cancel_event=None,
    ) -> None:
        deadline = time.monotonic() + timeout
        last_state = ""
        while not self._worker_ready.wait(timeout=0.2):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("사용자가 작업을 취소했습니다.")
            state, error = self.status()
            if state != last_state and progress:
                progress(state)
                last_state = state
            if error:
                raise RuntimeError(f"HyTrans 준비 실패: {error}")
            if (
                self._browser is not None
                and self._browser.poll() is not None
                and self._websocket is None
                and time.monotonic() - self._browser_started_at > 15
            ):
                raise RuntimeError("HyTrans 브라우저 worker가 예기치 않게 종료되었습니다.")
            if time.monotonic() >= deadline:
                raise TimeoutError("HyTrans 모델 준비 시간이 10분을 초과했습니다.")
        if progress:
            progress(self.status()[0])

    async def _translate_async(self, text: str, timeout: float) -> str:
        websocket = self._websocket
        if websocket is None or not self._worker_ready.is_set():
            raise RuntimeError("HyTrans 번역 모델이 준비되지 않았습니다.")
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending[request_id] = future
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "translate",
                        "id": request_id,
                        "text": text,
                        "max_new_tokens": 2048,
                    },
                    ensure_ascii=False,
                )
            )
            result = (await asyncio.wait_for(future, timeout=timeout)).strip()
            if not result:
                raise RuntimeError("HyTrans가 빈 번역 결과를 반환했습니다.")
            return result
        finally:
            self._pending.pop(request_id, None)

    def translate(self, text: str, timeout: float = 600, *, cancel_event=None) -> str:
        if self._loop is None:
            raise RuntimeError("HyTrans 이벤트 루프가 없습니다.")
        future = asyncio.run_coroutine_threadsafe(
            self._translate_async(text, timeout),
            self._loop,
        )
        deadline = time.monotonic() + timeout + 5
        while True:
            try:
                return future.result(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
            except concurrent.futures.TimeoutError:
                if cancel_event is not None and cancel_event.is_set():
                    future.cancel()
                    raise CancelledError("사용자가 작업을 취소했습니다.")
                if time.monotonic() >= deadline:
                    future.cancel()
                    raise TimeoutError("HyTrans 번역 시간이 초과되었습니다.")

    def _stop_browser(self) -> None:
        process = self._browser
        self._browser = None
        if process is None:
            return
        if os.name == "nt" and process.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                    check=False,
                    creationflags=_creation_flags(),
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass

    def _stop_profile_processes(self) -> None:
        """Stop an Edge compatibility relaunch that escaped the Popen handle."""

        profile = self._profile_dir
        if os.name != "nt" or profile is None:
            return
        environment = os.environ.copy()
        environment["REAZON_SUBTITLE_CHROME_PROFILE"] = str(profile)
        script = (
            "$needle=$env:REAZON_SUBTITLE_CHROME_PROFILE; "
            "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe' OR Name='chrome.exe'\" "
            "| Where-Object { $_.CommandLine -and $_.CommandLine.Contains($needle) } "
            "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
                creationflags=_creation_flags(),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def stop(self) -> None:
        self._worker_ready.clear()
        self._stop_browser()
        self._stop_profile_processes()
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=10)
        self._server = None
        self._server_thread = None
        profile = self._profile_dir
        self._profile_dir = None
        if profile is not None:
            try:
                root = chrome_profile_root().resolve()
                resolved = profile.resolve()
                if resolved != root and resolved.is_relative_to(root):
                    def remove_readonly(function, path, _error) -> None:
                        os.chmod(path, stat.S_IWRITE)
                        function(path)

                    for _attempt in range(30):
                        try:
                            shutil.rmtree(resolved, onerror=remove_readonly)
                        except OSError:
                            pass
                        if not resolved.exists():
                            break
                        time.sleep(0.25)
            except OSError:
                pass
        self._set_state("종료됨")

    def __enter__(self) -> "BrowserTranslator":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
