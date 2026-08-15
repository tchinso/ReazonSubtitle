from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from audio_pipeline import extract_original_audio, probe_media
from resource_paths import assets_dir
from subtitle_processor import process_video
from translator_service import BrowserTranslator


class LocalAssetSmokeTests(unittest.TestCase):
    def test_uvicorn_config_is_safe_without_console_streams(self) -> None:
        translator = BrowserTranslator("mt2")
        with (
            mock.patch.object(sys, "stdout", None),
            mock.patch.object(sys, "stderr", None),
        ):
            config = translator._uvicorn_config()
        self.assertIsNone(config.log_config)
        self.assertFalse(config.access_log)

    def test_delayed_audio_pts_is_preserved_as_leading_silence(self) -> None:
        root = assets_dir()
        ffmpeg = root / "ffmpeg" / "ffmpeg.exe"
        ffprobe = root / "ffmpeg" / "ffprobe.exe"
        if not ffmpeg.is_file() or not ffprobe.is_file():
            self.skipTest("FFmpeg assets are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            video = folder / "delayed.mp4"
            wav = folder / "delayed.wav"
            completed = subprocess.run(
                [
                    str(ffmpeg),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=160x90:r=24:d=4",
                    "-itsoffset",
                    "1",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=2",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-t",
                    "4",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    str(video),
                ],
                capture_output=True,
                check=False,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
            media = probe_media(video, ffprobe)
            extract_original_audio(video, wav, ffmpeg, media.duration)
            with wave.open(str(wav), "rb") as source:
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
            self.assertGreaterEqual(len(samples), 47_000)
            self.assertLess(float(np.abs(samples[:12_000]).mean()), 2.0)
            self.assertGreater(float(np.abs(samples[18_000:30_000]).mean()), 100.0)

    def test_silent_video_produces_empty_srt_without_translation_worker(self) -> None:
        root = assets_dir()
        ffmpeg = root / "ffmpeg" / "ffmpeg.exe"
        if not ffmpeg.is_file():
            self.skipTest("FFmpeg asset is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            video = folder / "silent.mp4"
            output = folder / "silent.ko.srt"
            completed = subprocess.run(
                [
                    str(ffmpeg),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=320x180:r=24:d=2",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=48000:cl=stereo:d=2",
                    "-shortest",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    str(video),
                ],
                capture_output=True,
                check=False,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
            summary = process_video(video, output)
            self.assertEqual(summary.recognized, 0)
            self.assertEqual(output.read_text(encoding="utf-8-sig"), "")

    @unittest.skipUnless(
        os.environ.get("REAZON_RUN_TRANSLATION_SMOKE") == "1",
        "set REAZON_RUN_TRANSLATION_SMOKE=1 for the multi-gigabyte model smoke test",
    )
    def test_hytrans_local_model(self) -> None:
        with BrowserTranslator("mt2") as translator:
            translator.wait_ready(timeout=600, progress=print)
            result = translator.translate("こんにちは。", timeout=600)
            self.assertTrue(result.strip())
            print("HYTrans smoke result:", result)


if __name__ == "__main__":
    unittest.main()
