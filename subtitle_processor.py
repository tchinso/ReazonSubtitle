from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from audio_pipeline import (
    CancelledError,
    SubtitleEntry,
    build_fast_segments,
    collect_vad_intervals,
    create_reazon_recognizer,
    extract_original_audio,
    probe_media,
    recognize_segments,
    wav_to_float32_memmap,
    write_srt_atomic,
)
from resource_paths import assets_dir, work_root
from translator_service import BrowserTranslator


@dataclass(frozen=True)
class ProcessSummary:
    output_path: Path
    media_duration: float
    vad_intervals: int
    recognized: int
    translated: int
    translation_failures: int
    elapsed: float


StatusCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]


_ASSISTANT_PREFIX = re.compile(r"^assistant\s*[:：]?\s*", re.IGNORECASE)


def _threads() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _clean_translation(text: str) -> str:
    result = _ASSISTANT_PREFIX.sub("", str(text).strip()).strip()
    if len(result) >= 2 and result[0] == result[-1] and result[0] in {'"', "'"}:
        result = result[1:-1].strip()
    return result


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("사용자가 작업을 취소했습니다.")


def process_video(
    input_path: Path,
    output_path: Path,
    *,
    precision: str = "int8",
    translation_model: str = "mt2",
    status: StatusCallback | None = None,
    log: LogCallback | None = None,
    cancel_event=None,
) -> ProcessSummary:
    started = time.perf_counter()
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    root = assets_dir()
    ffmpeg = root / "ffmpeg" / "ffmpeg.exe"
    ffprobe = root / "ffmpeg" / "ffprobe.exe"
    vad_model = root / "vad" / "silero_vad.onnx"
    for required in (ffmpeg, ffprobe, vad_model):
        if not required.is_file():
            raise FileNotFoundError(f"필수 파일이 없습니다: {required}")
    if not input_path.is_file():
        raise FileNotFoundError(f"영상 파일이 없습니다: {input_path}")
    if input_path == output_path:
        raise ValueError("입력 영상과 출력 SRT 경로가 같습니다.")

    if status:
        status(0.0, "영상의 원본 오디오 스트림을 확인하고 있습니다.")
    media = probe_media(input_path, ffprobe)
    if log:
        log(
            f"오디오 스트림 #{media.audio_stream_index} ({media.codec_name}), "
            f"영상 길이 {media.duration:.2f}초"
        )

    translator: BrowserTranslator | None = None
    audio: np.memmap | None = None
    with tempfile.TemporaryDirectory(prefix="job-", dir=work_root()) as temporary:
        temporary_dir = Path(temporary)
        wav_path = temporary_dir / "source-audio-16k.wav"
        raw_path = temporary_dir / "source-audio-16k.f32"
        try:
            extract_original_audio(
                input_path,
                wav_path,
                ffmpeg,
                media.duration,
                progress=(
                    (lambda ratio, message: status(0.02 + ratio * 0.13, message))
                    if status
                    else None
                ),
                cancel_event=cancel_event,
            )
            _check_cancel(cancel_event)
            if status:
                status(0.16, "추출한 오디오를 정밀 타임라인으로 변환하고 있습니다.")
            audio = wav_to_float32_memmap(wav_path, raw_path)

            intervals = collect_vad_intervals(
                audio,
                vad_model,
                progress=(
                    (lambda ratio, message: status(0.18 + ratio * 0.17, message))
                    if status
                    else None
                ),
                cancel_event=cancel_event,
                num_threads=_threads(),
            )
            segments = build_fast_segments(audio, intervals)
            if log:
                log(f"FAST VAD: 원시 구간 {len(intervals)}개, 인식 구간 {len(segments)}개")
            if not segments:
                write_srt_atomic([], output_path)
                if status:
                    status(1.0, "음성이 없어 빈 SRT를 저장했습니다.")
                return ProcessSummary(
                    output_path=output_path,
                    media_duration=media.duration,
                    vad_intervals=len(intervals),
                    recognized=0,
                    translated=0,
                    translation_failures=0,
                    elapsed=time.perf_counter() - started,
                )

            # Model loading overlaps with Reazon recognition. The worker stays
            # private to this job and is always stopped in the finally block.
            translator = BrowserTranslator(translation_model)
            translator.start()
            if status:
                status(0.36, "Reazon 일본어 인식 모델을 불러오고 있습니다.")
            recognizer = create_reazon_recognizer(root, precision, _threads())
            results = recognize_segments(
                recognizer,
                segments,
                progress=(
                    (lambda ratio, message: status(0.38 + ratio * 0.27, message))
                    if status
                    else None
                ),
                cancel_event=cancel_event,
            )
            del recognizer
            if log:
                for result in results:
                    log(
                        f"[{result.start_sample / 16000:.3f}–"
                        f"{result.end_sample / 16000:.3f}] JA  {result.japanese}"
                    )
            if not results:
                write_srt_atomic([], output_path)
                if status:
                    status(1.0, "인식된 일본어가 없어 빈 SRT를 저장했습니다.")
                return ProcessSummary(
                    output_path=output_path,
                    media_duration=media.duration,
                    vad_intervals=len(intervals),
                    recognized=0,
                    translated=0,
                    translation_failures=0,
                    elapsed=time.perf_counter() - started,
                )

            if status:
                status(0.66, "HyTrans 번역 모델이 준비되기를 기다리고 있습니다.")
            translator.wait_ready(
                timeout=600,
                progress=(
                    (lambda message: status(0.68, message)) if status else None
                ),
                cancel_event=cancel_event,
            )

            entries: list[SubtitleEntry] = []
            failures = 0
            for index, result in enumerate(results, 1):
                _check_cancel(cancel_event)
                if status:
                    status(
                        0.70 + (index - 1) / len(results) * 0.28,
                        f"한국어 번역 중 ({index}/{len(results)})",
                    )
                translated = ""
                last_error: Exception | None = None
                for attempt in range(2):
                    try:
                        translated = _clean_translation(
                            translator.translate(
                                result.japanese,
                                timeout=600,
                                cancel_event=cancel_event,
                            )
                        )
                        if not translated:
                            raise RuntimeError("빈 번역 결과")
                        break
                    except Exception as exc:
                        if isinstance(exc, CancelledError):
                            raise
                        last_error = exc
                        if attempt == 0:
                            time.sleep(0.2)
                if not translated:
                    failures += 1
                    translated = f"[번역 실패] {result.japanese}"
                    if log:
                        log(f"번역 실패: {last_error}")
                elif log:
                    log(f"KO  {translated}")
                entries.append(
                    SubtitleEntry(
                        start_sample=result.start_sample,
                        end_sample=result.end_sample,
                        korean=translated,
                        japanese=result.japanese,
                    )
                )

            _check_cancel(cancel_event)
            count = write_srt_atomic(entries, output_path)
            if status:
                status(1.0, f"완료: 한국어 자막 {count}개를 저장했습니다.")
            return ProcessSummary(
                output_path=output_path,
                media_duration=media.duration,
                vad_intervals=len(intervals),
                recognized=len(results),
                translated=count - failures,
                translation_failures=failures,
                elapsed=time.perf_counter() - started,
            )
        finally:
            if translator is not None:
                translator.stop()
            if isinstance(audio, np.memmap):
                try:
                    audio._mmap.close()
                except Exception:
                    pass
