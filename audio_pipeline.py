from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


SAMPLE_RATE = 16_000

FAST_VAD: dict[str, float] = {
    "threshold": 0.50,
    "min_speech_duration": 0.20,
    "min_silence_duration": 0.25,
    "max_segment_duration": 20.0,
    "pre_padding": 0.15,
    "post_padding": 0.35,
    "merge_gap": 0.15,
    "merge_short_under": 0.80,
    "forced_cut_overlap": 0.40,
}


class CancelledError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    audio_stream_index: int
    codec_name: str


@dataclass
class SpeechSegment:
    id: int
    start_sample: int
    end_sample: int
    audio: np.ndarray
    forced_cut: bool
    overlap_samples: int = 0

    @property
    def duration(self) -> float:
        return (self.end_sample - self.start_sample) / SAMPLE_RATE


@dataclass(frozen=True)
class RecognitionResult:
    segment_id: int
    start_sample: int
    end_sample: int
    japanese: str
    latency: float


@dataclass(frozen=True)
class SubtitleEntry:
    start_sample: int
    end_sample: int
    korean: str
    japanese: str = ""


ProgressCallback = Callable[[float, str], None]


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("사용자가 작업을 취소했습니다.")


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def probe_media(input_path: Path, ffprobe_path: Path) -> MediaInfo:
    command = [
        str(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index,codec_name,duration:format=duration",
        "-of",
        "json",
        str(input_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "FFprobe가 파일을 읽지 못했습니다."
        raise RuntimeError(detail)
    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams") or []
        if not streams:
            raise ValueError("audio stream missing")
        stream = streams[0]
        raw_duration = stream.get("duration") or payload.get("format", {}).get("duration")
        duration = max(0.0, float(raw_duration or 0.0))
        return MediaInfo(
            duration=duration,
            audio_stream_index=int(stream.get("index", 0)),
            codec_name=str(stream.get("codec_name") or "unknown"),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("영상에서 오디오 스트림 정보를 확인할 수 없습니다.") from exc


def extract_original_audio(
    input_path: Path,
    wav_path: Path,
    ffmpeg_path: Path,
    duration: float,
    *,
    progress: ProgressCallback | None = None,
    cancel_event=None,
) -> None:
    """Decode only the first source audio stream to analysis PCM.

    ``aresample=async=1:first_pts=0`` preserves gaps represented by timestamps
    as silence and trims negative timestamps. The resulting sample index is
    therefore directly usable as the SRT timeline without recording the
    speaker output or touching the video stream.
    """

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg_path),
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        "aresample=async=1:first_pts=0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-progress",
        "pipe:1",
        "-nostats",
        str(wav_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
    )
    messages: list[str] = []
    assert process.stdout is not None
    try:
        while True:
            _check_cancel(cancel_event)
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                time.sleep(0.02)
                continue
            line = line.strip()
            if not line:
                continue
            key, separator, value = line.partition("=")
            if separator and key in {"out_time_us", "out_time_ms"}:
                try:
                    # FFmpeg currently reports both fields in microseconds.
                    elapsed = int(value) / 1_000_000.0
                    ratio = min(1.0, elapsed / duration) if duration > 0 else 0.0
                    if progress:
                        progress(ratio, "원본 오디오를 추출하고 있습니다.")
                except ValueError:
                    pass
            elif key not in {"progress", "bitrate", "speed", "total_size"}:
                messages.append(line)
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        raise
    finally:
        process.stdout.close()
    if return_code != 0 or not wav_path.is_file() or wav_path.stat().st_size <= 44:
        detail = "\n".join(messages[-8:]).strip()
        raise RuntimeError(detail or "FFmpeg 오디오 추출에 실패했습니다.")
    if progress:
        progress(1.0, "원본 오디오 추출을 완료했습니다.")


def wav_to_float32_memmap(wav_path: Path, raw_path: Path) -> np.memmap:
    with wave.open(str(wav_path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("추출된 오디오가 mono PCM16 형식이 아닙니다.")
        if source.getframerate() != SAMPLE_RATE:
            raise ValueError(f"추출된 오디오 샘플레이트가 {SAMPLE_RATE} Hz가 아닙니다.")
        frame_count = source.getnframes()
        if frame_count <= 0:
            raise ValueError("추출된 오디오가 비어 있습니다.")
        target = np.memmap(raw_path, dtype=np.float32, mode="w+", shape=(frame_count,))
        written = 0
        while written < frame_count:
            block = source.readframes(min(SAMPLE_RATE * 30, frame_count - written))
            if not block:
                break
            samples = np.frombuffer(block, dtype="<i2").astype(np.float32)
            samples *= 1.0 / 32768.0
            target[written : written + len(samples)] = samples
            written += len(samples)
        target.flush()
    return np.memmap(raw_path, dtype=np.float32, mode="r", shape=(written,))


def collect_vad_intervals(
    audio: np.ndarray,
    vad_model: Path,
    *,
    progress: ProgressCallback | None = None,
    cancel_event=None,
    num_threads: int = 4,
) -> list[tuple[int, int]]:
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_model)
    config.silero_vad.threshold = FAST_VAD["threshold"]
    config.silero_vad.min_silence_duration = FAST_VAD["min_silence_duration"]
    config.silero_vad.min_speech_duration = FAST_VAD["min_speech_duration"]
    config.silero_vad.window_size = 512
    config.silero_vad.max_speech_duration = FAST_VAD["max_segment_duration"]
    config.sample_rate = SAMPLE_RATE
    config.num_threads = num_threads
    config.provider = "cpu"
    detector = sherpa_onnx.VoiceActivityDetector(config, 60.0)
    intervals: list[tuple[int, int]] = []

    def drain() -> None:
        while not detector.empty():
            item = detector.front
            start = max(0, int(item.start))
            end = min(len(audio), start + len(item.samples))
            if end > start:
                intervals.append((start, end))
            detector.pop()

    total_windows = max(1, math.ceil(len(audio) / 512))
    for index, start in enumerate(range(0, len(audio), 512)):
        if index % 64 == 0:
            _check_cancel(cancel_event)
            if progress:
                progress(index / total_windows, "VAD로 음성과 타임스탬프를 추적하고 있습니다.")
        window = np.asarray(audio[start : start + 512], dtype=np.float32)
        if len(window) < 512:
            window = np.pad(window, (0, 512 - len(window)))
        detector.accept_waveform(window)
        drain()
    detector.flush()
    drain()
    if progress:
        progress(1.0, f"VAD가 음성 구간 {len(intervals)}개를 찾았습니다.")
    return intervals


def build_fast_segments(
    audio: np.ndarray,
    intervals: Iterable[tuple[int, int]],
) -> list[SpeechSegment]:
    minimum = round(FAST_VAD["min_speech_duration"] * SAMPLE_RATE)
    merge_gap = round(FAST_VAD["merge_gap"] * SAMPLE_RATE)
    pre = round(FAST_VAD["pre_padding"] * SAMPLE_RATE)
    post = round(FAST_VAD["post_padding"] * SAMPLE_RATE)
    maximum = round(FAST_VAD["max_segment_duration"] * SAMPLE_RATE)
    overlap = round(FAST_VAD["forced_cut_overlap"] * SAMPLE_RATE)

    speech = [
        (max(0, int(start)), min(len(audio), int(end)))
        for start, end in intervals
        if int(end) - int(start) >= minimum
    ]
    merged: list[list[int]] = []
    for start, end in speech:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    padded: list[list[int]] = []
    for start, end in merged:
        candidate = [max(0, start - pre), min(len(audio), end + post)]
        if padded and candidate[0] <= padded[-1][1]:
            padded[-1][1] = max(padded[-1][1], candidate[1])
        else:
            padded.append(candidate)

    output: list[SpeechSegment] = []
    segment_id = 1
    for start, end in padded:
        cursor = start
        previous_overlap = 0
        while cursor < end:
            cut_end = min(cursor + maximum, end)
            forced = cut_end < end
            output.append(
                SpeechSegment(
                    id=segment_id,
                    start_sample=cursor,
                    end_sample=cut_end,
                    audio=np.asarray(audio[cursor:cut_end], dtype=np.float32),
                    forced_cut=forced or previous_overlap > 0,
                    overlap_samples=previous_overlap,
                )
            )
            segment_id += 1
            if not forced:
                break
            cursor = max(cursor + 1, cut_end - overlap)
            previous_overlap = overlap
    return output


def _validate_tokens(tokens_path: Path) -> None:
    seen: set[int] = set()
    for line_number, line in enumerate(
        tokens_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        fields = line.rsplit(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Reazon 토큰 파일 {line_number}행의 형식이 잘못되었습니다.")
        try:
            token_id = int(fields[1])
        except ValueError as exc:
            raise ValueError(f"Reazon 토큰 파일 {line_number}행의 ID가 잘못되었습니다.") from exc
        if token_id < 0 or token_id in seen:
            raise ValueError(f"Reazon 토큰 파일 {line_number}행의 ID가 중복되었습니다.")
        seen.add(token_id)
    if not seen or seen != set(range(max(seen) + 1)):
        raise ValueError("Reazon 토큰 파일의 ID가 연속적이지 않습니다.")


def reazon_model_paths(asset_root: Path, precision: str = "int8") -> dict[str, Path]:
    root = asset_root / "reazonspeech-ja"
    use_int8 = str(precision).lower() == "int8"
    paths = {
        "tokens": root / "tokens.txt",
        "encoder": root / (
            "encoder-epoch-99-avg-1.int8.onnx"
            if use_int8
            else "encoder-epoch-99-avg-1.onnx"
        ),
        "decoder": root / "decoder-epoch-99-avg-1.onnx",
        "joiner": root / (
            "joiner-epoch-99-avg-1.int8.onnx"
            if use_int8
            else "joiner-epoch-99-avg-1.onnx"
        ),
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Reazon 모델 파일이 없습니다: {missing[0]}")
    return paths


def create_reazon_recognizer(asset_root: Path, precision: str = "int8", num_threads: int = 4):
    import sherpa_onnx

    models = reazon_model_paths(asset_root, precision)
    _validate_tokens(models["tokens"])
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(models["encoder"]),
        decoder=str(models["decoder"]),
        joiner=str(models["joiner"]),
        tokens=str(models["tokens"]),
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
        model_type="",
    )


def _remove_overlap(previous: str, current: str, limit: int = 40) -> str:
    previous = previous.strip()
    current = current.strip()
    for length in range(min(limit, len(previous), len(current)), 1, -1):
        if previous[-length:] == current[:length]:
            return current[length:].lstrip()
    return current


def recognize_segments(
    recognizer,
    segments: Iterable[SpeechSegment],
    *,
    progress: ProgressCallback | None = None,
    cancel_event=None,
) -> list[RecognitionResult]:
    items = list(segments)
    results: list[RecognitionResult] = []
    previous_text = ""
    for index, segment in enumerate(items, 1):
        _check_cancel(cancel_event)
        started = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, segment.audio)
        recognizer.decode_stream(stream)
        text = str(stream.result.text).strip()
        if segment.overlap_samples:
            text = _remove_overlap(previous_text, text)
        if text:
            # Forced chunks overlap only to preserve ASR context. The duplicate
            # context must not create overlapping SRT cues.
            effective_start = min(
                segment.end_sample,
                segment.start_sample + segment.overlap_samples,
            )
            results.append(
                RecognitionResult(
                    segment_id=segment.id,
                    start_sample=effective_start,
                    end_sample=segment.end_sample,
                    japanese=text,
                    latency=time.perf_counter() - started,
                )
            )
            previous_text = text
        if progress:
            progress(index / max(1, len(items)), f"Reazon 일본어 인식 중 ({index}/{len(items)})")
    return results


def samples_to_srt_timestamp(samples: int) -> str:
    total_ms = max(0, (int(samples) * 1000 + SAMPLE_RATE // 2) // SAMPLE_RATE)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


_SPACE_RE = re.compile(r"[ \t\u3000]+")


def clean_subtitle_text(text: str) -> str:
    lines = []
    for line in str(text).replace("\r", "\n").split("\n"):
        cleaned = _SPACE_RE.sub(" ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()


def write_srt_atomic(entries: Iterable[SubtitleEntry], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    blocks: list[str] = []
    for entry in entries:
        text = clean_subtitle_text(entry.korean)
        if not text:
            continue
        index = len(blocks) + 1
        start = samples_to_srt_timestamp(entry.start_sample)
        end = samples_to_srt_timestamp(max(entry.start_sample + 1, entry.end_sample))
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    temporary.write_text("\n".join(blocks), encoding="utf-8-sig", newline="\n")
    os.replace(temporary, destination)
    return len(blocks)
