from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_pipeline import (
    SAMPLE_RATE,
    SubtitleEntry,
    build_fast_segments,
    samples_to_srt_timestamp,
    write_srt_atomic,
)


class AudioPipelineTests(unittest.TestCase):
    def test_timestamp_rounds_from_absolute_samples(self) -> None:
        self.assertEqual(samples_to_srt_timestamp(0), "00:00:00,000")
        self.assertEqual(samples_to_srt_timestamp(SAMPLE_RATE + 8), "00:00:01,001")
        self.assertEqual(
            samples_to_srt_timestamp((3 * 3600 + 2 * 60 + 1) * SAMPLE_RATE),
            "03:02:01,000",
        )

    def test_fast_forced_cut_keeps_context_overlap(self) -> None:
        audio = np.zeros(25 * SAMPLE_RATE, dtype=np.float32)
        segments = build_fast_segments(audio, [(0, len(audio))])
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].end_sample, 20 * SAMPLE_RATE)
        self.assertEqual(segments[1].start_sample, int(19.6 * SAMPLE_RATE))
        self.assertEqual(segments[1].overlap_samples, int(0.4 * SAMPLE_RATE))

    def test_srt_is_utf8_bom_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "subtitle.srt"
            count = write_srt_atomic(
                [
                    SubtitleEntry(
                        start_sample=0,
                        end_sample=2 * SAMPLE_RATE,
                        korean="안녕하세요.",
                    )
                ],
                output,
            )
            self.assertEqual(count, 1)
            self.assertTrue(output.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertIn("안녕하세요.", output.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
