from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from stream import render_report

from benchmarks.stream_report import (
    EVENT_NAMES,
    paired_outcome,
    validate_measurement,
    validate_raw_record,
    validate_silent_binary,
)


def measurement() -> dict[str, object]:
    return {
        "schema_version": 2,
        "timing_origin": "benchmark_cli_call_entry",
        "benchmark_safety_marker": "KOKORO_STREAM_MUTED_OUTPUT_V1",
        "event_seconds": dict(zip(EVENT_NAMES, (0.5, 0.6, 0.61, 0.8), strict=True)),
        "synthesis_wall_seconds": 0.4,
        "synthesis_thread_cpu_seconds": 0.38,
        "audio_queue_open_wall_seconds": 0.03,
        "audio_queue_open_complete_seconds": 0.4,
        "audio_queue_start_api_seconds": 0.1,
        "submit_backpressure_wall_seconds": 0.0,
        "playback_wall_seconds": 2.0,
        "process_wall_seconds": 2.6,
        "process_cpu_seconds": 0.6,
        "generated_sample_count": 48000,
        "generated_audio_seconds": 2.0,
        "chunk_count": 2,
        "queue_bound_bytes": 49152,
        "queue_bound_audio_seconds": 0.512,
        "queue_starvation_observations": 0,
        "acoustic_onset_seconds": None,
        "acoustic_onset_status": "unavailable",
        "audio_queue_volume": 0.0,
        "audio_queue_mute_status": "succeeded",
    }


def sample_record() -> dict[str, object]:
    return {
        "workload": "hello_from_terminal",
        "voice": "af_heart",
        "command": ["kokoro-stream-benchmark", "say", "hello", "--stream"],
        "driver_json": '{"schema_version":1}',
        "measurement": measurement(),
    }


def raw_record() -> dict[str, object]:
    sample = sample_record()
    pair = {
        "order": ["baseline", "candidate"],
        "baseline": {"driver_json": sample["driver_json"], "measurement": measurement()},
        "candidate": {"driver_json": sample["driver_json"], "measurement": measurement()},
    }
    return {
        "schema_version": 1,
        "artifact_directory": "artifacts/18-streaming-benchmark-fixture",
        "provenance": {
            "commit": "a" * 40,
            "source_sha256": "b" * 64,
            "host": {},
            "builds": {},
            "model": {},
            "voices": {},
        },
        "profile": {
            "status": "captured",
            "file": "profile.txt",
            "sha256": "c" * 64,
            "command": "sample 123 3 100",
        },
        "device_access": {
            "per_queue_volume_control": "AudioQueueSetParameter(kAudioQueueParam_Volume, 0.0f)",
            "system_output_volume_changed": False,
            "sandbox_probe": {
                "command": ["kokoro-stream-benchmark", "say", "hello"],
                "operation": "AudioQueueStart",
                "osstatus": -66680,
                "per_queue_mute_status": "succeeded before PCM enqueue",
                "system_output_volume_changed": False,
            },
            "escalated_muted_collection": "succeeded",
        },
        "streaming_samples": [sample],
        "optimization": {
            "description": "Profile-guided candidate description.",
            "paired_runs": [pair],
            "outcomes": {},
            "disposition": "Rejected under fixed parity limits.",
            "candidate_evidence": {
                "patch": {"file": "candidate.patch", "sha256": "d" * 64},
                "check_log": {"file": "candidate-check.log", "sha256": "e" * 64},
            },
        },
        "verification": {
            "checks": [{"command": "make check", "result": "passed"}],
            "independent_review": "approved",
        },
        "created_at_utc": "2026-10-12T16:19:21Z",
        "rerun_command": "set -e\ntrap cleanup EXIT",
    }


class StreamingReportTests(unittest.TestCase):
    def test_validates_events_process_scopes_and_audio_queue_bounds(self) -> None:
        validate_measurement(measurement())

    def test_rejects_missing_event_or_invalid_partial_order(self) -> None:
        value = measurement()
        value["event_seconds"] = dict(value["event_seconds"])
        value["event_seconds"]["first_buffer_accepted"] = 0.45
        with self.assertRaisesRegex(ValueError, "partial order"):
            validate_measurement(value)

    def test_reclamation_during_start_api_call_is_valid(self) -> None:
        value = measurement()
        value["event_seconds"] = dict(value["event_seconds"])
        # AudioQueue callbacks may run before AudioQueueStart returns successfully.
        value["event_seconds"]["first_buffer_reclaimed"] = 0.605
        validate_measurement(value)

    def test_rejects_nonzero_device_volume_and_fabricated_acoustic_timing(self) -> None:
        value = measurement()
        value["audio_queue_volume"] = 0.5
        with self.assertRaisesRegex(ValueError, "volume of zero"):
            validate_measurement(value)
        value = measurement()
        value["acoustic_onset_seconds"] = 0.1
        with self.assertRaisesRegex(ValueError, "independent"):
            validate_measurement(value)
        value = measurement()
        value["audio_queue_mute_status"] = "unavailable"
        with self.assertRaisesRegex(ValueError, "successful per-queue mute"):
            validate_measurement(value)

    def test_requires_fail_closed_silent_binary_marker(self) -> None:
        validate_silent_binary(b"benchmark KOKORO_STREAM_MUTED_OUTPUT_V1 binary")
        with self.assertRaisesRegex(ValueError, "fail-closed silent-output"):
            validate_silent_binary(b"ordinary executable")
        value = measurement()
        del value["benchmark_safety_marker"]
        value["schema_version"] = 1
        validate_measurement(value)

    def test_raw_record_retains_command_provenance_samples_and_pairs(self) -> None:
        validate_raw_record(raw_record())
        value = raw_record()
        value["streaming_samples"][0]["driver_json"] = ""
        with self.assertRaisesRegex(ValueError, "raw driver JSON"):
            validate_raw_record(value)

    def test_pair_threshold_comes_from_observed_sample_range(self) -> None:
        pairs = []
        for before, after in ((1.00, 0.95), (1.02, 0.96), (0.99, 0.94)):
            pairs.append(
                {
                    "baseline": {"measurement": {"synthesis_wall_seconds": before}},
                    "candidate": {"measurement": {"synthesis_wall_seconds": after}},
                }
            )
        result = paired_outcome(pairs)
        self.assertEqual(result["pair_count"], 3)
        self.assertGreater(result["contemporaneous_range_threshold_pct"], 0)
        self.assertTrue(result["clears_observed_variation"])

    def test_report_keeps_probe_outside_tables_and_links_rejected_evidence(self) -> None:
        report = render_report(raw_record())
        self.assertLess(report.index("The sandbox probe"), report.index("## Streaming samples"))
        self.assertIn("## Synthesis and AudioQueue timing", report)
        self.assertIn("## Generated audio and bounded queue", report)
        self.assertIn("potential playback-gap condition", report)
        self.assertIn("[`candidate.patch`](candidate.patch)", report)
        self.assertIn("[`candidate-check.log`](candidate-check.log)", report)
        self.assertIn("## Verification", report)
        self.assertIn("`make check` — passed", report)
        self.assertIn("Independent review: approved", report)


if __name__ == "__main__":
    unittest.main()
