import unittest
import json
import tempfile
import ctypes
from types import SimpleNamespace
from unittest.mock import patch

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import compare


def sample(workload, mode, seconds, *, implementation="native", text="same text"):
    return {
        "workload": workload,
        "mode": mode,
        "implementation": implementation,
        "input": text,
        "voice": "af_heart",
        "speed": 1.0,
        "provenance_sha256": "provenance",
        "wall_seconds": seconds,
    }


class ComparisonTests(unittest.TestCase):
    def test_comparison_keeps_raw_sample_count_and_only_calls_separated_ranges_a_win(self):
        result = compare.compare_samples(
            [sample("short", "english", 1.0), sample("short", "english", 1.1)],
            [sample("short", "english", 2.0, implementation="coreml"), sample("short", "english", 2.1, implementation="coreml")],
            "short",
            "english",
            "wall_seconds",
        )
        self.assertEqual(result["native"]["sample_count"], 2)
        self.assertEqual(result["coreml"]["sample_count"], 2)
        self.assertEqual(result["outcome"], "native_faster")

    def test_resident_native_and_coreml_counts_can_differ_by_design(self):
        result = compare.compare_samples(
            [sample("short", "english", 1.0 + index / 100) for index in range(25)],
            [sample("short", "english", 2.0, implementation="coreml")],
            "short",
            "english",
            "wall_seconds",
        )
        self.assertEqual(result["native"]["sample_count"], 25)
        self.assertEqual(result["coreml"]["sample_count"], 1)
        self.assertEqual(result["outcome"], "native_faster")

    def test_incomplete_or_mismatched_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            compare.compare_samples(
                [sample("short", "english", 1.0)], [], "short", "english", "wall_seconds"
            )
        with self.assertRaises(ValueError):
            compare.compare_samples(
                [sample("short", "english", 1.0)],
                [sample("short", "ipa", 2.0, implementation="coreml")],
                "short",
                "english",
                "wall_seconds",
            )
        with self.assertRaises(ValueError):
            compare.compare_samples(
                [sample("short", "english", 1.0)],
                [sample("short", "english", 2.0, implementation="coreml", text="other")],
                "short",
                "english",
                "wall_seconds",
            )
        incomplete = sample("short", "english", 1.0)
        del incomplete["wall_seconds"]
        with self.assertRaises(ValueError):
            compare.compare_samples(
                [incomplete],
                [sample("short", "english", 2.0, implementation="coreml")],
                "short",
                "english",
                "wall_seconds",
            )

    def test_different_provenance_hashes_are_rejected(self):
        native = sample("short", "english", 1.0)
        coreml = sample("short", "english", 2.0, implementation="coreml")
        coreml["provenance_sha256"] = "different provenance"
        with self.assertRaisesRegex(ValueError, "different provenance_sha256"):
            compare.compare_samples([native], [coreml], "short", "english", "wall_seconds")

    def test_overlapping_ranges_and_mixed_workload_winners_have_no_global_winner(self):
        overlap = compare.compare_samples(
            [sample("short", "english", 1.0), sample("short", "english", 3.0)],
            [sample("short", "english", 2.0, implementation="coreml"), sample("short", "english", 4.0, implementation="coreml")],
            "short",
            "english",
            "wall_seconds",
        )
        self.assertEqual(overlap["outcome"], "inconclusive_overlapping_observed_ranges")
        workloads = [
            {"name": "short", "fresh": {"english": {"native": [sample("short", "english", 1)], "coreml": [sample("short", "english", 2, implementation="coreml")]}}},
            {"name": "long", "fresh": {"english": {"native": [sample("long", "english", 3)], "coreml": [sample("long", "english", 2, implementation="coreml")]}}},
        ]
        self.assertIsNone(compare.comparisons(workloads)["global_winner"])

    def test_throughput_winner_uses_larger_ratio_and_remains_separate(self):
        result = compare.compare_throughput([6.0, 6.1], [4.0, 4.2])
        self.assertEqual(result["outcome"], "native_higher_audio_throughput")
        self.assertEqual(result["native"]["sample_count"], 2)

    def test_resident_throughput_uses_matching_waveform_ready_times(self):
        native, coreml = compare.resident_throughput(
            [{"audio_seconds": 1.8, "request_to_waveform_seconds": 0.3, "synthesis_seconds": 0.25}],
            [{"audio": {"audio_seconds": 1.5}, "cli_synthesis_seconds_rounded_to_ms": 0.3}],
        )
        self.assertEqual(native, [6.0])
        self.assertEqual(coreml, [5.0])

    def test_unsupported_metrics_are_scoped_and_do_not_fabricate_zero(self):
        metric = compare.unsupported("peak RSS", "CoreML service PID", "counter unavailable")
        self.assertEqual(metric["status"], "unsupported")
        self.assertIsNone(metric["value"])
        self.assertIsNone(metric["unit"])
        self.assertIn("CoreML service PID", metric["scope"])
        self.assertIn("counter unavailable", metric["reason"])
        parsed = compare.time_metrics("")
        self.assertEqual(parsed["os_peak_rss_bytes"]["status"], "unsupported")
        self.assertIsNone(parsed["os_peak_rss_bytes"]["value"])

    def test_time_l_cpu_rounding_is_below_resolution_not_zero(self):
        parsed = compare.time_metrics(
            "0.01 real 0.00 user 0.00 sys\n 123 maximum resident set size\n"
        )
        self.assertEqual(parsed["cpu_user_seconds"]["status"], "below_resolution")
        self.assertIsNone(parsed["cpu_user_seconds"]["value"])
        self.assertEqual(parsed["cpu_user_seconds"]["upper_bound_seconds"], 0.005)
        self.assertEqual(parsed["cpu_total_seconds"]["status"], "below_resolution")
        self.assertEqual(parsed["cpu_total_seconds"]["upper_bound_seconds"], 0.01)
        self.assertEqual(parsed["os_peak_rss_bytes"]["status"], "measured")
        self.assertIn("below time -l resolution", compare.cpu_text([{"process_resources": parsed}]))

    def test_non_one_to_one_mach_timebase_converts_ticks_to_seconds(self):
        self.assertEqual(
            compare.mach_ticks_to_seconds(24_000_000, {"numerator": 125, "denominator": 3}),
            1.0,
        )

    def test_owned_daemon_is_stopped_if_resource_setup_fails(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 42)", 42),
            ("Daemon is running (PID 42)", 42),
            ("Daemon is not running", None),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(compare, "mach_timebase", side_effect=OSError("unavailable")),
                patch.object(
                    compare.subprocess,
                    "run",
                    side_effect=[SimpleNamespace(stdout="daemon ready (PID 42)", stderr=""), None],
                ) as run,
            ):
                with self.assertRaisesRegex(OSError, "unavailable"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(
            run.call_args_list[1].args[0], [str(compare.COREML_CLI), "daemon", "stop"]
        )

    def test_owned_daemon_is_stopped_after_transient_startup_status_failure(self):
        statuses = [
            ("Daemon is not running", None),
            OSError("temporary status failure"),
            ("Daemon is running (PID 42)", 42),
            ("Daemon is not running", None),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(
                    compare.subprocess,
                    "run",
                    side_effect=[SimpleNamespace(stdout="daemon ready (PID 42)", stderr=""), None],
                ) as run,
            ):
                with self.assertRaisesRegex(OSError, "temporary status failure"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(
            run.call_args_list[1].args[0], [str(compare.COREML_CLI), "daemon", "stop"]
        )

    def test_startup_pid_mismatch_leaves_unverified_daemon_untouched(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 43)", 43),
            ("Daemon is running (PID 43)", 43),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(
                    compare.subprocess,
                    "run",
                    return_value=SimpleNamespace(stdout="daemon ready (PID 42)", stderr=""),
                ) as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "left untouched"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(run.call_count, 1)

    def test_unparseable_start_pid_never_stops_an_unverified_daemon(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 43)", 43),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(
                    compare.subprocess,
                    "run",
                    return_value=SimpleNamespace(stdout="unexpected startup output", stderr=""),
                ) as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "ownership is unverified"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(run.call_count, 1)

    def test_external_output_directory_is_rejected_before_benchmarking(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "inside the repository"):
                compare.validate_output_dir(Path(directory) / "comparison")

    def test_rusage_v4_layout_matches_macos_sdk_size(self):
        self.assertEqual(ctypes.sizeof(compare.RusageV4), 296)
        self.assertEqual(compare.RusageV4.user_time.offset, 16)
        self.assertEqual(compare.RusageV4.resident_size.offset, 64)
        self.assertEqual(compare.RusageV4.runnable_time.offset, 288)

    def test_cli_commands_write_files_without_playback_flags(self):
        output = Path("artifacts/sample.wav")
        for implementation in ("native", "coreml"):
            args = compare.cli_command(implementation, "hello", "ipa", output)
            self.assertIn("--output", args)
            self.assertIn(str(output), args)
            self.assertIn("--ipa", args)
            self.assertNotIn("--play", args)

    def test_report_generation_uses_fixture_data_without_host_or_model_access(self):
        report = {
            "captured_at_utc": "2026-10-12T00:00:00+00:00",
            "provenance": {
                "host": {"chip": "Apple test", "memory_bytes": 8 * 2**30, "os": {"version": "15", "build": "test"}},
                "native": {"commit": "abc", "cli_sha256": "n", "driver_sha256": "d", "source_sha256": "s", "model_pack_sha256": "p", "model_pack": {"pack_file_bytes": 123, "float32_weight_bytes": 100}, "build": {"compiler": "clang"}, "espeak_ng": "eSpeak NG 1.52.0"},
                "coreml": {
                    "version": "0.11.2", "wrapper": {"path": "/opt/homebrew/bin/kokoro"},
                    "executable": {"sha256": "c"}, "homebrew": {"formula_sha256": "h"},
                    "source": {"commit": "def", "version_note": "version note"},
                    "model": {"tag": "models-x", "file_count": 3, "total_bytes": 123,
                              "assets": [{"path": "voices/af_heart.bin", "size_bytes": 12}]},
                },
            },
            "comparisons": {"workloads": [], "global_winner": None},
            "execution": {
                "rerun_command": "uv run benchmark",
                "resident_sessions_per_workload": 5,
                "native_warm_repetitions_per_session": 5,
            },
            "post_capture_processing": {
                "source_raw_path": "raw-before-correction.json",
                "source_raw_sha256": "source-hash",
                "corrected_raw_path": "raw.json",
                "host_timebase": {"numerator": 125, "denominator": 3},
                "capture_runner_sha256": "capture-runner",
                "normalization_runner_sha256": "report-runner",
            },
            "workloads": [],
            "coreml_daemon": {},
        }
        rendered = compare.markdown(report)
        self.assertIn("Native and CoreML Kokoro comparison", rendered)
        self.assertIn("not reported; latency results differ by workload/path", rendered)
        self.assertIn("uv run benchmark", rendered)
        self.assertIn("Audio throughput", rendered)
        self.assertIn("CPU seconds and scope", rendered)
        self.assertIn("Path", rendered)
        self.assertIn("cpu_user_mach_ticks", rendered)

    def test_raw_report_is_saved_before_comparison_normalization(self):
        report = {"workloads": [{"name": "short", "raw_samples": [1, 2, 3]}]}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(compare, "comparisons", side_effect=ValueError("bad comparison")):
                with self.assertRaisesRegex(ValueError, "bad comparison"):
                    compare.finalize_report(report, Path(directory))
            saved = json.loads((Path(directory) / "raw.json").read_text())
        self.assertEqual(saved["comparisons"]["status"], "normalization_pending")
        self.assertEqual(saved["workloads"][0]["raw_samples"], [1, 2, 3])

    def test_normalized_report_preserves_sample_provenance(self):
        native = sample("short", "english", 1.0)
        coreml = sample("short", "english", 2.0, implementation="coreml")
        report = {
            "workloads": [
                {
                    "name": "short",
                    "fresh": {"english": {"native": [native], "coreml": [coreml]}},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            compare.finalize_report(report, Path(directory))
            saved = json.loads((Path(directory) / "raw.json").read_text())
        samples = saved["workloads"][0]["fresh"]["english"]
        self.assertEqual(samples["native"][0]["provenance_sha256"], "provenance")
        self.assertEqual(samples["coreml"][0]["provenance_sha256"], "provenance")


if __name__ == "__main__":
    unittest.main()
