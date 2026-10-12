import unittest
import json
import subprocess
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


class FakeStartProcess:
    def __init__(self, stdout, stderr, *, returncode=0, output_stdout="", output_stderr="", wait_actions=()):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = None
        self.result = returncode
        self.output_stdout = output_stdout
        self.output_stderr = output_stderr
        self.wait_actions = list(wait_actions)
        self.wait_calls = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.output_written = False

    def _write_output(self):
        if self.output_written:
            return
        self.stdout.write(self.output_stdout)
        self.stderr.write(self.output_stderr)
        self.stdout.flush()
        self.stderr.flush()
        self.output_written = True

    def terminate(self):
        self.terminate_calls += 1
        self._write_output()
        self.returncode = -15

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        action = self.wait_actions.pop(0) if self.wait_actions else self.result
        if isinstance(action, BaseException):
            raise action
        self._write_output()
        self.returncode = action
        return self.returncode

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9


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
        processes = []

        def popen(args, **kwargs):
            process = FakeStartProcess(
                kwargs["stdout"], kwargs["stderr"], output_stdout="Kokoro daemon ready (PID 42)\n"
            )
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(compare, "mach_timebase", side_effect=OSError("unavailable")),
                patch.object(compare.subprocess, "Popen", side_effect=popen) as popen_mock,
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(OSError, "unavailable"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(processes[0].terminate_calls, 0)
        self.assertEqual(popen_mock.call_args.args[0], [str(compare.COREML_CLI), "daemon", "start"])
        kill.assert_called_once_with(42, compare.signal.SIGTERM)

    def test_owned_daemon_is_stopped_after_transient_startup_status_failure(self):
        statuses = [
            ("Daemon is not running", None),
            OSError("temporary status failure"),
            ("Daemon is running (PID 42)", 42),
            ("Daemon is not running", None),
        ]
        process_holder = []

        def popen(args, **kwargs):
            process = FakeStartProcess(
                kwargs["stdout"], kwargs["stderr"], output_stdout="Kokoro daemon ready (PID 42)\n"
            )
            process_holder.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(compare.subprocess, "Popen", side_effect=popen),
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(OSError, "temporary status failure"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(process_holder[0].terminate_calls, 0)
        kill.assert_called_once_with(42, compare.signal.SIGTERM)

    def test_cleanup_status_failure_reports_owned_daemon_left_untouched(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 42)", 42),
            OSError("status unavailable"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(
                    compare.subprocess,
                    "Popen",
                    side_effect=lambda args, **kwargs: FakeStartProcess(
                        kwargs["stdout"], kwargs["stderr"], output_stdout="Kokoro daemon ready (PID 42)\n"
                    ),
                ),
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(RuntimeError, "left untouched"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        kill.assert_not_called()

    def test_start_failure_after_ready_pid_still_stops_owned_daemon(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 42)", 42),
            ("Daemon is not running", None),
        ]
        processes = []

        def popen(args, **kwargs):
            process = FakeStartProcess(
                kwargs["stdout"],
                kwargs["stderr"],
                returncode=1,
                output_stdout="Kokoro daemon ready (PID 42)\n",
            )
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(compare.subprocess, "Popen", side_effect=popen),
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(subprocess.CalledProcessError, "returned non-zero"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(processes[0].terminate_calls, 0)
        kill.assert_called_once_with(42, compare.signal.SIGTERM)

    def test_interrupt_after_popen_recovers_output_before_cleanup(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 42)", 42),
            ("Daemon is not running", None),
        ]
        processes = []
        wait_calls = 0
        wait_start = compare._wait_for_daemon_start

        def popen(args, **kwargs):
            process = FakeStartProcess(
                kwargs["stdout"], kwargs["stderr"], output_stdout="Kokoro daemon ready (PID 42)\n"
            )
            processes.append(process)
            return process

        def interrupt_before_wait(process, args, deadline):
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                raise KeyboardInterrupt()
            return wait_start(process, args, deadline)

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(compare.subprocess, "Popen", side_effect=popen),
                patch.object(compare, "_wait_for_daemon_start", side_effect=interrupt_before_wait),
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(wait_calls, 2)
        self.assertEqual(len(processes[0].wait_calls), 1)
        self.assertAlmostEqual(processes[0].wait_calls[0], 12, places=3)
        kill.assert_called_once_with(42, compare.signal.SIGTERM)

    def test_interrupt_during_popen_waits_for_detached_launcher_pid_output(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 42)", 42),
            ("Daemon is not running", None),
        ]

        def interrupted_popen(args, **kwargs):
            raise KeyboardInterrupt()

        def publish_ready_output(_delay):
            output = popen.call_args.kwargs["stdout"]
            output.write("Kokoro daemon ready (PID 42)\n")
            output.flush()

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(compare.subprocess, "Popen", side_effect=interrupted_popen) as popen,
                patch.object(compare.time, "sleep", side_effect=publish_ready_output) as sleep,
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        sleep.assert_called_once_with(0.1)
        kill.assert_called_once_with(42, compare.signal.SIGTERM)

    def test_interrupt_during_popen_without_pid_reports_unknown_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", return_value=("Daemon is not running", None)),
                patch.object(compare.subprocess, "Popen", side_effect=KeyboardInterrupt) as popen,
                patch.object(compare.time, "perf_counter", side_effect=(1.0, 2.0, 14.0)),
                patch.object(compare.time, "sleep") as sleep,
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(RuntimeError, "any possibly started daemon was left untouched"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        popen.assert_called_once()
        sleep.assert_called_once_with(0.1)
        kill.assert_not_called()

    def test_interrupt_before_pid_parse_recovers_completed_launcher_output(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 42)", 42),
            ("Daemon is not running", None),
        ]
        processes = []
        read_calls = 0
        read_output = compare._read_start_output

        def popen(args, **kwargs):
            process = FakeStartProcess(
                kwargs["stdout"], kwargs["stderr"], output_stdout="Kokoro daemon ready (PID 42)\n"
            )
            processes.append(process)
            return process

        def interrupt_first_read(stream):
            nonlocal read_calls
            read_calls += 1
            if read_calls == 1:
                raise KeyboardInterrupt()
            return read_output(stream)

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(compare.subprocess, "Popen", side_effect=popen),
                patch.object(compare, "_read_start_output", side_effect=interrupt_first_read),
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(len(processes[0].wait_calls), 1)
        self.assertAlmostEqual(processes[0].wait_calls[0], 12, places=3)
        kill.assert_called_once_with(42, compare.signal.SIGTERM)

    def test_startup_interrupt_waits_for_ready_pid_then_stops_owned_daemon(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 42)", 42),
            ("Daemon is not running", None),
        ]
        processes = []

        def popen(args, **kwargs):
            process = FakeStartProcess(
                kwargs["stdout"],
                kwargs["stderr"],
                output_stdout="Kokoro daemon ready (PID 42)\n",
                wait_actions=(KeyboardInterrupt(), 0),
            )
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(compare.subprocess, "Popen", side_effect=popen),
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(len(processes[0].wait_calls), 2)
        self.assertEqual(processes[0].terminate_calls, 0)
        kill.assert_called_once_with(42, compare.signal.SIGTERM)

    def test_start_timeout_captures_pid_from_stderr_and_stops_owned_daemon(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 42)", 42),
            ("Daemon is not running", None),
        ]
        processes = []

        def popen(args, **kwargs):
            process = FakeStartProcess(
                kwargs["stdout"],
                kwargs["stderr"],
                output_stderr="Daemon started (PID 42) but readiness timed out\n",
                wait_actions=(subprocess.TimeoutExpired(args, 12),),
            )
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(compare.subprocess, "Popen", side_effect=popen),
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(TimeoutError, "did not finish within 12 seconds"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(processes[0].terminate_calls, 1)
        kill.assert_called_once_with(42, compare.signal.SIGTERM)

    def test_already_running_output_does_not_claim_foreign_daemon_ownership(self):
        statuses = [
            ("Daemon is not running", None),
            ("Daemon is running (PID 43)", 43),
        ]
        processes = []

        def popen(args, **kwargs):
            process = FakeStartProcess(
                kwargs["stdout"],
                kwargs["stderr"],
                output_stdout="Daemon already running (PID 43)\n",
            )
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=statuses),
                patch.object(compare.subprocess, "Popen", side_effect=popen),
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(RuntimeError, "ownership is unverified, so it was left untouched"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(processes[0].terminate_calls, 0)
        kill.assert_not_called()

    def test_status_pid_mismatch_leaves_foreign_daemon_untouched(self):
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
                    "Popen",
                    side_effect=lambda args, **kwargs: FakeStartProcess(
                        kwargs["stdout"],
                        kwargs["stderr"],
                        output_stdout="Kokoro daemon ready (PID 42)\n",
                    ),
                ),
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(RuntimeError, "left untouched"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        kill.assert_not_called()

    def test_pid_replacement_after_ownership_check_signals_only_owned_pid(self):
        status_pids = iter((None, 42, 42, 43))
        calls = 0

        def status():
            nonlocal calls
            pid = next(status_pids)
            calls += 1
            return ("Daemon is not running" if pid is None else f"Daemon is running (PID {pid})", pid)

        process_holder = []

        def popen(args, **kwargs):
            process = FakeStartProcess(
                kwargs["stdout"], kwargs["stderr"], output_stdout="Kokoro daemon ready (PID 42)\n"
            )
            process_holder.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(compare, "daemon_status", side_effect=status),
                patch.object(compare.subprocess, "Popen", side_effect=popen),
                patch.object(compare.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed from owned PID 42 to 43"):
                    compare.coreml_resident(Path(directory), (), 1, "provenance")
        self.assertEqual(calls, 4)
        kill.assert_called_once_with(42, compare.signal.SIGTERM)

    def test_preexisting_daemon_is_not_spawned_or_stopped(self):
        with (
            patch.object(compare, "daemon_status", return_value=("Daemon is running (PID 43)", 43)),
            patch.object(compare.subprocess, "Popen") as popen,
            patch.object(compare.os, "kill") as kill,
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing to stop it"):
                compare.coreml_resident(Path("unused"), (), 1, "provenance")
        popen.assert_not_called()
        kill.assert_not_called()

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
