import unittest
from unittest.mock import patch

from run import (
    ROOT,
    add_rate_metrics,
    build_provenance,
    host_provenance,
    summarize,
    validate_audio_sample,
)


class BenchmarkReportTests(unittest.TestCase):
    def test_summary_reports_median_range_and_sample_count(self) -> None:
        self.assertEqual(
            summarize([1.0, 2.0, 3.0]),
            {
                "sample_count": 3,
                "median_seconds": 2.0,
                "min_seconds": 1.0,
                "max_seconds": 3.0,
                "range_pct_of_median": 100.0,
            },
        )

    def test_throughput_and_realtime_factor_are_inverses(self) -> None:
        sample = add_rate_metrics(
            {"audio_seconds": 4.0, "synthesis_seconds": 2.0}
        )
        self.assertEqual(
            sample["audio_throughput_audio_seconds_per_synthesis_second"], 2.0
        )
        self.assertEqual(
            sample["realtime_factor_synthesis_seconds_per_audio_second"], 0.5
        )

    def test_audio_duration_matches_sample_count_at_24khz(self) -> None:
        validate_audio_sample({"sample_count": 48000, "audio_seconds": 2.0})
        with self.assertRaisesRegex(ValueError, "sample count"):
            validate_audio_sample({"sample_count": 48000, "audio_seconds": 1.0})

    def test_build_provenance_uses_effective_make_overrides(self) -> None:
        values = {
            "KOKORO_BENCH_COMPILER": "custom-clang --target=arm64-apple-macos",
            "KOKORO_BENCH_BASE_CFLAGS": "-std=c17 -Werror -DOVERRIDE=1",
            "KOKORO_BENCH_OPT_CFLAGS": "-O2",
            "KOKORO_BENCH_CPPFLAGS": "-Icustom-src",
            "KOKORO_BENCH_LDLIBS": "-framework Accelerate -lcustom",
            "KOKORO_BENCH_BUILD_DIR": "custom build",
        }
        with patch.dict("os.environ", values), patch(
            "run.command", return_value="custom clang 21.0"
        ) as command:
            result = build_provenance()
        command.assert_called_once_with(
            ["custom-clang", "--target=arm64-apple-macos", "--version"]
        )
        self.assertEqual(result["compiler_command"], values["KOKORO_BENCH_COMPILER"])
        self.assertEqual(result["base_cflags"], values["KOKORO_BENCH_BASE_CFLAGS"])
        self.assertEqual(result["opt_cflags"], values["KOKORO_BENCH_OPT_CFLAGS"])
        self.assertEqual(result["cli_preprocessor_flags"], values["KOKORO_BENCH_CPPFLAGS"])
        self.assertEqual(result["link_flags"], values["KOKORO_BENCH_LDLIBS"])
        self.assertEqual(result["build_directory"], str(ROOT / "custom build"))

    def test_host_provenance_returns_hardware_and_os(self) -> None:
        outputs = iter(
            [
                "17179869184\nApple M5",
                "macOS",
                "27.0.1",
                "26A434",
            ]
        )
        with patch("run.command", side_effect=lambda _: next(outputs)), patch(
            "run.platform.machine", return_value="arm64"
        ):
            result = host_provenance()
        self.assertEqual(result["chip"], "Apple M5")
        self.assertEqual(result["memory_bytes"], 17179869184)
        self.assertEqual(result["architecture"], "arm64")
        self.assertEqual(
            result["os"], {"name": "macOS", "version": "27.0.1", "build": "26A434"}
        )

    def test_summary_rejects_empty_or_invalid_timings(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            summarize([])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            summarize([1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
