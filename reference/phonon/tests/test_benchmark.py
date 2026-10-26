"""Measurement-label invariants from ticket32 and its coordinator handoff."""
import argparse
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).resolve().parents[1] / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

class BenchmarkValidation(unittest.TestCase):
    def setUp(self):
        self.report = {"schema": 1, "acoustic_onset_s": None, "repeats": 5,
            "provenance": {"source_commit": "a"*40, "source_dirty": False, "pack_sha256": "b"*64, "voice_sha256": "c"*64}, "runs": []}
        for w in benchmark.WORKLOADS:
            for m in ("fresh_process", "warm_resident"):
                for i in range(5):
                    self.report["runs"].append(dict(workload=w, mode=m, repetition=i,
                        first_pcm_s=.1, synthesis_s=.2, cpu_s=.3, engine_prepare_s=.01,
                        voice_open_s=.01, samples=19200, frames=10, chunks=1,
                        rss_high_water_bytes=1000000, output_duration_s=.8, rtf=.25,
                        pcm_f32_sha256="d"*64, profiled=False, consumer="synchronous_memory_copy"))

    def test_unsupported_repeats_fail_before_measurement(self):
        for repeats in (4, 20):
            with self.subTest(repeats=repeats), self.assertRaises(SystemExit):
                benchmark.run(argparse.Namespace(repeats=repeats))

    def test_complete_consistent_measurements(self):
        benchmark.validate(self.report)

    def test_rejects_mislabeled_or_inconsistent_measurements(self):
        mutations = [
            lambda r: r.update(acoustic_onset_s=0),
            lambda r: r['provenance'].update(source_dirty=True),
            lambda r: r['runs'].pop(),
            lambda r: r['runs'].append(copy.deepcopy(r['runs'][0])),
            lambda r: r['runs'][0].update(first_pcm_s=.4),
            lambda r: r['runs'][0].update(synthesis_s=float('nan')),
            lambda r: r['runs'][0].update(samples=19201),
            lambda r: r['runs'][0].update(frames=1.5, samples=2880, output_duration_s=.12, rtf=.2/.12),
            lambda r: r['runs'][0].update(frames=-1, samples=-1920, output_duration_s=-.08, rtf=-2.5),
            lambda r: r['runs'][0].update(output_duration_s=1),
            lambda r: r['runs'][0].update(rtf=4),
            lambda r: r['runs'][0].update(profiled=True),
            lambda r: r['runs'][0].update(consumer='audioqueue'),
            lambda r: r['runs'][0].update(rss_high_water_bytes=0),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                report = copy.deepcopy(self.report)
                mutate(report)
                with self.assertRaises(ValueError):
                    benchmark.validate(report)

if __name__ == '__main__':
    unittest.main()
