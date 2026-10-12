"""Repeated local measurements for the native Kokoro CLI and resident engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/kokoro-v1_0-af_heart.kpack"
BUILD_DIR = Path(os.environ.get("KOKORO_BENCH_BUILD_DIR", ".build"))
if not BUILD_DIR.is_absolute():
    BUILD_DIR = ROOT / BUILD_DIR
CLI_PATH = BUILD_DIR / "kokoro"
DRIVER_PATH = BUILD_DIR / "kokoro-benchmark"
SAMPLE_RATE_HZ = 24000
VOICE = "af_heart"
SPEED = 1.0
SEED = 104729
STAGE_NAMES = (
    "context_encoder",
    "duration_prediction",
    "pitch_noise_prediction",
    "text_encoder",
    "decoder_conditioning",
    "neural_generator",
    "istft",
)
SHORT_TEXT = "I can help with that."
PARAGRAPH_TEXT = (
    "Good morning. I can help you organize the meeting notes, summarize the main decisions, "
    "and draft a clear follow-up for everyone who joined."
)


def summarize(values: list[float]) -> dict[str, float | int]:
    """Report median/min/max and range divided by median as a percentage."""
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("timing samples must be finite and positive")
    median = statistics.median(values)
    minimum = min(values)
    maximum = max(values)
    return {
        "sample_count": len(values),
        "median_seconds": median,
        "min_seconds": minimum,
        "max_seconds": maximum,
        "range_pct_of_median": (maximum - minimum) / median * 100.0,
    }


def add_rate_metrics(sample: dict[str, float | int]) -> dict[str, float | int]:
    audio = float(sample["audio_seconds"])
    synthesis = float(sample["synthesis_seconds"])
    if not math.isfinite(audio) or not math.isfinite(synthesis) or audio <= 0 or synthesis <= 0:
        raise ValueError("audio and synthesis durations must be finite and positive")
    sample["audio_throughput_audio_seconds_per_synthesis_second"] = audio / synthesis
    sample["realtime_factor_synthesis_seconds_per_audio_second"] = synthesis / audio
    return sample


def validate_audio_sample(sample: dict[str, object]) -> None:
    count = int(sample["sample_count"])
    seconds = float(sample["audio_seconds"])
    if count <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("generated audio must have a positive sample count and duration")
    if not math.isclose(seconds, count / SAMPLE_RATE_HZ, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("audio duration does not match sample count and sample rate")


def command(arguments: list[str], capture: bool = True) -> str:
    result = subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
    )
    return result.stdout.strip() if capture else ""


def host_provenance() -> dict[str, object]:
    values = command(["sysctl", "-n", "hw.memsize", "machdep.cpu.brand_string"]).splitlines()
    if len(values) != 2:
        raise RuntimeError("sysctl did not return memory size and Apple chip name")
    os_info = {
        key: command(["sw_vers", option])
        for key, option in (
            ("name", "-productName"),
            ("version", "-productVersion"),
            ("build", "-buildVersion"),
        )
    }
    return {
        "chip": values[1],
        "memory_bytes": int(values[0]),
        "architecture": platform.machine(),
        "os": os_info,
    }


def build_provenance() -> dict[str, str]:
    compiler_command = os.environ.get("KOKORO_BENCH_COMPILER", "")
    compiler_arguments = shlex.split(compiler_command)
    if not compiler_arguments:
        raise RuntimeError("benchmark compiler provenance was not passed by Make")
    base_cflags = os.environ.get("KOKORO_BENCH_BASE_CFLAGS", "")
    opt_cflags = os.environ.get("KOKORO_BENCH_OPT_CFLAGS", "")
    cppflags = os.environ.get("KOKORO_BENCH_CPPFLAGS", "")
    ldlibs = os.environ.get("KOKORO_BENCH_LDLIBS", "")
    configured_build_dir = os.environ.get("KOKORO_BENCH_BUILD_DIR", "")
    if not all((base_cflags, opt_cflags, cppflags, ldlibs, configured_build_dir)):
        raise RuntimeError("benchmark build flags were not passed by Make")
    build_dir = Path(configured_build_dir)
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir
    return {
        "build_directory": str(build_dir),
        "compiler_command": compiler_command,
        "compiler": command([*compiler_arguments, "--version"]).splitlines()[0],
        "base_cflags": base_cflags,
        "opt_cflags": opt_cflags,
        "cli_preprocessor_flags": cppflags,
        "native_driver_preprocessor_flags": f"{cppflags} -DKOKORO_BENCHMARK_TIMING",
        "link_flags": ldlibs,
        "benchmark_define": "KOKORO_BENCHMARK_TIMING",
    }


def pack_provenance(path: Path) -> dict[str, object]:
    """Read the pack header/index only; leave tensor payload pages for the timed loader."""
    path = path.resolve()
    header = struct.Struct("<8sIIIIQQ")
    entry = struct.Struct("<HBBI")
    dimension = struct.Struct("<Q")
    byte_range = struct.Struct("<QQ")
    with path.open("rb") as source:
        raw = source.read(header.size)
        if len(raw) != header.size:
            raise RuntimeError("model pack header is truncated")
        magic, version, tensor_count, index_size, metadata_size, payload_size, reserved = header.unpack(raw)
        if magic != b"KOKORO1\0" or version != 1 or reserved != 0:
            raise RuntimeError("model pack is not the supported v1 format")
        index = source.read(index_size)
        metadata = json.loads(source.read(metadata_size).decode("utf-8"))
        if len(index) != index_size:
            raise RuntimeError("model pack index is truncated")

    cursor = 0
    dtype_sizes = {"f32": 0, "i64": 0}
    for _ in range(tensor_count):
        name_size, dtype_code, rank, entry_reserved = entry.unpack_from(index, cursor)
        cursor += entry.size
        if entry_reserved != 0:
            raise RuntimeError("model pack index has a reserved value")
        cursor += rank * dimension.size
        _, size = byte_range.unpack_from(index, cursor)
        cursor += byte_range.size + name_size
        if dtype_code == 1:
            dtype_sizes["f32"] += size
        elif dtype_code == 2:
            dtype_sizes["i64"] += size
        else:
            raise RuntimeError(f"unsupported tensor dtype code: {dtype_code}")
    if cursor != len(index):
        raise RuntimeError("model pack index size does not match its entries")
    return {
        "format": metadata["format"],
        "model": metadata["model"],
        "model_revision": metadata["model_revision"],
        "inference_source_revision": metadata["source_revision"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "config_sha256": metadata["config_sha256"],
        "voice_sha256": metadata["voice_sha256"],
        "payload_sha256": metadata["payload_sha256"],
        "tensor_count": tensor_count,
        "payload_bytes": payload_size,
        "tensor_bytes_by_dtype": dtype_sizes,
        "float32_weight_bytes": dtype_sizes["f32"],
        "pack_file_bytes": path.stat().st_size,
        "pack_path": str(path.relative_to(ROOT)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256() -> str:
    paths = [
        ROOT / "Makefile",
        *sorted((ROOT / "src").rglob("*.c")),
        *sorted((ROOT / "src").rglob("*.h")),
        *sorted((ROOT / "benchmarks").glob("*.c")),
        *sorted((ROOT / "benchmarks").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def driver_session(mode: str, text: str, wav_path: Path, repetitions: int) -> dict[str, object]:
    arguments = [
        str(DRIVER_PATH),
        "--mode", mode,
        "--text", text,
        "--model", str(MODEL_PATH),
        "--voice", VOICE,
        "--speed", str(SPEED),
        "--seed", str(SEED),
        "--warmups", "1",
        "--repetitions", str(repetitions),
        "--wav", str(wav_path),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        arguments,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    events: list[dict[str, object]] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            event = json.loads(line)
            if event.get("kind") == "cold_audio":
                event["parent_start_to_waveform_event_seconds"] = time.perf_counter() - started
                validate_audio_sample(event)
            elif event.get("kind") in {"cold_wav", "warm_sample"}:
                validate_audio_sample(event)
                add_rate_metrics(event)
            events.append(event)
        error_text = process.stderr.read() if process.stderr is not None else ""
        status = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if status != 0:
        raise RuntimeError(f"native benchmark driver failed: {error_text.strip()}")
    if not events or events[0].get("kind") != "cold_audio":
        raise RuntimeError("driver did not report the first complete waveform")
    cold_wav = next((event for event in events if event.get("kind") == "cold_wav"), None)
    warm = [event for event in events if event.get("kind") == "warm_sample"]
    if cold_wav is None or len(warm) != repetitions:
        raise RuntimeError("driver returned an incomplete measurement set")
    return {
        "cold_audio": events[0],
        "cold_wav": cold_wav,
        "warm_samples": warm,
        "driver_process_wall_seconds": time.perf_counter() - started,
    }


def cli_sample(mode: str, text: str, wav_path: Path) -> float:
    arguments = [
        str(CLI_PATH), "say", text, "--voice", VOICE, "--speed", str(SPEED),
        "--output", str(wav_path),
    ]
    if mode == "ipa":
        arguments.insert(3, "--ipa")
    started = time.perf_counter()
    command(arguments, capture=False)
    return time.perf_counter() - started


def pack_open_sample() -> float:
    output = command([str(DRIVER_PATH), "--probe-pack", str(MODEL_PATH)])
    record = json.loads(output)
    if record.get("kind") != "pack_open":
        raise RuntimeError("pack-open probe returned an unexpected record")
    return float(record["seconds"])


def version_sample() -> float:
    started = time.perf_counter()
    command([str(CLI_PATH), "--version"], capture=False)
    return time.perf_counter() - started


def timing_summary(samples: list[dict[str, object]], key: str) -> dict[str, float | int]:
    return summarize([float(sample[key]) for sample in samples])


def stage_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    stage_values = {
        name: [float(sample["stage_seconds"][name]) for sample in samples]
        for name in STAGE_NAMES
    }
    summaries = {name: summarize(values) for name, values in stage_values.items()}
    dominant = max(summaries, key=lambda name: summaries[name]["median_seconds"])
    synthesis_median = statistics.median(float(sample["synthesis_seconds"]) for sample in samples)
    dominant_median = float(summaries[dominant]["median_seconds"])
    return {
        "per_stage": summaries,
        "dominant_stage": dominant,
        "dominant_stage_median_seconds": dominant_median,
        "dominant_stage_share_of_synthesis_median_pct": dominant_median / synthesis_median * 100.0,
    }


def summary_text(summary: dict[str, float | int], unit: str) -> str:
    scale = 1000.0 if unit == "ms" else 1.0
    median = float(summary["median_seconds"]) * scale
    minimum = float(summary["min_seconds"]) * scale
    maximum = float(summary["max_seconds"]) * scale
    variation = float(summary["range_pct_of_median"])
    samples = int(summary["sample_count"])
    return f"{median:.2f} {unit} [{minimum:.2f}, {maximum:.2f}] ({variation:.1f}%, n={samples})"


def markdown_report(report: dict[str, object]) -> str:
    host = report["host"]
    model = report["model"]
    execution = report["execution"]
    lines = [
        "# Native Kokoro performance baseline",
        "",
        f"Captured: {report['captured_at_utc']}",
        "",
        f"Hardware: {host['chip']}, {host['memory_bytes'] // (1024**3)} GiB, "
        f"{host['os']['name']} {host['os']['version']} ({host['os']['build']}).",
        f"Source commit: {report['source']['commit_sha']}; source SHA-256: "
        f"{report['source']['native_tree_sha256']}.",
        f"Build: {report['build']['compiler_command']} ({report['build']['compiler']}); "
        f"{report['build']['base_cflags']} {report['build']['opt_cflags']}; CLI flags "
        f"{report['build']['cli_preprocessor_flags']}; native driver flags "
        f"{report['build']['native_driver_preprocessor_flags']}; link flags "
        f"{report['build']['link_flags']}.",
        f"Build output directory: {report['build']['build_directory']}.",
        f"Model revision: {model['model_revision']}; pack {model['pack_file_bytes']:,} bytes; "
        f"{model['float32_weight_bytes']:,} float32 weight bytes; pack SHA-256 "
        f"{model['pack_sha256']}.",
        f"Voice: {report['voice']['name']} style {report['voice']['style_source']}; official voice "
        f"SHA-256 {report['voice']['official_voice_file_sha256']}; speed {execution['speed']}; "
        f"seed {execution['seed']}.",
        "",
        "Each entry gives median, [min, max], and (max - min) / median variation. "
        "No performance target is assumed.",
        "",
        "| Workload | Cold CLI end-to-end | Cold process to full waveform | Engine create | Frontend | Warm request to full waveform | Native synthesis | WAV close | Audio duration | Audio throughput | RTF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for workload in report["workloads"]:
        values = workload["summaries"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{workload['name']} ({workload['mode']})",
                    summary_text(values["cold_cli_end_to_end_seconds"], "ms"),
                    summary_text(values["cold_process_to_full_waveform_seconds"], "ms"),
                    summary_text(values["engine_create_seconds"], "ms"),
                    summary_text(values["warm_frontend_seconds"], "ms"),
                    summary_text(values["warm_request_to_waveform_seconds"], "ms"),
                    summary_text(values["warm_synthesis_seconds"], "ms"),
                    summary_text(values["warm_wav_write_completion_seconds"], "ms"),
                    summary_text(values["generated_audio_seconds"], "s"),
                    summary_text(values["audio_throughput_audio_seconds_per_synthesis_second"], "×"),
                    summary_text(values["realtime_factor_synthesis_seconds_per_audio_second"], "×"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Warm native synthesis stage medians:",
            "",
            "| Workload | Largest measured stage | Share of median native synthesis | Stage medians |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for workload in report["workloads"]:
        stage_data = workload["stage_summary"]
        stage_cells = ", ".join(
            f"{name}: {summary_text(stage_data['per_stage'][name], 'ms')}"
            for name in STAGE_NAMES
        )
        lines.append(
            f"| {workload['name']} ({workload['mode']}) | {stage_data['dominant_stage']} | "
            f"{stage_data['dominant_stage_share_of_synthesis_median_pct']:.1f}% | {stage_cells} |"
        )
    lines.extend(
        [
            "",
            "Process and model-load probes:",
            "",
            f"- Fresh CLI process (--version): "
            f"{summary_text(report['summaries']['fresh_process_startup_seconds'], 'ms')}",
            f"- Model pack open plus full payload SHA-256 validation: "
            f"{summary_text(report['summaries']['model_pack_open_seconds'], 'ms')}",
            "",
            "Engine creation includes pack open and validation, voice validation, and stage setup. "
            "Each first-request sample starts a new native driver process; cold refers to process "
            "and engine state, while OS file-page-cache state is uncontrolled. Warm repetitions "
            "reuse that session's resident engine. The separate "
            "pack-open probe runs after synthesis sessions, is not subtracted from engine creation, "
            "and does not estimate cold storage I/O. No cache purge was used.",
            "",
            "First audio means the complete in-memory mono waveform is available after all chunks "
            "have been synthesized and combined. The current engine does not stream. WAV time is "
            "the writer returning after file close; it does not measure fsync or durable-media flush.",
            "",
            "Audio throughput is generated audio seconds divided by native synthesis seconds. RTF "
            "is native synthesis seconds divided by generated audio seconds. Warm request time "
            "includes the English frontend or IPA tokenizer, synthesis, and waveform combination; "
            "it excludes WAV writing.",
            "",
            "Stage timings bracket the seven existing synthesis calls in a benchmark-only build. "
            "They add no runtime API or product-build instrumentation. Each stage adds two "
            "monotonic clock reads and one accumulator callback per chunk; that instrumentation "
            "is included in total synthesis time and the JSON other_synthesis_overhead residual, "
            "so stage shares are approximate.",
            "",
            "Raw samples and complete provenance are in the adjacent JSON report.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(runs: int, warm_repetitions: int) -> dict[str, object]:
    if sys.platform != "darwin":
        raise RuntimeError("the native baseline requires macOS")
    for path in (MODEL_PATH, CLI_PATH, DRIVER_PATH):
        if not path.is_file():
            raise RuntimeError(f"required file is missing: {path.relative_to(ROOT)}")
    host = host_provenance()
    build = build_provenance()
    espeak_version = command(["espeak-ng", "--version"]).splitlines()[0]
    pins = json.loads((ROOT / "reference/pins.json").read_text(encoding="utf-8"))
    model = pack_provenance(MODEL_PATH)
    phonemes = command(
        ["espeak-ng", "-q", "--ipa=3", "-v", "en-us", SHORT_TEXT]
    ).replace("\u200d", "").strip()

    specs = [
        {"name": "short_assistant_reply", "mode": "english", "text": SHORT_TEXT},
        {"name": "paragraph", "mode": "english", "text": PARAGRAPH_TEXT},
        {
            "name": "short_assistant_reply_prepared_phonemes",
            "mode": "ipa",
            "text": phonemes,
            "source_text": SHORT_TEXT,
        },
    ]
    run_dir = BUILD_DIR / "performance-baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    startup_samples = [version_sample() for _ in range(runs)]
    workloads = []
    for spec in specs:
        sessions = []
        cli_samples = []
        warm_samples = []
        for _ in range(runs):
            cli_samples.append(cli_sample(spec["mode"], spec["text"], run_dir / "cli.wav"))
            session = driver_session(
                spec["mode"], spec["text"], run_dir / "native.wav", warm_repetitions
            )
            sessions.append(session)
            warm_samples.extend(session["warm_samples"])
        cold_audio = [session["cold_audio"] for session in sessions]
        cold_wav = [session["cold_wav"] for session in sessions]
        counts = {int(sample["sample_count"]) for sample in warm_samples}
        if len(counts) != 1:
            raise RuntimeError(f"generated sample count varied for {spec['name']}")
        metrics = {
            "cold_process_to_full_waveform_seconds": timing_summary(
                cold_audio, "parent_start_to_waveform_event_seconds"
            ),
            "cold_engine_create_seconds": timing_summary(cold_audio, "engine_create_seconds"),
            "cold_frontend_seconds": timing_summary(cold_audio, "frontend_seconds"),
            "cold_native_synthesis_seconds": timing_summary(cold_audio, "synthesis_seconds"),
            "cold_wav_write_completion_seconds": timing_summary(
                cold_wav, "wav_write_completion_seconds"
            ),
            "cold_cli_end_to_end_seconds": summarize(cli_samples),
            "engine_create_seconds": timing_summary(cold_audio, "engine_create_seconds"),
            "warm_frontend_seconds": timing_summary(warm_samples, "frontend_seconds"),
            "warm_request_to_waveform_seconds": timing_summary(
                warm_samples, "request_to_waveform_seconds"
            ),
            "warm_synthesis_seconds": timing_summary(warm_samples, "synthesis_seconds"),
            "warm_wav_write_completion_seconds": timing_summary(
                warm_samples, "wav_write_completion_seconds"
            ),
            "generated_audio_seconds": timing_summary(warm_samples, "audio_seconds"),
            "audio_throughput_audio_seconds_per_synthesis_second": timing_summary(
                warm_samples, "audio_throughput_audio_seconds_per_synthesis_second"
            ),
            "realtime_factor_synthesis_seconds_per_audio_second": timing_summary(
                warm_samples, "realtime_factor_synthesis_seconds_per_audio_second"
            ),
        }
        stage_data = stage_summary(warm_samples)
        workloads.append(
            {
                "name": spec["name"],
                "mode": spec["mode"],
                "source_text": spec.get("source_text", spec["text"]),
                "prepared_phonemes": spec["text"] if spec["mode"] == "ipa" else None,
                "voice": VOICE,
                "speed": SPEED,
                "seed": SEED,
                "runs": sessions,
                "cold_cli_end_to_end_seconds": cli_samples,
                "warm_samples": warm_samples,
                "sample_count_per_request": counts.pop(),
                "generated_audio_seconds_per_request": warm_samples[0]["audio_seconds"],
                "summaries": metrics,
                "stage_summary": stage_data,
            }
        )

    # Keep the standalone model-load probe after synthesis so it cannot pre-warm measured runs.
    model_open_samples = [pack_open_sample() for _ in range(runs)]
    model["pack_sha256"] = sha256_file(MODEL_PATH)
    model["source_model_revision"] = pins["model"]["revision"]
    model["reference_inference_revision"] = pins["inference_source"]["revision"]
    git_commit = command(["git", "rev-parse", "HEAD"])
    if command(["git", "status", "--porcelain=v1"]):
        raise RuntimeError("benchmark provenance requires a clean source tree")

    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"commit_sha": git_commit, "native_tree_sha256": source_sha256()},
        "host": host,
        "build": {
            **build,
            "cli_sha256": sha256_file(CLI_PATH),
            "native_driver_sha256": sha256_file(DRIVER_PATH),
            "espeak_ng_version": espeak_version,
            "uv_version": command(["uv", "--version"]),
            "python_version": sys.version.split()[0],
        },
        "model": model,
        "voice": {
            "name": VOICE,
            "style_source": "embedded in the model pack",
            "embedded_tensor": "voice.af_heart.style",
            "official_voice_file_sha256": pins["model"]["voice"]["sha256"],
        },
        "execution": {
            "voice": VOICE,
            "speed": SPEED,
            "seed": SEED,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sessions_per_workload": runs,
            "warmups_per_session": 1,
            "warm_repetitions_per_session": warm_repetitions,
            "warm_sample_count_per_workload": runs * warm_repetitions,
            "cache_policy": (
                "Each fresh CLI sample and native session uses a new process; warm repetitions "
                "share that session's resident engine. OS file-page-cache state is uncontrolled; "
                "no destructive cache flush or purge is performed."
            ),
        },
        "semantics": {
            "engine_create": (
                "Total kokoro_engine_create time, including pack open and validation, "
                "voice validation, and stage creation."
            ),
            "model_pack_open": (
                "Separate kokoro_pack_open call, including mmap and full tensor-payload "
                "SHA-256 validation; not subtracted from engine creation."
            ),
            "cold_cli_end_to_end": (
                "Parent wall time for a fresh CLI process through WAV file close; includes "
                "frontend, engine creation, synthesis, and WAV writing. OS file-page-cache state "
                "is uncontrolled."
            ),
            "cold_native_session": (
                "Each session starts a fresh native driver process and measures its first request "
                "before warmup; OS file-page-cache state is uncontrolled, so this does not mean "
                "cold storage."
            ),
            "first_audio_availability": (
                "Complete in-memory waveform after all input chunks are synthesized and "
                "combined; no streaming."
            ),
            "warm_request": (
                "English frontend or IPA tokenization, native synthesis, and full waveform "
                "combination; WAV writing is excluded."
            ),
            "wav_write_completion": (
                "kokoro_wav_write_pcm16 returned after closing the file; no fsync/durability claim."
            ),
            "audio_throughput": "generated_audio_seconds / native_synthesis_seconds.",
            "realtime_factor": "native_synthesis_seconds / generated_audio_seconds.",
            "variation": "(maximum - minimum) / median * 100 percent.",
            "stage_timing": (
                "Benchmark-only monotonic timestamps around the seven existing inference-stage "
                "calls. This adds two monotonic clock reads and one accumulator callback per "
                "stage per chunk; that instrumentation is included in total synthesis time and "
                "the residual other_synthesis_overhead, so stage shares are approximate."
            ),
        },
        "summaries": {
            "fresh_process_startup_seconds": summarize(startup_samples),
            "model_pack_open_seconds": summarize(model_open_samples),
        },
        "fresh_process_startup_samples_seconds": startup_samples,
        "model_pack_open_samples_seconds": model_open_samples,
        "workloads": workloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="fresh process sessions per workload")
    parser.add_argument(
        "--warm-repetitions", type=int, default=5, help="measured resident requests per session"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/12-performance-baseline.json",
        help="machine-readable report path; Markdown companion is written beside it",
    )
    args = parser.parse_args()
    if args.runs < 1 or args.warm_repetitions < 1:
        parser.error("runs and warm repetitions must be positive")
    try:
        report = run_benchmark(args.runs, args.warm_repetitions)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    companion = output.with_suffix(".md")
    companion.write_text(markdown_report(report), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {companion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
