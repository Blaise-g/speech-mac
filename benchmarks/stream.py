"""Collect muted, repeatable native streaming timing records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import run as native_run
from stream_report import (
    EVENT_NAMES,
    event_summary,
    paired_outcome,
    validate_raw_record,
    validate_silent_binary,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/kokoro-v1_0-af_heart.kpack"
VOICE_PATHS = {
    "af_heart": None,
    "am_adam": ROOT / "models/voices/am_adam.kvoice",
}
WORKLOADS = (
    {
        "name": "hello_from_terminal",
        "text": "hello from the terminal",
    },
    {
        "name": "representative_paragraph",
        "text": (
            "Good morning. I can help you organize the meeting notes, summarize the main "
            "decisions, and draft a clear follow-up for everyone who joined."
        ),
    },
)
VOICES = ("af_heart", "am_adam")
SEED = 104729
SPEED = "1.0"
DEFAULT_BINARY = ROOT / ".build/kokoro-stream-benchmark"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def command(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths() -> list[Path]:
    return sorted(
        {
            ROOT / "Makefile",
            *ROOT.joinpath("src").rglob("*.c"),
            *ROOT.joinpath("src").rglob("*.h"),
            *ROOT.joinpath("benchmarks").glob("*.c"),
            *ROOT.joinpath("benchmarks").glob("*.h"),
            *ROOT.joinpath("benchmarks").glob("*.py"),
            ROOT / "reference/tests/test_stream_report.py",
        }
    )


def source_fingerprint() -> tuple[str, dict[str, str]]:
    manifest: dict[str, str] = {}
    digest = hashlib.sha256()
    for path in source_paths():
        relative = path.relative_to(ROOT).as_posix()
        file_digest = file_sha256(path)
        manifest[relative] = file_digest
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest(), manifest


def build_record(path: Path, phase: str, source_sha256: str) -> dict[str, object]:
    compiler = os.environ.get("KOKORO_STREAM_COMPILER", "xcrun clang")
    compiler_args = shlex.split(compiler)
    compiler_version = command([*compiler_args, "--version"]).splitlines()[0]
    base_flags = os.environ.get("KOKORO_STREAM_BASE_CFLAGS", "-std=c17 -Wall -Wextra -Wpedantic -Werror")
    opt_flags = os.environ.get("KOKORO_STREAM_OPT_CFLAGS", "-O2")
    preprocessor_flags = os.environ.get("KOKORO_STREAM_CPPFLAGS", "-Isrc")
    common_defines = "-DKOKORO_STREAM_BENCHMARK_TIMING -DKOKORO_STREAM_SILENT_OUTPUT"
    phase_flags = (
        os.environ.get("KOKORO_STREAM_CANDIDATE_FLAGS", "")
        if phase == "candidate"
        else ""
    )
    combined_preprocessor_flags = " ".join(
        value for value in (preprocessor_flags, phase_flags) if value
    )
    link_flags = os.environ.get("KOKORO_STREAM_LDLIBS", "-framework AudioToolbox -framework Accelerate")
    try:
        build_directory = path.parent.relative_to(ROOT).as_posix()
    except ValueError:
        build_directory = str(path.parent)
    build_command = shlex.join(
        [
            "make",
            "stream-benchmark-build",
            f"BUILD_DIR={build_directory}",
            f"CPPFLAGS={combined_preprocessor_flags}",
        ]
    )
    return {
        "phase": phase,
        "binary_path": str(path.resolve()),
        "binary_sha256": file_sha256(path),
        "source_sha256": source_sha256,
        "compiler_command": compiler,
        "compiler_version": compiler_version,
        "base_cflags": base_flags,
        "optimization_flags": opt_flags,
        "preprocessor_flags": preprocessor_flags,
        "benchmark_defines": common_defines,
        "phase_defines": phase_flags,
        "combined_preprocessor_flags": combined_preprocessor_flags,
        "link_flags": link_flags,
        "build_command": build_command,
    }


def require_silent_benchmark_binary(path: Path) -> None:
    try:
        validate_silent_binary(path.read_bytes())
    except OSError as error:
        raise RuntimeError(f"cannot inspect benchmark binary for silent-output marker: {path}") from error


def model_provenance() -> dict[str, object]:
    # The worktree links model assets to the coordinator's shared real-model tree.
    # Reuse the ticket 15 parser while giving its relative path calculation the resolved root.
    original_root = native_run.ROOT
    native_run.ROOT = MODEL_PATH.resolve().parents[1]
    try:
        pack = native_run.pack_provenance(MODEL_PATH)
    finally:
        native_run.ROOT = original_root
    pack["pack_sha256"] = file_sha256(MODEL_PATH)
    voices: dict[str, object] = {
        "af_heart": {
            "style_source": "embedded in native model pack",
            "embedded_tensor": "voice.af_heart.style",
            "sha256": pack["voice_sha256"],
        }
    }
    adam_path = VOICE_PATHS["am_adam"]
    assert adam_path is not None
    voices["am_adam"] = {
        "style_source": "separate pinned native voice artifact",
        "path": str(adam_path.relative_to(ROOT)),
        "size_bytes": adam_path.stat().st_size,
        "sha256": file_sha256(adam_path),
    }
    return {"pack": pack, "voices": voices}


def provenance(baseline_binary: Path, candidate_binary: Path | None) -> dict[str, object]:
    source_sha, source_manifest = source_fingerprint()
    commit = command(["git", "rev-parse", "HEAD"])
    tree_state = command(["git", "status", "--short", "--untracked-files=all"])
    builds = {"baseline": build_record(baseline_binary, "baseline", source_sha)}
    if candidate_binary is not None:
        builds["candidate"] = build_record(candidate_binary, "candidate", source_sha)
    model = model_provenance()
    return {
        "commit": commit,
        "working_tree_state": "dirty" if tree_state else "clean",
        "source_sha256": source_sha,
        "source_files": source_manifest,
        "host": native_run.host_provenance(),
        "native_build": {
            "compiler": builds["baseline"]["compiler_version"],
            "build_directory": str(baseline_binary.parent.resolve()),
            "defines": builds["baseline"]["benchmark_defines"],
        },
        "builds": builds,
        "model": model["pack"],
        "voices": model["voices"],
        "seed": SEED,
        "speed": float(SPEED),
        "sample_rate_hz": 24000,
        "espeak_ng_version": command(["espeak-ng", "--version"]).splitlines()[0],
        "python_version": sys.version.split()[0],
        "uv_version": command(["uv", "--version"]),
    }


def cli_argv(binary: Path, workload: dict[str, str], voice: str) -> list[str]:
    return [
        str(binary.resolve()),
        "say",
        workload["text"],
        "--voice",
        voice,
        "--speed",
        SPEED,
        "--stream",
    ]


def run_one(binary: Path, workload: dict[str, str], voice: str, sample_index: int) -> dict[str, object]:
    arguments = cli_argv(binary, workload, voice)
    result = subprocess.run(
        arguments,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"stream driver failed ({result.returncode}) for {workload['name']}/{voice}: "
            f"{result.stderr.strip()}"
        )
    output = result.stdout.strip()
    if "\n" in output:
        raise RuntimeError("stream driver returned more than one raw JSON line")
    measurement = json.loads(output)
    from stream_report import validate_measurement

    validate_measurement(measurement)
    return {
        "sample_index": sample_index,
        "workload": workload["name"],
        "text": workload["text"],
        "voice": voice,
        "command": arguments,
        "command_display": shlex.join(arguments),
        "return_code": result.returncode,
        "driver_json": output,
        "measurement": measurement,
        "stderr": result.stderr,
    }


def measurement_groups(records: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record["workload"]), str(record["voice"]))
        groups.setdefault(key, []).append(record)
    return groups


def format_range(summary: dict[str, float | int]) -> str:
    return (
        f"{float(summary['median']):.3f} s "
        f"[{float(summary['min']):.3f}, {float(summary['max']):.3f}]"
    )


def format_count_range(summary: dict[str, float | int]) -> str:
    return (
        f"{float(summary['median']):.0f} "
        f"[{float(summary['min']):.0f}, {float(summary['max']):.0f}]"
    )


def render_report(raw: dict[str, object]) -> str:
    lines = [
        "# Native streaming timing and optimization attempt",
        "",
        f"Captured: `{raw['created_at_utc']}` · source commit `{raw['provenance']['commit']}`.",
        "",
        "Each event is measured from the benchmark process immediately before `kokoro_cli_run`. "
        "The benchmark invokes the production CLI parser and stream path. Per-queue volume is set "
        "to 0.0 before any PCM submission; it does not change system output volume.",
        "",
        "The successful-start event is timestamped after `AudioQueueStart` returns successfully. "
        "An AudioQueue callback can run while that SDK call is still in progress, so the first "
        "buffer reclamation may precede the successful-start timestamp; validation checks only "
        "the ordering constraints implied by the adapter.",
        "",
        "The callback reclamation timestamp is a buffer reuse event, not evidence that audio was "
        "rendered. Acoustic onset is unavailable because no output loopback or device-clock "
        "measurement was used. The queue-starvation count is an adapter observation that the "
        "pending AudioQueue buffer count reached zero while the producer submission phase was "
        "still open; actual "
        "device underruns remain unavailable without independent device evidence. On long inputs "
        "it flags a potential playback-gap condition, not a confirmed audible gap.",
        "",
    ]
    sandbox_probe = raw["device_access"]["sandbox_probe"]
    if sandbox_probe is not None:
        lines.extend(
            [
                f"The sandbox probe reached `AudioQueueStart` and failed with OSStatus "
                f"`{sandbox_probe['osstatus']}` after per-queue mute succeeded before PCM enqueue. "
                "The repeated collection used the authorized escalated local device access; all "
                "benchmark queues remained muted and system output volume was unchanged.",
                "",
            ]
        )
    lines.extend(
        [
            "## Streaming samples",
            "",
            "All values are medians with observed sample ranges. Event times include full frontend and "
            "model initialization from the benchmark CLI-call entry.",
            "",
            "| Workload | Voice | Samples | First chunk ready | First buffer accepted | Start success | First buffer reclaimed |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    groups = sorted(measurement_groups(raw["streaming_samples"]).items())
    for (workload, voice), records in groups:
        summary = event_summary(records)
        event_values = [format_range(summary[name]) for name in EVENT_NAMES]
        lines.append(
            f"| {workload} | {voice} | {len(records)} | {' | '.join(event_values)} |"
        )

    lines.extend(
        [
            "",
            "## Synthesis and AudioQueue timing",
            "",
            "Synthesis wall and thread CPU cover synthesis calls only. AudioQueue open completion "
            "is measured from CLI-call entry; open span and start API are separate SDK intervals.",
            "",
            "| Workload | Voice | Samples | Synthesis wall / thread CPU | AQ open span / completion from entry | Start API | Submit backpressure | Playback wall | Process CPU / wall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for (workload, voice), records in groups:
        summary = event_summary(records)
        synthesis = (
            f"{format_range(summary['synthesis_wall_seconds'])} / "
            f"{format_range(summary['synthesis_thread_cpu_seconds'])}"
        )
        queue_open = (
            f"{format_range(summary['audio_queue_open_wall_seconds'])} / "
            f"{format_range(summary['audio_queue_open_complete_seconds'])}"
        )
        process = (
            f"{format_range(summary['process_cpu_seconds'])} / "
            f"{format_range(summary['process_wall_seconds'])}"
        )
        lines.append(
            f"| {workload} | {voice} | {len(records)} | {synthesis} | {queue_open} | "
            f"{format_range(summary['audio_queue_start_api_seconds'])} | "
            f"{format_range(summary['submit_backpressure_wall_seconds'])} | "
            f"{format_range(summary['playback_wall_seconds'])} | {process} |"
        )

    lines.extend(
        [
            "",
            "## Generated audio and bounded queue",
            "",
            "The buffer-reclamation callback and queue-starvation observation do not prove device "
            "render completion or an acoustic gap.",
            "",
            "| Workload | Voice | Samples | Generated audio | Queue bound | Queue-starvation observations |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for (workload, voice), records in groups:
        summary = event_summary(records)
        sample = records[0]["measurement"]
        starvation = format_count_range(summary["queue_starvation_observations"])
        lines.append(
            f"| {workload} | {voice} | {len(records)} | "
            f"{format_range(summary['generated_audio_seconds'])} | "
            f"{sample['queue_bound_bytes']:,} B / {float(sample['queue_bound_audio_seconds']):.3f} s | "
            f"{starvation} |"
        )

    optimization = raw["optimization"]
    lines.extend(["", "## Profile-guided attempt", ""])
    if raw["profile"]["status"] == "captured":
        lines.append(
            "The qualitative native profile is retained in [`profile.txt`](profile.txt), with its "
            "capture command and SHA-256 in `raw.json`. It identifies the sampled hot path but "
            "does not provide cycle-accurate attribution."
        )
    else:
        lines.append("A native profile sample was unavailable; see `raw.json` for the reason.")
    lines.append("")
    lines.append(optimization["description"])
    lines.append("")
    candidate_evidence = optimization["candidate_evidence"]
    if candidate_evidence:
        lines.append(
            f"The failed numerical candidate is preserved as "
            f"[`{candidate_evidence['patch']['file']}`]({candidate_evidence['patch']['file']})"
            f" (SHA-256 `{candidate_evidence['patch']['sha256']}`); its fixed-budget check output is "
            f"[`{candidate_evidence['check_log']['file']}`]({candidate_evidence['check_log']['file']}) "
            f"(SHA-256 `{candidate_evidence['check_log']['sha256']}`)."
        )
        lines.append("")
    if optimization["paired_runs"]:
        lines.extend(
            [
                "Paired same-host measurements use `af_heart`, speed 1.0, seed 104729, and the "
                "ticket 15 short and paragraph text. The threshold is each workload's larger "
                "contemporaneous baseline/candidate full-range variation.",
                "",
                "| Workload | Pairs | Baseline synthesis median [range] | Candidate synthesis median [range] | "
                "Observed variation threshold | Median gain | Clears variation |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for workload, outcome in sorted(optimization["outcomes"].items()):
            lines.append(
                f"| {workload} | {outcome['pair_count']} | "
                f"{float(outcome['baseline_median_seconds']):.3f} s "
                f"[{float(outcome['baseline_min_seconds']):.3f}, "
                f"{float(outcome['baseline_max_seconds']):.3f}] | "
                f"{float(outcome['candidate_median_seconds']):.3f} s "
                f"[{float(outcome['candidate_min_seconds']):.3f}, "
                f"{float(outcome['candidate_max_seconds']):.3f}] | "
                f"{float(outcome['contemporaneous_range_threshold_pct']):.2f}% | "
                f"{float(outcome['median_gain_pct']):.2f}% | "
                f"{'yes' if outcome['clears_observed_variation'] else 'no'} |"
            )
        lines.extend(["", optimization["disposition"]])
    else:
        lines.append("No paired optimization samples were captured in this run.")
    lines.extend(
        [
            "",
            "## Provenance and rerun",
            "",
            f"Raw measurements and full source/build/host/model provenance: [`raw.json`](raw.json). "
            f"The benchmark source fingerprint is `{raw['provenance']['source_sha256']}`; "
            "baseline and candidate executable hashes, per-file source hashes, commands, and "
            "raw driver JSON lines are retained there. The final source fingerprint after the "
            f"rejected candidate was removed is `{raw['provenance'].get('final_source_sha256', 'pending')}`.",
            "The rerun command installs the rejected patch only for the experiment and reverses "
            "it with an `EXIT` trap. If the shell is interrupted before cleanup, run "
            f"`git apply --reverse {raw['artifact_directory']}/candidate.patch` from the repository root.",
            "",
            "```sh",
            raw["rerun_command"],
            "```",
            "",
            "## Verification",
            "",
        ]
    )
    verification = raw["verification"]
    if verification["checks"]:
        for check in verification["checks"]:
            lines.append(f"- `{check['command']}` — {check['result']}.")
    else:
        lines.append("No check results were recorded when this report was generated.")
    lines.extend([f"Independent review: {verification['independent_review']}.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="baseline streaming samples per voice/input")
    parser.add_argument("--optimization-pairs", type=int, default=0)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--candidate-patch", type=Path)
    parser.add_argument("--candidate-check-log", type=Path)
    parser.add_argument("--profile-record", type=Path)
    parser.add_argument("--profile-command", default="")
    parser.add_argument("--sandbox-probe-osstatus", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--optimization-description", default="No profile-guided attempt was supplied.")
    parser.add_argument("--optimization-disposition", default="Optimization criterion remains open.")
    args = parser.parse_args()
    if args.runs <= 0 or args.optimization_pairs < 0:
        parser.error("--runs must be positive and --optimization-pairs cannot be negative")
    if args.optimization_pairs and args.candidate_binary is None:
        parser.error("--candidate-binary is required when paired optimization samples are requested")
    if not args.binary.is_file():
        parser.error(f"baseline benchmark binary does not exist: {args.binary}")
    if args.candidate_binary is not None and not args.candidate_binary.is_file():
        parser.error(f"candidate benchmark binary does not exist: {args.candidate_binary}")
    try:
        require_silent_benchmark_binary(args.binary)
        if args.candidate_binary is not None:
            require_silent_benchmark_binary(args.candidate_binary)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output or ROOT / f"artifacts/18-streaming-benchmark-{stamp}"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    raw: dict[str, object] = {
        "schema_version": 1,
        "artifact_directory": output_dir.relative_to(ROOT).as_posix(),
        "created_at_utc": utc_now(),
        "provenance": provenance(args.binary, args.candidate_binary),
        "inputs": {
            "workloads": list(WORKLOADS),
            "voices": list(VOICES),
            "speed": float(SPEED),
            "seed": SEED,
            "sample_rate_hz": 24000,
            "speaker_output": "AudioQueue volume 0.0 set for each benchmark queue",
            "queue_bound_bytes": 49152,
            "queue_bound_audio_seconds": 0.512,
        },
        "streaming_samples": [],
        "profile": {"status": "unavailable", "reason": "No profile sample was supplied."},
        "device_access": {
            "per_queue_volume_control": "AudioQueueSetParameter(kAudioQueueParam_Volume, 0.0f)",
            "system_output_volume_changed": False,
            "sandbox_probe": None,
            "escalated_muted_collection": "succeeded",
        },
        "optimization": {
            "description": args.optimization_description,
            "paired_runs": [],
            "outcomes": {},
            "disposition": args.optimization_disposition,
            "candidate_evidence": {},
        },
        "verification": {"checks": [], "independent_review": "pending"},
        "rerun_command": "",
    }
    if args.sandbox_probe_osstatus is not None:
        raw["device_access"]["sandbox_probe"] = {
            "command": cli_argv(args.binary, WORKLOADS[0], "af_heart"),
            "operation": "AudioQueueStart",
            "osstatus": args.sandbox_probe_osstatus,
            "per_queue_mute_status": "succeeded before PCM enqueue",
            "system_output_volume_changed": False,
        }
    if args.candidate_patch is not None or args.candidate_check_log is not None:
        if (
            args.candidate_patch is None
            or args.candidate_check_log is None
            or not args.candidate_patch.is_file()
            or not args.candidate_check_log.is_file()
        ):
            parser.error("candidate evidence requires an existing patch and check log")
        candidate_evidence = raw["optimization"]["candidate_evidence"]
        for key, path, filename in (
            ("patch", args.candidate_patch, "candidate.patch"),
            ("check_log", args.candidate_check_log, "candidate-check.log"),
        ):
            target = output_dir / filename
            shutil.copyfile(path, target)
            candidate_evidence[key] = {"file": filename, "sha256": file_sha256(target)}

    records: list[dict[str, object]] = raw["streaming_samples"]
    sample_index = 0
    for workload in WORKLOADS:
        for voice in VOICES:
            for _ in range(args.runs):
                records.append(run_one(args.binary, workload, voice, sample_index))
                sample_index += 1

    if args.profile_record is not None:
        if not args.profile_record.is_file() or not args.profile_command:
            parser.error("--profile-record requires an existing file and --profile-command")
        profile_path = output_dir / "profile.txt"
        shutil.copyfile(args.profile_record, profile_path)
        raw["profile"] = {
            "status": "captured",
            "file": "profile.txt",
            "sha256": file_sha256(profile_path),
            "command": args.profile_command,
            "scope": "qualitative native inference stack sample; not cycle attribution",
        }

    pairs: list[dict[str, object]] = raw["optimization"]["paired_runs"]
    if args.optimization_pairs:
        assert args.candidate_binary is not None
        for workload in WORKLOADS:
            for pair_index in range(args.optimization_pairs):
                order = (
                    ("baseline", "candidate")
                    if pair_index % 2 == 0
                    else ("candidate", "baseline")
                )
                pair: dict[str, object] = {
                    "workload": workload["name"],
                    "voice": "af_heart",
                    "pair_index": pair_index,
                    "order": list(order),
                }
                for phase in order:
                    binary = args.binary if phase == "baseline" else args.candidate_binary
                    pair[phase] = run_one(binary, workload, "af_heart", pair_index)
                pairs.append(pair)
        outcomes: dict[str, object] = raw["optimization"]["outcomes"]
        for workload in WORKLOADS:
            workload_pairs = [pair for pair in pairs if pair["workload"] == workload["name"]]
            outcomes[workload["name"]] = paired_outcome(workload_pairs)

    artifact_dir = output_dir.relative_to(ROOT).as_posix()
    if args.optimization_pairs:
        candidate_build = raw["provenance"]["builds"]["candidate"]
        baseline_build = raw["provenance"]["builds"]["baseline"]
        rerun_steps = [
            "set -e",
            f"candidate_patch={shlex.quote(f'{artifact_dir}/candidate.patch')}",
            'git apply "$candidate_patch"',
            'trap \'git apply --reverse "$candidate_patch"\' EXIT',
            baseline_build["build_command"],
            candidate_build["build_command"],
        ]
        runner_args = [
            "uv",
            "run",
            "--project",
            "reference",
            "--locked",
            "--no-sync",
            "python",
            "benchmarks/stream.py",
            "--runs",
            str(args.runs),
            "--optimization-pairs",
            str(args.optimization_pairs),
            "--binary",
            str(args.binary),
            "--candidate-binary",
            str(args.candidate_binary),
            "--candidate-patch",
            f"{artifact_dir}/candidate.patch",
            "--candidate-check-log",
            f"{artifact_dir}/candidate-check.log",
            "--optimization-description",
            args.optimization_description,
            "--optimization-disposition",
            args.optimization_disposition,
            "--output",
            "artifacts/18-streaming-benchmark-rerun",
        ]
        if args.profile_record is not None:
            runner_args.extend(
                [
                    "--profile-record",
                    f"{artifact_dir}/profile.txt",
                    "--profile-command",
                    args.profile_command,
                ]
            )
        if args.sandbox_probe_osstatus is not None:
            runner_args.extend(["--sandbox-probe-osstatus", str(args.sandbox_probe_osstatus)])
        environment = [
            f"KOKORO_STREAM_CPPFLAGS={shlex.quote(baseline_build['preprocessor_flags'])}",
            f"KOKORO_STREAM_CANDIDATE_FLAGS={shlex.quote(candidate_build['phase_defines'])}",
            "UV_CACHE_DIR=/private/tmp/kokoro-uv-cache",
        ]
        rerun_steps.append(f"{' '.join(environment)} {shlex.join(runner_args)}")
        raw["rerun_command"] = "\n".join(rerun_steps)
    else:
        raw["rerun_command"] = (
            f"make stream-benchmark-build && UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run "
            f"--project reference --locked --no-sync python benchmarks/stream.py --runs {args.runs} "
            f"--output artifacts/18-streaming-benchmark-rerun"
        )
    validate_raw_record(raw)
    raw_path = output_dir / "raw.json"
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = output_dir / "report.md"
    report_path.write_text(render_report(raw), encoding="utf-8")
    print(f"Wrote {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
