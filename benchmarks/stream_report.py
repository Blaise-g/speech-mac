"""Validation and summaries for native streaming measurements."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

SAMPLE_RATE_HZ = 24_000
QUEUE_BOUND_BYTES = 48 * 1024
QUEUE_BOUND_SECONDS = 0.512
SILENT_BINARY_MARKER = b"KOKORO_STREAM_MUTED_OUTPUT_V1"
EVENT_NAMES = (
    "first_complete_chunk_ready",
    "first_buffer_accepted",
    "first_playback_start_request",
    "first_buffer_reclaimed",
)
METRIC_NAMES = (
    *EVENT_NAMES,
    "synthesis_wall_seconds",
    "synthesis_thread_cpu_seconds",
    "audio_queue_open_wall_seconds",
    "audio_queue_open_complete_seconds",
    "audio_queue_start_api_seconds",
    "submit_backpressure_wall_seconds",
    "playback_wall_seconds",
    "process_wall_seconds",
    "process_cpu_seconds",
    "generated_audio_seconds",
    "queue_starvation_observations",
)


def _positive_finite(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and positive")
    return number


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    samples = list(values)
    if not samples:
        raise ValueError("measurement samples must not be empty")
    for value in samples:
        _positive_finite(value, "measurement sample", allow_zero=True)
    return {
        "sample_count": len(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def validate_measurement(sample: dict[str, object]) -> None:
    required = {
        "schema_version",
        "timing_origin",
        "event_seconds",
        "synthesis_wall_seconds",
        "synthesis_thread_cpu_seconds",
        "audio_queue_open_wall_seconds",
        "audio_queue_open_complete_seconds",
        "audio_queue_start_api_seconds",
        "submit_backpressure_wall_seconds",
        "playback_wall_seconds",
        "process_wall_seconds",
        "process_cpu_seconds",
        "generated_sample_count",
        "generated_audio_seconds",
        "chunk_count",
        "queue_bound_bytes",
        "queue_bound_audio_seconds",
        "queue_starvation_observations",
        "acoustic_onset_seconds",
        "acoustic_onset_status",
        "audio_queue_volume",
        "audio_queue_mute_status",
    }
    missing = required - sample.keys()
    if missing:
        raise ValueError(f"measurement is missing fields: {', '.join(sorted(missing))}")
    if sample["schema_version"] not in (1, 2):
        raise ValueError("unsupported measurement schema version")
    if sample["timing_origin"] != "benchmark_cli_call_entry":
        raise ValueError("measurement timing origin must be the CLI call entry")
    if sample["schema_version"] == 2 and (
        sample.get("benchmark_safety_marker") != SILENT_BINARY_MARKER.decode("ascii")
    ):
        raise ValueError("measurement must come from the fail-closed silent benchmark build")
    if "benchmark_safety_marker" in sample and (
        sample["benchmark_safety_marker"] != SILENT_BINARY_MARKER.decode("ascii")
    ):
        raise ValueError("measurement contains an invalid silent-output build marker")

    events = sample["event_seconds"]
    if not isinstance(events, dict) or set(events) != set(EVENT_NAMES):
        raise ValueError("measurement must report all four streaming event timestamps")
    event_times = {
        name: _positive_finite(events[name], name)
        for name in EVENT_NAMES
    }
    if not (
        event_times["first_complete_chunk_ready"]
        <= event_times["first_buffer_accepted"]
        <= event_times["first_playback_start_request"]
        and event_times["first_buffer_accepted"]
        <= event_times["first_buffer_reclaimed"]
    ):
        raise ValueError("streaming event timestamps violate the required partial order")

    for name in (
        "synthesis_wall_seconds",
        "synthesis_thread_cpu_seconds",
        "playback_wall_seconds",
        "process_wall_seconds",
        "process_cpu_seconds",
        "generated_audio_seconds",
    ):
        _positive_finite(sample[name], name)
    _positive_finite(sample["audio_queue_open_wall_seconds"], "AudioQueue open wall time", allow_zero=True)
    _positive_finite(sample["audio_queue_open_complete_seconds"], "AudioQueue open completion")
    _positive_finite(sample["audio_queue_start_api_seconds"], "AudioQueue start API call", allow_zero=True)
    _positive_finite(sample["submit_backpressure_wall_seconds"], "submit backpressure", allow_zero=True)
    _positive_finite(sample["queue_bound_audio_seconds"], "queue bound duration")
    if sample["process_wall_seconds"] < sample["playback_wall_seconds"]:
        raise ValueError("process wall time cannot be shorter than playback wall time")

    for name in ("generated_sample_count", "chunk_count", "queue_bound_bytes"):
        value = sample[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    starvation = sample["queue_starvation_observations"]
    if isinstance(starvation, bool) or not isinstance(starvation, int) or starvation < 0:
        raise ValueError("queue starvation observations must be a non-negative integer")
    if sample["queue_bound_bytes"] != QUEUE_BOUND_BYTES or not math.isclose(
        float(sample["queue_bound_audio_seconds"]), QUEUE_BOUND_SECONDS, abs_tol=1e-9
    ):
        raise ValueError("reported queue bound does not match the configured three-buffer pool")
    duration = int(sample["generated_sample_count"]) / SAMPLE_RATE_HZ
    if not math.isclose(float(sample["generated_audio_seconds"]), duration, abs_tol=1e-8):
        raise ValueError("generated audio duration does not match the sample count")
    if sample["audio_queue_volume"] != 0.0:
        raise ValueError("device measurements must use an inaudible per-queue volume of zero")
    if sample["audio_queue_mute_status"] != "succeeded":
        raise ValueError("device measurement requires successful per-queue mute confirmation")
    if sample["acoustic_onset_seconds"] is not None or sample["acoustic_onset_status"] != "unavailable":
        raise ValueError("acoustic timing requires independent render or loopback evidence")


def validate_raw_record(raw: dict[str, object]) -> None:
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported streaming report schema version")
    artifact_directory = raw.get("artifact_directory")
    if not isinstance(artifact_directory, str) or not artifact_directory.startswith("artifacts/"):
        raise ValueError("raw record must retain its artifact directory")
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("raw record must retain provenance")
    for name in ("commit", "source_sha256", "host", "builds", "model", "voices"):
        if name not in provenance:
            raise ValueError(f"provenance is missing {name}")
    verification = raw.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("raw record must retain verification status")
    checks = verification.get("checks")
    if not isinstance(checks, list):
        raise ValueError("verification must retain a list of check results")
    for check in checks:
        if (
            not isinstance(check, dict)
            or not isinstance(check.get("command"), str)
            or not check["command"]
            or check.get("result") not in ("passed", "failed")
        ):
            raise ValueError("verification check must retain its command and result")
    if not isinstance(verification.get("independent_review"), str):
        raise ValueError("verification must retain independent review disposition")
    profile = raw.get("profile")
    if not isinstance(profile, dict) or profile.get("status") not in ("captured", "unavailable"):
        raise ValueError("raw record must state whether a profile sample was captured")
    if profile["status"] == "captured" and (
        not isinstance(profile.get("file"), str)
        or not profile.get("sha256")
        or not profile.get("command")
    ):
        raise ValueError("profile record must retain its file, hash, and exact command")
    device_access = raw.get("device_access")
    if not isinstance(device_access, dict):
        raise ValueError("raw record must retain device mute and access status")
    if (
        device_access.get("per_queue_volume_control")
        != "AudioQueueSetParameter(kAudioQueueParam_Volume, 0.0f)"
        or device_access.get("system_output_volume_changed") is not False
        or device_access.get("escalated_muted_collection") not in ("succeeded", "unavailable")
    ):
        raise ValueError("device measurement must record fail-closed per-queue muting")
    sandbox_probe = device_access.get("sandbox_probe")
    if sandbox_probe is not None and (
        not isinstance(sandbox_probe, dict)
        or not isinstance(sandbox_probe.get("command"), list)
        or sandbox_probe.get("operation") != "AudioQueueStart"
        or not isinstance(sandbox_probe.get("osstatus"), int)
        or sandbox_probe.get("per_queue_mute_status") != "succeeded before PCM enqueue"
        or sandbox_probe.get("system_output_volume_changed") is not False
    ):
        raise ValueError("sandbox AudioQueueStart failure must retain its muted command and status")
    records = raw.get("streaming_samples")
    if not isinstance(records, list) or not records:
        raise ValueError("raw record must retain streaming samples")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("streaming sample must be an object")
        for name in ("workload", "voice", "command", "driver_json", "measurement"):
            if name not in record:
                raise ValueError(f"streaming sample is missing {name}")
        if not isinstance(record["command"], list) or not record["command"]:
            raise ValueError("streaming sample must retain its exact command argv")
        if not isinstance(record["driver_json"], str) or not record["driver_json"]:
            raise ValueError("streaming sample must retain the raw driver JSON line")
        validate_measurement(record["measurement"])
    optimization = raw.get("optimization")
    if not isinstance(optimization, dict) or not isinstance(optimization.get("paired_runs"), list):
        raise ValueError("raw record must retain paired optimization measurements")
    candidate_evidence = optimization.get("candidate_evidence")
    if not isinstance(candidate_evidence, dict):
        raise ValueError("optimization record must retain rejected candidate evidence status")
    if candidate_evidence:
        for name in ("patch", "check_log"):
            evidence = candidate_evidence.get(name)
            if (
                not isinstance(evidence, dict)
                or not isinstance(evidence.get("file"), str)
                or not evidence.get("sha256")
            ):
                raise ValueError("candidate evidence must retain its file name and SHA-256")
    for pair in optimization["paired_runs"]:
        if not isinstance(pair, dict) or pair.get("order") not in (
            ["baseline", "candidate"],
            ["candidate", "baseline"],
        ):
            raise ValueError("optimization run must retain paired order")
        for phase in ("baseline", "candidate"):
            if phase not in pair or "driver_json" not in pair[phase]:
                raise ValueError("optimization pair must retain both raw sample lines")
            validate_measurement(pair[phase]["measurement"])


def event_summary(records: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for name in EVENT_NAMES:
        summaries[name] = summarize(
            float(record["measurement"]["event_seconds"][name]) for record in records
        )
    for name in METRIC_NAMES[len(EVENT_NAMES) :]:
        summaries[name] = summarize(float(record["measurement"][name]) for record in records)
    return summaries


def validate_silent_binary(binary: bytes) -> None:
    if SILENT_BINARY_MARKER not in binary:
        raise ValueError("benchmark binary lacks the fail-closed silent-output build marker")


def paired_outcome(pairs: list[dict[str, object]]) -> dict[str, float | int | bool]:
    if not pairs:
        raise ValueError("paired optimization runs must not be empty")
    before = [float(pair["baseline"]["measurement"]["synthesis_wall_seconds"]) for pair in pairs]
    after = [float(pair["candidate"]["measurement"]["synthesis_wall_seconds"]) for pair in pairs]
    before_summary = summarize(before)
    after_summary = summarize(after)
    threshold_pct = max(
        (float(before_summary["max"]) - float(before_summary["min"]))
        / float(before_summary["median"]),
        (float(after_summary["max"]) - float(after_summary["min"]))
        / float(after_summary["median"]),
    ) * 100.0
    gain_pct = (float(before_summary["median"]) - float(after_summary["median"])) / float(
        before_summary["median"]
    ) * 100.0
    return {
        "pair_count": len(pairs),
        "baseline_median_seconds": float(before_summary["median"]),
        "baseline_min_seconds": float(before_summary["min"]),
        "baseline_max_seconds": float(before_summary["max"]),
        "candidate_median_seconds": float(after_summary["median"]),
        "candidate_min_seconds": float(after_summary["min"]),
        "candidate_max_seconds": float(after_summary["max"]),
        "contemporaneous_range_threshold_pct": threshold_pct,
        "median_gain_pct": gain_pct,
        "clears_observed_variation": gain_pct > threshold_pct,
    }
