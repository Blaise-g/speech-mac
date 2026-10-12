"""Compare the native CLI with the already-installed CoreML Kokoro CLI."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import run as native_run  # noqa: E402

COREML_CLI = Path("/opt/homebrew/bin/kokoro")
COREML_BINARY = Path("/opt/homebrew/Cellar/kokoro/0.11.2/libexec/kokoro")
COREML_MODELS = Path.home() / "Library/Application Support/kokoro-coreml/models/kokoro"
COREML_TAG = "v0.11.2"
COREML_COMMIT = "6bfc02afb237a311caee927482e58c028edb6acb"
COREML_MODELS_URL = (
    "https://github.com/Jud/kokoro-coreml/releases/download/"
    "models-2026-03-23/kokoro-models.tar.gz"
)
TIME = Path("/usr/bin/time")
SHORT_TEXT = "hello from the terminal"
LONG_TEXT = (
    "Good morning. I can help you organize the meeting notes, summarize the main "
    "decisions, and draft a clear follow-up for everyone who joined."
)
# eSpeak NG 1.52.0 output for SHORT_TEXT after removing its U+200D joiners.
SHORT_IPA = "həlˈoʊ fɹʌmðə tˈɜːmɪnəl"
WORKLOADS = (
    {"name": "hello_from_terminal", "mode": "english", "text": SHORT_TEXT},
    {"name": "representative_paragraph", "mode": "english", "text": LONG_TEXT},
)

# Match Makefile's release benchmark defaults so provenance works in a direct rerun too.
for key, value in {
    "KOKORO_BENCH_COMPILER": "xcrun clang",
    "KOKORO_BENCH_BASE_CFLAGS": "-std=c17 -Wall -Wextra -Wpedantic -Werror",
    "KOKORO_BENCH_OPT_CFLAGS": "-O2",
    "KOKORO_BENCH_CPPFLAGS": "-Isrc",
    "KOKORO_BENCH_LDLIBS": "-framework Accelerate",
    "KOKORO_BENCH_BUILD_DIR": ".build",
}.items():
    os.environ.setdefault(key, value)


def provenance() -> dict[str, object]:
    brew = json.loads(native_run.command(["brew", "info", "--json=v2", "kokoro"]))["formulae"][0]
    model_files = [path for path in COREML_MODELS.rglob("*") if path.is_file()]
    metadata = {}
    for name in ("kokoro_frontend", "kokoro_backend"):
        item = json.loads(
            (COREML_MODELS / f"{name}.mlmodelc/metadata.json").read_text()
        )[0]
        metadata[name.removeprefix("kokoro_")] = {
            key: item.get(key)
            for key in ("generatedClassName", "storagePrecision", "computePrecision")
        }
    assets = [
        COREML_MODELS / ".model-tag",
        COREML_MODELS / "kokoro_frontend.mlmodelc/metadata.json",
        COREML_MODELS / "kokoro_frontend.mlmodelc/model.mil",
        COREML_MODELS / "kokoro_frontend.mlmodelc/weights/weight.bin",
        COREML_MODELS / "kokoro_backend.mlmodelc/metadata.json",
        COREML_MODELS / "kokoro_backend.mlmodelc/model.mil",
        COREML_MODELS / "kokoro_backend.mlmodelc/weights/weight.bin",
        COREML_MODELS / "voices/af_heart.bin",
    ]
    coreml = {
        "version": native_run.command([str(COREML_CLI), "--version"]),
        "wrapper": {"path": str(COREML_CLI), "sha256": native_run.sha256_file(COREML_CLI)},
        "executable": {
            "path": str(COREML_BINARY),
            "sha256": native_run.sha256_file(COREML_BINARY),
        },
        "homebrew": {
            "formula": brew["full_name"],
            "version": brew["versions"]["stable"],
            "formula_sha256": brew["urls"]["stable"]["checksum"],
        },
        "source": {
            "repository": "https://github.com/Jud/kokoro-coreml",
            "tag": COREML_TAG,
            "commit": COREML_COMMIT,
            "models_url": COREML_MODELS_URL,
            "version_note": (
                "Installed executable reports 0.11.2; v0.11.2 source still declares "
                "0.10.0 in KokoroSay.swift."
            ),
            "compute_policy": "frontend .cpuOnly; backend .all",
        },
        "model": {
            "directory": str(COREML_MODELS),
            "tag": (COREML_MODELS / ".model-tag").read_text().strip(),
            "file_count": len(model_files),
            "total_bytes": sum(path.stat().st_size for path in model_files),
            "voice_asset_count": len(list((COREML_MODELS / "voices").glob("*.bin"))),
            "assets": [
                {
                    "path": str(path.relative_to(COREML_MODELS)),
                    "size_bytes": path.stat().st_size,
                    "sha256": native_run.sha256_file(path),
                }
                for path in assets
            ],
            "metadata": metadata,
        },
    }
    build = native_run.build_provenance()
    native = {
        "commit": native_run.command(["git", "rev-parse", "HEAD"]),
        "cli_path": str(native_run.CLI_PATH.resolve()),
        "cli_sha256": native_run.sha256_file(native_run.CLI_PATH),
        "driver_path": str(native_run.DRIVER_PATH.resolve()),
        "driver_sha256": native_run.sha256_file(native_run.DRIVER_PATH),
        "source_sha256": build_source_sha256(),
        "model_pack": native_run.pack_provenance(native_run.MODEL_PATH),
        "model_pack_sha256": native_run.sha256_file(native_run.MODEL_PATH),
        "build": build,
        "espeak_ng": native_run.command(["espeak-ng", "--version"]).splitlines()[0],
    }
    host = native_run.host_provenance()
    record = {
        "host": host,
        "native": native,
        "coreml": coreml,
        "comparison_runner_sha256": native_run.sha256_file(Path(__file__)),
        "comparison_test_sha256": native_run.sha256_file(ROOT / "reference/tests/test_compare.py"),
    }
    record["sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record


def build_source_sha256() -> str:
    paths = [
        ROOT / "Makefile",
        *sorted((ROOT / "src").rglob("*.c")),
        *sorted((ROOT / "src").rglob("*.h")),
        ROOT / "benchmarks/native_driver.c",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def measured(value: int | float, unit: str, scope: str) -> dict[str, object]:
    return {"status": "measured", "value": value, "unit": unit, "scope": scope}


def unsupported(metric: str, scope: str, reason: str) -> dict[str, object]:
    return {
        "status": "unsupported",
        "value": None,
        "unit": None,
        "metric": metric,
        "scope": scope,
        "reason": reason,
    }


def below_resolution(metric: str, scope: str, resolution: float, reason: str) -> dict[str, object]:
    return {
        "status": "below_resolution",
        "value": None,
        "unit": "seconds",
        "metric": metric,
        "scope": scope,
        "resolution_seconds": resolution,
        "upper_bound_seconds": resolution / 2,
        "reason": reason,
    }


def time_metrics(stderr: str) -> dict[str, object]:
    timing = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", stderr)
    rss = re.search(r"\s(\d+)\s+maximum resident set size", stderr)
    footprint = re.search(r"\s(\d+)\s+peak memory footprint", stderr)
    scope = "/usr/bin/time -l counters for the directly timed process"
    result = {
        "cpu_user_seconds": unsupported("user CPU", scope, "not reported by time -l"),
        "cpu_system_seconds": unsupported("system CPU", scope, "not reported by time -l"),
        "os_peak_rss_bytes": unsupported("OS-reported peak RSS", scope, "not reported by time -l"),
        "interval_sampled_peak_rss_bytes": unsupported(
            "interval-sampled RSS", scope, "the process was not interval-sampled"
        ),
    }
    if timing:
        user, system = float(timing[2]), float(timing[3])
        resolution = 0.01
        result["cpu_user_seconds"] = (
            measured(user, "seconds", scope)
            if user
            else below_resolution(
                "user CPU", scope, resolution, "time -l displayed 0.00 seconds"
            )
        )
        result["cpu_system_seconds"] = (
            measured(system, "seconds", scope)
            if system
            else below_resolution(
                "system CPU", scope, resolution, "time -l displayed 0.00 seconds"
            )
        )
        result["cpu_total_seconds"] = (
            measured(user + system, "seconds", scope)
            if user or system
            else below_resolution(
                "total CPU",
                scope,
                2 * resolution,
                "user and system CPU both displayed 0.00 seconds; each is below 0.005 seconds",
            )
        )
    if rss:
        result["os_peak_rss_bytes"] = measured(int(rss[1]), "bytes", scope)
    if footprint:
        result["os_peak_memory_footprint_bytes"] = measured(
            int(footprint[1]), "bytes", scope
        )
    return result


def audio(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as wav:
        result = {
            "channels": wav.getnchannels(),
            "sample_rate_hz": wav.getframerate(),
            "sample_width_bytes": wav.getsampwidth(),
            "sample_count": wav.getnframes(),
            "compression": wav.getcomptype(),
        }
    if result["channels"] != 1 or result["sample_rate_hz"] != 24000:
        raise ValueError("WAV must be mono 24 kHz")
    if result["sample_width_bytes"] != 2 or result["compression"] != "NONE":
        raise ValueError("WAV must be uncompressed PCM16")
    result.update(
        {
            "audio_seconds": result["sample_count"] / result["sample_rate_hz"],
            "size_bytes": path.stat().st_size,
            "sha256": native_run.sha256_file(path),
            "path": str(path.relative_to(ROOT)),
        }
    )
    return result


def cli_command(implementation: str, text: str, mode: str, output: Path) -> list[str]:
    cli = native_run.CLI_PATH if implementation == "native" else COREML_CLI
    args = [str(cli), "say"]
    if mode == "ipa":
        args.append("--ipa")
    args.extend([text, "--voice", "af_heart", "--speed", "1.0", "--output", str(output)])
    return args


def timed(arguments: list[str], sampler: "ProcessSampler | None" = None) -> dict[str, object]:
    wrapped = [str(TIME), "-l", *arguments]
    started = time.perf_counter()
    rss_index = len(sampler.rss) if sampler else 0
    before = sampler.read() if sampler else None
    if sampler:
        sampler.start()
    try:
        process = subprocess.run(wrapped, cwd=ROOT, text=True, capture_output=True)
    finally:
        if sampler:
            sampler.stop()
    after = sampler.read() if sampler else None
    wall = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {shlex.join(wrapped)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return {
        "command": wrapped,
        "rerun_command": shlex.join(wrapped),
        "return_code": process.returncode,
        "wall_seconds": wall,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "process_resources": time_metrics(process.stderr),
        "sampler_before": before,
        "sampler_after": after,
        "sampler_rss_bytes": sampler.rss[rss_index:] if sampler else [],
    }


def fresh_sample(
    report_dir: Path,
    spec: dict[str, str],
    mode: str,
    text: str,
    implementation: str,
    index: int,
    provenance_sha: str,
) -> dict[str, object]:
    path = report_dir / "audio" / f"fresh-{implementation}-{spec['name']}-{mode}-{index:02}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    result = timed(cli_command(implementation, text, mode, path))
    wav = audio(path)
    return {
        "workload": spec["name"],
        "mode": mode,
        "implementation": implementation,
        "sample_index": index,
        "input": text,
        "voice": "af_heart",
        "speed": 1.0,
        "provenance_sha256": provenance_sha,
        **result,
        "audio": wav,
        "audio_throughput_seconds_per_wall_second": wav["audio_seconds"] / result["wall_seconds"],
    }


def driver_command(text: str, output: Path, repetitions: int) -> list[str]:
    return [
        str(native_run.DRIVER_PATH), "--mode", "english", "--text", text,
        "--model", str(native_run.MODEL_PATH), "--voice", "af_heart", "--speed", "1.0",
        "--seed", "104729", "--warmups", "1", "--repetitions", str(repetitions),
        "--wav", str(output),
    ]


def native_resident(
    report_dir: Path,
    spec: dict[str, str],
    session: int,
    repetitions: int,
    provenance_sha: str,
) -> dict[str, object]:
    path = report_dir / "audio" / f"resident-native-{spec['name']}-{session:02}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    args = driver_command(spec["text"], path, repetitions)
    result = timed(args)
    events = [json.loads(line) for line in str(result["stdout"]).splitlines() if line]
    warm = [event for event in events if event.get("kind") == "warm_sample"]
    if len(warm) != repetitions:
        raise ValueError(f"native driver returned {len(warm)} warm samples, expected {repetitions}")
    for event in warm:
        native_run.validate_audio_sample(event)
        native_run.add_rate_metrics(event)
    return {
        "session_index": session,
        "provenance_sha256": provenance_sha,
        "command": result["command"],
        "rerun_command": result["rerun_command"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "process_resources": result["process_resources"],
        "resource_scope": "driver process lifetime including load, first request, warm-up, repeats, and WAV writing",
        "events": events,
        "warm_samples": warm,
        "wav": audio(path),
    }


class RusageV4(ctypes.Structure):
    # Mirrors macOS sys/resource.h's struct rusage_info_v4 through ri_runnable_time.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size "
            "resident_size phys_footprint proc_start_abstime proc_exit_abstime "
            "child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups "
            "child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten "
            "cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background "
            "cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated "
            "cpu_time_qos_user_interactive billed_system_time serviced_system_time "
            "logical_writes lifetime_max_phys_footprint instructions cycles billed_energy "
            "serviced_energy interval_max_phys_footprint runnable_time"
        ).split()
    ]


class MachTimebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def mach_timebase() -> dict[str, int]:
    timebase = MachTimebase()
    mach = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    read_timebase = mach.mach_timebase_info
    read_timebase.argtypes = [ctypes.POINTER(MachTimebase)]
    read_timebase.restype = ctypes.c_int
    status = read_timebase(ctypes.byref(timebase))
    if status != 0 or not timebase.numer or not timebase.denom:
        raise OSError(status, "mach_timebase_info failed")
    return {"numerator": int(timebase.numer), "denominator": int(timebase.denom)}


def mach_ticks_to_seconds(ticks: int, timebase: dict[str, int]) -> float:
    return ticks * timebase["numerator"] / timebase["denominator"] / 1e9


def validate_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"output directory must be inside the repository: {output_dir}") from error
    return output_dir


def daemon_measurement(pid: int) -> dict[str, int]:
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    read_usage = libproc.proc_pid_rusage
    read_usage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(RusageV4)]
    read_usage.restype = ctypes.c_int
    usage = RusageV4()
    if read_usage(pid, 4, ctypes.byref(usage)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), f"proc_pid_rusage({pid})")
    return {
        "pid": pid,
        "cpu_user_mach_ticks": usage.user_time,
        "cpu_system_mach_ticks": usage.system_time,
        "rss_bytes": usage.resident_size,
        "phys_footprint_bytes": usage.phys_footprint,
        "lifetime_peak_phys_footprint_bytes": usage.lifetime_max_phys_footprint,
    }


class ProcessSampler:
    """Sample one owned CoreML daemon's RSS while request clients are running."""

    def __init__(self, pid: int, interval: float = 0.05):
        self.pid, self.interval = pid, interval
        self.rss: list[int] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def read(self) -> dict[str, int]:
        usage = daemon_measurement(self.pid)
        self.rss.append(usage["rss_bytes"])
        return usage

    def start(self) -> None:
        def sample() -> None:
            while not self.stop_event.wait(self.interval):
                try:
                    self.read()
                except OSError:
                    return

        self.stop_event.clear()
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.thread = None


def daemon_status() -> tuple[str, int | None]:
    output = native_run.command([str(COREML_CLI), "daemon", "status"])
    match = re.search(r"(?:Daemon is running|Kokoro daemon ready) \(PID (\d+)\)", output)
    if "not running" in output.lower():
        return output, None
    if not match:
        raise ValueError(f"unexpected CoreML daemon status: {output!r}")
    return output, int(match[1])


def _read_start_output(stream: TextIO) -> str:
    stream.flush()
    stream.seek(0)
    return stream.read()


def _started_daemon_pid(output: str) -> int | None:
    match = re.search(r"(?:Kokoro daemon ready|Daemon started) \(PID (\d+)\)", output, re.IGNORECASE)
    return int(match[1]) if match else None


def _wait_for_daemon_start(
    process: subprocess.Popen[str], start_args: list[str], deadline: float
) -> tuple[int, KeyboardInterrupt | None, bool]:
    interrupted: KeyboardInterrupt | None = None
    timed_out = False
    while True:
        try:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(start_args, 12)
            returncode = process.wait(timeout=remaining)
            break
        except KeyboardInterrupt as error:
            if interrupted is None:
                interrupted = error
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            while True:
                try:
                    returncode = process.wait(timeout=3)
                    break
                except KeyboardInterrupt as error:
                    if interrupted is None:
                        interrupted = error
                except subprocess.TimeoutExpired:
                    process.kill()
                    while True:
                        try:
                            returncode = process.wait()
                            break
                        except KeyboardInterrupt as error:
                            if interrupted is None:
                                interrupted = error
                    break
            break
    return returncode, interrupted, timed_out


def _stop_owned_daemon(owned_pid: int) -> str:
    try:
        current_status, current_pid = daemon_status()
    except Exception as error:
        raise RuntimeError(
            f"could not verify owned daemon PID {owned_pid} during cleanup; left untouched"
        ) from error
    if current_pid is None:
        return current_status
    if current_pid != owned_pid:
        raise RuntimeError(
            f"daemon PID changed from owned PID {owned_pid} to {current_pid}; left untouched"
        )

    os.kill(owned_pid, signal.SIGTERM)
    deadline = time.perf_counter() + 12
    while True:
        try:
            current_status, current_pid = daemon_status()
        except Exception as error:
            raise RuntimeError(
                f"could not confirm shutdown of owned daemon PID {owned_pid} after SIGTERM"
            ) from error
        if current_pid is None:
            return current_status
        if current_pid != owned_pid:
            raise RuntimeError(
                f"daemon PID changed from owned PID {owned_pid} to {current_pid}; left untouched"
            )
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"owned CoreML daemon PID {owned_pid} remained running after SIGTERM")
        time.sleep(0.1)


def coreml_resident(
    report_dir: Path,
    specs: tuple[dict[str, str], ...],
    runs: int,
    provenance_sha: str,
) -> dict[str, object]:
    initial_output, initial_pid = daemon_status()
    if initial_pid is not None:
        raise RuntimeError(f"CoreML daemon already running (PID {initial_pid}); refusing to stop it")
    start_args = [str(COREML_CLI), "daemon", "start"]
    owned_pid: int | None = None
    start_stdout = ""
    start_stderr = ""
    start_seconds = 0.0
    start_process: subprocess.Popen[str] | None = None
    startup_deadline: float | None = None
    startup_interrupted: KeyboardInterrupt | None = None
    sampler: ProcessSampler | None = None
    requests: dict[str, list[dict[str, object]]] = {}
    warmups: list[dict[str, object]] = []
    stdout = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    try:
        started = time.perf_counter()
        startup_deadline = started + 12
        start_process = subprocess.Popen(
            start_args,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        # Defer Ctrl-C through the bounded readiness window because the detached
        # launcher may only print its service PID after startup completes.
        returncode, startup_interrupted, startup_timed_out = _wait_for_daemon_start(
            start_process, start_args, startup_deadline
        )
        start_seconds = time.perf_counter() - started
        start_stdout = _read_start_output(stdout)
        start_stderr = _read_start_output(stderr)
        owned_pid = _started_daemon_pid(f"{start_stdout}\n{start_stderr}")
        if owned_pid is None:
            try:
                _, reported_pid = daemon_status()
            except Exception as error:
                raise RuntimeError(
                    "daemon start produced no verifiable PID; ownership is unverified, "
                    "so any running daemon was left untouched; status probe failed"
                ) from error
            if reported_pid is not None:
                raise RuntimeError(
                    "daemon start produced no verifiable PID; "
                    f"status reports PID {reported_pid}, ownership is unverified, "
                    "so it was left untouched"
                )
            if startup_interrupted is not None:
                raise startup_interrupted
            if startup_timed_out:
                raise TimeoutError("daemon start did not finish within 12 seconds")
            if returncode:
                raise subprocess.CalledProcessError(
                    returncode, start_args, output=start_stdout, stderr=start_stderr
                )
            raise RuntimeError(f"daemon start did not report its PID: {start_stdout!r}")
        if startup_interrupted is not None:
            raise startup_interrupted
        if startup_timed_out:
            raise TimeoutError("daemon start did not finish within 12 seconds")
        if returncode:
            raise subprocess.CalledProcessError(
                returncode, start_args, output=start_stdout, stderr=start_stderr
            )
        status, reported_pid = daemon_status()
        if reported_pid != owned_pid:
            raise RuntimeError(f"started PID {owned_pid}, but daemon status reports {reported_pid}; left untouched")
        sampler = ProcessSampler(owned_pid)
        timebase = mach_timebase()
        lifetime_start = sampler.read()
        for spec in specs:
            # First request pays any initial CoreML compilation/loading and is kept out of repeats.
            warm_path = report_dir / "audio" / f"resident-coreml-{spec['name']}-warmup.wav"
            warm_path.parent.mkdir(parents=True, exist_ok=True)
            warm = timed(cli_command("coreml", spec["text"], "english", warm_path), sampler)
            if "[af_heart daemon]" not in str(warm["stdout"]):
                raise RuntimeError("CoreML CLI did not report daemon-backed synthesis")
            warmups.append({"workload": spec["name"], **warm, "audio": audio(warm_path)})
            requests[spec["name"]] = []
            for index in range(1, runs + 1):
                path = report_dir / "audio" / f"resident-coreml-{spec['name']}-{index:02}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                before = sampler.read()
                result = timed(cli_command("coreml", spec["text"], "english", path), sampler)
                after = sampler.read()
                if "[af_heart daemon]" not in str(result["stdout"]):
                    raise RuntimeError("CoreML CLI did not report daemon-backed synthesis")
                audio_record = audio(path)
                synth_match = re.search(r"\[af_heart daemon\]\s+(\d+)ms synth", str(result["stdout"]))
                if not synth_match:
                    raise RuntimeError("CoreML CLI output omitted daemon synthesis time")
                synth = int(synth_match[1]) / 1000.0
                cpu_ticks = (
                    after["cpu_user_mach_ticks"]
                    + after["cpu_system_mach_ticks"]
                    - before["cpu_user_mach_ticks"]
                    - before["cpu_system_mach_ticks"]
                )
                requests[spec["name"]].append(
                    {
                        "workload": spec["name"],
                        "mode": "english",
                        "implementation": "coreml",
                        "sample_index": index,
                        "input": spec["text"],
                        "voice": "af_heart",
                        "speed": 1.0,
                        "provenance_sha256": provenance_sha,
                        **result,
                        "audio": audio_record,
                        "cli_synthesis_seconds_rounded_to_ms": synth,
                        "audio_throughput_seconds_per_synthesis_second": audio_record["audio_seconds"] / synth,
                        "daemon_request_cpu_seconds": measured(
                            mach_ticks_to_seconds(cpu_ticks, timebase),
                            "seconds",
                            f"daemon PID {owned_pid} user+system CPU delta during request; client excluded; mach_timebase_info conversion applied",
                        ),
                        "daemon_request_cpu_mach_ticks": cpu_ticks,
                        "daemon_rss_samples_bytes": result["sampler_rss_bytes"],
                        "daemon_rss_sample_interval_seconds": sampler.interval,
                    }
                )
        lifetime_end = sampler.read()
    except KeyboardInterrupt as error:
        if startup_interrupted is None:
            startup_interrupted = error
        raise
    finally:
        try:
            if sampler:
                sampler.stop()
        finally:
            try:
                if startup_interrupted is not None and owned_pid is None:
                    if start_process is not None and start_process.poll() is None:
                        _wait_for_daemon_start(
                            start_process,
                            start_args,
                            startup_deadline if startup_deadline is not None else time.perf_counter(),
                        )
                    elif start_process is None:
                        # Popen itself can be interrupted after spawn but before its
                        # process handle is assigned; keep checking captured output
                        # through the same bounded startup window.
                        deadline = startup_deadline if startup_deadline is not None else time.perf_counter() + 12
                        while time.perf_counter() < deadline:
                            start_stdout = _read_start_output(stdout)
                            start_stderr = _read_start_output(stderr)
                            owned_pid = _started_daemon_pid(f"{start_stdout}\n{start_stderr}")
                            if owned_pid is not None:
                                break
                            time.sleep(0.1)
                    start_stdout = _read_start_output(stdout)
                    start_stderr = _read_start_output(stderr)
                    owned_pid = _started_daemon_pid(f"{start_stdout}\n{start_stderr}")
                    if owned_pid is None:
                        try:
                            _, reported_pid = daemon_status()
                        except Exception as error:
                            raise RuntimeError(
                                "startup was interrupted before ownership could be verified; "
                                "any running daemon was left untouched; status probe failed"
                            ) from error
                        if reported_pid is not None:
                            raise RuntimeError(
                                "startup was interrupted without verifiable PID output; "
                                f"status reports PID {reported_pid}, ownership is unverified, "
                                "so it was left untouched"
                            )
                        raise RuntimeError(
                            "startup was interrupted before any service PID could be verified; "
                            "no stop was attempted, and any possibly started daemon was left untouched"
                        ) from startup_interrupted
                if owned_pid is not None:
                    stopped_status = _stop_owned_daemon(owned_pid)
                else:
                    stopped_status = "Daemon ownership could not be verified; no stop attempted"
            finally:
                start_stdout = _read_start_output(stdout)
                start_stderr = _read_start_output(stderr)
                stdout.close()
                stderr.close()
    return {
        "initial_status": initial_output,
        "start_command": start_args,
        "start_stdout": start_stdout,
        "start_stderr": start_stderr,
        "start_seconds": start_seconds,
        "owned_pid": owned_pid,
        "running_status": status,
        "warmups": warmups,
        "requests": requests,
        "daemon_lifetime_cpu_seconds": measured(
            mach_ticks_to_seconds(
                lifetime_end["cpu_user_mach_ticks"]
                + lifetime_end["cpu_system_mach_ticks"]
                - lifetime_start["cpu_user_mach_ticks"]
                - lifetime_start["cpu_system_mach_ticks"],
                timebase,
            ),
            "seconds",
            f"daemon PID {owned_pid} user+system CPU lifetime delta from ready through last measured request; includes warmups, excludes CLI clients; mach_timebase_info conversion applied",
        ),
        "daemon_interval_sampled_rss": {
            "sample_count": len(sampler.rss),
            "interval_seconds": sampler.interval,
            "peak_rss_bytes": max(sampler.rss),
            "scope": f"CoreML daemon PID {owned_pid}; boundary reads plus interval samples during warmups and requests, gaps unsampled",
        },
        "daemon_lifetime_peak_phys_footprint_bytes": measured(
            lifetime_end["lifetime_peak_phys_footprint_bytes"],
            "bytes",
            f"daemon PID {owned_pid}; OS rusage lifetime peak physical footprint, not RSS",
        ),
        "daemon_os_peak_rss": unsupported(
            "OS-reported peak RSS",
            f"daemon PID {owned_pid}",
            "proc_pid_rusage exposes current RSS and peak physical footprint, not peak RSS",
        ),
        "cleanup_status": stopped_status,
        "cpu_timebase": {
            **timebase,
            "source": "mach_timebase_info",
            "formula": "Mach ticks × numerator ÷ denominator = nanoseconds; divide by 1e9 for seconds",
            "cpu_scope": "CoreML daemon PID only; user+system process CPU, excludes CLI clients and GPU/ANE execution",
        },
        "daemon_lifetime_cpu_mach_ticks": (
            lifetime_end["cpu_user_mach_ticks"]
            + lifetime_end["cpu_system_mach_ticks"]
            - lifetime_start["cpu_user_mach_ticks"]
            - lifetime_start["cpu_system_mach_ticks"]
        ),
    }


def validate_samples(samples: list[dict[str, object]], workload: str, mode: str) -> None:
    if not samples:
        raise ValueError("comparison requires samples from both implementations")
    for sample in samples:
        if sample.get("workload") != workload or sample.get("mode") != mode:
            raise ValueError("sample workload or mode does not match comparison")
        if not sample.get("provenance_sha256"):
            raise ValueError("sample provenance is missing")


def compare_samples(
    native: list[dict[str, object]],
    coreml: list[dict[str, object]],
    workload: str,
    mode: str,
    key: str,
) -> dict[str, object]:
    validate_samples(native, workload, mode)
    validate_samples(coreml, workload, mode)
    if any(key not in sample for sample in (*native, *coreml)):
        raise ValueError(f"comparison samples must contain {key}")
    for key_name in ("input", "voice", "speed", "provenance_sha256"):
        native_values = {sample.get(key_name) for sample in native}
        coreml_values = {sample.get(key_name) for sample in coreml}
        if native_values != coreml_values or len(native_values) != 1:
            raise ValueError(f"matched samples have different {key_name}")
    n = native_run.summarize([float(sample[key]) for sample in native])
    c = native_run.summarize([float(sample[key]) for sample in coreml])
    if n["max_seconds"] < c["min_seconds"]:
        outcome = "native_faster"
    elif c["max_seconds"] < n["min_seconds"]:
        outcome = "coreml_faster"
    else:
        outcome = "inconclusive_overlapping_observed_ranges"
    return {
        "workload": workload,
        "mode": mode,
        "metric": key,
        "native": n,
        "coreml": c,
        "outcome": outcome,
    }


def comparisons(workloads: list[dict[str, object]]) -> dict[str, object]:
    outcomes = []
    for workload in workloads:
        for mode, pair in workload["fresh"].items():
            outcomes.append(
                compare_samples(
                    pair["native"], pair["coreml"], workload["name"], mode, "wall_seconds"
                )
            )
        if "resident" in workload:
            resident = workload["resident"]
            outcomes.append(
                compare_samples(
                    resident["native_samples"],
                    resident["coreml_samples"],
                    workload["name"],
                    "english",
                    "waveform_seconds",
                )
            )
    winners = {item["outcome"] for item in outcomes}
    global_winner = next(iter(winners)) if len(winners) == 1 else None
    if global_winner == "inconclusive_overlapping_observed_ranges":
        global_winner = None
    # The aggregate is intentionally absent when individual workload/mode results differ.
    return {"workloads": outcomes, "global_winner": global_winner}


def finalize_report(report: dict[str, object], output_dir: Path) -> dict[str, object]:
    """Checkpoint all raw events before deriving comparison outcomes."""
    raw_path = output_dir / "raw.json"
    report["comparisons"] = {"status": "normalization_pending"}
    raw_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    report["comparisons"] = comparisons(report["workloads"])
    raw_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def values_text(values: list[float], scale: float = 1.0, unit: str = "s") -> str:
    if not values:
        return "unsupported"
    result = native_run.summarize(values)
    return (
        f"{result['median_seconds'] / scale:.3f} "
        f"[{result['min_seconds'] / scale:.3f}, {result['max_seconds'] / scale:.3f}] "
        f"{unit}, n={result['sample_count']}"
    )


def cpu_text(samples: list[dict[str, object]]) -> str:
    metrics = [sample["process_resources"].get("cpu_total_seconds") for sample in samples]
    values = [float(metric["value"]) for metric in metrics if metric and metric["status"] == "measured"]
    below = [metric for metric in metrics if metric and metric["status"] == "below_resolution"]
    if not values:
        if below:
            bound = max(float(metric["upper_bound_seconds"]) for metric in below)
            return f"<{bound:.3f} s (below time -l resolution), n={len(below)}"
        return "unsupported"
    rendered = values_text(values)
    if below:
        rendered += f"; {len(below)} below resolution"
    return rendered


def process_metric(sample: dict[str, object], name: str) -> float | None:
    metric = sample["process_resources"].get(name)
    return float(metric["value"]) if metric and metric["status"] == "measured" else None


def compare_throughput(native_values: list[float], coreml_values: list[float]) -> dict[str, object]:
    native_summary = native_run.summarize(native_values)
    coreml_summary = native_run.summarize(coreml_values)
    if native_summary["min_seconds"] > coreml_summary["max_seconds"]:
        outcome = "native_higher_audio_throughput"
    elif coreml_summary["min_seconds"] > native_summary["max_seconds"]:
        outcome = "coreml_higher_audio_throughput"
    else:
        outcome = "inconclusive_overlapping_observed_ranges"
    return {"native": native_summary, "coreml": coreml_summary, "outcome": outcome}


def resident_throughput(
    native_events: list[dict[str, object]], coreml_requests: list[dict[str, object]]
) -> tuple[list[float], list[float]]:
    native = [
        float(event["audio_seconds"]) / float(event["request_to_waveform_seconds"])
        for event in native_events
    ]
    coreml = [
        float(request["audio"]["audio_seconds"])
        / float(request["cli_synthesis_seconds_rounded_to_ms"])
        for request in coreml_requests
    ]
    return native, coreml


def markdown(report: dict[str, object]) -> str:
    p, ex = report["provenance"], report["execution"]
    coreml_model = p["coreml"]["model"]
    processing = report.get("post_capture_processing")
    voice_count = coreml_model.get("voice_asset_count")
    if voice_count is None and processing:
        voice_count = processing.get("coreml_voice_asset_count")
    voice_count_text = (
        f"{voice_count} separate voice assets" if voice_count is not None else "voice assets"
    )
    resource_rows = []
    throughput_rows = []
    daemon = report.get("coreml_daemon", {})

    for workload in report["workloads"]:
        name = workload["name"]
        for mode, pair in workload["fresh"].items():
            for tool, samples in pair.items():
                rss = [process_metric(sample, "os_peak_rss_bytes") for sample in samples]
                resource_rows.append(
                    (
                        name, "Fresh CLI", mode, tool,
                        values_text([sample["audio"]["audio_seconds"] for sample in samples]),
                        values_text([sample["audio_throughput_seconds_per_wall_second"] for sample in samples], unit="×"),
                        f"{cpu_text(samples)}; direct process",
                        f"{values_text([value for value in rss if value is not None], 1024**3, 'GiB')}; OS peak RSS",
                    )
                )
            throughput_rows.append(
                (
                    name, "Fresh CLI", mode,
                    compare_throughput(
                        [sample["audio_throughput_seconds_per_wall_second"] for sample in pair["native"]],
                        [sample["audio_throughput_seconds_per_wall_second"] for sample in pair["coreml"]],
                    ),
                )
            )

        native_sessions = workload.get("native_resident_sessions", [])
        native_events = [event for session in native_sessions for event in session["warm_samples"]]
        if native_events:
            rss = [process_metric(session, "os_peak_rss_bytes") for session in native_sessions]
            coreml_requests = daemon.get("requests", {}).get(name, [])
            rates, server_rates = resident_throughput(native_events, coreml_requests)
            resource_rows.append(
                (
                    name, "Resident", "english", "native",
                    values_text([event["audio_seconds"] for event in native_events]),
                    values_text(rates, unit="×"),
                    f"{cpu_text(native_sessions)}; driver lifetime incl load and warm-up",
                    f"{values_text([value for value in rss if value is not None], 1024**3, 'GiB')}; driver OS peak RSS",
                )
            )
            if coreml_requests:
                server_cpu = [sample["daemon_request_cpu_seconds"]["value"] for sample in coreml_requests]
                client_rss = [process_metric(sample, "os_peak_rss_bytes") for sample in coreml_requests]
                server_rss = [max(sample["daemon_rss_samples_bytes"]) for sample in coreml_requests]
                resource_rows.append(
                    (
                        name, "Resident", "english", "coreml",
                        values_text([sample["audio"]["audio_seconds"] for sample in coreml_requests]),
                        values_text(server_rates, unit="×"),
                        f"client {cpu_text(coreml_requests)}; daemon {values_text(server_cpu)}",
                        f"client peak {values_text([value for value in client_rss if value is not None], 1024**3, 'GiB')}; daemon sampled/request {values_text(server_rss, 1024**3, 'GiB')}",
                    )
                )
                throughput_rows.append(
                    (
                        name, "Resident", "english",
                        compare_throughput(rates, server_rates),
                    )
                )

    model_voice_bytes = next(
        asset["size_bytes"] for asset in coreml_model["assets"]
        if asset["path"] == "voices/af_heart.bin"
    )
    lines = [
        "# Native and CoreML Kokoro comparison",
        "",
        f"Captured: {report['captured_at_utc']}",
        f"Host: {p['host']['chip']}, {p['host']['memory_bytes'] // 2**30} GiB, macOS {p['host']['os']['version']} ({p['host']['os']['build']}).",
        f"Installed CoreML CLI `{p['coreml']['version']}` at `{p['coreml']['wrapper']['path']}`; executable SHA-256 `{p['coreml']['executable']['sha256']}`; Homebrew formula SHA-256 `{p['coreml']['homebrew']['formula_sha256']}`. Source commit `{p['coreml']['source']['commit']}`. {p['coreml']['source']['version_note']}",
        f"Native commit `{p['native']['commit']}`; CLI SHA-256 `{p['native']['cli_sha256']}`; driver SHA-256 `{p['native']['driver_sha256']}`; build inputs SHA-256 `{p['native']['source_sha256']}`; model pack SHA-256 `{p['native']['model_pack_sha256']}`. Native pack is {p['native']['model_pack']['pack_file_bytes']:,} bytes ({p['native']['model_pack']['float32_weight_bytes']:,} float32 weight bytes) and includes the tested `af_heart` style. Build flags/compiler: `{p['native']['build']}`.",
        f"CoreML model tag `{coreml_model['tag']}` contains {coreml_model['file_count']} files ({coreml_model['total_bytes']:,} bytes), with {voice_count_text}; `af_heart` voice asset is {model_voice_bytes:,} bytes.",
        "",
        f"Both tools use `af_heart`, speed `1.0`, explicit output WAV paths, and no playback. English text uses native {p['native']['espeak_ng']} versus CoreML's bundled BART G2P. Both outputs are mono PCM16 at 24 kHz; native writes WAV through its C writer and CoreML through AVAudioFile. The same prepared IPA characters are sent to both direct CLI paths, but each uses a separate tokenizer/model pipeline and does not guarantee identical model token IDs.",
        f"For the shared IPA string `{SHORT_IPA}`, the measured audio duration differed (see table). CoreML source applies adaptive BOS/EOS trims and output processing after tokenization; native has its own chunk/tokenizer path without that same trim step. These differences can affect duration, but this run did not isolate the cause or quantify each contribution. English and IPA results are implementation comparisons, not equal-duration or audio-quality claims.",
        "Fresh process means a new CLI process; it does not mean cold model or hardware caches. No caches were purged.",
        "",
        "## Latency outcomes",
        "",
        "A faster outcome is reported only when observed ranges are disjoint; each side shows its own sample count.",
        "",
        "| Workload | Path | Mode | Native median [min, max], n | CoreML median [min, max], n | Outcome |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["comparisons"]["workloads"]:
        show = lambda item: f"{item['median_seconds']:.3f} [{item['min_seconds']:.3f}, {item['max_seconds']:.3f}] s, n={item['sample_count']}"
        path = "Fresh CLI completion" if row["metric"] == "wall_seconds" else "Resident waveform"
        lines.append(f"| {row['workload']} | {path} | {row['mode']} | {show(row['native'])} | {show(row['coreml'])} | {row['outcome']} |")
    lines.extend(
        [
            "",
            f"Resident latency compares native driver's complete `request_to_waveform_seconds` with CoreML's daemon CLI-reported synthesis time rounded to milliseconds; both exclude WAV writing. Native has {ex['resident_sessions_per_workload'] * ex['native_warm_repetitions_per_session']} measured warm requests per workload ({ex['resident_sessions_per_workload']} sessions × {ex['native_warm_repetitions_per_session']}); CoreML has {ex['resident_sessions_per_workload']} resident requests after one unmeasured setup request per workload.",
            "",
            "## Audio output and process resources",
            "",
            "Throughput is generated audio seconds per wall second for Fresh CLI and per request-to-waveform second for Resident. Native resident time includes its frontend and inference; CoreML's pinned v0.11.2 `synthesisTime` starts before text chunk preparation and ends after waveform post-processing, so it includes its text frontend and inference. Both exclude WAV writing. CPU is user plus system time. `/usr/bin/time -l` prints CPU to 0.01 s; a displayed 0.00 is recorded as below resolution, not as zero. CoreML resident client and service measurements are reported separately.",
            "",
            "| Workload | Path | Mode | Tool | Audio seconds | Audio throughput | CPU seconds and scope | RSS and scope |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in resource_rows)
    lines.extend(
        [
            "",
            "### Audio-throughput outcomes",
            "",
            "Throughput uses output duration as the numerator; higher is better. This is separate from the latency table.",
            "",
            "| Workload | Path | Mode | Native median [min, max], n | CoreML median [min, max], n | Outcome |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name, path, mode, result in throughput_rows:
        show = lambda item: f"{item['median_seconds']:.3f} [{item['min_seconds']:.3f}, {item['max_seconds']:.3f}] ×, n={item['sample_count']}"
        lines.append(f"| {name} | {path} | {mode} | {show(result['native'])} | {show(result['coreml'])} | {result['outcome']} |")

    if "daemon_lifetime_cpu_seconds" in daemon:
        lifetime_cpu = daemon["daemon_lifetime_cpu_seconds"]["value"]
        sampled_rss = daemon["daemon_interval_sampled_rss"]
        footprint = daemon["daemon_lifetime_peak_phys_footprint_bytes"]["value"]
        lines.extend(
            [
                "",
                f"CoreML daemon lifetime CPU from ready through the last measured request was {lifetime_cpu:.3f} s (includes two setup requests and excludes CLI clients). Its interval-sampled RSS peak was {sampled_rss['peak_rss_bytes'] / 1024**3:.3f} GiB from {sampled_rss['sample_count']} readings at {sampled_rss['interval_seconds']:.2f} s during setup/request windows; gaps were unsampled. OS lifetime peak physical footprint was {footprint / 1024**3:.3f} GiB; it is not RSS.",
            ]
        )
    lines.extend(
        [
            "",
            "## Measurement limits",
            "",
            "`/usr/bin/time -l` CPU and peak RSS apply to the directly timed process. Native resident process totals include model load, first request, warm-up, repeated requests, and WAV writing. CoreML resident client resources are distinct from daemon per-request CPU deltas and interval-sampled RSS. OS-reported peak RSS for the daemon is unavailable; exact GPU/ANE attribution and memory accounting are also unavailable. CoreML backend requests `.all` compute units and its compiled metadata reports mixed precision/storage; native uses float32 weights through Accelerate. No listening judgment or audio-quality winner is claimed.",
            "CoreML daemon CPU is the daemon PID's user+system process CPU, excluding the CLI client. `proc_pid_rusage`'s rusage v4 CPU counters are Mach time units; they are converted with the recorded host `mach_timebase_info` numerator/denominator. This process CPU does not attribute GPU/ANE execution.",
            "",
            "CoreML direct IPA is supported, but resident IPA is unsupported because the daemon request contains text, voice, and speed; `--ipa` bypasses the daemon.",
            "",
            f"Global winner: {report['comparisons']['global_winner'] or 'not reported; latency results differ by workload/path and throughput is a separate measure'}.",
            "",
            "## Rerun",
            "",
            "```sh",
            ex["rerun_command"],
            "```",
            "",
            "Commands, full stdout/stderr, per-sample provenance, WAV hashes/formats, and raw timings are in `raw.json`. Smoke/setup notes and the exact actual benchmark command are also retained there.",
            "",
            "Presentation regenerated from the saved raw samples after capture; no benchmark was rerun for this report update.",
        ]
    )
    if processing:
        timebase = processing["host_timebase"]
        lines.extend(
            [
                "",
                f"Post-capture resource correction: the original raw capture is preserved as `{processing['source_raw_path']}` (SHA-256 `{processing['source_raw_sha256']}`). The corrected `{processing['corrected_raw_path']}` multiplies the two CoreML service CPU metrics by {timebase['numerator']}/{timebase['denominator']} from `mach_timebase_info`, reparses `/usr/bin/time -l` CPU fields from saved stderr so displayed 0.00 values are below resolution, and renames sampled `cpu_user_ns`/`cpu_system_ns` fields to `cpu_user_mach_ticks`/`cpu_system_mach_ticks` without changing their tick values. Capture runner SHA-256: `{processing['capture_runner_sha256']}`; report-generation runner SHA-256: `{processing['normalization_runner_sha256']}`. Timing samples were unchanged and no inference was rerun.",
            ]
        )
    return "\n".join(lines) + "\n"


def run_benchmark(output_dir: Path, fresh_runs: int, resident_runs: int, warm_repetitions: int) -> dict[str, object]:
    output_dir = validate_output_dir(output_dir)
    if sys.platform != "darwin":
        raise RuntimeError("CoreML comparison requires macOS")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    p = provenance()
    prov_sha = str(p["sha256"])
    workloads = []
    for spec in WORKLOADS:
        fresh = {}
        modes = [("english", spec["text"])]
        if spec["name"] == "hello_from_terminal":
            modes.append(("ipa", SHORT_IPA))
        for mode, text in modes:
            pair = {implementation: [] for implementation in ("native", "coreml")}
            for index in range(1, fresh_runs + 1):
                order = ("native", "coreml") if index % 2 else ("coreml", "native")
                for implementation in order:
                    pair[implementation].append(
                        fresh_sample(output_dir, spec, mode, text, implementation, index, prov_sha)
                    )
            fresh[mode] = pair
        native_sessions = [
            native_resident(output_dir, spec, i, warm_repetitions, prov_sha)
            for i in range(1, resident_runs + 1)
        ]
        workloads.append(
            {
                **spec,
                "voice": "af_heart",
                "speed": 1.0,
                "fresh": fresh,
                "native_resident_sessions": native_sessions,
            }
        )
    daemon = coreml_resident(output_dir, tuple(WORKLOADS), resident_runs, prov_sha)
    for workload in workloads:
        native_samples = [
            {
                "workload": workload["name"],
                "mode": "english",
                "input": workload["text"],
                "voice": "af_heart",
                "speed": 1.0,
                "provenance_sha256": session["provenance_sha256"],
                "session_index": session["session_index"],
                "waveform_seconds": event["request_to_waveform_seconds"],
            }
            for session in workload["native_resident_sessions"]
            for event in session["warm_samples"]
        ]
        coreml_samples = [
            {
                "workload": workload["name"],
                "mode": "english",
                "input": workload["text"],
                "voice": "af_heart",
                "speed": 1.0,
                "provenance_sha256": request["provenance_sha256"],
                "sample_index": request["sample_index"],
                "waveform_seconds": request["cli_synthesis_seconds_rounded_to_ms"],
            }
            for request in daemon["requests"][workload["name"]]
        ]
        workload["resident"] = {
            "native_samples": native_samples,
            "coreml_samples": coreml_samples,
            "resident_ipa": unsupported(
                "resident IPA inference",
                "CoreML daemon request protocol",
                "daemon requests contain text, voice, and speed; CLI --ipa bypasses the daemon",
            ),
        }
    rerun = (
        "UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --offline "
        "--no-sync --locked python benchmarks/compare.py"
    )
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": p,
        "execution": {
            "fresh_runs_per_workload_mode": fresh_runs,
            "resident_sessions_per_workload": resident_runs,
            "native_warm_repetitions_per_session": warm_repetitions,
            "voice": "af_heart",
            "speed": 1.0,
            "sample_rate_hz": 24000,
            "playback": "disabled; every sample specifies --output",
            "cache_policy": "fresh process only; OS and model caches uncontrolled and not purged",
            "rerun_command": rerun,
            "actual_benchmark_command": (
                f"{rerun} --fresh-runs {fresh_runs} --resident-runs {resident_runs} "
                f"--warm-repetitions {warm_repetitions} --output-dir {output_dir.relative_to(ROOT)}"
            ),
        },
        "setup_notes": [
            {
                "stage": "sandboxed CoreML smoke",
                "result": "failed; excluded from samples and resource comparisons",
                "stderr": "IOSurfaceSharedEventAddEventListener failed: 10000003; CoreML failed to build the model execution plan for kokoro_backend.mlmodelc/model.mil with error code -6.",
                "retry": "same fixed smoke was rerun with macOS CoreML service access and succeeded",
            },
            {
                "stage": "authorized CoreML-service smoke",
                "result": "passed with installed 0.11.2 and unchanged model assets",
                "command": [str(COREML_CLI), "say", "-o", "/private/tmp/kokoro-15-smoke/coreml.wav", SHORT_TEXT],
                "audio_sha256": native_run.sha256_file(Path("/private/tmp/kokoro-15-smoke/coreml.wav")),
                "stdout": "[af_heart] 4247ms synth, 1.5s audio, 0.3x RT",
            },
            {
                "stage": "prepared IPA smoke",
                "input": SHORT_IPA,
                "result": "both direct CLI paths generated mono PCM16 24 kHz WAV",
                "resident": "unsupported for CoreML daemon because --ipa uses the direct engine route",
            },
        ],
        "workloads": workloads,
        "coreml_daemon": daemon,
        "limitations": {
            "gpu_ane": unsupported("GPU/ANE attribution and memory accounting", "CoreML backend", "available process counters do not expose per-device usage"),
            "listening": "not performed; no audio-quality judgment",
        },
    }
    return finalize_report(report, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-runs", type=int, default=5)
    parser.add_argument("--resident-runs", type=int, default=5)
    parser.add_argument("--warm-repetitions", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.fresh_runs, args.resident_runs, args.warm_repetitions) < 1:
        parser.error("run counts must be positive")
    output = args.output_dir or ROOT / "artifacts" / f"15-coreml-comparison-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if not output.is_absolute():
        output = ROOT / output
    rerun = (
        "UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --offline "
        "--no-sync --locked python benchmarks/compare.py"
    )
    report = run_benchmark(output, args.fresh_runs, args.resident_runs, args.warm_repetitions)
    report["execution"]["rerun_command"] = (
        f"{rerun} --fresh-runs {args.fresh_runs} --resident-runs {args.resident_runs} "
        f"--warm-repetitions {args.warm_repetitions}"
    )
    report["execution"]["actual_benchmark_command"] = (
        f"{report['execution']['rerun_command']} --output-dir {output.relative_to(ROOT)}"
    )
    (output / "raw.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    print(f"Raw samples and provenance: {output.relative_to(ROOT)}/raw.json")
    print(f"Human comparison: {output.relative_to(ROOT)}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
