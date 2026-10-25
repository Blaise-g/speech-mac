"""Compare native closed-loop traces with the pinned HF head on identical inputs."""
from pathlib import Path
import sys
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle

TRACE = ROOT / ".build" / "phonon-flow-generated-trace-v1.bin"
FIXTURE = HERE / "fixtures" / "phonon-hello-q8-natural.npz"
BUDGETS = json.loads((HERE / "budgets.json").read_text())["native_parity_gate"]
STAGE_BUDGETS = BUDGETS["native_q8_stage_targets"]
PCM_BUDGETS = BUDGETS["native_pcm_targets"]


def within(actual: np.ndarray, expected: np.ndarray, budget: dict) -> tuple[bool, float, int]:
    error = np.abs(actual - expected)
    limit = budget["atol"] + budget["rtol"] * np.abs(expected)
    return bool(np.all(error <= limit)), float(error.max()), int(np.count_nonzero(error > limit))


def read_trace() -> tuple[np.ndarray, ...]:
    raw = TRACE.read_bytes()
    if len(raw) < 12 or raw[:8] != b"PHGEN1\0\0":
        raise ValueError("native generation trace has an invalid header")
    frames = int.from_bytes(raw[8:12], "little")
    if frames != 32:
        raise ValueError(f"expected 32 natural frames, got {frames}")
    values = np.frombuffer(raw, dtype="<f4", offset=12)
    record = 512 + 32 + 32 + 1 + 32
    if values.size != frames * record:
        raise ValueError("native generation trace has an invalid byte count")
    rows = values.reshape(frames, record)
    return rows[:, :512], rows[:, 512:544], rows[:, 544:576], rows[:, 576], rows[:, 577:]


def main() -> None:
    conditions, noise, native_normalized, native_eos, native_latents = read_trace()
    with np.load(FIXTURE) as frozen:
        frozen_noise = frozen["noise_draws"][:, 0, :]
        frozen_conditions = frozen["flow_condition"][:, 0, :]
        frozen_integration = frozen["flow_integration"].reshape(-1, 32)
        frozen_latents = frozen["latents"][0]
        frozen_eos = frozen["eos_logits"].reshape(-1)
        frozen_raw_pcm = frozen["raw_pcm"][0]
        frozen_final_pcm = frozen["final_pcm"][0]
        frozen_raw_count = int(frozen["raw_pcm_sample_count"][0])
        frozen_final_count = int(frozen["final_pcm_sample_count"][0])
    _, model, codec, processor = oracle.load("q8")
    model.eval()
    torch.set_num_threads(1)
    with torch.inference_mode():
        torch_condition = torch.from_numpy(conditions.copy())
        torch_noise = torch.from_numpy(noise.copy()) * np.sqrt(0.8)
        times_start = torch.zeros((32, 1), dtype=torch.float32)
        times_end = torch.ones((32, 1), dtype=torch.float32)
        hf_velocity = model.flow_head(torch_noise, torch_condition, times=[times_start, times_end])
        hf_normalized = (torch_noise + hf_velocity).cpu().numpy()
        hf_eos = model.eos_head(torch_condition).reshape(-1).cpu().numpy()
    flow_budget = STAGE_BUDGETS["flow_integration"]
    eos_budget = STAGE_BUDGETS["eos_logits"]
    head_ok, head_max, head_bad = within(native_normalized, hf_normalized, flow_budget)
    eos_same_ok, eos_same_max, eos_same_bad = within(native_eos, hf_eos, eos_budget)
    condition_ok, condition_max, condition_bad = within(
        conditions, frozen_conditions, STAGE_BUDGETS["flow_condition"]
    )
    integration_ok, integration_max, integration_bad = within(
        native_normalized, frozen_integration, flow_budget
    )
    eos_ok, eos_max, eos_bad = within(native_eos, frozen_eos, eos_budget)
    latent_ok, latent_max, latent_bad = within(
        native_latents, frozen_latents, STAGE_BUDGETS["latents"]
    )
    finite_ok = all(
        np.isfinite(value).all()
        for value in (conditions, noise, native_normalized, native_eos, native_latents)
    )
    noise_exact = np.array_equal(noise, frozen_noise)
    latent_tensor = torch.from_numpy(native_latents.copy()).unsqueeze(0)
    latent_mask = torch.ones((1, 32), dtype=torch.long)
    with torch.inference_mode():
        raw_pcm = codec.decode(latent_tensor, return_dict=True).audio_values[:, 0].float().cpu().numpy()[0]
        final_pcm = processor.decode(latent_tensor, latent_mask)[0].float().cpu().numpy()
    if len(raw_pcm) != frozen_raw_count or len(final_pcm) != frozen_final_count:
        raise SystemExit(
            f"native latent PCM counts differ: raw={len(raw_pcm)}/{frozen_raw_count}, "
            f"final={len(final_pcm)}/{frozen_final_count}"
        )
    raw_error = raw_pcm - frozen_raw_pcm[:frozen_raw_count]
    final_error = final_pcm - frozen_final_pcm[:frozen_final_count]
    raw_max = float(np.max(np.abs(raw_error)))
    final_max = float(np.max(np.abs(final_error)))
    raw_nrmse = float(np.sqrt(np.mean(raw_error**2)) / np.sqrt(np.mean(frozen_raw_pcm[:frozen_raw_count]**2)))
    final_nrmse = float(np.sqrt(np.mean(final_error**2)) / np.sqrt(np.mean(frozen_final_pcm[:frozen_final_count]**2)))
    pcm_finite = bool(np.isfinite(raw_pcm).all() and np.isfinite(final_pcm).all())
    print(
        f"saved Gaussian replay exact={noise_exact}; finite native/model PCM="
        f"{finite_ok and pcm_finite}"
    )
    print(
        "native head on identical dynamic inputs: "
        f"max_abs={head_max:.9g}, over_budget={head_bad}/1024; "
        f"EOS max_abs={eos_same_max:.9g}, over_budget={eos_same_bad}/32"
    )
    print(
        "autonomous reference trajectory: "
        f"condition max_abs={condition_max:.9g}, over_budget={condition_bad}/16384; "
        f"normalized integration max_abs={integration_max:.9g}, over_budget={integration_bad}/1024; "
        f"EOS max_abs={eos_max:.9g}, over_budget={eos_bad}/32; "
        f"denormalized latent max_abs={latent_max:.9g}, over_budget={latent_bad}/1024"
    )
    print(
        f"decoded PCM counts raw/final={len(raw_pcm)}/{len(final_pcm)} exact; "
        f"raw nrmse/max_abs={raw_nrmse:.9g}/{raw_max:.9g}; "
        f"final nrmse/max_abs={final_nrmse:.9g}/{final_max:.9g}"
    )
    stage_failures = []
    if not noise_exact:
        stage_failures.append("exact saved-noise replay")
    component_checks = (
        (head_ok, "same-input flow integration"),
        (eos_same_ok, "same-input EOS projection"),
        (condition_ok, "flow condition"),
        (integration_ok, "normalized flow integration"),
        (eos_ok, "EOS trajectory"),
        (latent_ok, "denormalized latent"),
    )
    stage_failures.extend(name for ok, name in component_checks if not ok)
    if (raw_nrmse > PCM_BUDGETS["normalized_rmse_max"] or
            final_nrmse > PCM_BUDGETS["normalized_rmse_max"] or
            raw_max > PCM_BUDGETS["max_abs"] or final_max > PCM_BUDGETS["max_abs"]):
        stage_failures.append("Mimi PCM")
    if not finite_ok or not pcm_finite:
        stage_failures.append("non-finite native trace")
    if stage_failures:
        raise SystemExit("native closed-loop stages exceed frozen budgets: " + ", ".join(stage_failures))


if __name__ == "__main__":
    main()
