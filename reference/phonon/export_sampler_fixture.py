"""Export fixed and natural Q8 flow-head replay arrays for native C checks."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).parent
out = Path(".build/phonon-flow-sampler-v1.bin")
out.parent.mkdir(parents=True, exist_ok=True)
records = []
for name in ("phonon-hello-q8.npz", "phonon-hello-q8-natural.npz"):
    with np.load(root / "fixtures" / name) as data:
        conditions = data["flow_condition"][:, 0, :].astype("<f4")
        noise = data["noise_draws"][:, 0, :].astype("<f4")
        integration = data["flow_integration"].astype("<f4")
        latents = data["latents"][0].astype("<f4")
        eos = data["eos_logits"].reshape(-1).astype("<f4")
    if not (conditions.shape == (len(noise), 512) and noise.shape == integration.shape == latents.shape):
        raise ValueError(f"unexpected sampler fixture dimensions in {name}")
    records.append((conditions, noise, integration, latents, eos))
with out.open("wb") as stream:
    stream.write(b"PHSAMP1\0")
    stream.write(np.asarray([len(records)], dtype="<u4").tobytes())
    for conditions, noise, integration, latents, eos in records:
        stream.write(np.asarray([len(noise)], dtype="<u4").tobytes())
        for frame in range(len(noise)):
            for arr in (conditions[frame], noise[frame], integration[frame], latents[frame], eos[frame]):
                stream.write(np.asarray(arr, dtype="<f4").tobytes())
print(json.dumps({"fixture": str(out), "runs": [len(r[1]) for r in records]}))
