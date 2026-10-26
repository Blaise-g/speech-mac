# Reference assets: licenses and provenance

- `gradium/phonon-40m-en` at `cc738b7e4d182c2ba1bafb8062aa4846af000637`: CC BY-NC 4.0, as identified by the pinned Hugging Face repository metadata. This checkpoint is gated; users must accept the model terms before downloading it. The default voice is `default-voice.safetensors` from that same revision. No external named-voice catalogue entry is included in this ticket's fixture.
- `gradium/phonon-mimi` at `33519be1a8e61c6fa00c6e154e2d38e76a367919`: Apache-2.0 according to the pinned repository metadata. Its 48 transformed tensors are checked bitwise against the `mimi.*` tensors in the Phonon model bundle through the pinned Transformers converter mapping.
- `gradium-ai/xn-ptts` at `ac0e1524cfd9e71d57aadca0de542796f942d775`: Apache-2.0 OR MIT. Used as pinned source provenance for runtime preprocessing and the PR209 discrepancy; no Rust runtime is linked or shipped.
- Hugging Face Transformers at `50e35cc0ca18ec19b80653690de553e6decf74f5`: Apache-2.0. Used only by the locked development/reference project.
- PR209 (`https://github.com/gradium-ai/xn-ptts/pull/209`) records that upstream Rust's `delta <= context` retained `context + 1` key positions while trained Python used `delta < context`; the Phonon reference sets Mimi's local window to 250 and verifies the actual HF mask at the boundary.

All model/config/tokenizer/voice/processor/Mimi file sizes and SHA-256 hashes are committed in `pins.json`, derived from revision-specific Hugging Face LFS/API provenance. `verify_assets.py` checks the shared read-only asset files against those committed pins before any model load.
