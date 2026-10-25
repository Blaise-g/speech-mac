import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import json
import hashlib
import numpy as np
import torch
from gguf import GGUFReader
from safetensors import safe_open
import os
import tempfile
import quantization
from context import allowed_positions
from oracle import verify, verify_q8, mimi_batch_probe, mimi_temporal_probe, load
from verify_assets import verify as verify_assets

class ReferenceContractTests(unittest.TestCase):
    def test_pinned_tensor_types_match_quantization_manifest(self):
        root=Path(__file__).parents[1]
        summary=json.loads((root/'tensor-types.json').read_text())
        assets=Path(os.environ.get('PHONON_MODEL_DIR','/Volumes/Code/Personal/speech-mac-models/phonon-40m-en'))
        with safe_open(str(assets/'model.safetensors'),framework='pt') as source:
            self.assertEqual({k:sum(source.get_slice(name).get_dtype()==k for name in source.keys()) for k in ('BF16','F32')},summary['source_model_safetensors'])
        counts={}
        for tensor in GGUFReader(str(assets/'model.q8.gguf')).tensors: counts[tensor.tensor_type.name]=counts.get(tensor.tensor_type.name,0)+1
        self.assertEqual(counts,summary['published_gguf'])
        self.assertEqual(counts,{'BF16':173,'Q8_0':36})

    def test_q8_derived_cache_recovers_from_corruption(self):
        original=quantization.OUT
        with tempfile.TemporaryDirectory(prefix='phonon-q8-cache-test-') as temp:
            quantization.OUT=Path(temp)/'cache'
            try:
                out=quantization.prepare_q8_checkpoint()
                checkpoint=out/'model.safetensors'; marker=json.loads((out/'derived-manifest.json').read_text())
                self.assertEqual(quantization.sha256(checkpoint),marker['output_sha256'])
                checkpoint.write_bytes(b'corrupt partial cache')
                quantization.prepare_q8_checkpoint()
                regenerated=json.loads((out/'derived-manifest.json').read_text())
                self.assertEqual(quantization.sha256(checkpoint),regenerated['output_sha256'])
                self.assertNotEqual(regenerated['output_sha256'],hashlib.sha256(b'corrupt partial cache').hexdigest())
                config=out/'config.json'; original_config=config.read_bytes()
                config.write_bytes(b'{"corrupt":true}')
                quantization.prepare_q8_checkpoint()
                self.assertEqual(config.read_bytes(),original_config)
                regenerated=json.loads((out/'derived-manifest.json').read_text())
                self.assertEqual(quantization.sha256(config),regenerated['config_sha256'])
            finally: quantization.OUT=original

    def test_q8_fixed_frame_replay_is_bitwise(self):
        verify_q8()

    def test_fixture_file_manifest_checks_npz_hashes(self):
        root=Path(__file__).parents[1]/'fixtures'; manifest=json.loads((root/'files-manifest.json').read_text())
        for name,record in manifest['files'].items():
            data=(root/name).read_bytes()
            self.assertEqual(len(data),record['size_bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(),record['sha256'])

    def test_committed_native_stage_budgets_are_fixed_and_distinct(self):
        root=Path(__file__).parents[1]
        budgets=json.loads((root/'budgets.json').read_text())
        gate=budgets['native_parity_gate']
        self.assertEqual(gate['native_float_stage_targets'],gate['native_q8_stage_targets'])
        for values in gate['native_float_stage_targets'].values():
            self.assertEqual(values,{'rtol':1e-3,'atol':1e-5})
        q8=json.loads((root/'fixtures'/'q8-budgets.json').read_text())
        self.assertEqual(q8['pre_native_q8_stage_acceptance_targets'],gate['native_q8_stage_targets'])
        self.assertEqual(q8['same_q8_oracle_replay_budgets']['latents'],{'rtol':0.0,'atol':0.0})

    def test_pinned_assets_match_committed_hashes(self):
        self.assertEqual(verify_assets(),12)

    def test_context_window_is_strict_and_capped_at_250(self):
        keys=range(300)
        self.assertEqual(len(allowed_positions(248,keys)),249)
        self.assertEqual(len(allowed_positions(249,keys)),250)
        self.assertEqual(len(allowed_positions(250,keys)),250)
        self.assertEqual(allowed_positions(250,keys)[0],1)
        self.assertNotIn(0,allowed_positions(250,keys))

    def test_actual_hf_mimi_mask_has_250_keys_at_boundary(self):
        import transformers.models.pocket_tts_audio_tokenizer.modeling_pocket_tts_audio_tokenizer as module
        _,_,codec,_=load(); captured=[]; original=module.create_sliding_window_causal_mask
        def capture_mask(**kwargs):
            mask=original(**kwargs); captured.append(mask); return mask
        module.create_sliding_window_causal_mask=capture_mask
        try:
            with torch.inference_mode(): codec.decoder.decoder_transformer(torch.zeros(1,252,512),return_dict=True)
        finally: module.create_sliding_window_causal_mask=original
        mask=captured[0][0,0]
        for query in (249,250,251):
            allowed=torch.nonzero(mask[query]).flatten().tolist()
            expected=list(range(max(0,query-249),query+1))
            self.assertEqual(allowed,expected)
            self.assertEqual(len(allowed),min(query+1,250))

    def test_temporal_mimi_framewise_and_uneven_partitions(self):
        mimi_temporal_probe(False)

    def test_raw_and_final_pcm_sample_counts_and_zero_fade(self):
        root=Path(__file__).parents[1]/'fixtures'
        with np.load(root/'phonon-hello.npz') as fixture:
            self.assertEqual(fixture['final_pcm_sample_count'][0],int(fixture['stop_lengths'][0])*1920)
            self.assertGreaterEqual(fixture['raw_pcm_sample_count'][0],fixture['final_pcm_sample_count'][0])
            np.testing.assert_array_equal(fixture['final_pcm'],fixture['raw_pcm'][:,:fixture['final_pcm_sample_count'][0]])

    def test_fixture_manifest_checks_array_sha256(self):
        root=Path(__file__).parents[1]/'fixtures'; manifest=json.loads((root/'manifest.json').read_text())
        with np.load(root/manifest['fixture']) as data:
            for name,array in data.items():
                observed=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                self.assertEqual(observed,manifest['arrays'][name]['sha256'])

    def test_reference_replays_exactly(self):
        verify()

    def test_uneven_mimi_batch_beyond_context(self):
        mimi_batch_probe(False)

    def test_fixture_manifest_matches_saved_arrays(self):
        root=Path(__file__).parents[1]/'fixtures'
        manifest=json.loads((root/'manifest.json').read_text())
        with np.load(root/manifest['fixture']) as data:
            self.assertEqual(set(data.files),set(manifest['arrays']))
            for name,array in data.items():
                self.assertEqual(list(array.shape),manifest['arrays'][name]['shape'])
                self.assertEqual(str(array.dtype),manifest['arrays'][name]['dtype'])
                self.assertEqual(hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),manifest['arrays'][name]['sha256'])

if __name__=='__main__': unittest.main()
