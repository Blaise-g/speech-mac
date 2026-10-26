"""Deterministic Phonon reference trace; requires the pinned, locally verified model assets."""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, GradiumPhononTTSForConditionalGeneration
from transformers.models.pocket_tts.processing_pocket_tts import PocketTTSProcessor
from transformers.models.pocket_tts.feature_extraction_pocket_tts import PocketTTSFeatureExtractor
from transformers.models.pocket_tts_audio_tokenizer import PocketTTSAudioTokenizerConfig, PocketTTSAudioTokenizerModel

ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path(os.environ.get('PHONON_MODEL_DIR','models/phonon-40m-en'))
MIMI=ASSETS/'reference-phonon-mimi'
REV='cc738b7e4d182c2ba1bafb8062aa4846af000637'
MIMI_REV='33519be1a8e61c6fa00c6e154e2d38e76a367919'
HF_REV='50e35cc0ca18ec19b80653690de553e6decf74f5'
_ASSETS_VERIFIED=False

def load(weights="float"):
    global _ASSETS_VERIFIED
    if not _ASSETS_VERIFIED:
        from verify_assets import verify
        verify(); _ASSETS_VERIFIED=True
    from transformers.models.gradium_phonon_tts.convert_gradium_phonon_tts_to_hf import convert_tokenizer
    with tempfile.TemporaryDirectory(prefix='phonon-tokenizer-') as tok_tmp:
        tok_dir=Path(tok_tmp)
        convert_tokenizer(ASSETS,tok_dir)
        tok=AutoTokenizer.from_pretrained(tok_dir,local_files_only=True)
    model_path=ASSETS if weights=='float' else __import__('quantization').prepare_q8_checkpoint()
    model,model_load_info=GradiumPhononTTSForConditionalGeneration.from_pretrained(model_path,local_files_only=True,dtype=torch.float32,output_loading_info=True)
    model=model.eval()
    if any(model_load_info.values()): raise RuntimeError(f'checkpoint-to-HF mapping incomplete: {model_load_info}')
    cfg=PocketTTSAudioTokenizerConfig.from_pretrained(MIMI)
    cfg.sliding_window=250  # documented PR209 correction: trained Python predicate is delta < context.
    codec=PocketTTSAudioTokenizerModel.from_pretrained(MIMI,config=cfg,local_files_only=True).eval()
    # Prove the separate codec snapshot is the exact decoder subset of the pinned Phonon bundle.
    from transformers.models.gradium_phonon_tts.convert_gradium_phonon_tts_to_hf import build_audio_tokenizer_config, convert_audio_tokenizer_state_dict
    original=load_file(str(ASSETS/'model.safetensors'))
    codec_config=build_audio_tokenizer_config(json.loads((ASSETS/'config.json').read_text()))
    if weights=='q8':
        q8_state=load_file(str(Path(model_path)/'model.safetensors'))
        mapped=convert_audio_tokenizer_state_dict(q8_state,codec_config)
        actual=codec.state_dict()
        if mapped.keys()!=actual.keys(): raise RuntimeError('GGUF Mimi tensors do not fully map to decoder model')
        result=codec.load_state_dict(mapped,strict=True)
        if result.missing_keys or result.unexpected_keys: raise RuntimeError(f'Q8 Mimi map incomplete: {result}')
        if any(not torch.equal(codec.state_dict()[k],mapped[k]) for k in mapped): raise RuntimeError('loaded Q8 Mimi tensors differ from published GGUF expansion')
    else:
        mapped=convert_audio_tokenizer_state_dict(original,codec_config)
        actual=codec.state_dict()
        if mapped.keys()!=actual.keys() or any(not torch.equal(mapped[k],actual[k]) for k in mapped):
            raise RuntimeError('pinned Mimi codec differs from Mimi tensors embedded in Phonon model')
    pc=json.loads((ASSETS/'processor_config.json').read_text())
    gen=json.loads((ASSETS/'generation_config.json').read_text())
    model.generation_config.do_sample=True
    model.generation_config.temperature=0.8
    model.generation_config.eos_threshold=gen['eos_threshold']
    model.generation_config.num_flow_steps=gen['num_flow_steps']
    model.generation_config.min_frames_before_eos=gen['min_frames_before_eos']
    proc=PocketTTSProcessor(PocketTTSFeatureExtractor(**pc['feature_extractor']),tok,codec,chat_template=(ASSETS/'chat_template.jinja').read_text(),fade_in_duration=pc['fade_in_duration'],frames_after_eos=pc['frames_after_eos'],frames_after_eos_short_text=pc['frames_after_eos_short_text'],short_text_max_words=pc['short_text_max_words'])
    tok.chat_template=(ASSETS/'chat_template.jinja').read_text()
    return tok,model,codec,proc

def build_inputs(tok,model,proc,text,voice):
    result=proc.apply_chat_template([{'role':'user','content':[{'type':'audio'},{'type':'text','text':text}]}],tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True,processor_kwargs={'audio_prompt_latents':[voice]})
    # Tokenizer JSON represents model special IDs as appended AddedTokens. The checkpoint config is authoritative;
    # translate those three prompt markers back to the reserved model vocabulary IDs before embedding lookup.
    result['input_ids']=result['input_ids'].clone()
    for marker,idx in [('<|audio_prompt|>',model.config.audio_prompt_token_id),('<|audio_bos|>',model.config.audio_bos_token_id),('<|audio_latent|>',model.config.audio_latent_token_id)]:
        result['input_ids'][result['input_ids']==tok.convert_tokens_to_ids(marker)]=idx
    return result

def capture(text='“Hello”—from Phonon, today!',seed=123,frames=40,weights="float",eos_threshold=None):
    tok,model,codec,proc=load(weights); voice=load_file(str(ASSETS/'default-voice.safetensors'))['speaker_wavs'][0].T.float()
    inputs=build_inputs(tok,model,proc,text,voice)
    normalized=tok.backend_tokenizer.normalizer.normalize_str(text)
    traces={'input_ids':inputs['input_ids'].numpy(),'attention_mask':inputs['attention_mask'].numpy(),'voice_prompt_latents':voice.numpy(),'raw_text':np.asarray([text],dtype=np.str_),'tokenizer_normalized_text':np.asarray([normalized],dtype=np.str_),'audio_marker_ids':np.asarray([model.config.audio_bos_token_id,model.config.audio_prompt_token_id,model.config.audio_latent_token_id],dtype=np.int64)}
    draws=[]; backbone=[]; flow=[]; flow_conditions=[]; mimi_features=[]
    def capture_backbone(_m,_i,out): backbone.append(out.last_hidden_state.detach().cpu().float().numpy().copy())
    def capture_mimi(_m,_i,out): mimi_features.append(out.last_hidden_state.detach().cpu().float().numpy().copy())
    model.model.register_forward_hook(capture_backbone)
    voice_projection=[]
    model.model.audio_prompt_proj.register_forward_hook(lambda _m,_i,out: voice_projection.append(out.detach().cpu().float().numpy().copy()))
    codec.decoder.decoder_transformer.register_forward_hook(lambda _m,_i,out: mimi_features.append(out.last_hidden_state.detach().cpu().float().numpy().copy()))
    original_sample=model.sample_latent
    def sample_latent(*args,**kwargs):
        flow_conditions.append(args[0].detach().cpu().float().numpy().copy()); value=original_sample(*args,**kwargs); flow.append(value.detach().cpu().float().numpy().copy()); return value
    model.sample_latent=sample_latent
    draws=[]; original=torch.randn
    def recording_randn(*a,**kw):
        x=original(*a,**kw); draws.append(x.detach().cpu().float().numpy().copy()); return x
    torch.randn=recording_randn
    try:
        torch.manual_seed(seed); torch.set_num_threads(1)
        with torch.inference_mode():
            model.generation_config.temperature=0.8
            model.generation_config.do_sample=True
            generation={'max_new_tokens':frames,'return_dict_in_generate':True,'output_scores':True}
            if eos_threshold is not None: generation['eos_threshold']=eos_threshold
            out=model.generate(**inputs,**generation)
    finally: torch.randn=original
    if len(draws)!=out.audio_latents.shape[1]: raise RuntimeError('captured noise draw count differs from generated frame count')
    traces.update({'latents':out.audio_latents.float().numpy(),'latent_mask':out.audio_latents_mask.numpy(),'eos_logits':torch.stack(out.scores,dim=0).cpu().numpy(),'noise_draws':np.stack(draws),'flow_input_noise':np.stack(draws)*np.sqrt(0.8),'temperature':np.asarray([0.8],dtype=np.float32),'flow_condition':np.stack(flow_conditions),'voice_projection':voice_projection[0],'prefix_states':backbone[0],'backbone_output':np.concatenate(backbone,axis=1),'flow_integration':np.concatenate(flow,axis=0),'stop_lengths':out.audio_latents_mask.sum(-1).cpu().numpy()})
    with torch.inference_mode():
        raw=codec.decode(out.audio_latents.float(),return_dict=True).audio_values[:,0].float().cpu().numpy()
    traces['raw_pcm']=raw
    traces['raw_pcm_sample_count']=np.asarray([raw.shape[-1]],dtype=np.int64)
    traces['final_pcm']=np.stack([x.numpy() for x in proc.decode(out.audio_latents,out.audio_latents_mask)])
    final_count=int(out.audio_latents_mask.sum().item())*1920
    if traces['final_pcm'].shape[-1]!=final_count or raw.shape[-1]<final_count:
        raise RuntimeError('Mimi postprocessing PCM count disagrees with generated frame stop length')
    np.testing.assert_array_equal(traces['final_pcm'],raw[:,:final_count])
    traces['final_pcm_sample_count']=np.asarray([traces['final_pcm'].shape[-1]],dtype=np.int64)
    if len(mimi_features)!=2 or not np.array_equal(mimi_features[0],mimi_features[1]): raise RuntimeError('raw and processor Mimi feature traces disagree')
    traces['mimi_features']=mimi_features[0]
    return traces,model,codec

def update_fixture_manifest():
    root=Path(__file__).parent/'fixtures'
    records={}
    for path in sorted(root.glob('*.npz')):
        data=path.read_bytes()
        records[path.name]={'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    manifest={'schema_version':1,'model_revision':REV,'mimi_revision':MIMI_REV,'transformers_revision':HF_REV,'files':records}
    (root/'files-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def write():
    arrays,_,_=capture()
    out=Path(__file__).parent/'fixtures'
    np.savez_compressed(out/'phonon-hello.npz',**arrays)
    manifest={'schema_version':1,'model_revision':REV,'mimi_revision':MIMI_REV,'transformers_revision':HF_REV,'mimi_context_positions':250,'context_patch':'override sliding_window from upstream converter 251 to trained Python semantics delta < 250; provenance xn-ptts PR209','seed':123,'fixture':'phonon-hello.npz','temperature':0.8,'generation_settings':json.loads((ASSETS/'generation_config.json').read_text()) | {'temperature':0.8,'eos_threshold':4.0,'num_flow_steps':1,'do_sample':True},'processor_settings':json.loads((ASSETS/'processor_config.json').read_text()),'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k,v in sorted(arrays.items())}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    update_fixture_manifest()
    print(json.dumps({'fixture':str(out/'phonon-hello.npz'),'arrays':list(arrays)},indent=2))

def verify():
    with np.load(Path(__file__).parent/'fixtures/phonon-hello.npz') as fixture: saved={k:fixture[k] for k in fixture.files}
    current,_,_=capture()
    assert saved.keys()==current.keys()
    for k in saved: np.testing.assert_array_equal(saved[k],current[k],err_msg=k)
    print('deterministic replay: all trace arrays bitwise equal')

def mimi_batch_probe(write=False):
    """Decode deterministic uneven [263,257] rows, both past Mimi's 250-key context."""
    _,_,codec,_=load()
    generator=torch.Generator().manual_seed(8128)
    latents=torch.randn((2,263,32),generator=generator,dtype=torch.float32)
    lengths=np.asarray([263,257],dtype=np.int64); latents[1,257:]=0
    with torch.inference_mode():
        batched=codec.decode(latents,return_dict=True).audio_values.float().cpu().numpy()
        single0=codec.decode(latents[:1],return_dict=True).audio_values.float().cpu().numpy()
        single1=codec.decode(latents[1:2,:257],return_dict=True).audio_values.float().cpu().numpy()
    pcm=batched[:,0,:]
    ref=np.zeros_like(pcm); ref[0,:263*1920]=single0[0,0,:263*1920]; ref[1,:257*1920]=single1[0,0,:257*1920]
    error=max(float(np.max(np.abs(pcm[0,:263*1920]-ref[0,:263*1920]))),float(np.max(np.abs(pcm[1,:257*1920]-ref[1,:257*1920]))))
    if error>2e-7: raise RuntimeError(f'uneven batch Mimi PCM mismatch: {error}')
    if write:
        path=Path(__file__).parent/'fixtures'/'mimi-uneven-263-257.npz'
        np.savez_compressed(path,latents=latents.numpy(),lengths=lengths,pcm=pcm,pcm_single=ref)
        print(f'{path}: max_abs_batch_vs_single={error:.9g}')
        update_fixture_manifest()
    else:
        with np.load(Path(__file__).parent/'fixtures'/'mimi-uneven-263-257.npz') as data:
            np.testing.assert_array_equal(data['latents'],latents.numpy())
            np.testing.assert_allclose(data['pcm'][0,:263*1920],pcm[0,:263*1920],rtol=0,atol=2e-7)
            np.testing.assert_allclose(data['pcm'][1,:257*1920],pcm[1,:257*1920],rtol=0,atol=2e-7)
            np.testing.assert_allclose(data['pcm_single'],ref,rtol=0,atol=2e-7)
        print(f'Mimi >250 positions, uneven batch repeat verified; max_abs={error:.9g}')


def mimi_temporal_probe(write=False):
    """Compare whole decoding with framewise and uneven temporal KV-cache partitions."""
    _,_,codec,_=load(); generator=torch.Generator().manual_seed(451)
    latents=torch.randn((1,263,32),generator=generator,dtype=torch.float32)
    with torch.inference_mode():
        hidden=codec.decoder.upsample(codec.latent_proj(latents.transpose(1,2))).transpose(1,2)
        whole=codec.decoder.decoder_transformer(hidden,return_dict=True).last_hidden_state
        def chunked(frame_chunks):
            cache=None; outputs=[]; offset=0
            for frames in frame_chunks:
                end=offset+frames*16
                out=codec.decoder.decoder_transformer(hidden[:,offset:end],past_key_values=cache,use_cache=True,return_dict=True)
                cache=out.past_key_values; outputs.append(out.last_hidden_state); offset=end
            if offset!=hidden.shape[1]: raise RuntimeError('temporal partition did not consume all Mimi positions')
            return torch.cat(outputs,dim=1).float()
        whole_pcm=codec.decoder.decoder_seanet(whole.transpose(1,2)).float()
        def decode_stream(features,frame_chunks):
            emitted=[]; chunk_samples=[]; start_frame=0
            for frame_count in frame_chunks:
                end_frame=start_frame+frame_count
                start_pos=start_frame*16; end_pos=end_frame*16
                overlap_pos=max(0,start_pos-250)
                segment=codec.decoder.decoder_seanet(features[:,overlap_pos:end_pos].transpose(1,2)).float()
                trim=(start_pos-overlap_pos)*120
                expected=frame_count*1920
                chunk=segment[...,trim:trim+expected]
                if chunk.shape[-1]!=expected: raise RuntimeError(f'Mimi chunk length {chunk.shape[-1]} != {expected}')
                emitted.append(chunk); chunk_samples.append(expected); start_frame=end_frame
            if start_frame!=263: raise RuntimeError('Mimi output chunks do not cover the complete latent sequence')
            return torch.cat(emitted,dim=-1),np.asarray(chunk_samples,dtype=np.int64)
        frame_features=chunked([1]*263)
        sizes=[1,7,113,142]; uneven_features=chunked(sizes)
        frame_pcm,frame_chunk_samples=decode_stream(frame_features,[1]*263)
        uneven_pcm,uneven_chunk_samples=decode_stream(uneven_features,sizes)
    feature_error=float((whole.float()-frame_features).abs().max()); pcm_error=float((whole_pcm-frame_pcm).abs().max())
    uneven_feature_error=float((whole.float()-uneven_features).abs().max()); uneven_pcm_error=float((whole_pcm-uneven_pcm).abs().max())
    if max(feature_error,pcm_error,uneven_feature_error,uneven_pcm_error)>1e-5: raise RuntimeError('Mimi chunk mismatch')
    if write:
        path=Path(__file__).parent/'fixtures'/'mimi-temporal-263.npz'
        np.savez_compressed(path,latents=latents.numpy(),whole_features=whole.float().numpy(),framewise_features=frame_features.numpy(),uneven_features=uneven_features.numpy(),whole_pcm=whole_pcm.numpy(),framewise_pcm=frame_pcm.numpy(),uneven_pcm=uneven_pcm.numpy(),uneven_frame_chunks=np.asarray(sizes,dtype=np.int64),framewise_chunk_samples=frame_chunk_samples,uneven_chunk_samples=uneven_chunk_samples,total_output_samples=np.asarray([frame_pcm.shape[-1]],dtype=np.int64),flush_samples=np.asarray([0],dtype=np.int64))
        print(f'{path}: feature max_abs framewise={feature_error:.9g} uneven={uneven_feature_error:.9g}; PCM framewise={pcm_error:.9g} uneven={uneven_pcm_error:.9g}')
        update_fixture_manifest()
    else:
        with np.load(Path(__file__).parent/'fixtures'/'mimi-temporal-263.npz') as fixture:
            np.testing.assert_array_equal(fixture['latents'],latents.numpy())
            if fixture['total_output_samples'].item()!=263*1920 or fixture['flush_samples'].item()!=0: raise RuntimeError('Mimi delay/flush sample contract changed')
            np.testing.assert_array_equal(fixture['framewise_chunk_samples'],np.full(263,1920,dtype=np.int64))
            np.testing.assert_array_equal(fixture['uneven_chunk_samples'],np.asarray(sizes,dtype=np.int64)*1920)
            for key,current in [('whole_features',whole.float().numpy()),('framewise_features',frame_features.numpy()),('uneven_features',uneven_features.numpy()),('whole_pcm',whole_pcm.numpy()),('framewise_pcm',frame_pcm.numpy()),('uneven_pcm',uneven_pcm.numpy())]: np.testing.assert_allclose(fixture[key],current,rtol=0,atol=1e-5)
            print(f'Mimi framewise/uneven cache and PCM emission verified; samples={frame_pcm.shape[-1]}, explicit flush samples=0, budget=1e-5')


def write_q8():
    float_arrays,_,_=capture(frames=40,eos_threshold=1e9)
    q8_arrays,_,_=capture(frames=40,weights="q8",eos_threshold=1e9)
    q8_natural,_,_=capture(weights="q8")
    np.savez_compressed(Path(__file__).parent/'fixtures'/'phonon-hello-q8.npz',**q8_arrays)
    np.savez_compressed(Path(__file__).parent/'fixtures'/'phonon-hello-q8-natural.npz',**q8_natural)
    budgets={'exact_replay':{'rtol':0.0,'atol':0.0},'quantization_observed_vs_float':{}}
    for name,reference in float_arrays.items():
        quantized=q8_arrays[name]
        if reference.shape!=quantized.shape: raise RuntimeError(f'Q8 comparison shape differs at {name}: {reference.shape} != {quantized.shape}')
        if reference.dtype.kind in 'iu':
            if not np.array_equal(reference,quantized): raise RuntimeError(f'Q8 changed discrete stage {name}')
            budgets['quantization_observed_vs_float'][name]={'exact':True}
        elif reference.dtype.kind=='f':
            diff=np.abs(reference.astype(np.float64)-quantized.astype(np.float64))
            budgets['quantization_observed_vs_float'][name]={'observed_max_abs':float(diff.max()),'observed_rmse':float(np.sqrt(np.mean(diff*diff)))}
    native_gates=json.loads((Path(__file__).parent/'budgets.json').read_text())['native_parity_gate']
    (Path(__file__).parent/'fixtures'/'q8-budgets.json').write_text(json.dumps({'schema_version':1,'model_revision':REV,'q8_tensor_format':'all 209 tensors expanded from GGUF Q8_0 + BF16','fixed_stage_run':'40 frames, EOS disabled for aligned stages; exact recorded Gaussian draws and temperature=0.8','natural_run':{'temperature':0.8,'eos_threshold':4.0,'stop_lengths':q8_natural['stop_lengths'].tolist(),'raw_pcm_sample_counts':q8_natural['final_pcm_sample_count'].tolist()},'same_q8_oracle_replay_budgets':{name:budgets['exact_replay'] for name in q8_arrays},'pre_native_q8_stage_acceptance_targets':native_gates['native_q8_stage_targets'],'pre_native_pcm_acceptance_targets':native_gates['native_pcm_targets'],'quantization_effect_measurements_not_acceptance_tolerances':budgets['quantization_observed_vs_float']},indent=2)+'\n')
    update_fixture_manifest()
    print(json.dumps(budgets,indent=2))

def verify_q8():
    fixed,_,_=capture(frames=40,weights="q8",eos_threshold=1e9)
    natural,_,_=capture(weights="q8")
    with np.load(Path(__file__).parent/'fixtures'/'phonon-hello-q8.npz') as saved:
        for name,value in fixed.items(): np.testing.assert_array_equal(saved[name],value,err_msg=f'fixed:{name}')
    with np.load(Path(__file__).parent/'fixtures'/'phonon-hello-q8-natural.npz') as saved:
        for name,value in natural.items(): np.testing.assert_array_equal(saved[name],value,err_msg=f'natural:{name}')
    print('Q8 fixed-frame and natural-stop replays: all arrays bitwise equal')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['write','verify','write-mimi','verify-mimi','write-temporal','verify-temporal','write-q8','verify-q8','write-manifest']); args=parser.parse_args(); (write if args.command=='write' else verify if args.command=='verify' else write_q8 if args.command=='write-q8' else verify_q8 if args.command=='verify-q8' else update_fixture_manifest if args.command=='write-manifest' else lambda: mimi_batch_probe(args.command=='write-mimi') if 'mimi' in args.command else mimi_temporal_probe(args.command=='write-temporal'))()
