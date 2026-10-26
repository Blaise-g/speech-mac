import pathlib,re
p=pathlib.Path('artifacts/34-acceptance-20261026/acceptance.log'); text=p.read_text()
assert re.findall(r'engine actual record=(\d+) samples=(\d+) chunks=(\d+) expected=(\d+)/(\d+)',text)==[(str(i),str(n),str(c),str(n),str(c)) for i,(n,c) in enumerate([(30720,1),(49920,1),(72960,1),(503040,2),(1242240,2)])]
assert len(re.findall(r'actual q8 record=\d+ partition=\d+ raw',text))==33
assert all(f'real Mimi run=2 partition={i}' in text for i in range(3))
for required in ['Phonon native pack: 209 tensors, 36 Q8 projections matched oracle','actual CLI positional/file/stdin/default/named/local/projected/native controls/WAV/playback cleanup: ok','actual stream order/bounds/waits/generation+submit+drain+signal cancellation/errors/resident reuse/CLI: ok','Ran 23 tests','Mimi >250 positions, uneven batch repeat verified','raw nrmse/max_abs=0/0; final nrmse/max_abs=0/0']:
    assert required in text, required
skips=[line for line in text.splitlines() if re.search('skip',line,re.I)]
assert skips==['SKIP muted device: explicitly select PHONON_STREAM_DEVICE=1 with silent-output build'],skips
print('PASS full actual fixture coverage: five prepared-engine cases, 33 raw PCM partitions, long Mimi, all23 locked reference tests, all36Q8 oracles, exact frozen closed loop; only separate-device skip present')
