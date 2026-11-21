from pathlib import Path
import subprocess,sys,json
binary=Path(sys.argv[1]).resolve()
common=[str(binary),'not-a-model.ninfer','--prompt','x','--context-value-fp16']
cases=[([], 'packed FP16 values require'),
       (['--context-headscale-qk','--spec','mtp'], 'headscale-QK requires'),
       (['--context-headscale-qk','--full-context-compute','--context-parallel-gqa','--prefix-snapshot-dir','unused','--spec','none'], 'packed FP16 values require'),
       (['--context-headscale-qk','--full-context-compute','--context-parallel-gqa','--prefix-snapshot-dir','unused','--spec','mtp','--mtp-depth','1'], 'headscale-QK requires'),
       (['--context-headscale-qk','--full-context-compute','--context-parallel-gqa','--prefix-snapshot-dir','unused','--spec','mtp','--kv-dtype','bf16'], 'headscale-QK requires'),
       (['--context-headscale-qk','--full-context-compute','--context-parallel-gqa','--spec','mtp'], 'headscale-QK requires'),
       (['--context-headscale-qk','--full-context-compute','--context-parallel-gqa','--prefix-snapshot-dir','unused','--spec','mtp','--sessions','2'], 'packed FP16 values require')]
for args,expected in cases:
    r=subprocess.run(common+args,capture_output=True,text=True,timeout=10)
    if r.returncode!=2 or expected not in r.stderr+r.stdout:
        raise RuntimeError(str((args,r.returncode,r.stdout,r.stderr)))
print('PASS 7 incompatible packed-value/target/topology/cache CLI combinations rejected before model loading')
