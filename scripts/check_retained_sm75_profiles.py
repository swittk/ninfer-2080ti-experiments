#!/usr/bin/env python3
"""Verify retained NInfer headline evidence/profile boundaries; never a current-performance certificate."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
REQUIRED=frozenset({
"c1_strict_p0m79","c1_exact80","c1_d3_margin7","c1_margin7_current_recert",
"dual_short_margin7","dual_short_65k","q45_65k_exact107","dual_genuine262",
"exact_pv_genuine262","single_multimodal","dual_multimodal","short_media_policy",
"physical_c8_first_pass","native_c8_clean256","q5_c8_measured271","native_c8_16k",
"native_c8_current","strict_c1_full_vocab"})
def check(root:Path,manifest:dict,check_binaries:bool=False)->list[str]:
    errors=[];root=root.resolve();profiles=manifest.get('profiles',[])
    ids=[p.get('id') for p in profiles]
    if len(ids)!=len(set(ids)) or not REQUIRED.issubset(ids):errors.append('accepted headline profile missing or duplicated')
    c=manifest.get('canonical_c1',{})
    if (c.get('draft_tokens'),c.get('greedy_margin_delta'),c.get('proposal_head'),c.get('acceptance_policy'),c.get('minimum_useful_tps'))!=(3,-1.0,'optimized','strict',100.0):errors.append('canonical strict C1 topology/policy/gate changed')
    r=manifest.get('retained_relaxed_c1',{})
    if (r.get('draft_tokens'),r.get('greedy_margin_delta'),r.get('proposal_head'),r.get('gate_credit'))!=(3,7.0,'optimized',False):errors.append('retained margin7 preservation/non-gating contract changed')
    if c.get('retained_mtp_packed_prefix') is not True:errors.append('retained C1 draft arithmetic changed')
    h=manifest.get('owner_headlines',{})
    if (h.get('c1_strict_current_p0m_tps'),h.get('c1_legacy_certified_single80_tps'),h.get('c1_margin7_tps'),h.get('c1_required_strict_tps'),h.get('c1_margin7_gate_credit'))!=(79.01,80.185,107.27,100.0,False):errors.append('C1 headline/gate contract changed')
    gates=manifest.get('hard_gates',{})
    if (gates.get('dual_short_tps'),gates.get('populated_65k_tps'),gates.get('genuine_262k_tps'),gates.get('physical_c8_aggregate_tps'),gates.get('physical_c8_min_per_user_tps'))!=(100.0,100.0,100.0,160.0,20.0):errors.append('owner non-C1 hard gates changed')
    if (gates.get('c1_strict_useful_tps'),gates.get('c1_draft_tokens'),gates.get('c1_greedy_margin_delta'),gates.get('c1_language_soup'),gates.get('c1_relaxed_policy_may_satisfy_gate'))!=(100.0,3,-1.0,'HARD_FAIL',False):errors.append('owner strict C1 hard gate changed')
    def evidence(name,path,digest):
        target=(root/path).resolve()
        if not target.is_relative_to(root) or not target.is_file():errors.append(name+': missing evidence '+path);return
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:errors.append(name+': evidence changed; review required')
    for p in profiles:
        if p.get('must_preserve') is not True:errors.append(str(p.get('id'))+': preservation disabled')
        evidence(p['id'],p['evidence'],p['evidence_sha256'])
        for i,a in enumerate(p.get('additional_evidence',[])):evidence(f"{p['id']}[{i}]",a['path'],a['sha256'])
    fv=next((p for p in profiles if p.get('id')=='strict_c1_full_vocab'),{})
    if fv.get('headline_credit') is not False:errors.append('full-vocab server candidate incorrectly credited as strict79 headline')
    for pid in ('c1_d3_margin7','c1_margin7_current_recert'):
        p=next((x for x in profiles if x.get('id')==pid),{})
        if p.get('gate_credit') is not False:errors.append(pid+': relaxed C1 path incorrectly granted owner gate credit')
    sp=next((p for p in profiles if p.get('id')=='c1_strict_p0m79'),{})
    if sp.get('gate_credit') is not True or sp.get('gate_met') is not False:errors.append('strict C1 reference gate state changed')
    if check_binaries:
        for name,p in manifest.get('preserved_binaries',{}).items():evidence(name,p['preserved'],p['sha256'])
    return errors
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest',type=Path,default=Path('tests/retained_sm75_profiles.json'))
    parser.add_argument('--check-binaries',action='store_true')
    args=parser.parse_args();manifest=json.loads((args.root/args.manifest).read_text())
    errors=check(args.root,manifest,args.check_binaries)
    if errors:
        print('\n'.join(errors));raise SystemExit(1)
    print(f"PASS {len(manifest['profiles'])} retained headline witnesses; not a current-performance certificate")
