from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('retained_guard',ROOT/'scripts/check_retained_sm75_profiles.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class PreservationGuard(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.m=json.loads((ROOT/'tests/retained_sm75_profiles.json').read_text())
  self.m['preserved_binaries']={}
  for p in self.m['profiles']:
   p['evidence']=p['id']+'.json';data=json.dumps({'id':p['id']}).encode();(self.root/p['evidence']).write_bytes(data);p['evidence_sha256']=hashlib.sha256(data).hexdigest()
   for i,a in enumerate(p.get('additional_evidence',[])):
    a['path']=p['id']+f'-extra-{i}.json';(self.root/a['path']).write_bytes(data);a['sha256']=hashlib.sha256(data).hexdigest()
 def test_intact(self):self.assertEqual(module.check(self.root,self.m),[])
 def test_removal(self):
  self.m['profiles'].pop(0);self.assertTrue(module.check(self.root,self.m))
 def test_duplicate(self):
  self.m['profiles'].append(copy.deepcopy(self.m['profiles'][0]));self.assertTrue(module.check(self.root,self.m))
 def test_disable(self):
  self.m['profiles'][0]['must_preserve']=False;self.assertTrue(module.check(self.root,self.m))
 def test_evidence_mutation(self):
  (self.root/self.m['profiles'][0]['evidence']).write_text('changed');self.assertTrue(module.check(self.root,self.m))
 def test_missing(self):
  (self.root/self.m['profiles'][0]['evidence']).unlink();self.assertTrue(module.check(self.root,self.m))
 def test_policy_substitution(self):
  for key,value in [('draft_tokens',1),('greedy_margin_delta',7),('acceptance_policy','relaxed'),('minimum_useful_tps',99),('retained_mtp_packed_prefix',False)]:
   m=copy.deepcopy(self.m);m['canonical_c1'][key]=value;self.assertTrue(module.check(self.root,m))
 def test_relaxed_profile_is_retained_but_not_gating(self):
  for key,value in [('draft_tokens',1),('greedy_margin_delta',-1),('gate_credit',True)]:
   m=copy.deepcopy(self.m);m['retained_relaxed_c1'][key]=value;self.assertTrue(module.check(self.root,m))
  m=copy.deepcopy(self.m)
  next(p for p in m['profiles'] if p['id']=='c1_d3_margin7')['gate_credit']=True
  self.assertTrue(module.check(self.root,m))
 def test_strict_gate_cannot_be_lowered_or_replaced(self):
  for key,value in [('c1_strict_useful_tps',99),('c1_greedy_margin_delta',7),('c1_language_soup','allowed'),('c1_relaxed_policy_may_satisfy_gate',True)]:
   m=copy.deepcopy(self.m);m['hard_gates'][key]=value;self.assertTrue(module.check(self.root,m))
 def test_escape(self):
  self.m['profiles'][0]['evidence']='../not-a-repo-witness';self.assertTrue(module.check(self.root,self.m))
 def test_optional_binary(self):
  self.m['preserved_binaries']={'old':{'preserved':'missing','sha256':'0'*64}}
  self.assertFalse(module.check(self.root,self.m));self.assertTrue(module.check(self.root,self.m,True))
if __name__=='__main__':unittest.main()
