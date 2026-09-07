"""Runner tests use isolated temporary Git repos and tiny CPU commands."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runner_under_test',TOOLS/'run_experiment.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
spec2=importlib.util.spec_from_file_location('checker_under_test',TOOLS/'check_setup.py')
checker=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(checker)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.g('init','-q');self.g('config','user.name','Local test');self.g('config','user.email','local-test@example.invalid')
        (self.root/'.gitignore').write_text('.runtime/\n__pycache__/\n')
        self.plan={'schema_version':1,'run_id':'E-test-001','approval':{'status':'approved','approved_by':'TEST ONLY'},
            'argv':['{python}','-c','print("toy")'],'timeout_seconds':5,
            'hypothesis':'toy command completes','resource':{'kind':'cpu','cuda_visible_devices':None},
            'dataset_revision':'synthetic','split_id':'synthetic','feature_spec':'synthetic',
            'predictor_spec':'none','primary_metric':'exit-code','seed':1,'expected_outputs':[]}
        self.path=self.root/'plan.json'
    def tearDown(self):self.temp.cleanup()
    def g(self,*args):
        return subprocess.check_output(['git','-C',str(self.root),*args],stderr=subprocess.STDOUT,text=True).strip()
    def prepare(self):
        self.path.write_text(json.dumps(self.plan));self.g('add','--','.gitignore','plan.json');self.g('commit','-qm','test plan')
    def run_plan(self):self.prepare();return runner.execute(self.path,self.root)
    def test_success_and_record(self):
        result=self.run_plan();self.assertEqual(result['status'],'succeeded');self.assertEqual(result['scientific_status'],'unreviewed')
        self.assertEqual(result['code_sha'],self.g('rev-parse','HEAD'))
        stored=json.loads((self.root/'MEMORY/runs/E-test-001/execution.json').read_text())
        self.assertEqual(stored,result);self.assertFalse((self.root/'.git/aimo-experiment.lock').exists())
    def test_cpu_hides_gpu_env(self):
        self.plan['argv']=['{python}','-c','import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == ""']
        self.assertEqual(self.run_plan()['status'],'succeeded')
    def test_pending_rejected(self):
        self.plan['approval']['status']='pending';self.prepare()
        with self.assertRaises(runner.PlanError):runner.execute(self.path,self.root)
    def test_placeholder_rejected(self):
        self.plan['split_id']='UNSET';self.prepare()
        with self.assertRaises(runner.PlanError):runner.execute(self.path,self.root)
    def test_dirty_rejected(self):
        self.prepare();(self.root/'untracked.txt').write_text('dirty')
        with self.assertRaises(runner.PlanError):runner.execute(self.path,self.root)
    def test_invalid_timeout(self):
        for n in (0,-1,True,float('nan'),86401):
            self.plan['timeout_seconds']=n
            with self.assertRaises(runner.PlanError):runner.validate_plan(self.plan,self.root)
    def test_traversal_rejected(self):
        for name in ('../escape','x/y','', '.', '..'):
            self.plan['run_id']=name
            with self.assertRaises(runner.PlanError):runner.validate_plan(self.plan,self.root)
    def test_output_escape_rejected(self):
        self.plan['expected_outputs']=['../escape']
        with self.assertRaises(runner.PlanError):runner.validate_plan(self.plan,self.root)
    def test_gpu_needs_explicit_devices(self):
        self.plan['resource']={'kind':'gpu','cuda_visible_devices':None}
        with self.assertRaises(runner.PlanError):runner.validate_plan(self.plan,self.root)
    def test_failed_command_recorded(self):
        self.plan['argv']=['{python}','-c','raise SystemExit(3)']
        result=self.run_plan();self.assertEqual(result['status'],'failed');self.assertEqual(result['exit_code'],3)
    def test_missing_program_recorded(self):
        self.plan['argv']=['AIMO_TEST_NONEXISTENT_EXECUTABLE']
        self.assertEqual(self.run_plan()['status'],'failed')
    def test_timeout_recorded(self):
        self.plan['argv']=['{python}','-c','import time; time.sleep(60)'];self.plan['timeout_seconds']=.2
        self.assertEqual(self.run_plan()['status'],'timed_out')
    def test_missing_expected_output(self):
        self.plan['expected_outputs']=['.runtime/no-result.json']
        self.assertEqual(self.run_plan()['status'],'missing_outputs')
    def test_stale_output_rejected(self):
        self.plan["expected_outputs"]=[".runtime/stale.txt"];self.prepare()
        (self.root/".runtime").mkdir();(self.root/".runtime/stale.txt").write_text("old")
        with self.assertRaises(runner.PlanError):runner.execute(self.path,self.root)
    def test_output_hash(self):
        self.plan["expected_outputs"]=[".runtime/result.txt"]
        self.plan["argv"]=["{python}","-c",'from pathlib import Path; Path(".runtime/result.txt").write_text("abc")']
        result=self.run_plan();self.assertEqual(result["status"],"succeeded")
        self.assertEqual(result["outputs"][0]["sha256"],"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    def test_lock_blocks(self):
        self.prepare();lock=self.root/'.git/aimo-experiment.lock';lock.write_text('owned')
        with self.assertRaises(runner.PlanError):runner.execute(self.path,self.root)
        self.assertEqual(lock.read_text(),'owned')
    def test_no_overwrite(self):
        self.run_plan();self.g('add','MEMORY');self.g('commit','-qm','record')
        with self.assertRaises(runner.PlanError):runner.execute(self.path,self.root)
    def test_source_changes_detected(self):
        self.plan['argv']=['{python}','-c','from pathlib import Path; Path(".gitignore").write_text("changed")']
        self.assertEqual(self.run_plan()['status'],'source_changed')
    def test_shared_worktree_lock(self):
        self.prepare()
        with tempfile.TemporaryDirectory() as second:
            work=Path(second)/'wt';self.g('worktree','add','-qb','other',str(work))
            lock=self.root/'.git/aimo-experiment.lock';lock.write_text('owned')
            with self.assertRaises(runner.PlanError):runner.execute(work/'plan.json',work)


class SetupTests(unittest.TestCase):
    def test_setup(self):self.assertEqual(checker.check(),[])


if __name__=='__main__':unittest.main()
