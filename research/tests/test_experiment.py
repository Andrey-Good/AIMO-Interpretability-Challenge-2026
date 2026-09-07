"""Offline integration checks: the external LLM is a deliberately tiny test double."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import torch
from torch import nn
from research import _support as io
from research.experiment import run
from research.features import extract_features
from research.predictor import build_predictor, normalize, predict


class ToyTokenizer:
    def __init__(self, left=False):
        self.left = left
    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("No test chat template")
    def __call__(self, texts, **kwargs):
        rows = [[int(w) for w in s.split()][:kwargs["max_length"]] for s in texts]
        size = max(map(len, rows))
        ids, masks = [], []
        for r in rows:
            pad = [0] * (size-len(r))
            ids.append(pad+r if self.left else r+pad)
            masks.append([0]*len(pad)+[1]*len(r) if self.left else [1]*len(r)+[0]*len(pad))
        return dict(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks))


class ToyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1, 2)
    def get_input_embeddings(self):
        return self.embedding
    def forward(self, input_ids, attention_mask, **kwargs):
        assert kwargs == dict(output_hidden_states=True, return_dict=True, use_cache=False)
        h = torch.stack((input_ids.float(), -input_ids.float()), -1)
        return SimpleNamespace(hidden_states=(h, h+10))


def context(args):
    return ToyTokenizer(), dict(model="toy", revision="test", local=False, layers=args.layers,
        max_length=args.max_length, template=args.template, padding="right", truncation="right",
        dtype=args.dtype, device=args.device, torch=str(torch.__version__), transformers="test",
        tokenizer_sha256="test", extractor_sha256=io.file_hash(Path(io.__file__).with_name("features.py")))


class CoreTests(unittest.TestCase):
    def test_padding_order_and_values(self):
        for left in (False, True):
            x = extract_features(ToyLLM(), ToyTokenizer(left), ["1 2", "4"], layers=[1, 0], max_length=10, template="plain")
            torch.testing.assert_close(x, torch.tensor([[[12.,8.],[2.,-2.]],[[14.,6.],[4.,-4.]]]))
            self.assertFalse(x.requires_grad)
    def test_batch_invariance(self):
        model, tok = ToyLLM(), ToyTokenizer()
        a = extract_features(model, tok, ["1 2", "4"], layers=[0], max_length=10, template="plain")
        b = extract_features(model, tok, ["4"], layers=[0], max_length=10, template="plain")
        torch.testing.assert_close(a[1:], b)
    def test_no_template_fallback(self):
        with self.assertRaises(ValueError):
            extract_features(ToyLLM(), ToyTokenizer(), ["1"], layers=[0], max_length=10, template="chat")
    def test_invalid_layers(self):
        for layers in ([], [0,0], [-1], [3], [True]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                extract_features(ToyLLM(), ToyTokenizer(), ["1"], layers=layers, max_length=10, template="plain")
    def test_empty_input(self):
        for texts in ([], [""], [42]):
            with self.assertRaises(ValueError):
                extract_features(ToyLLM(), ToyTokenizer(), texts, layers=[0], max_length=10, template="plain")
    def test_shapes_and_threshold(self):
        for width in (0, 4):
            head=build_predictor(4,width)
            with torch.no_grad():
                for p in head.parameters():p.zero_()
            x=torch.zeros(2,2,2)
            self.assertEqual(head(normalize(x,torch.zeros(4),torch.ones(4))).shape,(2,1))
            self.assertEqual(predict(head,x,torch.zeros(4),torch.ones(4)).tolist(),[.5,.5])
    def test_bad_predictor_dimensions(self):
        for dims in ((0,0),(2,-1),(True,3)):
            with self.assertRaises(ValueError):build_predictor(*dims)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.root_patch=patch.object(io,"ROOT",self.root);self.root_patch.start()
        self.context_patch=patch.object(io,"model_context",side_effect=context);self.context_patch.start()
        self.load_patch=patch.object(io,"load_backbone",side_effect=lambda *a:ToyLLM())
        self.loader=self.load_patch.start()
        self.train=self.root/'train.jsonl';self.valid=self.root/'validation.jsonl'
        self.write(self.train,[self.row(i) for i in range(12)])
        self.write(self.valid,[self.row(i) for i in range(20,24)])
    def tearDown(self):
        self.load_patch.stop();self.context_patch.stop();self.root_patch.stop();self.temp.cleanup()
    @staticmethod
    def row(i):
        return dict(id=str(i),family_id=f'g{i}',problem=f'{i} {9 if i%2 else 1}',is_robust=bool(i%2),model_id='toy')
    @staticmethod
    def write(path,rows):
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def args(self,name='E001'):
        return SimpleNamespace(train=self.train,validation=self.valid,model='toy',revision='main',name=name,
            layers=[0,1],max_length=20,template='plain',dtype='float32',device='cpu',batch_size=4,
            epochs=30,width=0,learning_rate=.05,threshold=.5,seed=7,max_seconds=30)
    def execute(self,args=None):
        args=args or self.args()
        with io.journal(args) as record:run(args,record)
        return record
    def test_full_training_and_files(self):
        r=self.execute()
        self.assertEqual(r['status'],'completed');self.assertEqual(r['accuracy'],1.0)
        self.assertEqual(r['majority_accuracy'],.5);self.assertEqual(r['scientific_status'],'unreviewed')
        self.assertLess(r['train_loss'][-1],r['train_loss'][0])
        self.assertEqual(r['shape'],[16,2,2]);self.assertTrue((self.root/'.runtime/E001/head.pt').is_file())
        self.assertEqual(json.loads((self.root/'experiments/E001/result.json').read_text()),r)
    def test_cache_reused_for_other_head(self):
        self.execute();a=self.args('E002');a.width=3;self.execute(a)
        self.assertEqual(self.loader.call_count,1)
    def test_feature_change_invalidates_cache(self):
        self.execute();a=self.args('E002');a.layers=[1];self.execute(a)
        self.assertEqual(self.loader.call_count,2)
    def test_normalizer_train_only(self):
        self.execute();p=torch.load(self.root/'.runtime/E001/head.pt',weights_only=True)
        torch.testing.assert_close(p['mean'],torch.tensor([5.,-5.,15.,5.]))
    def test_group_overlap_rejected_before_llm(self):
        rows=[self.row(20),self.row(21)];rows[0]['family_id']='g0';self.write(self.valid,rows)
        with self.assertRaisesRegex(ValueError,'family_id'):self.execute()
        self.loader.assert_not_called()
        self.assertEqual(json.loads((self.root/'experiments/E001/result.json').read_text())['status'],'failed')
    def test_identical_text_overlap(self):
        row=self.row(20);row['problem']='0 1';self.write(self.valid,[row])
        with self.assertRaisesRegex(ValueError,'problem'):self.execute()
    def test_label_and_family_required(self):
        for field,value in [('is_robust',1),('family_id',None)]:
            row=self.row(0);row[field]=value;self.write(self.train,[row])
            with self.assertRaises(ValueError):io.read_rows(self.train)
    def test_model_mismatch(self):
        with self.assertRaises(ValueError):io.load_splits(self.train,self.valid,'other')
    def test_prepare_grouped_not_rows(self):
        source=self.root/'source.jsonl';rows=[self.row(i) for i in range(30)]
        rows.extend(dict(self.row(i),id=f'{i}b',problem=f'{i} 2 {9 if i%2 else 1}') for i in range(30))
        self.write(source,rows)
        a=SimpleNamespace(source=source,out=self.root/'prepared',model='toy',seed=42,validation_fraction=.2)
        io.prepare(a)
        train,valid=io.load_splits(a.out/'train.jsonl',a.out/'validation.jsonl','toy')
        self.assertEqual(len(train)+len(valid),60)
        self.assertFalse({r['family_id'] for r in train}&{r['family_id'] for r in valid})
        with self.assertRaises(FileExistsError):io.prepare(a)
    def test_timeout_recorded(self):
        a=self.args();a.max_seconds=-1
        with self.assertRaises(TimeoutError):self.execute(a)
        self.assertEqual(json.loads((self.root/'experiments/E001/result.json').read_text())['error_type'],'TimeoutError')
        self.assertFalse((self.root/'.runtime/aimo.lock').exists())
    def test_no_overwrite(self):
        self.execute()
        with self.assertRaises(FileExistsError):self.execute()
    def test_lock_blocks(self):
        (self.root/'.runtime').mkdir();(self.root/'.runtime/aimo.lock').write_text('other')
        with self.assertRaises(FileExistsError):self.execute()
        self.assertEqual((self.root/'.runtime/aimo.lock').read_text(),'other')
    def test_interruption_recorded(self):
        with self.assertRaises(KeyboardInterrupt),io.journal(self.args()):raise KeyboardInterrupt()
        self.assertEqual(json.loads((self.root/'experiments/E001/result.json').read_text())['status'],'interrupted')
    def test_bundle_predictor_roundtrip(self):
        self.execute();p=self.root/'.runtime/E001'
        self.assertEqual(io.predict_bundle(p,'toy',['1 1','1 9']),[False,True])
        self.assertEqual(io.predict_bundle(p,'toy',[]),[])
        with self.assertRaises(ValueError):io.predict_bundle(p,'wrong',['1'])
    def test_export_has_no_private_inputs(self):
        self.execute();out=self.root/'solution.zip';io.export(SimpleNamespace(name='E001',output=out))
        with ZipFile(out) as z:
            names=z.namelist();self.assertIn('solution.py',names);self.assertIn('research/features.py',names)
            self.assertFalse(any('predictions' in n or 'train.jsonl' in n for n in names))
        with self.assertRaises(FileExistsError):io.export(SimpleNamespace(name='E001',output=out))
    def test_changed_head_refuses_export(self):
        self.execute();(self.root/'.runtime/E001/head.pt').write_bytes(b'bad')
        with self.assertRaises(ValueError):io.export(SimpleNamespace(name='E001',output=self.root/'s.zip'))
    def test_changed_source_refuses_export_without_partial_zip(self):
        self.execute();p=self.root/'.runtime/E001/source/features.py';p.write_text('changed')
        out=self.root/'broken.zip'
        with self.assertRaises(ValueError):io.export(SimpleNamespace(name='E001',output=out))
        self.assertFalse(out.exists())
    def test_cli_run_uses_the_same_pipeline(self):
        argv=['experiment','run','--train',str(self.train),'--validation',str(self.valid),
              '--model','toy','--name','CLI001','--layers','0','--template','plain','--epochs','2']
        with patch.object(sys,'argv',argv):io.main(run)
        r=json.loads((self.root/'experiments/CLI001/result.json').read_text())
        self.assertEqual(r['status'],'completed');self.assertEqual(len(r['train_loss']),2)
    def test_clean_room_solution_import_and_inference(self):
        self.execute();out=self.root/'solution.zip';io.export(SimpleNamespace(name='E001',output=out))
        room=self.root/'room';room.mkdir()
        with ZipFile(out) as z:z.extractall(room)
        # Only the external model provider is substituted; shipped imports/head/formulas are real.
        helpers=Path(__file__).read_text().split('class CoreTests')[0]
        (room/'toy_helpers.py').write_text(helpers)
        script="""from pathlib import Path
from research import _support as io
from toy_helpers import context, ToyLLM
io.model_context=context
io.load_backbone=lambda *a: ToyLLM()
import solution, research
assert Path(research.__file__).resolve().is_relative_to(Path.cwd())
assert solution.are_robust('toy',['1 1','1 9']) == [False,True]
assert solution.are_robust('toy',[]) == []
print('clean-room passed')
"""
        env=dict(os.environ,PYTHONPATH=str(room),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        p=subprocess.run([sys.executable,'-c',script],cwd=room,env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)
    def test_cli_help(self):
        p=subprocess.run([sys.executable,'-m','research.experiment','--help'],cwd=Path(__file__).parents[2],
                         capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr);self.assertIn('prepare',p.stdout)


class ExtraChecks(unittest.TestCase):
    def test_configuration_and_skills(self):
        import ast, re, tomllib
        root=Path(__file__).resolve().parents[2]
        cfg=tomllib.loads((root/'.codex/config.toml').read_text())
        self.assertEqual(cfg['model'],'gpt-6-astra');self.assertEqual(cfg['model_reasoning_effort'],'medium')
        expected={'researcher':('gpt-6-astra','high'),'worker':('gpt-5.6-terra','medium'),'explorer':('gpt-5.6-luna','low')}
        for name,(model,effort) in expected.items():
            role=tomllib.loads((root/f'.codex/agents/{name}.toml').read_text())
            self.assertEqual(role['name'],name);self.assertEqual((role['model'],role['model_reasoning_effort']),(model,effort))
            self.assertFalse(role['agents']['enabled']);self.assertTrue(role['developer_instructions'])
        for p in (root/'.agents/skills').glob('*/SKILL.md'):
            header=p.read_text().split('---',2)[1]
            self.assertIn(f'name: {p.parent.name}',header);self.assertIn('description:',header)
        for p in (root/'research').rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
        for p in [root/'README.md',root/'AGENTS.md',*(root/'.agents').rglob('*.md')]:
            for target in re.findall(r'\]\(([^)\s]+)\)',p.read_text()):
                if '://' not in target and not target.startswith('#'):
                    self.assertTrue((p.parent/target.split('#')[0]).exists(),f'{p}: {target}')
        self.assertLess((root/'AGENTS.md').stat().st_size,6000)

if __name__=='__main__':unittest.main()
