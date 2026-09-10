import unittest
import json
import signal
import torch
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from research._activation_store import Welford, bf16_bits, bits_bf16, completed_manifests, geometry_channels, plan_cases, require_cuda, write_completed
from research.features import segment_generation, stable_positions, output_statistics
from research._collector import collect_case, StreamingSelector, _output_row, HookCapture


class ActivationStoreTests(unittest.TestCase):
    class TinyChatTokenizer:
        chat_template = "tiny"
        eos_token_id = 0
        def _ids(self, text): return [1 + (ord(c) % 47) for c in text]
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            text = "U:" + messages[0]["content"] + "\\nA:"
            return self._ids(text) if tokenize else text
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            out = {"input_ids": [9,10] if text == "</think>" else self._ids(text)}
            if return_offsets_mapping: out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
            return out
        def decode(self, ids, skip_special_tokens=False): return "x" * len(ids)

    def test_bf16_bits_roundtrip(self):
        x = torch.tensor([1.25, -3.5], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(x, bits_bf16(bf16_bits(x))))

    def test_welford_matches_fp64(self):
        x = torch.tensor([[1., 4.], [2., 8.], [3., 12.]])
        a, b = Welford.empty(2), Welford.empty(2)
        a.update(x[:1]); b.update(x[1:]); a.merge(b)
        self.assertEqual(a.n, 3)
        torch.testing.assert_close(a.mean, x.double().mean(0).float())
        torch.testing.assert_close(a.m2, ((x-x.mean(0)).square().sum(0)))

    def test_plan_cases(self):
        original = {"tasks":[{"family_id":"f", "original_problem":"q"}]}
        variations = {"tasks":[{"family_id":"f", "variants":[{"variant_id":f"v{i:02d}","problem":f"q{i}"} for i in range(1,11)]}]}
        cases = plan_cases(original, variations)
        self.assertEqual(len(cases), 11)
        self.assertEqual(cases[0]["variant_id"], "original")

    def test_geometry_uses_actual_u_and_h(self):
        r=torch.tensor([[1.,0.]]); a=torch.tensor([[2.,0.]]); m=torch.tensor([[0.,3.]])
        u=torch.tensor([[2.5,0.]]); h=torch.tensor([[3.,4.]])
        got=geometry_channels(r,a,m,u,h)[0]
        self.assertEqual(got.numel(),13); self.assertEqual(float(got[8]),6.25); self.assertEqual(float(got[12]),20.)

    def test_all_geometry_channels_match_independent_formula(self):
        r=torch.tensor([[1.,2.]]); a=torch.tensor([[3.,5.]]); m=torch.tensor([[7.,11.]])
        u=torch.tensor([[13.,17.]]); h=torch.tensor([[19.,23.]])
        got=geometry_channels(r,a,m,u,h)[0].double()
        dot=lambda x,y: float((x.double()*y.double()).sum())
        expected=torch.tensor([dot(r,r),dot(a,a),dot(m,m),dot(r,a),dot(r,m),dot(a,m),dot(a+m,a+m),dot(h,h),dot(u,u),dot(r,u),dot(u,h),dot(r,h),dot(h-r,h-r)],dtype=torch.float64)
        torch.testing.assert_close(got, expected)

    def test_geometry_uses_rounded_bf16_residuals_for_all_channels(self):
        r=torch.tensor([[1.003,2.007]],dtype=torch.bfloat16); a=torch.tensor([[.333,.667]],dtype=torch.bfloat16)
        m=torch.tensor([[.111,.222]],dtype=torch.bfloat16); u=torch.tensor([[1.25,2.75]],dtype=torch.bfloat16); h=torch.tensor([[1.5,3.0]],dtype=torch.bfloat16)
        got=geometry_channels(r,a,m,u,h)[0]
        # Independent FP64 calculation uses the stored (rounded) u/h, not r+a.
        vals=[x.float().double() for x in (r,a,m,u,h)]; rr,aa,mm,uu,hh=vals
        dot=lambda x,y:(x*y).sum(-1)
        want=torch.stack((dot(rr,rr),dot(aa,aa),dot(mm,mm),dot(rr,aa),dot(rr,mm),dot(aa,mm),dot(aa+mm,aa+mm),dot(hh,hh),dot(uu,uu),dot(rr,uu),dot(uu,hh),dot(rr,hh),dot(hh-rr,hh-rr)),-1)[0].float()
        torch.testing.assert_close(got,want); self.assertFalse(torch.equal(u.float(),(r.float()+a.float())))

    def test_tiny_qwen_cache_matches_full_prefix_and_last_hook(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg=Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
        model=Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16).eval(); prefix=torch.tensor([[3,4,5]])
        with torch.inference_mode(), HookCapture(model) as hooks:
            first=model(input_ids=prefix,use_cache=True,return_dict=True); hooks.take()
            cached=model(input_ids=torch.tensor([[6]]),past_key_values=first.past_key_values,use_cache=True,return_dict=True); cached_hook=hooks.take()[35]["h"][0,0].float()
        with torch.inference_mode(), HookCapture(model) as hooks:
            full=model(input_ids=torch.tensor([[3,4,5,6]]),use_cache=False,return_dict=True); full_hook=hooks.take()[35]["h"][0,-1].float()
        torch.testing.assert_close(cached.logits[0,-1].float(),full.logits[0,-1].float(),rtol=2e-2,atol=2e-2)
        torch.testing.assert_close(cached_hook,full_hook,rtol=2e-2,atol=2e-2)

    def test_marker_eos_phase_moments_match_fp64_offline(self):
        # multi-token marker positions and EOS are service, so only 1,4 enter R/A.
        ids=[1,9,10,4,0]; roles,status=segment_generation(ids,[9,10],eos_id=0)
        self.assertEqual((roles,status),(['R','service','service','A','service'],'confirmed'))
        rows=torch.tensor([[1.,2.],[9.,9.],[10.,10.],[4.,8.],[0.,0.]])
        r,a=rows[[0]].double(),rows[[3]].double()
        for row in (r,a):
            w=Welford.empty(2); w.update(row.float()); torch.testing.assert_close(w.mean,row.mean(0).float()); torch.testing.assert_close(w.m2,torch.zeros(2))

    def test_collect_case_forced_marker_eos_persists_phase_fp64_moments(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg=Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
        model=Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16).eval(); args=SimpleNamespace(max_prompt_tokens=32,max_new_tokens=7,do_sample=False,temperature=1.,selection_seed=1)
        tokens=[1,2,9,10,3,4,0]; it=iter(tokens)
        case={"case_id":"f:original","family_id":"f","variant_id":"original","text":"q"}
        with tempfile.TemporaryDirectory() as d, patch("research._collector.torch.argmax",side_effect=lambda _:torch.tensor(next(it))):
            path=collect_case(model,self.TinyChatTokenizer(),case,args,{"model":"tiny"},Path(d)/"x.h5",is_raw=True)
            import h5py
            with h5py.File(path) as h:
                self.assertEqual(h["phase_h_n"][:3].tolist(),[1,2,2])
                self.assertEqual(h["generated_ids"][:].tolist(),tokens)
                raw=bits_bf16(torch.from_numpy(h["raw_g_h_bf16"][:])).double()
                pos=h["raw_g_positions"][:].tolist(); by={p:raw[i] for i,p in enumerate(pos)}
                for phase,indices in ((1,[0,1]),(2,[4,5])):
                    rows=torch.stack([by[p] for p in indices]); mean=rows.mean(0); m2=((rows-mean).square()).sum(0)
                    torch.testing.assert_close(torch.from_numpy(h["phase_h_mean"][phase]).reshape(36,16),mean.float(),rtol=2e-3,atol=2e-3)
                    torch.testing.assert_close(torch.from_numpy(h["phase_h_m2"][phase]).reshape(36,16),m2.float(),rtol=2e-3,atol=2e-3)
                    self.assertGreater(float(m2.sum()),0.)

    def test_collect_sigterm_restores_previous_handler(self):
        from research import experiment
        old=signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); original={"tasks":[{"family_id":"f","original_problem":"q"}]}; variations={"tasks":[{"family_id":"f","variants":[{"variant_id":f"v{i:02d}","problem":"q"} for i in range(1,11)]}]}
            (root/'o.json').write_text(json.dumps(original)); (root/'v.json').write_text(json.dumps(variations))
            args=SimpleNamespace(originals=root/'o.json',variations=root/'v.json',out=root/'out',dtype='bfloat16',device='cuda',allow_download=False,cpu_offload=False,gpu_memory_gib=None,offload_folder=root/'offload')
            tokenizer=SimpleNamespace(chat_template='x')
            def fail(*a,**k):
                self.assertIsNot(signal.getsignal(signal.SIGTERM),old); raise RuntimeError('stop')
            with patch("research.experiment.require_cuda"), patch("research.experiment.io.model_context",return_value=(tokenizer,{"dtype":"bfloat16"})), patch("research.experiment.io.load_backbone",side_effect=fail):
                with self.assertRaises(RuntimeError): experiment.collect(args)
        self.assertIs(signal.getsignal(signal.SIGTERM),old)

    def test_production_output_statistics_uses_full_vocab_and_temperature(self):
        logits=torch.linspace(-3,4,31); chosen=7
        ids, values, metrics=_output_row(logits,chosen,False,.7)
        z=logits.double(); logz=torch.logsumexp(z,0); logp=z-logz
        self.assertTrue(torch.equal(ids[:20].to(torch.long),torch.topk(logits,20).indices))
        torch.testing.assert_close(values[:20],torch.topk(logits,20).values)
        expected=torch.tensor([logz, -(logp.exp()*logp).sum(), logp[chosen], torch.log_softmax(z/.7,0)[chosen]],dtype=torch.float32)
        torch.testing.assert_close(metrics,expected)

    def test_streaming_selector_has_fixed_vector_bound(self):
        selector=StreamingSelector(seed=1,run_id="x",raw=True)
        state={i:{"h":torch.zeros(1,dtype=torch.bfloat16),"a":torch.zeros(1,dtype=torch.bfloat16),"m":torch.zeros(1,dtype=torch.bfloat16)} for i in range(36)}
        for p in range(10000): selector.add({"position":p,"states":state,"geometry":torch.zeros(36*13)})
        self.assertLessEqual(selector.vector_frames_held,112)
        self.assertLessEqual(len(selector.sparse()),32); self.assertLessEqual(len(selector.raw_frames()),96)

    def test_streaming_sparse_selection_matches_offline_contract(self):
        selector=StreamingSelector(seed=7,run_id="same",raw=False)
        state={i:{"h":torch.zeros(1,dtype=torch.bfloat16),"a":torch.zeros(1,dtype=torch.bfloat16),"m":torch.zeros(1,dtype=torch.bfloat16)} for i in range(36)}
        for p in range(1000): selector.add({"position":p,"states":state,"geometry":torch.zeros(36*13)})
        self.assertEqual([frame["position"] for frame,_ in selector.sparse()], [p for p,_ in stable_positions(range(1000),seed=7,run_id="same")])

    def test_sparse_positions_and_logits(self):
        selected=stable_positions(list(range(100)),seed=7,run_id='x')
        self.assertLessEqual(len(selected),32); self.assertEqual(selected[0][0],0); self.assertEqual(selected[-1][0],99)
        row=output_statistics(torch.tensor([1.,2.,3.]),2,greedy=True)
        self.assertEqual(row.numel(),10); self.assertEqual(float(row[-1]),0.)

    def test_marker_segmentation(self):
        self.assertEqual(segment_generation([1,9,10,2],[9,10])[0],['R','service','service','A'])
        self.assertEqual(segment_generation([1,2,0],[9],eos_id=0),(['unknown','unknown','service'],'absent'))
        self.assertEqual(segment_generation([1,9,10,2,9,10,0],[9,10],eos_id=0),(['unknown','service','service','unknown','service','service','service'],'ambiguous'))

    def test_cpu_collector_refuses_before_model(self):
        with patch('research._activation_store.torch.version.cuda', None), patch('research._activation_store.torch.cuda.is_available', return_value=False):
            with self.assertRaises(RuntimeError): require_cuda()

    def test_partial_and_corrupt_payload_never_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=write_completed(root/'one.h5',{'id':'x'},{'x':torch.tensor([1,2])})
            self.assertEqual(list(completed_manifests(root)),[{'id':'x'}])
            path.with_suffix('.partial.h5').write_bytes(path.read_bytes())
            self.assertEqual(list(completed_manifests(root)),[{'id':'x'}])
            import h5py
            with h5py.File(path,'a') as h: del h['x']
            with self.assertRaises(ValueError): list(completed_manifests(root))

    def test_tiny_real_qwen_hooks_write_resumable_archive(self):
        """This uses the real Qwen module tree, not a mocked hook target."""
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=36,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
        model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        args = SimpleNamespace(max_prompt_tokens=32, max_new_tokens=3, do_sample=False, temperature=1.0,
                               selection_seed=7)
        case = {"case_id": "f:original", "family_id": "f", "variant_id": "original", "text": "2+2?"}
        with tempfile.TemporaryDirectory() as d:
            path = collect_case(model, self.TinyChatTokenizer(), case, args, {"model":"tiny"}, Path(d) / "one.h5", is_raw=True)
            self.assertTrue(path.exists())
            import h5py
            with h5py.File(path) as h:
                self.assertEqual(h["anchors_h_bf16"].shape, (3, 36, 16))
                self.assertEqual(h["phase_geometry_mean"].shape, (4, 36 * 13))
                self.assertEqual(h["phase_h_mean"].shape, (4, 36 * 16))
                self.assertLessEqual(h["sparse_h_bf16"].shape[0], 32)
                self.assertEqual(h["raw_q_h_bf16"].shape[1:], (36, 16))
                self.assertEqual(h["raw_g_h_bf16"].shape[1:], (36, 16))
                self.assertEqual(h["prompt_geometry"].shape[0], len(h["prompt_ids"]))
                self.assertEqual(h["generated_geometry"].shape[0], len(h["generated_ids"]))
                self.assertEqual(int(h["phase_h_n"][0]), int(h["q_mask"][:].sum()))
                self.assertGreaterEqual(int(h["phase_h_n"][3]), 1)  # no </think> => G_unsegmented
                self.assertEqual(int(h["anchor_present"][2]), 1)  # final emitted token was processed
                self.assertLessEqual(json.loads(h.attrs["manifest"])["peak_vector_frames"],112)
            self.assertEqual([m["case_id"] for m in completed_manifests(Path(d))], ["f:original"])


if __name__ == "__main__":
    unittest.main()
