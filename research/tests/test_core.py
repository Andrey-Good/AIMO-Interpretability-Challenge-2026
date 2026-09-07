"""CPU mathematical/contract tests. FakeLLM is NOT the contest checkpoint."""
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from research.features import extract_features
from research.predictor import build_predictor
from research.method import predict
from research.train import fit_predictor


class FakeLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, input_ids, attention_mask, **kwargs):
        assert kwargs == dict(output_hidden_states=True, return_dict=True, use_cache=False)
        x = input_ids.float().unsqueeze(-1) * self.scale
        return SimpleNamespace(hidden_states=(x.expand(-1,-1,2), (x+10).expand(-1,-1,2)))


def batch():
    return {"input_ids": torch.tensor([[1,2,0],[0,3,4]]), "attention_mask": torch.tensor([[1,1,0],[0,1,1]])}


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.llm = FakeLLM()
    def test_left_and_right_padding(self):
        x=extract_features(self.llm,batch(),[0,1])
        torch.testing.assert_close(x,torch.tensor([[[2.,2.],[12.,12.]],[[4.,4.],[14.,14.]]]))
    def test_order_and_batch_invariance(self):
        b=batch();x=extract_features(self.llm,b,[1,0])
        for i in range(2):
            one={k:v[i:i+1] for k,v in b.items()}
            torch.testing.assert_close(extract_features(self.llm,one,[1,0]),x[i:i+1])
    def test_llm_mode_restored_and_no_gradient(self):
        self.llm.train();x=extract_features(self.llm,batch(),[1])
        self.assertTrue(self.llm.training);self.assertFalse(x.requires_grad)
        self.assertIsNone(self.llm.scale.grad)
    def test_all_padding_rejected(self):
        b=batch();b['attention_mask'][0]=0
        with self.assertRaises(ValueError):extract_features(self.llm,b,[0])
    def test_nonbinary_mask(self):
        b=batch();b['attention_mask'][0,0]=2
        with self.assertRaises(ValueError):extract_features(self.llm,b,[0])
    def test_label_injection_rejected(self):
        b=batch();b['labels']=torch.tensor([True,False])
        with self.assertRaises(ValueError):extract_features(self.llm,b,[0])
    def test_layer_bounds(self):
        for layers in ([],[-1],[2],[0,0],[False]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):extract_features(self.llm,batch(),layers)
    def test_empty_batch(self):
        with self.assertRaises(ValueError):extract_features(self.llm,{k:v[:0] for k,v in batch().items()},[0])
    def test_bad_token_dtype(self):
        b=batch();b['input_ids']=b['input_ids'].float()
        with self.assertRaises(ValueError):extract_features(self.llm,b,[0])
    def test_predictor_shapes(self):
        for width in (0,3):self.assertEqual(build_predictor(2,2,width)(torch.zeros(3,2,2)).shape,(3,1))
    def test_predictor_dimensions(self):
        for dims in ((0,2,0),(2,0,0),(2,2,-1),(2,True,0)):
            with self.assertRaises(ValueError):build_predictor(*dims)
    def test_threshold_equality(self):
        head=build_predictor(2,1)
        with torch.no_grad():
            for p in head.parameters():p.zero_()
        self.assertEqual(predict(self.llm,head,batch(),[0]),[True,True])
        self.assertTrue(head.training)
    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):predict(self.llm,build_predictor(2,1),batch(),[0],float('nan'))
    def test_nonfinite_features(self):
        with torch.no_grad():self.llm.scale.fill_(float('nan'))
        with self.assertRaises(ValueError):extract_features(self.llm,batch(),[0])
    def test_training_learns_toy_separation(self):
        x=torch.tensor([-2.,-1.,1.,2.]).reshape(4,1,1).requires_grad_()
        y=torch.tensor([0.,0.,1.,1.]);head=build_predictor(1,1)
        losses=fit_predictor(head,x,y,epochs=40,batch_size=4,learning_rate=.1,seed=4)
        self.assertLess(losses[-1],losses[0]);self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is not None for p in head.parameters()))
        self.assertEqual((head(x).detach().squeeze()>0).tolist(),[False,False,True,True])
    def test_train_invalid_labels(self):
        with self.assertRaises(ValueError):fit_predictor(build_predictor(1,1),torch.ones(2,1,1),torch.tensor([0.,2.]))


if __name__=='__main__':unittest.main()
