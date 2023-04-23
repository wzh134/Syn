#%%
import torch
import torch.nn as nn

from torch.nn import Module, Linear
from torch.nn.init import xavier_uniform_
from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Dropout, Linear, Module, ReLU, Sequential, Sigmoid

#%%
class Syn(Module):
    def __init__(self):
        super().__init__()
        self.gamma = 0.5
    def forward(self, input):
        x = input.contiguous()
        n = torch.sqrt(torch.sum(input**2, dim=-1, keepdim=True))
        input = input / n
        D = 2.0
        output = input + self.gamma*input*(1.0-D+input**2)
        return x + (output-x).detach()

class SynReverse(Module):
    def __init__(self):
        super().__init__()
        self.gamma = 0.5
    def forward(self, input):
        x = input.contiguous()
        n = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))
        input = input / n
        b = input/(2*self.gamma)
        a = ((b**2+((1-self.gamma)/(3*self.gamma))**3)**0.5+b)**(1/3)
        output = a + (self.gamma-1)/(3*self.gamma*a)
        return x + (output-x).detach()

class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)

class SynPool(Module):
    def __init__(self, input_dim, embed_dim, output_dim, steps, num_heads, scaling=1.0, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.steps = steps
        self.num_heads = num_heads
        self.scaling = scaling
        self.dropout = Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_dim)
        self.in_proj_kv = Linear(input_dim, 2*embed_dim)
        self.in_proj_q = Linear(input_dim, embed_dim)

        self.q = nn.Parameter(torch.empty(size=(1,input_dim)), requires_grad=True)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, output_dim)
        self.syn = Syn() if steps >= 0 else SynReverse()
        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.in_proj_kv.weight)
        nn.init.constant_(self.in_proj_kv.bias, 0.0)
        xavier_uniform_(self.in_proj_q.weight)
        nn.init.constant_(self.in_proj_q.bias, 0.0)
        xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        xavier_uniform_(self.q)

    def forward(self, input):
        data = input.transpose(0,1)
        bsz = data.shape[1]
        data = self.layernorm(data.reshape(shape=(-1, data.shape[2]))).reshape(shape=data.shape)
        k, v = data, data
        q = self.q.expand(bsz,-1).unsqueeze(0)
        tgt_len = q.shape[0]
        head_dim = self.embed_dim // self.num_heads

        k, v = self.in_proj_kv(k).chunk(2, dim=-1)
        q = self.in_proj_q(q)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)

        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = nn.functional.softmax(attn, dim=-1)

        for _ in range(self.steps):
            attn = self.syn(attn)

        attn = self.dropout(attn)

        output = torch.bmm(attn, v)
        output = output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        output = self.out_proj(output)
        return output, attn

class Net(Module):    
    def __init__(self, embed_layer_num=1, input_dim=166, hidden_dim=32, embed_dim=32, steps=0, num_heads=2, scaling=1.0):
        super().__init__()
        self.dropout1 = Dropout(0.75)
        blocks = [Linear(input_dim, hidden_dim), ReLU()]
        for _ in range(embed_layer_num-1):
            blocks += [Linear(hidden_dim, hidden_dim), ReLU()]
        self.feature_extractor = Sequential(*blocks)
        self.pooling = SynPool(hidden_dim, embed_dim, hidden_dim, steps=steps, num_heads=num_heads, scaling=scaling)
        self.classifier = Sequential(
            Linear(hidden_dim, 1),
            Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        x = x.squeeze(0)
        x = self.dropout1(x)
        x = self.feature_extractor(x)
        x = x.unsqueeze(0)
        P, A = self.pooling(x)
        H = P.squeeze(0)
        Y = self.classifier(H)
        return Y, A

    def calculate_classification_error(self, input: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        Y = target.float()
        Y_prob, _ = self.forward(input)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, input: Tensor, target: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        Y = target.float()
        Y_prob, A = self.forward(input)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=(1.0 - 1e-5))
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob))

        return neg_log_likelihood, A

#%%