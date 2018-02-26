import torch
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
from torch.nn import init
import numpy as np
from ..utils.helper import torchauto, tensorauto
from ..generator import generator_act_fn

class BaseAttention(Module) :
    NEG_NUM = -10000.0
    def __init__(self) :
        super(BaseAttention, self).__init__()
        self.state = None

    def apply_mask(self, score, mask, mask_value=NEG_NUM) :
        # TODO inplace masked_fill_
        return score.masked_fill(mask == 0, mask_value)

    def calc_expected_context(self, p_ctx, ctx) :
        """
        p_ctx = (batch x srcL)
        ctx = (batch x srcL x dim)
        """
        p_ctx_3d = p_ctx.unsqueeze(1) # (batch x 1 x enc_len)
        expected_ctx = torch.bmm(p_ctx_3d, ctx).squeeze(1) # (batch x dim)
        return expected_ctx

    def reset(self) :
        self.state = None
        pass


class BilinearAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size) :
        super(BilinearAttention, self).__init__()
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.W = nn.Linear(query_size, ctx_size, bias=False)
        self.out_features = self.ctx_size
        pass

    def forward(self, input) :
        ctx = input['ctx'] # batch x enc_len x enc_dim #
        query = input['query'] # batch x dec_dim #
        mask = input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()
        score_ctx = ctx.bmm(self.W(query).unsqueeze(2))
        score_ctx = score_ctx.squeeze(2)
        if mask is not None :
            score_ctx = self.apply_mask(score_ctx, mask)
        p_ctx = F.softmax(score_ctx, dim=-1)
        expected_ctx = self.calc_expected_context(p_ctx, ctx)
        return expected_ctx, p_ctx

    def __call__(self, *input, **kwargs) :
        result = super(BilinearAttention, self).__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return {"p_ctx":p_ctx, "expected_ctx":expected_ctx}

class MLPAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size, att_hid_size=256, act_fn=F.tanh) :
        super(MLPAttention, self).__init__()
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.att_hid_size = att_hid_size
        self.act_fn = act_fn
        self.lin_in2proj = nn.Linear(ctx_size + query_size, att_hid_size)
        self.lin_proj2score = nn.Linear(att_hid_size, 1)
        self.out_features = self.ctx_size
        pass

    def forward(self, input) :
        ctx = input['ctx'] # batch x enc_len x enc_dim #
        query = input['query'] # batch x dec_dim #
        mask = input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()
        
        combined_input = torch.cat([ctx, query.unsqueeze(1).expand(batch, enc_len, self.query_size)], 2) # batch x enc_len x (enc_dim + dec_dim) #
        combined_input_2d = combined_input.view(batch * enc_len, -1)
        score_ctx = self.lin_proj2score(self.act_fn(self.lin_in2proj(combined_input_2d)))
        score_ctx = score_ctx.view(batch, enc_len) # batch x enc_len #
        if mask is not None :
            score_ctx = self.apply_mask(score_ctx, mask)
        p_ctx = F.softmax(score_ctx, dim=-1)
        expected_ctx = self.calc_expected_context(p_ctx, ctx)
        return expected_ctx, p_ctx
        pass

    def __call__(self, *input, **kwargs) :
        result = super(MLPAttention, self).__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return {"p_ctx":p_ctx, "expected_ctx":expected_ctx}
    pass

