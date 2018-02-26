from torch import nn
from tamamo import nn as nnev
from torch.nn import functional as F
import torch
import tamamo 

"""
RNN generator
"""
def generator_rnn(config) :
    mod_type = config['type'].lower()
    mod_args = config.get('args', [])
    mod_kwargs = config.get('kwargs', {})

    if mod_type == 'lstm' :
        _lyr = nn.LSTM
    elif mod_type == 'gru' :
        _lyr = nn.GRU
    elif mod_type == 'rnn' :
        _lyr = nn.RNN
    elif mod_type == 'lstmcell' :
        _lyr = nn.LSTMCell
    elif mod_type == 'grucell' :
        _lyr = nn.GRUCell
    elif mod_type == 'rnncell' :
        _lyr = nn.RNNCell
    elif mod_type =='stateful_lstmcell' :
        _lyr = nnev.StatefulLSTMCell
    else :
        raise NotImplementedError("rnn class {} is not implemented/existed".format(mod_type))
    return _lyr(*mod_args, **mod_kwargs)

def generator_attention(config) :
    mod_type = config['type'].lower()
    mod_args = config.get('args', [])
    mod_kwargs = config.get('kwargs', {})
    if mod_type == 'bilinear' :
        _lyr = tamamo.custom.attention.BilinearAttention
    elif mod_type == 'mlp' :
        _lyr = tamamo.custom.attention.MLPAttention
    else :
        raise NotImplementedError()
    return _lyr(*mod_args, **mod_kwargs)

def generator_act_fn(name) :
    act_fn = None
    if name is None or name.lower() in ['none', 'null'] :
        act_fn = (lambda x : x)
    else :
        try :
            act_fn = getattr(F, name)
        except AttributeError :
            act_fn = getattr(torch, name)
    return act_fn
