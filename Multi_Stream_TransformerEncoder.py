import torch.nn as nn
from torch.nn.modules import LayerNorm
from torch.nn.modules import ModuleList
import copy
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention
from torch.nn.modules import Linear,Dropout
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer


class Multi_Stream_TransformerEncoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, use_norm=True, layer_norm_eps: float = 1e-5, d_model=1024):
        super(Multi_Stream_TransformerEncoder, self).__init__()
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_norm = use_norm
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_e = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_t = LayerNorm(d_model, eps=layer_norm_eps)

    def _get_clones(self, module, N):
        # FIXME: copy.deepcopy() is not defined on nn.module
        return ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, src, mask = None, src_key_padding_mask = None, modality=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:

            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, modality=modality)
            if self.use_norm:
                if modality is None:
                    output = self.norm(output)
                elif modality == "e":
                    output = self.norm_e(output)
                elif modality == "t":
                    output = self.norm_t(output)
                # output = self.norm(output)
        return output


class Multi_Stream_TransformerEncoderLayer(nn.Module):

    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation= F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = True, layer_norm=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Multi_Stream_TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear1_e = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear1_t = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.linear2_e = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.linear2_t = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        # self.adaptive_rms_norm = adaptive_rms_norm
        self.layer_norm = layer_norm

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm1_e = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2_e = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm1_t = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2_t = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout = Dropout(dropout)
        self.dropout_e = Dropout(dropout)
        self.dropout_t = Dropout(dropout)

        self.dropout1 = Dropout(dropout)
        self.dropout1_e = Dropout(dropout)
        self.dropout1_t = Dropout(dropout)

        self.dropout2 = Dropout(dropout)
        self.dropout2_e = Dropout(dropout)
        self.dropout2_t = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(Multi_Stream_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask= None, modality=None, src_key_padding_mask= None):

        x = src
        if self.norm_first:
            if modality == None:
                x = x + self._sa_block(x=self.norm1(x), attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            elif modality == 'e':
                x = x + self._sa_block(x=self.norm1_e(x), attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
                x = x + self._ff_block(self.norm2_e(x))
                # print("come to the e")
            elif modality == 't':
                x = x + self._sa_block(x=self.norm1_t(x), attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
                x = x + self._ff_block(self.norm2_t(x))
        else:
            if modality == None:
                x = self.norm1(x + self._sa_block(x=x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,modality=modality))
                x = self.norm2(x + self._ff_block(x, modality=modality))
                # print("come to the none")
            elif modality == 'e':
                x = self.norm1_e(x + self._sa_block(x=x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,modality=modality))
                x = self.norm2_e(x + self._ff_block(x, modality=modality))
                # print("come to the e")
            elif modality == 't':
                x = self.norm1_t(x + self._sa_block(x=x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,modality=modality))
                x = self.norm2_t(x + self._ff_block(x, modality=modality))
                # print("come to the t")

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask,modality):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        # return self.dropout1(x)
        if modality == None:
            return self.dropout1(x)
        elif modality == 'e':
            return self.dropout1_e(x)
        elif modality == 't':
            return self.dropout1_t(x)

    # feed forward block
    def _ff_block(self, x, modality):
        if modality == None:
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            return self.dropout2(x)
        elif modality == 'e':
            x = self.linear2_e(self.dropout_e(self.activation(self.linear1_e(x))))
            return self.dropout2_e(x)
        elif modality == 't':
            x = self.linear2_t(self.dropout_t(self.activation(self.linear1_t(x))))
            return self.dropout2_t(x)
        # x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # return self.dropout2(x)

def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))