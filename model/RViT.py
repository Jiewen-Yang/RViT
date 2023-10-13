import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

#from visualizer import get_local
#get_local.activate()

MIN_NUM_PATCHES = 16

table = [
            # t, c, n, s, SE
            [1,  24,  2, 1, 0],
            [4,  48,  4, 2, 0],
            [4,  64,  4, 2, 0],
            [4, 128,  6, 2, 1],
            #[6, 160,  9, 1, 1],
            #[6, 256, 15, 2, 1],
        ]


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(input_channel, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output_channel),
        SiLU()
    )


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim / heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_qkv_h = nn.Linear(dim, dim * 3)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))
        self.reattn_norm = nn.Sequential(
                            Rearrange('b h i j -> b i j h'),
                            nn.LayerNorm(heads),
                            Rearrange('b i j h -> b h i j'))

        self.elu = nn.ELU(alpha=1)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),)
    
    def forward(self, x, hidden):
        b, n, _, h = *x.shape, self.heads

        qkv = torch.add(self.to_qkv(x), self.to_qkv_h(hidden)).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        #out = self.linear_attn(self._elu(q), self._elu(k), v)
        #out = self.re_attn(q, k, v)
        out = self.softmax_attn(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

    def _elu(self, x):
        return self.elu(x) + 1
    
    #@get_local('attn')
    def softmax_attn(self, q, k, v):
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out

    def linear_attn(self, q, k, v):
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out

    def re_attn(self, q, k, v):
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = torch.einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)
        return torch.einsum('b h i j, b h j d -> b h i d', attn, v)


class Encoder1DBlock(nn.Module):
    def __init__(self, input_shape, heads, mlp_dim, dtype=torch.float32, dropout_rate=0.1, attention_dropout_rate=0.1, deterministic=True, rnn_layer = True):
        super().__init__()

        self.rnn_layer = rnn_layer
        self.layer_norm_input = nn.LayerNorm(input_shape)
        self.layer_norm_hidden = nn.LayerNorm(input_shape)
        self.layer_norm_out = nn.LayerNorm(input_shape)

        self.attention = MultiHeadDotProductAttention(input_shape, heads = heads)
        self.mlp = FeedForward(input_shape, mlp_dim, dropout_rate)
        self.drop_out_attention  = nn.Dropout(attention_dropout_rate)

    def forward(self, x, h):
        
        residual_x, residual_h = x, h
        attn = self.attention(self.layer_norm_input(x), self.layer_norm_hidden(h))
        x = self.drop_out_attention(attn) + residual_x
        
        residual_x = x
        x = self.layer_norm_out(x)
        x = self.mlp(x)
        x += residual_x

        return x, attn + residual_h


class Encoder(nn.Module):
    def __init__(self, input_shape, num_layers, heads, mlp_dim, dropout_rate=0.1):
        super(Encoder, self).__init__()
        # encoder blocks
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([Encoder1DBlock(input_shape, heads, mlp_dim)]))

    def forward(self, x, hidden, pos_embedding):
        h = hidden.clone()
        for i, layer in enumerate(self.layers):
            x, h[i] = layer[0](self.dropout(x         + pos_embedding),
                                           (hidden[i] + pos_embedding),)

        return x, h


class RViT(nn.Module):
    """ Vision Transformer """
    def __init__(self, *, image_size, patch_size, num_classes, depth, length, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super(RViT, self).__init__()
        
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        hidden_size = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.length = length

        self.embedding = nn.Conv2d(channels, hidden_size, patch_size, patch_size)

        # positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        
        # class token
        self._cls = nn.Parameter(torch.randn(1, 1, hidden_size))
        self._seq = nn.Parameter(torch.randn(depth, 1, 1, hidden_size))
        self.dropout = nn.Dropout(emb_dropout)
        
        # transformer
        self.transformer = Encoder(hidden_size, depth, heads, mlp_dim, dropout_rate = dropout)
        
        # classfier
        self.to_cls_token = nn.Identity()

        self.fc = FC_class(hidden_size * (depth+1), num_classes)

    def forward(self, vid, hidden, labels=None, validation=False):
        class_token = None
        emb = self.embedding(vid)     # (n, c, gh, gw)
        emb = rearrange(emb, 'b c h w  -> b (h w) c')
        b, n, c = emb.shape

        # prepend cls token
        _cls = repeat(self._cls, '() n d -> b n d', b = b)
        _seq = repeat(self._seq, 'l () n d -> l b n d', b = b // self.length)
        emb = torch.cat((_cls, emb), dim = 1)
        emb = emb.reshape(b//self.length, self.length, -1, c)
        hidden = torch.cat((_seq, hidden), dim = 2)
        # transformer
        for i in range(self.length):
            feat, hidden = self.transformer(emb[:, i], hidden, self.dropout(self.pos_embedding))

            feat_t = hidden[:, :, 0].transpose(0,1).contiguous()
            feat_t = feat_t.view(feat_t.size(0), -1)
            feat_st = torch.cat((feat[:, 0], feat_t), dim=-1)

        cls, embedding = self.fc(self.dropout(self.to_cls_token(feat_st)))
        print(cls.shape)
        return cls

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1)


class FC_class(nn.Module):
    def __init__(self, input_dim, class_num):
        super(FC_class, self).__init__()
        self.mlp = FeedForward(input_dim, input_dim * 4, 0.1)
        self.class_ = nn.Linear(input_dim, class_num)

    def forward(self, embedding):
        mlp_embedding = self.mlp(embedding)
        return self.class_(mlp_embedding), mlp_embedding


def RViT_P8_L16_112(**kwargs):
    input_size = 112
    patch_size = 8
    num_layers = 2
    num_classes = 400
    length = 16
    if 'num_classes' in kwargs:
        num_classes = kwargs['num_classes']

    return RViT(
        image_size = input_size,
        patch_size = patch_size,
        num_classes = num_classes,
        depth = num_layers,
        length = length,
        heads = 8,
        mlp_dim = patch_size ** 2 * 3 * 4,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    

if __name__ == '__main__':
    import torch
    using_device = 'cuda:0'
    input_size = 112

    v = RViT_P8_L16_112().cuda(using_device)

    vid = torch.randn(1, 16, 3, input_size, input_size).cuda(using_device)
    hidden = torch.zeros(2, 1, 196, 192).cuda(using_device)
    label = torch.ones(1, 1).long().cuda(using_device)
    '''
    model_dict = v.state_dict()
    modelCheckpoint = torch.load('imagenet21k+imagenet2012_ViT-B_16-224.pth')
    new_dict = {k: v for k, v in modelCheckpoint.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    print('Total : {}, update: {}'.format(len(modelCheckpoint), len(new_dict)))
    v.load_state_dict(model_dict)
    print("loaded finished!")
    # v.load_state_dict(torch.load('imagenet21k+imagenet2012_ViT-B_16-224.pth'))
    '''
    # from thop import profile, clever_format
    # sum_ = 0
    # for name, param in v.named_parameters():
    #     mul = 1
    #     for size_ in param.shape:
    #         mul *= size_                            
    #     sum_ += mul                                 
    #     print('%14s : %s, %1.f' % (name, param.shape, mul))    
    #     # print('%s' % param)
    # print('Parameters Numbers：', sum_)

    preds = v(vid.view(-1, 3, input_size, input_size), hidden, label) # (1, 1000)
    print(preds.shape)