import math
import random
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 1.Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super(PatchEmbedding, self).__init__()
        self.in_channels = in_channel
        self.out_channels = embed_dim
        self.patch_size = patch_size
        
        self.input_embedding = nn.Conv2d(
            in_channels=in_channel, out_channels=embed_dim, kernel_size=(self.patch_size, 1), stride=(self.patch_size, 1)
        )
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
    
    def forward(self, long_term_history):
        # long_term_history (torch.Tensor): (B, N, 1, L*P)
        # Here P is the number of segments(patches)
        B, N, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1)
        # (B, N, 1, L*P, 1) -> (B*N, 1, L*P, 1)
        long_term_history = long_term_history.reshape(B*N, num_feat, len_time_series, 1)
        # use conv2d to get the number of patches
        output = self.input_embedding(long_term_history)  # (B*N, 1, L, 1)
        # norm
        output = self.norm_layer(output)
        output = output.squeeze(-1).view(B, N, self.out_channels, -1)
        assert output.shape[-1] == len_time_series / self.patch_size
        return output


# 2. S-T Positional Encoding
class PositioanlEncoding(nn.Module):
    def __init__(self):
        super(PositioanlEncoding, self).__init__()

    def forward(self, input_data):
        # input_data shape: (B, N, T, D)
        # when D = 2i, pos = sin(t/10000^(4i/D)), when D=2i+1, pos = cos(t/10000^(4i/D))
        tp_enc_2d = PositionalEncoding2D(input_data.shape[-1])
        input_data += tp_enc_2d(input_data)
        return input_data, tp_enc_2d(input_data)


# 3. Mask Generator
# 根据给定torken总数和掩码比例，随机生成（0-1分布）需要掩码和保留的token索引，用于自监督学习
# 生成的这些索引可用于创建输入的mask和unmask数据，帮助模型通过剩余的信息重建被掩码的部分
class MaskGenerator(nn.Module):
    def __init__(self, num_tokens, mask_ratio):
        super(MaskGenerator, self).__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True
        
    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens
    
    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens


# 4.Transformer Layers
class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, num_layers, mlp_ratio, num_heads=4, dropout=0.1):
        super(TransformerLayers, self).__init__()
        # mlp_ratio: 控制每个TransformerLayer中前馈神经网络的扩展比例
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(self.d_model, num_heads, hidden_dim * mlp_ratio, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.contiguous()
        src = src.view(B * N, L, D)
        src = src.transpose(0, 1)  # (L, B * N, D)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output


if __name__ == '__main__':
    # 1. Test PatchEmbedding Module
    # layer = PatchEmbedding(patch_size=12, in_channel=1, embed_dim=96, norm_layer=None)
    # long_term_history = torch.randn(8, 307, 1, 864)
    # output = layer(long_term_history)
    # print(output.shape)  # torch.Size([8, 307, 96, 72])
    
    # 2. Test PositioanlEncodings Module
    # layer = PositioanlEncodings()
    # output, enc_embed_2d = layer(torch.randn(8, 307, 72, 96))
    # print(output.shape, enc_embed_2d.shape)  
    
    # 3.Test MaskGenerator Module
    layer = MaskGenerator(num_tokens=72, mask_ratio=0.25)
    unmasked_tokens, masked_tokens = layer.uniform_rand()
    print(len(unmasked_tokens), len(masked_tokens))