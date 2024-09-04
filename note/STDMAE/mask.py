import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_
from modules import PatchEmbedding, MaskGenerator, PositioanlEncoding, TransformerLayers


class Mask(nn.Module):
    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, mask_ratio,
                 encoder_depth, decoder_depth, spatial=False, mode='pre-train'):
        super(Mask, self).__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode!"
        self.mode = mode
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.mask_ratio = mask_ratio
        self.dropout = dropout
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.spatial = spatial
        self.selected_feature = 0
        
        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat = None
        # encoder specifics
        # 1.Patch Embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # 2.S-T Positional Encoding
        self.positional_encoding = PositioanlEncoding()
        # 3.Encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
        # decoder specifics
        self.decoder = TransformerLayers(embed_dim, self.decoder_depth, mlp_ratio, num_heads, dropout)
        
        # predector layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()
    
    
    def initialize_weights(self):
        trunc_normal_(self.mask_token, std=.02)
    
        
    def encoding(self, long_term_history, mask=True):
        # long_term_history shape: (B, N, C, P * L)
        # mask (bool): True in pre-training stage and False in forecasting
        if mask:
            patches = self.patch_embedding(long_term_history)
            patches = patches.transpose(-1, -2)  # (B, N, L, d)
            batch_size, num_nodes, num_time, _ = patches.shape
            patches, self.pos_mat = self.positional_encoding(patches)
            # S-MAE
            if self.spatial:
                Maskg = MaskGenerator(patches.shape[1], self.mask_ratio)
                unmasked_token_index, masked_token_index = Maskg.uniform_rand()
                encoder_input = patches[:, unmasked_token_index, :, :]
                encoder_input = encoder_input.transpose(1, 2)  # (B, L, Nr, d)
                hidden_state_unmasked = self.encoder(encoder_input)
                hidden_state_unmasked = self.encoder_norm(hidden_state_unmasked).view(batch_size, num_time, -1, self.embed_dim)
            # T-MAE
            if not self.spatial:
                Maskg = MaskGenerator(patches.shape[2], self.mask_ratio)
                unmasked_token_index, masked_token_index = Maskg.uniform_rand()
                encoder_input = patches[:, :, unmasked_token_index, :] # (B, N, Lr, d)
                hidden_state_unmasked = self.encoder(encoder_input)
                hidden_state_unmasked = self.encoder_norm(hidden_state_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
        else:
            batch_size, num_nodes, _, _ = long_term_history.shape
            patches = self.patch_embedding(long_term_history)
            patches = patches.transpose(-1, -2)  # (B, N, L, d)
            # positional embedding
            patches, self.pos_mat = self.positional_encoding(patches)
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches  # (B, N, L, d)
            if self.spatial:
                encoder_input = encoder_input.transpose(1, 2)
            hidden_state_unmasked = self.encoder(encoder_input)
            if self.spatial:
                hidden_state_unmasked = hidden_state_unmasked.transpose(1, 2)
            hidden_state_unmasked = self.encoder_norm(hidden_state_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
        
        return hidden_state_unmasked, unmasked_token_index, masked_token_index
    
    
    def decoding(self, hidden_states_unmasked, masked_token_index):
        """
            hidden_states_unmasked shape:
                - SMAE: (B, L, Nr, D), Nr = N * (1 - r)
                - TMAE: (B, N, Lr, D), Lr = L * (1 - r)
        """
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)
        if self.spatial:
            batch_size, num_time, unmasked_num_nodes, _ = hidden_states_unmasked.shape
            unmasked_token_index = []
            for i in range(0, len(masked_token_index) + unmasked_num_nodes):
                if i not in masked_token_index:
                    unmasked_token_index.append(i)
            hidden_states_masked = self.pos_mat[:, masked_token_index, :, :]  # (B, N*r, L, d)
            hidden_states_masked = hidden_states_masked.transpose(1, 2) # (B, L, N*r, d)
            
            hidden_states_masked += self.mask_token.expand(batch_size, num_time, len(masked_token_index), hidden_states_unmasked.shape[-1])
            hidden_states_unmasked += self.pos_mat[:, unmasked_token_index, :, :].transpose(1, 2)  # (B, L, N*(1-r), d)
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=2)  # (B, L, N, d)
            
            # decoding
            hidden_states_full = self.decoder(hidden_states_full)  # (B, L, N, d)
            hidden_states_full = self.decoder_norm(hidden_states_full)
            # prediction (reconstruction)
            reconstruction_full = self.output_layer(hidden_states_full)  # (B, L, N, P)
        else:
            batch_size, num_nodes, unmasked_num_time, _ = hidden_states_unmasked.shape
            unmasked_token_index = []
            for i in range(0, len(masked_token_index) + unmasked_num_time):
                if i not in masked_token_index:
                    unmasked_token_index.append(i)
            hidden_states_masked = self.pos_mat[:, :, masked_token_index, :]       # (B, N, L*r, d)
            hidden_states_masked += self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1])
            hidden_states_unmasked += self.pos_mat[:, :, unmasked_token_index, :]  # (B, N, L*(1-r), d)
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=2)  # (B, N, L, d)
            
            # decoding
            hidden_states_full = self.decoder(hidden_states_full)  # (B, N, L, d)
            hidden_states_full = self.decoder_norm(hidden_states_full)
            # prediction (reconstruction)
            reconstruction_full = self.output_layer(hidden_states_full)  # (B, N, L, P)
        return reconstruction_full
        
    
    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        """
        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens, shape is (B, N, L, P) or (B, L, N, P).
            real_value_full (torch.Tensor): ground truth full tokens, shape is (B, N, 1, T), T = L * P.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.
        """
        if self.spatial:
            batch_size, num_time, num_nodes, _ = reconstruction_full.shape
            # Extract the mask of the spatial dimension and reconstruct the tokens.
            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]  # (B, L, N*r, P)
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_time, -1)  # (B, L, N*r*P)
            
            label_full = real_value_full.permute(0, 3, 1, 2)  # (B, T, N, 1)
            # 第一个参数指示对哪个维度进行操作，第二个参数表示滑动窗口的大小，第三个参数表示滑动的步幅（每次滑动不会重叠）
            label_full = label_full.unfold(1, self.patch_size, self.patch_size)  # (B, L, N, 1, P)
            label_full = label_full[:, :, :, self.selected_feature, :].transpose(1, 2)  # (B, N, L, P)
            label_masked_tokens = label_full[:, masked_token_index, :, :].transpose(1, 2).contiguous() # (B, L, N*r, P)
            label_masked_tokens = label_masked_tokens.view(batch_size, num_time, -1)  # (N, L, N*r*P)
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            batch_size, num_nodes, num_time, _ = reconstruction_full.shape
            # Extract the mask of the temporal dimension and reconstruct the tokens.
            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]  # (B, N, L*r, P)
            # (B, N, L*r*P)  ->  (B, L*r*P, N)
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)
            
            label_full = real_value_full.permute(0, 3, 1, 2)  # (B, T, N, 1)
            label_full = label_full.unfold(1, self.patch_size, self.patch_size)  # (B, L, N, 1, P)
            label_full = label_full[:, :, :, self.selected_feature, :].transpose(1, 2)     # (B, N, L, P)
            label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous()  # (B, N, L*r, P)
            # (B, N, L*r*P) ->  (B, L*r*P, N)
            label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)
            return reconstruction_masked_tokens, label_masked_tokens
        
        
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None):
        # history_data shape: (B, T, N, D)
        history_data = history_data.permute(0, 2, 3, 1)  # (B, T, N, D) -> (B, N, D, T)
        if self.mode == 'pre-train':
            # encoding
            hidden_state_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            # decoding
            reconstruction_full = self.decoding(hidden_state_unmasked, masked_token_index)
            # =================================================================================
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_state_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_state_full
        

if __name__ == '__main__':
    model = Mask(
        patch_size=12, 
        in_channel=1, 
        embed_dim=96, 
        num_heads=4, 
        mlp_ratio=4, 
        dropout=0.1, 
        mask_ratio=0.75,
        encoder_depth=4, 
        decoder_depth=1, 
        spatial=False,  # S-MAE or T-MAE
        mode='pre-train'
    )
    history_data = torch.randn(4, 864, 307, 1)
    hidden_state_full = model(history_data)
    print(hidden_state_full.shape)  # torch.Size([4, 307, 72, 96])
    # reconstruction_masked_tokens, label_masked_tokens = model(history_data)
    # print(reconstruction_masked_tokens.shape, label_masked_tokens.shape)