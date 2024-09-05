import torch
import torch.nn as nn
from mask import Mask
from predictor import GraphWaveNet as Predictor


class STDMAE(nn.Module):
    """
        Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting
        - mask_args:
            patch_size=12, in_channel=1, embed_dim=96, num_heads=4, mlp_ratio=4, dropout=0.1, mask_ratio=0.25,
            encoder_depth=4, decoder_depth=1, spatial=False, mode='forecasting'
        - backend_args:
            num_nodes, supports, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
            residual_channels=32, dilation_channels=32, skip_channels=256, end_chnanels=512, kernel_size=2, blocks=4, layers=2
    """
    def __init__(self, dataset_name, pre_trained_tmae_path, pre_trained_smae_path, mask_args, backend_args):
        super(STDMAE, self).__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tmae_path = pre_trained_tmae_path
        self.pre_trained_smae_path = pre_trained_smae_path
        # initalize spatio-temporal mask pre-training framework
        self.tmae = Mask(**mask_args)
        self.smae = Mask(**mask_args)
        
        # downstream task
        self.backend = Predictor(**backend_args)
        # load pre-trained model
        self.load_pre_trained_model()
        
    def load_pre_trained_model(self):
        # load pre-trained model's parameters
        checkpoint_dict = torch.load(self.pre_trained_tmae_path)
        self.tmae.load_state_dict(checkpoint_dict['model_state_dict'])
        
        checkpoint_dict = torch.load(self.pre_trained_smae_path)
        self.smae.load_state_dict(checkpoint_dict['model_state_dict'])
        
        # freeze parameters
        for param in self.tmae.parameters():
            param.requires_grad = False
        for param in self.smae.parameters():
            param.requires_grad = False
    
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor):
        """
        Args:
            history_data (torch.Tensor): Short-term history data. shape: [B, T, N, 3]
            long_history_data (torch.Tensor): Long-term history data. shape: [B, L * P, N, 3]
        """
        short_term_history = history_data
        
        hidden_states_t = self.tmae(long_history_data[..., [0]])  # (B, N, L, d)
        hidden_states_s = self.smae(long_history_data[..., [0]])  # (B, N, L, d)
        hidden_states = torch.cat([hidden_states_t, hidden_states_s], dim=-1)  # (B, N, L, 2*d)
        
        # enhance
        hidden_states = hidden_states[:, :, -1, :]   # (B, N, 2*d) Only take the features of the last time step.
        y_hat = self.backend(short_term_history, hidden_states=hidden_states)
        return y_hat  # (B, T, N, 1)