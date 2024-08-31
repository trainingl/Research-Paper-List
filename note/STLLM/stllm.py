import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class TemporalEmbedding(nn.Module):
    def __init__(self, times, features):
        super(TemporalEmbedding, self).__init__()
        self.times = times
        self.time_of_day = nn.Parameter(torch.empty(times, features))
        self.day_of_week = nn.Parameter(torch.empty(7, features))
        
        nn.init.xavier_uniform_(self.time_of_day)
        nn.init.xavier_uniform_(self.day_of_week)
        
    def forward(self, x):
        # x shape: (B, T, N, 3)
        day_emb = x[..., 1]
        time_of_day = self.time_of_day[
            (day_emb[:, -1, :] * self.times).type(torch.LongTensor)
        ] # (B, N, D) -> (B, D, N) -> (B, D, N, 1)
        time_of_day = time_of_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        day_of_week = self.day_of_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ] # (B, N, D) -> (B, D, N) -> (B, D, N, 1)
        day_of_week = day_of_week.transpose(1, 2).unsqueeze(-1)
        
        # temporal embeddings
        temp_emb = time_of_day + day_of_week  # (B, D, N, 1)
        return temp_emb
    

# The PFA(partially frozen attention) LLM has F+U layers.
# These layers are divided into the first F layers and the last U layers.
# And the first F layers are frozen, and the last U layers are unfrozen.
class PFA(nn.Module):
    def __init__(self, device='cuda:0', gpt_layers=6, U=1):
        super(PFA, self).__init__()
        # 注意：下载huggingface预训练好的模型，要挂代理
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True, cache_dir="./model_cache"
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U
        
        for idx, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if idx < gpt_layers - self.U:
                    if 'ln' in name or 'wpe' in name:
                        param.requires_grad = True   # Layer Norm
                    else:
                        param.requires_grad = False  # Others
                else:
                    if "mlp" in name:
                        param.requires_grad = False  # Feed Forward
                    else:
                        param.requires_grad = True   # Others
        
    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state
        

class ST_LLM(nn.Module):
    def __init__(self, device, times, input_dim=3, channels=64, num_nodes=170, input_len=12, 
                 output_len=12, dropout=0.1, U=1):
        super(ST_LLM, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.output_len = output_len
        self.U = U
        self.times = times
        
        gpt_channel = 256
        to_gpt_channel = 768
        
        self.Temb = TemporalEmbedding(self.times, gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)
        
        self.start_conv = nn.Conv2d(
            input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )
        self.gpt = PFA(device=self.device, gpt_layers=6, U=self.U)
        self.feature_fusion = nn.Conv2d(
            gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1)
        )
        
        # regression
        self.regressor = nn.Conv2d(
            gpt_channel * 3, self.output_len, kernel_size=(1, 1)
        )
    
    # return the total parameters of model
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    
    def forward(self, history_data):
        # history shape: (B, D, N, T)
        input_data = history_data
        batch_size, _, num_nodes, _ = input_data.shape
        history_data = history_data.permute(0, 3, 2, 1)
        
        temp_emb = self.Temb(history_data)
        node_emb = []
        node_emb.append(
            self.node_emb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )  # (B, D, N, 1)
        
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        ) # (B, D, N, 1)
        input_data = self.start_conv(input_data)
        
        data_st = torch.cat(
            [input_data] + [temp_emb] + node_emb, dim=1
        )  # concat along the channel axis
        data_st = self.feature_fusion(data_st)  # (B, D, N, 1)
        
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)  # (B, N, D)
        data_st = self.gpt(data_st)
        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)  # (B, D, N, 1)
        out = self.regressor(data_st)  # (B, T, N, 1)
        return out
        
        
if __name__ == '__main__':
    model = ST_LLM(
        device='cuda:0', times=48, input_dim=3, channels=64, num_nodes=170, 
        input_len=12, output_len=12, dropout=0.1, U=1
    )
    print("模型参数量: ", model.param_num())
    x = torch.randn(32, 12, 170, 1)
    tod = torch.rand(32, 12, 170, 1)  # range from 0 to 1, float
    dow = torch.randint(0, 6, size=(32, 12, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)
    x = x.transpose(1, 3)
    y = model(x)
    print(y.size())