import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvlc,vw->nwlc', (x, A))
        return x.contiguous()
    
    
class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, supports_len=2, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * supports_len + 1) * c_in
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order
    
    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        
        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    

class QKVAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, num_heads = 4):
        super(QKVAttention, self).__init__()
        self.query = nn.Linear(in_dim, hidden_dim)
        self.key = nn.Linear(in_dim, hidden_dim)
        self.value = nn.Linear(in_dim, hidden_dim)
        self.num_heads = num_heads
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        assert hidden_dim % num_heads == 0
        
    def forward(self, x, kv=None):
        # x shape may be (B, N, T, C) or (B, T, N, C)
        # able to get additional kv-input for Time-Enhanced Attention
        if kv is None:
            kv = x   # self-attention
        query = self.query(x)
        key = self.key(kv)
        value = self.value(kv)
        num_heads = self.num_heads
        # multi-head attention
        if num_heads > 1:
            query = torch.cat(torch.chunk(query, num_heads, dim=-1), dim=0)
            key = torch.cat(torch.chunk(key, num_heads, dim=-1), dim=0)
            value = torch.cat(torch.chunk(value, num_heads, dim=-1), dim=0)
        d = value.size(-1)
        # calculate the attention score
        att_score = torch.matmul(query, key.transpose(-1, -2))
        att_score = att_score / (d ** 0.5)
        score = torch.softmax(att_score, dim=-1)
        head_out = torch.matmul(score, value)
        out = torch.cat(torch.chunk(head_out, num_heads, dim=0), dim=-1)
        return self.dropout(self.proj(out))
    

class SkipConnection(nn.Module):
    def __init__(self, module, norm):
        super(SkipConnection, self).__init__()
        self.module = module
        self.norm = norm
        
    def forward(self, x, aux=None):
        return self.norm(x + self.module(x, aux))
    
    
class LayerNorm(nn.Module):
    # Assume the input shape is (B, N, T, C)
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(*normalized_shape))
        self.beta = nn.Parameter(torch.zeros(*normalized_shape))
        
    def forward(self, x):
        # dims = [-1, -2, -3, -4]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        # mean --> (B, C, H, W) --> (B)
        # mean with keepdims --> (B, C, H, W) --> (B, 1, 1, 1)
        mean = x.mean(dim=dims, keepdims=True)
        std = x.std(dim=dims, keepdims=True, unbiased=False)
        x_norm = (x - mean) / (std + self.eps)
        out = x_norm * self.gamma + self.beta
        return out
    
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, activation=nn.GELU()):
        super(PositionwiseFeedForward, self).__init__()
        self.act = activation
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, x, kv=None):
        # linear -> activate -> linear -> dropout
        return self.dropout(self.l2(self.act(self.l1(x))))


class TemporalInformationEmbedding(nn.Module):
    def __init__(self, hidden_dim, vocab_size, freq_act=torch.sin, n_freq=1):
        super(TemporalInformationEmbedding, self).__init__()
        """
        The input only contains temporal information with index.
        Arguments:
            - vocab_size: total number of temporal features.
            - freq_act: periodic activation function.
            - n_freq: number of hidden elements for frequency components.
                - 0: use linear
                - H: use linear and frequency component
                - 0 < and < H: both linear and frequecy component
        """
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.freq_act = freq_act
        self.n_freq = n_freq
        
    def forward(self, x):
        # x shape: (B, T)
        x_emb = self.embedding(x)  # (B, T, D)
        x_weight = self.linear(x_emb)  # linear
        if self.n_freq == 0:
            return x_weight
        if self.n_freq == self.hidden_dim:
            return self.freq_act(x_weight)
        # 0 < n_freq < H
        x_linear = x_weight[..., self.n_freq:]
        x_freq = self.freq_act(x_weight[..., :self.n_freq])
        return torch.cat([x_linear, x_freq], dim=-1)


class TemporalModel(nn.Module):
    def __init__(self, hidden_dim, num_nodes, layers, dropout, in_dim=1, out_dim=1, vocab_size=288, activation=nn.ReLU()):
        super(TemporalModel, self).__init__()
        self.vocab_size = vocab_size
        self.act = activation
        self.in_dim = in_dim
        self.embedding = TemporalInformationEmbedding(hidden_dim, vocab_size=vocab_size)
        self.feat_proj = nn.Linear(in_dim, hidden_dim)
        # concat speed information and TIM information
        self.feat_cat = nn.Linear(hidden_dim * 2, hidden_dim)
        
        module = QKVAttention(in_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        ff = PositionwiseFeedForward(in_dim=hidden_dim, hidden_dim= 4 * hidden_dim, dropout=dropout)
        norm = LayerNorm(normalized_shape=(hidden_dim,))
        
        self.node_features = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.temp_attn_layers = nn.ModuleList()
        self.ff = nn.ModuleList()
        for _ in range(layers):
            self.temp_attn_layers.append(SkipConnection(deepcopy(module), deepcopy(norm)))
            self.ff.append(SkipConnection(deepcopy(ff), deepcopy(norm)))
        
        self.proj = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x, feat=None):
        """
            x shape: (B, T), label's temporal index.
            feat shape: (B, T, N, 1), actual feature to predict.
        """
        TIM = self.embedding(x)
        x_nemb = torch.einsum('btc,nc->bntc', TIM, self.node_features)
        if feat is None:
            # 如果没有传入特征信息，则使用全零的token代替
            feat = torch.zeros_like(x_nemb[..., :self.in_dim])
        x_feat = self.feat_proj(feat)
        # 融合Time2Vec向量、可学习节点特征、交通量特征
        x_nemb = self.feat_cat(torch.cat([x_feat, x_nemb], dim=-1))
        
        attns = []
        # 多层Transformer结构
        for _, (temp_attn_layer, ff) in enumerate(zip(self.temp_attn_layers, self.ff)):
            # 1.Temporal Attention -> Add & Norm
            x_attn = temp_attn_layer(x_nemb)
            # 2. Feed Forward -> Add & Norm
            x_nemb = ff(x_attn)
            attns.append(x_nemb)
            
        out = self.proj(self.act(x_nemb))
        return out, attns
    

class STModel(nn.Module):
    def __init__(self, hidden_dim, supports_len, dropout, layers, out_dim=1, in_dim=2, spatial=False, activation=nn.ReLU()):
        super(STModel, self).__init__()
        self.spatial = spatial  # Flag that determine when spatial attention will be performed.
        self.act = activation
        self.out_dim = out_dim
        
        t_attn = QKVAttention(in_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        s_gcn = gcn(c_in=hidden_dim, c_out=hidden_dim, dropout=dropout, supports_len=supports_len, order=2)
        ff = PositionwiseFeedForward(in_dim=hidden_dim, hidden_dim=4 * hidden_dim, dropout=dropout)
        norm = LayerNorm(normalized_shape=(hidden_dim,))
        
        self.start_linear = nn.Linear(in_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim + out_dim)
        
        self.temporal_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()
        self.ed_layers = nn.ModuleList()
        self.ff = nn.ModuleList()
        
        for _ in range(layers):
            self.temporal_layers.append(SkipConnection(deepcopy(t_attn), deepcopy(norm)))
            self.spatial_layers.append(SkipConnection(deepcopy(s_gcn), deepcopy(norm)))
            self.ed_layers.append(SkipConnection(deepcopy(t_attn), deepcopy(norm)))
            self.ff.append(SkipConnection(deepcopy(ff), deepcopy(norm)))
        
    def forward(self, x, prev_hidden, supports):
        # (B, C, N, T) -> (B, N, T, C)
        x = self.start_linear(x.permute(0, 2, 3, 1))
        x_start = x
        hiddens = []
        for _, (temporal_layer, spatial_layer, ed_layer, ff) in enumerate(zip(self.temporal_layers, self.spatial_layers, self.ed_layers, self.ff)):
            # 1. Temporal Attention -> Add & Norm
            # 2. Spatial Attention -> Add & Norm
            if not self.spatial:
                x1 = temporal_layer(x)  # (B, N, T, C)
                x_attn = spatial_layer(x1, supports)  # (B, N, T, C)
            else:
                x1 = spatial_layer(x, supports)
                x_attn = temporal_layer(x1)
            # 3. Time-Enhanced Attention -> Add & Norm
            if prev_hidden is not None:
                x_attn = ed_layer(x_attn, prev_hidden)
            # 4. Feed Forward -> Add & Norm
            x = ff(x_attn)
            hiddens.append(x)
        
        out = self.proj(self.act(x))
        res, out = torch.split(out, [out.size(-1) - self.out_dim, self.out_dim], dim=-1)
        return x_start - res, out.contiguous(), hiddens
        
                
class AttentionModel(nn.Module):
    def __init__(self, hidden_dim, layers, dropout, in_dim=2, out_dim=1, spatial=False, activation=nn.ReLU()):
        super(AttentionModel, self).__init__()
        self.spatial = spatial
        self.act = activation
        
        base_model = SkipConnection(QKVAttention(hidden_dim, hidden_dim, dropout), LayerNorm(normalized_shape=(hidden_dim, )))
        ff = SkipConnection(PositionwiseFeedForward(hidden_dim, 4 * hidden_dim, dropout), LayerNorm(normalized_shape=(hidden_dim, )))
        
        self.start_linear = nn.Linear(in_dim, hidden_dim)
        self.spatial_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()
        self.ed_layers = nn.ModuleList()
        self.ff = nn.ModuleList()
        
        for i in range(layers):
            self.spatial_layers.append(deepcopy(base_model))
            self.temporal_layers.append(deepcopy(base_model))
            self.ed_layers.append(deepcopy(base_model))
            self.ff.append(deepcopy(ff))
            
        self.proj = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, prev_hidden=None):
        # x shape: (B, C, N, T)
        # (B, C, N, T) -> (B, N, T, C)
        x = self.start_linear(x.permute(0, 2, 3, 1))
        for i, (s_layer, t_layer, ff) in enumerate(zip(self.spatial_layers, self.temporal_layers, self.ff)):
            # 1. Temporal Attention -> Add & Norm
            # 2. Spatial Attention -> Add & Norm
            if not self.spatial:
                x1 = t_layer(x)  # Temporal Attention: (B, N, T, C) -> (B, N, T, T)
                x_attn = s_layer(x1.transpose(1, 2)) # Spatial Attention: (B, T, N, C) -> (B, T, N, N)
            else:
                x1 = s_layer(x.tranpose(1, 2))
                x_attn = t_layer(x.tranpose(1, 2)).transpose(1, 2)

            # 3. Time-Enhanced Attention -> Add & Norm
            if prev_hidden is not None:  
                x_attn = self.ed_layers[i](x_attn.transpose(1, 2), prev_hidden)
                x_attn = x_attn.transpose(1, 2)
            x = ff(x_attn.transpose(1, 2))
        
        return self.proj(self.act(x)), x
    

class MemoryGate(nn.Module):
    def __init__(self, hidden_dim, num_nodes, mem_hid=32, in_dim=2, out_dim=1, memory_size=20, 
                 sim=nn.CosineSimilarity(dim=-1), nodewise=False, attention_type='attention'):
        super(MemoryGate, self).__init__()
        self.attention_type = attention_type
        self.sim = sim
        self.nodewise = nodewise
        self.out_dim = out_dim
        """
        Arguments:
        - mem_hid, memory_size: hidden size and total number of memory items.
        - sim: similarity function to evaluate rooting probability.
        - nodewise: flag to determine routing level.
        """
        self.memory = nn.Parameter(torch.empty(memory_size, mem_hid))
        self.hid_query = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, mem_hid)) for _ in range(3)])
        self.key = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, mem_hid)) for _ in range(3)])
        self.value = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, mem_hid)) for _ in range(3)])
        
        self.input_query = nn.Parameter(torch.empty(in_dim, mem_hid))
        self.We1 = nn.Parameter(torch.empty(num_nodes, memory_size))
        self.We2 = nn.Parameter(torch.empty(num_nodes, memory_size))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        
    def query_mem(self, input):
        B, N, T, _ = input.size()
        mem = self.memory  # (memory_size, mem_hid)
        # generate query vector
        query = torch.matmul(input, self.input_query)  # (B, N, T, mem_hid)
        energy = torch.matmul(query, mem.T)   # (B, N, T, memory_size)
        score = torch.softmax(energy, dim=-1)
        out = torch.matmul(score, mem)  # (B, N, T, mem_hid)
        return out
    
    def attention(self, x, i):
        B, N, T, _ = x.size()
        # (B, N, T, mem_hid)
        query = torch.matmul(x, self.hid_query[i]) 
        key = torch.matmul(x, self.key[i])
        value = torch.matmul(x, self.value[i])
        if self.nodewise:
            query = query.sum(dim=-2, keepdim=True)  # (B, N, 1, mem_hid)
        energy = torch.matmul(query, key.transpose(-1, -2))
        score = torch.softmax(energy, dim=-1)
        out = torch.matmul(score, value)
        return out.expand_as(value)
    
    def topk_attention(self, x, i, k=3):
        B, N, T, _ = x.size()
        query = torch.matmul(x, self.hid_query[i])
        key = torch.matmul(x, self.key[i])
        value = torch.matmul(x, self.value[i])
        if self.nodewise:
            query = query.sum(dim=-2, keepdim=True)
        energy = torch.matmul(query, key.transpose(-1, -2))
        values, indices = torch.topk(energy, k=k, dim=-1)
        score = energy.zero_().scatter_(-1, indices, torch.relu(values))
        out = torch.matmul(score, value)
        return out.expand_as(value)
        
    def forward(self, input, hidden):
        # input shape: (B, N, T, 2), hidden = [h_identity, h_adaptive, h_attention]
        if self.attention_type == 'attention':
            attention = self.attention
        else:
            attention = self.topk_attention
        B, N, T, _ = input.size()
        memories = self.query_mem(input) # (B, N, T, mem_hid)
        scores = []
        for i, h in enumerate(hidden):
            hidden_att = attention(h, i)
            scores.append(self.sim(memories, hidden_att))
        
        scores = torch.stack(scores, dim=-1)
        return scores.unsqueeze(dim=-2).expand(B, N, T, self.out_dim, scores.size(-1))
        
        
class TESTAM(nn.Module):
    def __init__(self, num_nodes, dropout=0.3, in_dim=2, out_dim=1, hidden_dim=32, layers=3, 
                 prob_mul=False, max_time_index=288):
        super(TESTAM, self).__init__()
        self.dropout = dropout
        self.prob_mul = prob_mul
        self.supports_len = 2
        self.max_time_index = max_time_index
        
        self.identity_expert = TemporalModel(hidden_dim, num_nodes, in_dim=in_dim-1, out_dim=out_dim, layers=layers, dropout=dropout, vocab_size=max_time_index)
        self.adaptive_expert = STModel(hidden_dim, self.supports_len, in_dim=in_dim, out_dim=out_dim, layers=layers, dropout=dropout)
        self.attention_expert = AttentionModel(hidden_dim, in_dim=in_dim, out_dim=out_dim, layers=layers, dropout=dropout)
        
        self.gate_network = MemoryGate(hidden_dim, num_nodes, in_dim=in_dim, out_dim=out_dim)
        
        for model in [self.identity_expert, self.adaptive_expert, self.attention_expert]:
            for n, p in model.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
    
    def forward(self, input, gate_out=False):
        # input shape: (B, in_dim, N, T), the last dimension of input is temporal feature.
        n1 = torch.matmul(self.gate_network.We1, self.gate_network.memory)
        n2 = torch.matmul(self.gate_network.We2, self.gate_network.memory)
        g1 = torch.softmax(torch.relu(torch.mm(n1, n2.T)), dim=-1)
        g2 = torch.softmax(torch.relu(torch.mm(n2, n1.T)), dim=-1)
        new_supports = [g1, g2]
        
        # time index shape is (B, T), the range is between 0 and 1.
        time_index = input[:, -1, 0, :] 
        max_t = self.max_time_index
        cur_time_index = ((time_index * max_t) % max_t).long()
        next_time_index = ((time_index * max_t + time_index.size(-1)) % max_t).long()
        
        # Router-1: Just calculate the multi-head temporal attention (transformer).
        o_identity, h_identity = self.identity_expert(cur_time_index, input[:,:-1,...].permute(0,2,3,1))
        _, h_future = self.identity_expert(next_time_index)
        
        # Router-2: As shown in the original paper, including temporal attention, spatial model and time-enhanced attention.
        # Time2Vec is not used here. Instead, the original traffic features + time features are directly used.
        _, o_adaptive, h_adaptive = self.adaptive_expert(input, h_future[-1], new_supports)

        # Router-3: It is composed of temporal attention, spatial attention and time-enhanced attention.
        o_attention, h_attention = self.attention_expert(input, h_future[-1])

        # The outputs of the three experts are all of dimension (B, N, T, 1).
        ind_out = torch.stack([o_identity, o_adaptive, o_attention], dim = -1)  # (B, N, T, 3)
        B, N, T, _ = o_identity.size()
        
        gate_in = [h_identity[-1], h_adaptive[-1], h_attention]
        gate = torch.softmax(self.gate_network(input.permute(0, 2, 3, 1), gate_in), dim = -1)
        # print(gate.shape) # (B, N, T, 1, 3)，这里的3指的是三个专家的所得路由概率
        
        # ===============================================================================================
        outs = [o_identity, o_adaptive, o_attention]
        out = torch.zeros_like(o_identity).view(-1, 1)  # (B*N*T*D, 1)
        route_prob_max, routes = torch.max(gate, dim=-1)  # route_prob_max是最大路由概率，routes是索引[0, 1, 2]
        # print(route_prob_max.shape, routes.shape)
        route_prob_max = route_prob_max.view(-1)
        routes = routes.view(-1)
        
        for i in range(len(outs)):
            # Take the combination of output values ​​with the maximum routing probability as the final output.
            cur_out = outs[i].view(-1, 1)  # (B*N*T*D, 1)，依次遍历 o_identity -> o_adaptive -> o_attention
            indices = torch.eq(routes, i).nonzero(as_tuple = True)[0]  # 关键步骤1
            out[indices] = cur_out[indices]  # 关键步骤2
            
        if self.prob_mul:  # 是否将值与对应的概率相乘
            out = out * (route_prob_max).unsqueeze(dim = -1)
        # ===============================================================================================

        out = out.view(B, N, T, -1) 
        out = out.permute(0, 3, 1, 2)  # (B, 1, N, T)
        if self.training or gate_out:
            return out, gate, ind_out
        else:
            return out


if __name__ == '__main__':
    model = TESTAM(num_nodes = 207, in_dim = 2, out_dim = 1)
    x = torch.randn(8, 1, 207, 12)
    dow = torch.randint(0, 288, size=(8, 1, 207, 12))
    x = torch.cat([x, dow], dim=1)
    out, gate, ind_out = model(x, gate_out = True)
    print(out.shape, gate.shape, ind_out.shape)