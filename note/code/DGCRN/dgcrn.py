import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

class GConv_Hyper(nn.Module):
    def __init__(self, dims, gdep, alpha, beta, gamma):
        super(GConv_Hyper, self).__init__()
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.mlp = nn.Sequential(
            nn.Linear((gdep + 1) * dims[0], dims[1]),
            nn.Sigmoid(),
            nn.Linear(dims[1], dims[2]),
            nn.Sigmoid(),
            nn.Linear(dims[2], dims[3])
        )

    def forward(self, x, adj):
        # x shape: (B, N, D), adj shape: (N, N)
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + self.gamma * torch.einsum('bnc,nm->bmc', (h, adj))
            out.append(h)
        ho = torch.cat(out, dim=-1)  # concate along channel
        ho = self.mlp(ho)
        return ho


class GConv_RNN(nn.Module):
    def __init__(self, dims, gdep, alpha, beta, gamma):
        super(GConv_RNN, self).__init__()
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])
    
    def forward(self, x, adj):
        """
            x shape: (B, N, D), adj shape: [(B, N, N), (N, N)]
            The first one is a dynamic graph, the second one is a predefined graph.
        """
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x 
            + self.beta * torch.einsum('bnc,bnm->bmc', (h, adj[0])) 
            + self.gamma * torch.einsum('bnc,nm->bmc', (h, adj[1]))
            out.append(h)
        ho = torch.cat(out, dim=-1)  # concate along channel
        ho = self.mlp(ho)
        return ho
    

class DGCN(nn.Module):
    def __init__(self, dims_hyper, gcn_depth, predefined_A, alpha, list_weight):
        super(DGCN, self).__init__()
        self.predefined_A = predefined_A  # type: list
        self.alpha = alpha
        self.gcn1_hyper_1 = GConv_Hyper(dims_hyper, gcn_depth, *list_weight)
        self.gcn1_hyper_2 = GConv_Hyper(dims_hyper, gcn_depth, *list_weight)
        self.gcn2_hyper_1 = GConv_Hyper(dims_hyper, gcn_depth, *list_weight)
        self.gcn2_hyper_2 = GConv_Hyper(dims_hyper, gcn_depth, *list_weight)

    def normalize(self, adj, predefined_A):
        num_nodes = adj.size(1)
        adj = adj + torch.eye(num_nodes).to(adj.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_A]

    def forward(self, x, hidden_state, nodevec1, nodevec2):
        """
            The shape of x and hidden_state is [batch, num_node, hidden_dim]
            The shape of nodevec1 and nodevec2 is [num_nodes, node_dim]
        """
        hyper_input = torch.cat((x, hidden_state), dim=2)  # (B, N, 2*D)
        filter1 = self.gcn1_hyper_1(hyper_input, self.predefined_A[0]) + self.gcn1_hyper_2(hyper_input, self.predefined_A[1])
        filter2 = self.gcn2_hyper_1(hyper_input, self.predefined_A[0]) + self.gcn2_hyper_2(hyper_input, self.predefined_A[1])
        # (B, N, dim) * (N, dim) -> (B, N, dim)
        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))
        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
            nodevec2, nodevec1.transpose(2, 1)
        ) # (B, N, N)
        adp = F.relu(torch.tanh(self.alpha * a))  # (B, N, N)
        adp, adpT = self.normalize(adp, self.predefined_A[0]), self.normalize(adp.transpose(1, 2), self.predefined_A[1])
        return adp, adpT


class DGCRM(nn.Module):
    def __init__(self, num_node, hidden_dim, in_dim, hyperGNN_dim, middle_dim, 
                 node_dim, gcn_depth, predefined_A, alpha, list_weight, device):
        super(DGCRM, self).__init__()
        self.num_node = num_node
        self.hidden_dim = hidden_dim
        self.device = device
        dims_hyper = [hidden_dim + in_dim, hyperGNN_dim, middle_dim, node_dim]
        self.dgcn = DGCN(dims_hyper, gcn_depth, predefined_A, alpha, list_weight)
        dims = [hidden_dim + in_dim, hidden_dim]
        self.gz1 = GConv_RNN(dims, gcn_depth, *list_weight)
        self.gz2 = GConv_RNN(dims, gcn_depth, *list_weight)
        self.gr1 = GConv_RNN(dims, gcn_depth, *list_weight)
        self.gr2 = GConv_RNN(dims, gcn_depth, *list_weight)
        self.gc1 = GConv_RNN(dims, gcn_depth, *list_weight)
        self.gc2 = GConv_RNN(dims, gcn_depth, *list_weight)

    def forward(self, x, hidden_state, nodevec1, nodevec2):
        adp, adpT = self.dgcn(x, hidden_state, nodevec1, nodevec2)
        combined = torch.cat((x, hidden_state), dim=-1)
        z = F.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpT))
        r = F.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpT))
        candidate = torch.cat((x, torch.mul(r, hidden_state)), dim=-1)
        hc = F.tanh(self.gc1(candidate, adp) + self.gc2(candidate, adpT))
        h = torch.mul(z, hidden_state) + torch.mul(1 - z, hc)
        return h
    
    def init_hidden_state(self, batch_size):
        hidden_state = Variable(
            torch.zeros(batch_size, self.num_node, self.hidden_dim).to(self.device)
        )
        nn.init.orthogonal(hidden_state)
        return hidden_state
    

# The encoder and decoder in the source code all has only one layer.
class DGCRN(nn.Module):
    def __init__(self, 
                 gcn_depth, 
                 num_node, 
                 device, 
                 predefined_A=None, 
                 dropout=0.3,
                 node_dim=40,
                 middle_dim=2,
                 seq_length=12,
                 in_dim=2,
                 list_weight=[0.05, 0.95, 0.95],
                 tanhalpha=3,
                 cl_decay_steps=4000,
                 rnn_size=64,
                 hyperGNN_dim=16):
        super(DGCRN, self).__init__()
        self.output_dim = 1
        self.num_node = num_node
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.seq_length = seq_length
        self.rnn_size = rnn_size
        self.hyperGNN_dim = hyperGNN_dim
        self.middle_dim = middle_dim
        self.in_dim = in_dim
        self.hidden_dim = self.rnn_size
        self.alpha = tanhalpha
        self.device = device

        self.emb1 = nn.Embedding(self.num_node, node_dim)
        self.emb2 = nn.Embedding(self.num_node, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)        
        self.idx = torch.arange(self.num_node).to(self.device)
        
        self.encoder = DGCRM(self.num_node, self.hidden_dim, self.in_dim, self.hyperGNN_dim, self.middle_dim, 
                 node_dim, gcn_depth, self.predefined_A, self.alpha, list_weight, self.device)
        self.decoder = DGCRM(self.num_node, self.hidden_dim, self.in_dim, self.hyperGNN_dim, self.middle_dim, 
                 node_dim, gcn_depth, self.predefined_A, self.alpha, list_weight, self.device)
        self.fc_final = nn.Linear(self.hidden_dim, self.output_dim)

        self.use_curriculum_learning = True
        self.cl_decay_steps = cl_decay_steps
        self.gcn_depth = gcn_depth

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, input, ycl=None, batches_seen=None, task_level=12):
        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)
        batch_size = input.size(0)
        hidden_state = self.encoder.init_hidden_state(batch_size)

        # 1.DGCRM_Encoder
        output_hidden = []
        for t in range(self.seq_length):
            hidden_state = self.encoder(input[:, t, :, :], hidden_state, nodevec1, nodevec2)
            output_hidden.append(hidden_state)
        go_symbol = torch.zeros((batch_size, self.num_node, self.output_dim), device=self.device)
        timeofday = ycl[:, :, :, 1:]
        decoder_input = go_symbol
        outputs_final = []
        # 2.DGCRM_Decoder
        for i in range(task_level):
            try:
                decoder_input = torch.cat([decoder_input, timeofday[:, i, ...]], dim=-1)
            except:
                print(decoder_input.shape, timeofday.shape)
                sys.exit(0)
            hidden_state = self.decoder(decoder_input, hidden_state, nodevec1, nodevec2)
            decoder_output = self.fc_final(hidden_state)
            outputs_final.append(decoder_output)
            # curriculum learning
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = ycl[:, i, :, :1]
        
        outputs_final = torch.stack(outputs_final, dim=1)
        return outputs_final
    

if __name__ == '__main__':
    predefined_A = [torch.randn(207, 207), torch.randn(207, 207)]
    model = DGCRN(
        gcn_depth=2, 
        num_node=207, 
        device='cpu', 
        predefined_A=predefined_A, 
        dropout=0.3,
        node_dim=40,
        middle_dim=2,
        seq_length=12,
        in_dim=2,
        list_weight=[0.05, 0.95, 0.95],
        tanhalpha=3,
        cl_decay_steps=4000,
        rnn_size=64,
        hyperGNN_dim=16
    )
    # The first feature is the traffic volume, and the second feature is the time information.
    x = torch.randn(32, 12, 207, 2)
    ycl = torch.randn(32, 12, 207, 2) 
    batches_seen = 10
    y = model(x, ycl, batches_seen)
    print(y.size())