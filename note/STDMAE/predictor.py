import torch
import torch.nn as nn
import torch.nn.functional as F


# Hre the authors chose GWNet as the downstream predictor.
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()
        
    def forward(self, x, A):
        A = A.to(x.device)
        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        else:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()
    

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        
    def forward(self, x):
        return self.mlp(x)
    
    
class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
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
        h = torch.cat(out, dim=1)  # concat along channel dimention
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    

class GraphWaveNet(nn.Module):
    # Paper: Graph WaveNet for Deep Spatial-Temporal Grpah Modeling.
    def __init__(self, num_nodes, supports, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32, skip_channels=256, end_chnanels=512, kernel_size=2, blocks=4, layers=2):
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        
        self.fc_his_t = nn.Sequential(
            nn.Linear(96, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_his_s = nn.Sequential(
            nn.Linear(96, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        
        self.supports = supports
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        
        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1
        
        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope  # receptive_field: (1 + 2) * 4 + 1 = 13
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
        
        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_chnanels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_chnanels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        
        
    def forward(self, input, hidden_states):
        """
        Args:
            input (torch.Tensor): (B, T, N, C)
            hidden_states (torch.Tensor): (B, N, d)
        """
        input = input.transpose(1, 3)  # (B, C, N, T)
        input = input[:, 0:2, :, :]
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)

        # calculate the current adaptive adj matrix
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        
        skip = 0   # Here the skip variable collects the skip information of each layer.
        # WaveNet Layers
        for i in range(self.layers * self.blocks):
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            
            # residual connection
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        
        hidden_states_t = self.fc_his_t(hidden_states[:, :, :96])  # (B, N, D)
        hidden_states_t = hidden_states_t.transpose(1, 2).unsqueeze(-1)  # (B, D, N, 1)
        hidden_states_s = self.fc_his_s(hidden_states[:, :, 96:])  # (B, N, D)
        hidden_states_s = hidden_states_s.transpose(1, 2).unsqueeze(-1)  # (B, D, N, 1)
        skip = skip + hidden_states_t + hidden_states_s
        
        out = F.relu(skip)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)  # (B, T, N, 1)
        return out
        
        
if __name__ == '__main__':
    model = GraphWaveNet(
        num_nodes=307, 
        supports=[torch.randn(307, 307), torch.randn(307, 307)], 
        dropout=0.3, 
        gcn_bool=True, 
        addaptadj=True, 
        aptinit=None, 
        in_dim=2
    )
    x = torch.randn(32, 12, 307, 2)
    hidden_states = torch.randn(32, 307, 96*2)
    y = model(x, hidden_states)  
    print(y.size())