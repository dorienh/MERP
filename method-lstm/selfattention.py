

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init


class MutliHeadAttention1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, groups=1, position=True, bias=False):
        """kernel_size is the 1D local attention window size"""

        super().__init__()
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.position = position
        
        # Padding should always be (kernel_size-1)/2
        # Isn't it?
        self.padding = (kernel_size-1)//2
        self.groups = groups

        # Make sure the feature dim is divisible by the n_heads
        assert self.out_features % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        assert (kernel_size-1) % 2 == 0, "kernal size must be odd number"

        if self.position:
            # Relative position encoding
            self.rel = nn.Parameter(torch.randn(1, out_features, kernel_size), requires_grad=True)

        # Input shape = (batch, len, feat)
        
        # Increasing the channel deapth (feature dim) with Conv2D
        # kernel_size=1 such that it expands only the feature dim
        # without affecting other dimensions
        self.W_k = nn.Linear(in_features, out_features, bias=bias)
        self.W_q = nn.Linear(in_features, out_features, bias=bias)
        self.W_v = nn.Linear(in_features, out_features, bias=bias)

        self.reset_parameters()

    def forward(self, x):

        batch, seq_len, feat_dim = x.size()

        padded_x = F.pad(x, [0, 0, self.padding, self.padding])
        q_out = self.W_q(x)
        k_out = self.W_k(padded_x)
        v_out = self.W_v(padded_x)
        
        k_out = k_out.unfold(1, self.kernel_size, self.stride)
        # (batch, L, feature, local_window)
        
        v_out = v_out.unfold(1, self.kernel_size, self.stride)
        # (batch, L, feature, local_window)
        
        if self.position:
            k_out = k_out + self.rel # relative position?

        k_out = k_out.contiguous().view(batch, seq_len, self.groups, self.out_features // self.groups, -1)
        v_out = v_out.contiguous().view(batch, seq_len, self.groups, self.out_features // self.groups, -1)
        # (batch, L, n_heads, feature_per_head, local_window)
        
        # expand the last dimension s.t. it can multiple with the local att window
        q_out = q_out.view(batch, seq_len, self.groups, self.out_features // self.groups, 1)
        # (batch, L, n_heads, feature_per_head, 1)
        
        energy = (q_out * k_out).sum(-2, keepdim=True)
        
        attention = F.softmax(energy, dim=-1)
        # (batch, L, n_heads, 1, local_window)
        
        out = attention*v_out
#         out = torch.einsum('blnhk,blnhk -> blnh', attention, v_out).view(batch, seq_len, -1)
        # (batch, c, H, W)
        
        return out.sum(-1).flatten(2), attention.squeeze(3)

    def reset_parameters(self):
        init.kaiming_normal_(self.W_k.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.W_v.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.W_q.weight, mode='fan_out', nonlinearity='relu')
        if self.position:
            init.normal_(self.rel, 0, 1)


class self_attention(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super(self_attention, self).__init__()
        # self.hidden_dim = hidden_dim
        # self.input_dim = input_dim

        # self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attn = MutliHeadAttention1D(in_features, out_features, kernel_size)

        # self.dropout1 = nn.Dropout(drop_prob)
        self.act1 = nn.LeakyReLU(0.1)

        # self.fc = nn.Linear(hidden_dim, hidden_dim//2)
        # self.dropout2 = nn.Dropout(drop_prob)
        # self.act2 = nn.LeakyReLU(0.1)
        
        # self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        # self.dropout3 = nn.Dropout(drop_prob)
        # self.act3 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(out_features, 1)
        

    def forward(self, x):
        # print('x shape: ', x.shape)
        # lstm_out, lstm_h = self.lstm(x)
        attn_out, _ = self.attn(x)
        # print('lstm_out shape: ', lstm_out.shape)
        # lstm_out = self.dropout1(lstm_out)
        attn_out = self.act1(attn_out)
        
        # print(attn_out.shape)
        # out = self.fc(lstm_out)
        # out = self.dropout2(out)
        # out = self.act2(out)
        # # print('fc1 shape: ', out.shape)

        # out = self.fc2(out)
        # out = self.dropout3(out)
        # out = self.act3(out)
        # print('fc2 shape: ', out.shape)
        
        out = self.fc3(attn_out)
        # print('out shape: ', out.shape)
        return out