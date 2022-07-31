from torch import nn
import torch.nn.functional as F
import torch
import math

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_ff, dropout):
        super().__init__()
        self.multihead_attention = MultiheadAttention(d_model, num_heads, d_k, 0)
        self.drop_out_1 = nn.Dropout(dropout)
        self.layer_norm_1 = LayerNormalization(d_model)

        self.feed_forward = FeedFowardLayer(d_model, d_ff, dropout)
        self.drop_out_2 = nn.Dropout(dropout)
        self.layer_norm_2 = LayerNormalization(d_model)
    def forward(self, x, e_mask):
        x_1 = x + self.drop_out_1(self.multihead_attention(x, x, x, mask=e_mask))  # (B, L, d_model)
        x_1 = self.layer_norm_1(x_1)  # (B, L, d_model)

        x_2 = x_1 + self.drop_out_2(self.feed_forward(x_1)) # (B, L, d_model)
        x_2 = self.layer_norm_2(x_2)  # (B, L, d_model)

        return x_2 # (B, L, d_model)

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, d_k, d_ff, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_k, d_ff, dropout) for i in range(n_layers)])
        self.layer_norm = LayerNormalization(d_model)
        self.output_norm=False
    def forward(self, x, e_mask):
        for i in range(self.n_layers):
            x = self.layers[i](x, e_mask)
        if self.output_norm:
            return self.layer_norm(x)
        else:
            return x

class FeedFowardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        
        ###########################################

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.linear_1(x)) # (B, L, d_ff)
        ###########################################

        # x = self.dropout(x)
        x = self.linear_2(x) # (B, L, d_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine=False, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model

        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, d_model) # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        self.positional_encoding = pe_matrix.cuda().requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(self.d_model) # (B, L, d_model)
        x = x + self.positional_encoding # (B, L, d_model)

        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, dropout):
        super().__init__()
        self.inf = 1e9

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        ###########################################
        self.dropout = nn.Dropout(dropout)
        self.attn_softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_model = d_model

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)
        k = self.w_k(k).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)
        v = self.w_v(v).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask) # (B, num_heads, L, d_k)
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, self.d_model) # (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)
        ###########################################
        # attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, num_heads, L, d_k)

        return attn_values
class DotAttention(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model=d_model
        self.dropout=nn.Dropout(dropout)
        self.softmax=nn.Softmax(dim=2)
    def forward(self,q,k,v):
        attn=torch.bmm(q,k.transpose(1,2))
        attn=self.softmax(attn)
        attn=self.dropout(attn)
        context=torch.bmm(attn,v)
        return context,attn