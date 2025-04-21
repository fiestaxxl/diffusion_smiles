import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self,
                dim_model,
                num_heads):
        super().__init__()
        self.head_dim = dim_model // num_heads
        self.num_heads = num_heads
        assert self.head_dim * num_heads == dim_model, "embed_dim must be divisible by num_heads"

        self.Q = nn.Linear(dim_model, dim_model)
        self.K = nn.Linear(dim_model, dim_model)
        self.V = nn.Linear(dim_model, dim_model)
        self.ff = nn.Linear(dim_model, dim_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.ff.weight)

    def forward(self, q, k, v, mask=None):
        #q [bs, seq_len, dim_model]
        batch_size, seq_len, dim_model = q.size()
        q, k, v = self.Q(q), self.K(k), self.V(v) #[bs, seq_len, dim_model]

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


        scores = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim**0.5) #[bs, num_heads, seq_len, seq_len]

        if mask is not None:
            if mask.shape != scores.shape:
                mask = mask.unsqueeze(1)

            scores = scores.masked_fill(mask == 0, float('-inf'))
            #scores += mask

        attn_weights = nn.functional.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v) # (batch_size, num_heads, seq_length, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim_model)

        return self.ff(attn_output), attn_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 dim_model,
                 num_heads,
                 dim_ff,
                 alpha,
                 dropout=0.2):
        super().__init__()
        self.attention = MultiHeadAttention(dim_model, num_heads)
        self.alpha = alpha

        self.linear_net = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_ff, dim_model)
        )

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

                                
    def forward(self, x, mask=None):
        # Attention part
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x, mask=mask)
        x = x + self.dropout(attn_out)
        #x = self.norm1(x)

        # MLP part
        x = self.norm2(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        #x = self.norm2(x)

        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 dim_model,
                 num_heads,
                 dim_ff,
                 alpha,
                 dropout=0.2):
        super().__init__()
        self.self_attention = MultiHeadAttention(dim_model, num_heads)
        self.cross_attention = MultiHeadAttention(dim_model, num_heads)
        self.alpha = alpha

        self.linear_net = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_ff, dim_model)
        )


        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

                                
    def forward(self, x, k, v, mask=None):
        # Attention part
        x = self.norm1(x)
        attn_out, _ = self.self_attention(x, x, x, mask=mask)
        x = x + self.dropout(attn_out)
        #x = self.norm1(x)

        x = self.norm2(x)
        attn_out, _ = self.cross_attention(x, k, v, mask=None)
        x = x + self.dropout(attn_out)

        # MLP part
        x = self.norm3(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        #x = self.norm2(x)

        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self,
                 dim_model, 
                 num_heads,
                 dim_ff,
                 num_layers,
                 dropout=0.2):
        super().__init__()
        
        alpha = 0.81 * num_layers**0.25
        self.layers = nn.ModuleList([TransformerEncoderLayer(dim_model, num_heads, dim_ff, alpha, dropout) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):

        for layer in self.layers:
            x = layer(x, mask=mask)
        
        return x
 
class TransformerDecoder(nn.Module):
    def __init__(self,
                 dim_model, 
                 num_heads,
                 dim_ff,
                 num_layers,
                 dropout=0.2):
        super().__init__()
        
        alpha = 0.81 * num_layers**0.25
        self.layers = nn.ModuleList([TransformerDecoderLayer(dim_model, num_heads, dim_ff, alpha, dropout) for _ in range(num_layers)])
    
    def forward(self, x, t, mask=None):

        for layer in self.layers:
            x = layer(x, t, t, mask=mask)
        
        return x
       
class DiffusionTransformerEncoder(nn.Module):
    def __init__(self,
                 emb_dim,
                 time_dim,
                 dim_model, 
                 num_heads,
                 dim_ff,
                 num_layers,
                 vocab_size,
                 pad_idx,
                 dropout=0.2):
        super().__init__()

        self.smiles_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        #self.smiles_embedding = nn.Embedding(vocab_size, emb_dim)
        self.input_up_proj = nn.Sequential(
                                nn.Linear(emb_dim, dim_model),
                                nn.SiLU(), nn.Linear(dim_model, dim_model)
                                )
        self.positional_encoding = PositionalEncoding(d_model=dim_model)

        self.time_mlp = nn.Sequential(
                                SinusoidalPositionEmbeddings(time_dim),  # Convert time step to embedding
                                nn.Linear(time_dim, dim_model),
                                nn.GELU(),
                                nn.Linear(dim_model, dim_model),  # Match model dimension
                            )
        self.encoder = TransformerEncoder(dim_model, num_heads, dim_ff, num_layers, dropout)

        self.output_down_proj = nn.Sequential(nn.Linear(dim_model, dim_model),
                                              nn.SiLU(), nn.Linear(dim_model, emb_dim))
        
        self.lm_head = nn.Sequential(nn.Linear(emb_dim, emb_dim//2),
                                        nn.ReLU(),
                                        nn.Linear(emb_dim//2, vocab_size, bias=False))

        # self.lm_head = nn.Sequential(
        #                             nn.Linear(emb_dim, dim_model),
        #                             nn.Tanh(), 
        #                             nn.Linear(dim_model, emb_dim),
        #                             nn.Tanh(),
        #                             nn.Linear(emb_dim, vocab_size))
        
    def forward(self, x, time, mask=None):
        # x [bs, seq_len, emb_dim]
        # t [bs]

        _, seq_len, _ = x.shape

        x = self.input_up_proj(x) #bs seq_len dim_model
        x = self.positional_encoding(x)


        # Inject time information
        time_emb = self.time_mlp(time)  # [bs, 1, dim_model]
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [bs, seq_len, dim_model]
        x += time_emb  # Broadcast to match sequence length

        x = self.encoder(x, mask) # [bs, seq_len, dim_model]

        x = self.output_down_proj(x) # [bs, seq_len, emb_dim]

        return x
    
    def get_embeds(self, x):
        #x [bs, seq_len]
        return self.smiles_embedding(x)
    
    def get_logits(self, x):
        #x [bs, seq_len, emb_dim]
        return self.lm_head(x)

class DiffusionTransformerDecoder(nn.Module):
    def __init__(self,
                 emb_dim,
                 time_dim,
                 dim_model, 
                 num_heads,
                 dim_ff,
                 num_layers,
                 vocab_size,
                 pad_idx = None,
                 dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.smiles_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        self.input_up_proj = nn.Sequential(
                                nn.Linear(emb_dim, dim_model),
                                nn.SiLU(), nn.Linear(dim_model, dim_model)
                                )
        self.positional_encoding = PositionalEncoding(d_model=dim_model)

        self.time_mlp = nn.Sequential(
                                SinusoidalPositionEmbeddings(time_dim),  # Convert time step to embedding
                                nn.Linear(time_dim, dim_model),
                                nn.GELU(),
                                nn.Linear(dim_model, dim_model),  # Match model dimension
                            )
        self.decoder = TransformerDecoder(dim_model, num_heads, dim_ff, num_layers, dropout)

        self.output_down_proj = nn.Sequential(nn.Linear(dim_model, dim_model),
                                              nn.SiLU(), nn.Linear(dim_model, emb_dim))
        
        
        # self.lm_head = nn.Linear(emb_dim, vocab_size)
        # with torch.no_grad():
        #     self.lm_head.weight = self.smiles_embedding.weight
        
        self.lm_head = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                     nn.SiLU(),
                                     nn.Linear(emb_dim, vocab_size))
        
    def forward(self, x, time, mask=None):
        # x [bs, seq_len, emb_dim]
        # t [bs]

        _, seq_len, _ = x.shape

        x = self.input_up_proj(x) #bs seq_len dim_model
        x = self.positional_encoding(x)


        # Inject time information
        time_emb = self.time_mlp(time)  # [bs, 1, dim_model]
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [bs, seq_len, dim_model]

        x = self.decoder(x, time_emb, mask) # [bs, seq_len, dim_model]

        x = self.output_down_proj(x) # [bs, seq_len, emb_dim]

        return x
    
    def get_embeds(self, x):
        #x [bs, seq_len]
        return self.smiles_embedding(x)
    
    def get_logits(self, x):
        #x [bs, seq_len, emb_dim]
        return self.lm_head(x)


