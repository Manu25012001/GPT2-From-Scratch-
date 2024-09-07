import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, epsilon=1e-12):
        """
      Initialize LayerNorm module.
      """
        super().__init__()

        # Learnable weight parameter for scaling.
        self.weight = nn.Parameter(torch.ones(hidden_size))

        # Learnable bias parameter for shifting.
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        # Small value to avoid division by zero in normalization.
        self.epsilon = epsilon

    def forward(self, x):
        # Compute mean and variance along the last dimension.
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)

        # Normalize the input tensor.
        x = (x - u) / torch.sqrt(s + self.epsilon)

        # Scale and shift using learnable parameters.
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        """
      The CONV 1D layer can be thought of as a linear layer itself.
      It is casting an initial tensor x (having the final
      dimension of x.size(-1)) being passed to it to have a final dimension
      of size self.nf.

      We do this to be able to cast the input to query, key and value matrices.

      """
        super().__init__()

        # Number of filters (output channels).
        self.nf = nf

        # Initialize weight parameter with normal distribution and small standard deviation.
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)

        # Initialize bias parameter with zeros.
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        # Calculate the size of the output tensor.
        size_out = x.size()[:-1] + (self.nf,)

        # Perform 1D convolution using matrix multiplication.
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)

        # Reshape the tensor to the desired output size.
        x = x.view(*size_out)

        return x


class FeedForward(nn.Module):
    def __init__(self, dropout, d_model=768, nx=768 * 4):
        """
      Initialize Feedforward Layer module.
      """
        super().__init__()

        # 1D Convolutional Layer for the first linear transformation.
        self.c_fc = Conv1D(d_model, nx)

        # 1D Convolutional Layer for the second linear transformation.
        self.c_proj = Conv1D(nx, d_model)

        # Activation function (GELU).
        self.act = F.gelu

        # Dropout layer with specified dropout probability.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply the first linear transformation, activation, and dropout.
        x = self.dropout(self.c_proj(self.act(self.c_fc(x))))

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_head=12, n_ctx=1024, bias=True, scale=True):
        """
      Initialize Attention Layer module.

      """
        super().__init__()

        # Number of attention heads.
        self.n_head = n_head

        # Dimensionality of the model.
        self.d_model = d_model

        # 1D Convolutional Layer for attention weights computation.
        self.c_attn = Conv1D(d_model, d_model * 3)

        # Flag to scale attention scores.
        self.scale = scale

        # Softmax activation for attention scores.
        self.softmax = nn.Softmax(dim=-1)

        # Lower triangular bias matrix for masking future tokens.
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

        # Dropout layer for regularization.
        self.dropout = nn.Dropout(0.1)

        # 1D Convolutional Layer for output projection.
        self.c_proj = Conv1D(d_model, d_model)

    def split_heads(self, x):
        """
      Split the last dimension of the input tensor into multiple heads.

      """
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _attn(self, q, k, v, attn_mask=None):
        """
      Compute attention scores and apply attention to values.

      """
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.scale:
            scores = scores / math.sqrt(v.size(-1))
        nd, ns = scores.size(-2), scores.size(-1)
        if attn_mask is not None:
            scores = scores + attn_mask
        scores = self.softmax(scores)
        scores = self.dropout(scores)
        outputs = torch.matmul(scores, v)
        return outputs

    def merge_heads(self, x):
        """
      Merge the heads back to the original shape.
      """
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def forward(self, x):

        # Compute attention weights using 1D convolution.
        x = self.c_attn(x)

        # Split the tensor into query, key, and value.
        q, k, v = x.split(self.d_model, dim=2)

        # Split heads for query, key, and value.
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        # Apply attention mechanism.
        out = self._attn(q, k, v)

        # Merge the heads back to the original shape.
        out = self.merge_heads(out)

        # Apply output projection.
        out = self.c_proj(out)

        return out


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, d_head=768, n_ctx=1024, base=10_000):
        super().__init__()
        self.d_head = d_head
        self.n_ctx = n_ctx
        self.inv_freq = 1. / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self._construct_rope_matrices()

    def _construct_rope_matrices(self):
        """Constructs rotary embedding matrices for additive version.
        Configured for x beeing of shape
        (batch_size, seqlen, d_model).
        """
        assert self.d_head % 2 == 0
        # [t1, t1, t2, t2, t3, t3, ...]
        thetas = 1000 ** (
                -2.0 * torch.arange(1, self.d_head / 2 + 1) / self.d_head
        ).repeat_interleave(2)
        positions = torch.arange(1, self.n_ctx + 1).float()
        # [ [1t1, 1t1, 1t2, 1t2, ...],
        #   [2t1, 2t1, 2t2, 2t2, ...],
        #   [3t1, 3t1, 3t2, 3t2, ...],
        #   ...                       ]
        args = positions.reshape(-1, 1) @ thetas.reshape(1, -1)
        self.register_buffer("rope_sin", torch.sin(args))
        self.register_buffer("rope_cos", torch.cos(args))

    def _reorder_for_rope_sin(self, x):
        """Reorders the inputs for the
      multiplication with the sinus-part of the RoPE. Configured for x beeing
      having d_head as last dimension, should be of shape
      (batch_size, n_heads, seqlen, d_head).
      """
        # [x1, x3, x5, ...]
        x_odd = x[..., ::2]
        # [x2, x4, x6, ...]
        x_even = x[..., 1::2]
        # [[-x2, x1], [-x4, x3], [-x6, x5], ...]
        x_stacked = torch.stack([-x_even, x_odd], dim=-1)
        # [-x2, x1, -x4, x3, ...]
        return x_stacked.flatten(start_dim=-2)

    def forward(self, x):
        """Applies RoPE the inputs.
      Configured for x being of shape (batch_size, n_heads, seqlen, d_head).
      """
        T = x.shape[2]
        x_sin = self._reorder_for_rope_sin(x)
        x_rope = x * self.rope_cos[:T, :] + x_sin * self.rope_sin[:T, :]
        return x_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=768, n_head=12, n_ctx=1024, d_head=64, bias=True, scale=True, num_groups=2):
        """
        Initialize Grouped Query Attention Layer module.
        """
        super().__init__()

        # Number of attention heads.
        self.n_head = n_head

        # Dimensionality of the model.
        self.d_model = d_model

        # Number of query groups
        self.num_groups = num_groups

        # Check if the number of groups divides the dimensionality evenly
        assert d_model % num_groups == 0, "Number of groups must evenly divide the dimensionality."

        # Dimensionality of each group
        self.group_dim = d_model // num_groups

        # Dimensionality of each head
        self.head_dim = d_model // n_head

        # 1D Convolutional Layer for attention weights computation.
        self.c_attn = Conv1D(d_model, d_model * 3)

        # Flag to scale attention scores.
        self.scale = scale

        # Softmax activation for attention scores.
        self.softmax = nn.Softmax(dim=-1)

        # Lower triangular bias matrix for masking future tokens.
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

        # Dropout layer for regularization.
        self.dropout = nn.Dropout(0.1)

        # 1D Convolutional Layer for output projection.
        self.c_proj = Conv1D(d_model, d_model)

        # Rotary Position Embedding
        self.rpe = RotaryPositionalEmbedding(self.head_dim, n_ctx)

    def split_heads(self, x):
        """
        Split the last dimension of the input tensor into multiple heads.
        """
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _attn(self, q, k, v, attn_mask=None):
        """
        Compute attention scores and apply attention to values.
        """
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.scale:
            scores = scores / torch.sqrt(torch.tensor(self.group_dim).float())  # Scale by sqrt(group_dim)
        nd, ns = scores.size(-2), scores.size(-1)
        if attn_mask is not None:
            scores = scores + attn_mask
        scores = self.softmax(scores)
        scores = self.dropout(scores)
        outputs = torch.matmul(scores, v)
        return outputs

    def merge_heads(self, x):
        """
        Merge the heads back to the original shape.
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def group_queries(self, x):
        """
        Group queries based on the number of query groups.
        """
        return torch.split(x, self.group_dim, dim=-1)

    def forward(self, x):

        # Compute attention weights using 1D convolution.
        x = self.c_attn(x)

        # Split the tensor into query, key, and value.
        q, k, v = x.split(self.d_model, dim=2)

        # Split heads for query, key, and value.
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        # Apply RopE embeddings
        q = self.rpe(q)
        k = self.rpe(k)

        # Group queries
        grouped_queries = self.group_queries(q)

        # Apply grouped attention mechanism
        grouped_outputs = []
        for group_queries in grouped_queries:
            out = self._attn(group_queries, k, v)
            grouped_outputs.append(out)

        # Merge the grouped outputs
        out = torch.cat(grouped_outputs, dim=-1)

        # Merge the heads back to the original shape.
        out = self.merge_heads(out)

        # Apply output projection.
        out = self.c_proj(out)

        return out


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_head, n_ctx, use_gqa=False, num_groups=2, dropout=0.1):
        """
      Initialize Transformer Block module.
      """
        super().__init__()

        # GroupQuery-Attention Layer
        if use_gqa:
            self.attn = GroupedQueryAttention(d_model=d_model, n_head=n_head, d_head=64,
                                              n_ctx=n_ctx, bias=True, scale=True,
                                              num_groups=num_groups)
        else:
            # Self-Attention Layer
            self.attn = MultiHeadAttention(d_model=d_model, n_head=n_head, n_ctx=n_ctx, bias=True, scale=True)

        # Feedforward Layer
        self.feedforward = FeedForward(dropout=0.1, d_model=d_model, nx=d_model * 4)

        # Layer Normalization for the attention output
        self.ln_1 = LayerNorm(d_model)

        # Layer Normalization for the feedforward output
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x):
        # Self-Attention Layer with Layer Normalization and skip connection
        x = x + self.attn(self.ln_1(x))

        # Feedforward Layer with Layer Normalization and skip connection
        x = x + self.feedforward(self.ln_2(x))

        return x