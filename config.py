from dataclasses import dataclass

# Configuration data class for GPT-2 model.
@dataclass
class GPT2Config:
    # Vocabulary size for tokenization.
    vocab_size: int = 50257

    # Maximum number of positions in a sequence.
    n_positions: int = 1024

    # Dimensionality of the causal mask (usually same as n_positions)
    n_ctx: int = 1024

    # Dimensionality of the embedding vector for each token.
    d_model: int = 768

    # Number of transformer layers in the model.
    n_layer: int = 12

    # Number of attention heads in each transformer layer.
    n_head: int = 12

    # Epsilon value for layer normalization.
    layer_norm_epsilon: float = 1e-5

    # Range for weight initialization.
    initializer_range: float = 0.02

    # Use GQA with RoPE
    use_gqa: bool = False

    # Number of groups for GQA
    num_groups: int = 2


