from layers import LayerNorm, Conv1D, TransformerBlock
from config import GPT2Config
import torch.nn as nn
import torch
import copy


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        """
        Initialize GPT-2 model.
        """
        super().__init__()

        #GPT2 Configurations
        self.config = config

        # Number of transformer layers
        self.nlayers = config.n_layer

        # Transformer block as the basic building unit
        self.block = TransformerBlock(d_model=config.d_model, n_head=config.n_head,
                                      n_ctx=config.n_ctx, use_gqa=config.use_gqa,
                                      num_groups=config.num_groups,
                                      dropout=0.1)

        # List of transformer blocks forming the layers
        self.h = nn.ModuleList([copy.deepcopy(self.block) for i in range(self.nlayers)])

        # Word Embedding layer
        self.wte = nn.Embedding(config.vocab_size, config.d_model)

        # Position Embedding layer
        self.wpe = None
        if not config.use_gqa:
            self.wpe = nn.Embedding(config.n_ctx, config.d_model)

        # Dropout layer for regularization
        self.drop = nn.Dropout(0.1)

        # Layer Normalization for the final output
        self.ln_f = LayerNorm(config.d_model)

        # Linear layer for output predictions
        self.out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # CrossEntropyLoss for training
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        # Share weights between output layer and word embedding
        self.out.weight = self.wte.weight

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Custom weight initialization for linear, embedding, and convolutional layers.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, src, labels=None, pos_ids=None):

        inp = self.wte(src)

        if not self.config.use_gqa:
            # If pos_ids is not provided, create positional IDs
            if pos_ids is None:
                pos_ids = torch.arange(0, src.size(-1)).unsqueeze(0)

            # Apply position embeddings with dropout
            inp = self.drop((inp + self.wpe(pos_ids)))

        # Forward pass through transformer layers
        for i in range(self.nlayers):
            inp = self.h[i](inp)

        # Apply layer normalization to the final output
        inp = self.ln_f(inp)

        # Linear layer for output predictions
        logits = self.out(inp)

        # Prepare outputs
        outputs = (logits,) + (inp,)

        # If labels are provided, compute and return the loss
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs
            return outputs

        # Otherwise, return logits
        return logits