import torch.nn as nn
import torch.nn.init as init

class TunedLens(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.translators = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=True) for _ in range(num_layers)
        ])
        
        # Apply Xavier initialization
        for layer in self.translators:
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, hidden_state, layer_idx):
        return self.translators[layer_idx](hidden_state)