import torch
from torch import nn
from torch.nn import functional as F

from typing import Dict, List, Optional, Tuple, Union
import random
import math
import re
import time

class Lens:
    """
    Implementation of the logit and tuned lens method for visualizing intermediate layer predictions
    in transformer models.
    
    The logit and tuned lens allows us to decode hidden states at each layer of a transformer
    using the unembedding matrix to observe how predictions evolve through the network.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the LogitLens.
        
        Args:
            model: The transformer model
        """
        self.model = model
        self.unembed_weight = self.model.decoder.weight # weight of the Linear module in the implementation of the course
    
    def logit_lens(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply the standard logit lens to a hidden state.
        
        Args:
            hidden_state: Hidden state from an intermediate layer
            
        Returns:
            Logits obtained by projecting the hidden state through the unembedding matrix
        """
        # Project through the unembedding matrix
        logits = F.linear(hidden_state, self.unembed_weight)
        return logits
    
    def tuned_lens(self, 
                   hidden_state: torch.Tensor, 
                   translator: nn.Module) -> torch.Tensor:
        """
        Apply the tuned logit lens to a hidden state using a learned translator.
        
        Args:
            hidden_state: Hidden state from an intermediate layer
            translator: A learned affine transformation module
            
        Returns:
            Logits obtained by applying the translator and then projecting through
            the unembedding matrix
        """
        # Apply the translator
        translated_state = translator(hidden_state)
        
        # Project through the unembedding matrix
        logits = F.linear(translated_state, self.unembed_weight)
        return logits

class TranslatorModule(nn.Module):
    """A learned affine transformation for the tuned lens."""
    
    def __init__(self, hidden_size: int):
        """
        Initialize the translator module.
        
        Args:
            hidden_size: Size of the hidden state
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the translator to a hidden state."""
        return self.linear(x)

class TransformerWithLens(nn.Module):
    """
    A wrapper for a transformer model that captures intermediate hidden states
    and applies the lens method.
    """
    
    def __init__(self, 
                 transformer_model: nn.Module, 
                 num_layers: int,
                 hidden_size: int,
                 
                 use_tuned_lens: bool = False):
        """
        Initialize the wrapper.
        
        Args:
            transformer_model: The transformer model to wrap
            num_layers: Number of layers in the transformer
            hidden_size: Dimensionality of the hidden states
            use_tuned_lens: Whether to use the tuned lens with learned translators
        """
        super().__init__()
        self.transformer = transformer_model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_tuned_lens = use_tuned_lens

        # Freeze the parameters of the transformer
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Initialize the logit lens
        self.logit_lens = Lens(transformer_model)
        
        # Initialize one translator per layer for tuned lens if needed
        if use_tuned_lens:
            self.translators = nn.ModuleList([
                TranslatorModule(hidden_size) for _ in range(num_layers)
            ])
    
    def forward(self, 
                inputs: Dict[str, torch.Tensor]
                ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass that includes logit lens visualizations.
        
        Args:
            inputs: Input tensors for the transformer model
            
        Returns:
            A dictionary containing:
                - 'output': The original model output
                - 'logit_lens_outputs': List of logit lens outputs for each layer
                - 'tuned_lens_outputs': List of tuned lens outputs for each layer (if enabled)
        """
        # Get the original model output and the hidden states
        outputs, hidden_states = self.transformer(inputs)
                
        # Apply logit lens to each hidden state
        logit_lens_outputs = [
            self.logit_lens.logit_lens(hidden_state)
            for hidden_state in hidden_states
        ]
        
        # Apply tuned lens if enabled
        tuned_lens_outputs = None
        if self.use_tuned_lens:
            tuned_lens_outputs = [
                self.logit_lens.tuned_lens(hidden_states[i], self.translators[i])
                for i in range(self.num_layers)
            ]
        
        return {
            'output': outputs,
            'logit_lens_outputs': logit_lens_outputs,
            'tuned_lens_outputs': tuned_lens_outputs
        }

