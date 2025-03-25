import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tuned_lens import TunedLens, LogitLens

class ModelWrapper:
    """
    A wrapper class for loading a GPT-2 model with an associated lens (either Tuned Lens or Logit Lens) 
    for interpretability tasks.
    """
    def __init__(self, model_name='gpt2-large', lens_type: str = 'tuned_lens', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the ModelWrapper with GPT-2 and the specified lens type.
        
        Parameters:
        lens_type (str): Type of lens to use ('tuned_lens' or 'logit_lens').
        device (str): Device to load the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True).half().to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if lens_type == 'tuned_lens':
            self.lens = TunedLens.from_model_and_pretrained(self.model).to(self.device)
        elif lens_type == 'logit_lens':
            self.lens = LogitLens.from_model(self.model).to(self.device)
        else:
            raise ValueError("Invalid lens type. Use 'tuned_lens' or 'logit_lens'.")

    def get_model(self) -> GPT2LMHeadModel:
        """
        Returns the loaded GPT-2 model.
        
        Returns:
        GPT2LMHeadModel: The GPT-2 model instance.
        """
        return self.model

    def get_tokenizer(self) -> GPT2Tokenizer:
        """
        Returns the tokenizer associated with GPT-2.
        
        Returns:
        GPT2Tokenizer: The GPT-2 tokenizer instance.
        """
        return self.tokenizer

    def get_lens(self):
        """
        Returns the initialized lens for interpretability.
        
        Returns:
        TunedLens or LogitLens: The selected lens instance.
        """
        return self.lens
