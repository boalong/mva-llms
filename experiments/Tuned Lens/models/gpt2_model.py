from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch
from config.settings import Config

def initialize_components(device):
    model = AutoModelForCausalLM.from_pretrained('gpt2', 
                                               output_hidden_states=True,
                                               torch_dtype=torch.float16 if Config.mixed_precision else torch.float32)
    
    if Config.mixed_precision:
        model = model.half()
    
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer