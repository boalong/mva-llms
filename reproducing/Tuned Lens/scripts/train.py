from models.gpt2_model import initialize_components
from models.tuned_lens import TunedLens
from models.dataset import CustomDataset, tokenize_function
from config.settings import Config
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.cuda.amp import autocast, GradScaler

def train_loop(dataloader, model, tuned_lens, device='cuda', verbose=False):
    optimizer = torch.optim.AdamW(tuned_lens.parameters(), 
                                 lr=Config.lr,
                                 weight_decay=1e-4)
    
    scaler = GradScaler(enabled=Config.mixed_precision)
    
    for epoch in range(Config.num_epochs):
        model.module.eval()
        tuned_lens.module.train()
        epoch_loss = torch.tensor(0.0, device=device)
        
        optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.no_grad(), autocast(enabled=Config.mixed_precision):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
                final_logits = outputs.logits
            
            with autocast(enabled=Config.mixed_precision):
                losses = []
                for layer_idx, h in enumerate(hidden_states):
                    h_tuned = tuned_lens(h, layer_idx)
                    logits_tl = model.module.lm_head(h_tuned)
                    
                    loss = F.kl_div(
                        F.log_softmax(logits_tl / Config.temperature, dim=-1),
                        F.softmax(final_logits / Config.temperature, dim=-1),
                        reduction='batchmean',
                        log_target=False
                    )
                    losses.append(loss)
                
                loss_total = torch.stack(losses).mean()
            
            scaler.scale(loss_total).backward()
            epoch_loss += loss_total.detach()
            
            if (step + 1) % Config.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        
        if (step + 1) % Config.accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        avg_loss = epoch_loss.item() / len(dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

