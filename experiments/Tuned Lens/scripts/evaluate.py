import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def compute_perplexity_and_kl(model, tokenizer, tuned_lens, dataloader, device, temperature=2.0):
    model.eval()
    tuned_lens.eval()

    total_ppl_tuned = {layer_idx: 0.0 for layer_idx in range(model.module.config.n_layer)}
    total_kl_tuned = {layer_idx: 0.0 for layer_idx in range(model.module.config.n_layer)}

    total_ppl_logit = {layer_idx: 0.0 for layer_idx in range(model.module.config.n_layer)}
    total_kl_logit = {layer_idx: 0.0 for layer_idx in range(model.module.config.n_layer)}

    total_samples = 0


    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Tuned & Logit Lens"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get hidden states and final logits from the frozen model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states[1:]  # Skip embeddings
            final_logits = outputs.logits  # Final layer logits
            
            # Assume tokenizer has an EOS token
            eos_token_id = tokenizer.eos_token_id  # Get EOS token ID
            
            # Shift input IDs left and append EOS token
            #targets = torch.cat([input_ids[:, 1:], (eos_token_id * torch.ones((input_ids.shape[0], 1), dtype=torch.long)).to(device)], dim=1)
            targets = input_ids[:, 1:]
            final_logits = final_logits[:, :-1, :]  # Align with targets
            
            batch_size, seq_len, vocab_size = final_logits.shape
            total_samples += batch_size * seq_len
            
            for layer_idx, h in enumerate(hidden_states):
                # --- Tuned Lens ---
                with autocast():
                    h_tuned = tuned_lens.module(h, layer_idx)

                logits_tuned = model.module.lm_head(h_tuned)
                logits_tuned = logits_tuned[:, :-1, :]

                # Compute Perplexity for Tuned Lens
                loss_ppl_tuned = F.cross_entropy(logits_tuned.reshape(-1, vocab_size), 
                                                 targets.reshape(-1), 
                                                 ignore_index=tokenizer.pad_token_id, 
                                                 reduction='sum')
                total_ppl_tuned[layer_idx] += loss_ppl_tuned.item()

                # Compute KL Divergence (Tuned Lens vs Final Logits)
                kld_tuned = F.kl_div(
                    F.log_softmax(logits_tuned / temperature, dim=-1),
                    F.softmax(final_logits / temperature, dim=-1),
                    reduction='batchmean'
                )
                total_kl_tuned[layer_idx] += kld_tuned.item()

                # --- Logit Lens (Directly apply lm_head) ---
                logits_logit_lens = model.module.lm_head(h)
                logits_logit_lens = logits_logit_lens[:, :-1, :]

                # Compute Perplexity for Logit Lens
                loss_ppl_logit = F.cross_entropy(logits_logit_lens.reshape(-1, vocab_size), 
                                                 targets.reshape(-1), 
                                                 ignore_index=tokenizer.pad_token_id, 
                                                 reduction='sum')
                total_ppl_logit[layer_idx] += loss_ppl_logit.item()

                # Compute KL Divergence (Logit Lens vs Final Logits)
                kld_logit = F.kl_div(
                    F.log_softmax(logits_logit_lens / temperature, dim=-1),
                    F.softmax(final_logits / temperature, dim=-1),
                    reduction='batchmean'
                )
                total_kl_logit[layer_idx] += kld_logit.item()

    # Compute averaged values
    avg_ppl_tuned = {layer_idx: torch.exp(torch.tensor(total_ppl_tuned[layer_idx] / total_samples)) 
                     for layer_idx in total_ppl_tuned}
    avg_kl_tuned = {layer_idx: total_kl_tuned[layer_idx] / len(dataloader) for layer_idx in total_kl_tuned}

    avg_ppl_logit = {layer_idx: torch.exp(torch.tensor(total_ppl_logit[layer_idx] / total_samples)) 
                     for layer_idx in total_ppl_logit}
    avg_kl_logit = {layer_idx: total_kl_logit[layer_idx] / len(dataloader) for layer_idx in total_kl_logit}

    return avg_ppl_tuned, avg_kl_tuned, avg_ppl_logit, avg_kl_logit
