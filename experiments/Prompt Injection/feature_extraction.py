import torch
import numpy as np
from tuned_lens.plotting import PredictionTrajectory

def get_prediction_trajectory(prompt: str, model, tokenizer, lens, max_length: int = 1024):
    """
    Computes the prediction trajectory of a given prompt using a model and a lens.

    Parameters:
    prompt (str): The input text prompt.
    model: The language model (e.g., GPT-2).
    tokenizer: The tokenizer corresponding to the model.
    lens: The tuned or logit lens used for analysis.
    max_length (int, optional): The maximum sequence length. Defaults to 1024.

    Returns:
    np.ndarray: The mean forward KL divergence statistics for each token in the trajectory.
                Returns None if the input prompt is empty.
    """
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=max_length)

    if not input_ids:
        return None

    targets = input_ids[1:] + [tokenizer.eos_token_id]

    with torch.no_grad():
        pred_traj = PredictionTrajectory.from_lens_and_model(
            lens=lens, model=model, input_ids=input_ids, tokenizer=tokenizer, targets=targets
        )

    return np.mean(getattr(pred_traj, 'forward_kl')().stride(1).stats, axis=-1)
