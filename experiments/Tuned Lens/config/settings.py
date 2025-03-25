class Config:
    batch_size = 16
    accumulation_steps = 4
    num_epochs = 3
    lr = 1e-3
    temperature = 2.0
    mixed_precision = True
    gradient_checkpointing = True