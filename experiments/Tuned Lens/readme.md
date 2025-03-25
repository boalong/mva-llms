Here's a clean, comprehensive `README.md` file for your Tuned Lens project:

```markdown
# Tuned Lens Analysis for Transformer Models

This project implements and evaluates a "Tuned Lens" approach for analyzing hidden states in GPT-2 transformer models. The Tuned Lens provides layer-wise interpretation of model behavior.

## 🚀 Quick Start

1. **Clone the repository**


2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter notebooks**
   ```bash
   jupyter notebook notebooks/
   ```

## 📂 Project Structure

```
tuned-lens/
├── config/                # Configuration settings
│   └── settings.py        
├── data/                  # Data storage
├── models/                # Model implementations
│   ├── tuned_lens.py      # Tuned Lens module
│   ├── gpt2_model.py      # GPT-2 wrapper
│   └── dataset.py         # Data handling
├── notebooks/             # Interactive analysis
│   ├── tuned_lens.ipynb     # Training workflow
├── scripts/               # Command-line tools
│   ├── train.py           
│   └── evaluate.py        
├── utils/                 # Utilities
│   └── visualization.py   # Plotting functions
└── requirements.txt       # Dependencies
```

## 🧠 Key Features

- **Tuned Lens Implementation**: Custom module for analyzing transformer hidden states
- **Interactive Notebooks**: Complete workflow from training to visualization
- **Reproducible Research**: Configurable settings and modular design
- **Visual Analysis**: Layer-wise performance metrics and comparisons

## 📊 Notebook Workflow

1. **`tuned_lens.ipynb`**
    * Train Model
        - Load and preprocess the "high-quality-english-sentences" dataset
        - Initialize GPT-2 model with frozen parameters
        - Train the Tuned Lens adapter

    * Evaluate the Tuned Lens:
        - Compute layer-wise metrics:
            - Perplexity
            - KL divergence
        - Compare Tuned Lens vs standard Logit Lens

    * Visualize Results
        - Generate interactive visualizations
        - Analyze attention patterns
        - Interpret model behavior

## ⚙️ Configuration

Modify training parameters in `config/settings.py`:

```python
class Config:
    batch_size = 16          # Training batch size
    num_epochs = 3           # Training epochs
    learning_rate = 1e-3     # Optimizer learning rate
    temperature = 2.0        # Softmax temperature
    mixed_precision = True   # Use FP16 training
```

## 💻 Command Line Usage

**Training:**
```bash
python scripts/train.py \
    --batch_size 16 \
    --epochs 3 \
    --lr 1e-3
```

**Evaluation:**
```bash
python scripts/evaluate.py \
    --model_checkpoint ./models/checkpoint.pt \
    --output_dir ./results/
```

## 📈 Example Results

![Layer-wise KL Divergence](docs/images/kl_divergence.png)
*Comparison of KL divergence across model layers*

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request


## Author
Omar Arbi