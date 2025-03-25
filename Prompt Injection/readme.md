# Anomaly Detection & Prompt Analysis

This repository contains implementations of anomaly detection models and prompt analysis tools using machine learning techniques.

## Features
- **Anomaly Detection**: Uses Isolation Forest, Local Outlier Factor, and Mahalanobis distance for detecting outliers in trajectory data.
- **Prompt Dataset Management**: Loads and processes normal and adversarial prompts from Hugging Face datasets.
- **Model Wrapping**: Integrates tuned and logit lens approaches for analyzing transformer models.
- **Prediction Trajectory Analysis**: Computes forward KL divergence for analyzing model predictions.

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Anomaly Detection
```python
from anomaly_detector import AnomalyDetector
import numpy as np

# Train model
detector = AnomalyDetector()
normal_data = np.random.rand(100, 10)  # Example data
detector.train_detectors(normal_data)

# Evaluate on test set
test_data = np.random.rand(50, 10)
test_labels = np.random.randint(0, 2, size=50)
results = detector.evaluate(test_data, test_labels)
print(results)
```

### 2. Prompt Dataset Management
```python
from prompt_dataset import PromptDataset

dataset = PromptDataset.from_huggingface()
train_set, test_set = dataset.split()
```

### 3. Model Wrapper
```python
from model_wrapper import ModelWrapper

wrapper = ModelWrapper(lens_type='tuned_lens')
model = wrapper.get_model()
tokenizer = wrapper.get_tokenizer()
lens = wrapper.get_lens()
```

### 4. Prediction Trajectory Analysis
```python
from trajectory_analysis import get_prediction_trajectory

prompt = "Hello, how are you?"
trajectory = get_prediction_trajectory(prompt, model, tokenizer, lens)
print(trajectory)
```

## File Structure
```
/
├── anomaly_detector.py  # Anomaly detection models
├── prompt_dataset.py    # Dataset processing
├── model_wrapper.py     # Model loading and lens application
├── trajectory_analysis.py  # Prediction trajectory computation
├── utils.py                # Auxiliary functions 
├── requirements.txt     # Dependencies
├── README.md            # Documentation
```

## Author
Omar Arbi

