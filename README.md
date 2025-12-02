# DistilBERT Sentiment Classification on SST-2

A comprehensive tutorial notebook demonstrating how to fine-tune **DistilBERT** for binary sentiment classification using the **Stanford Sentiment Treebank v2 (SST-2)** dataset from the GLUE benchmark. This project includes parallel training experiments with both **TensorBoard** and **Weights & Biases (W&B)** logging for comparison.

## Overview

This project provides a step-by-step tutorial for:
- Fine-tuning DistilBERT (a lightweight BERT variant) for sentiment classification
- Working with the SST-2 dataset from the GLUE benchmark
- Comparing two popular experiment tracking tools: TensorBoard and Weights & Biases
- Implementing reproducible training pipelines with fixed random seeds

## Requirements

### Python Version
- Python 3.8 or higher (tested with Python 3.11.14)

### Required Packages

```
ipykernel>=6.0.0
transformers>=4.30.0
datasets>=2.12.0
torch>=2.0.0
tensorboard>=2.13.0
wandb>=0.15.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
evaluate>=0.4.0
accelerate>=0.20.0
```

## Installation

### Option 1: Using Conda (Recommended)

1. **Create a new conda environment:**
   ```bash
   conda create -n sentiment_classification python=3.11
   conda activate sentiment_classification
   ```

2. **Install ipykernel:**
   ```bash
   conda install ipykernel -y
   # OR
   pip install ipykernel
   ```

3. **Install required packages:**
   ```bash
   pip install transformers datasets torch tensorboard wandb scikit-learn pandas numpy evaluate accelerate
   ```

4. **Register the kernel with Jupyter:**
   ```bash
   python -m ipykernel install --user --name sentiment_classification --display-name "sentiment_classification"
   ```

### Option 2: Using pip only

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install packages:**
   ```bash
   pip install ipykernel transformers datasets torch tensorboard wandb scikit-learn pandas numpy evaluate accelerate
   ```

3. **Register the kernel:**
   ```bash
   python -m ipykernel install --user --name sentiment_classification --display-name "sentiment_classification"
   ```

### For Weights & Biases (Optional)

If you want to use W&B cloud logging, you'll need to login:

```bash
wandb login
```

Or set the API key as an environment variable:
```bash
export WANDB_API_KEY=your_api_key_here
```

## Usage

### Running the Notebook

1. **Open Jupyter Notebook or JupyterLab:**
   ```bash
   jupyter notebook
   # OR
   jupyter lab
   ```

2. **Select the kernel:**
   - When opening the notebook, select the `sentiment_classification` kernel (or your environment name)
   - In VS Code: Select the Python interpreter for your environment

3. **Run cells sequentially:**
   - Execute cells from top to bottom
   - The notebook is designed to be run cell-by-cell

### Viewing Logs

#### TensorBoard

After training, view TensorBoard logs:

```bash
tensorboard --logdir=./logs/tensorboard
```

Then open your browser to `http://localhost:6006`

#### Weights & Biases

1. Go to https://wandb.ai
2. Navigate to your project: `distilbert-sst2-tutorial`
3. View real-time metrics, hyperparameters, and system information

## Project Structure

```
sentiment_classification_DistilBERT/
│
├── distilbert_sst2_tutorial.ipynb  # Main tutorial notebook
├── README.md                       # This file
│
├── logs/                           # Generated during training
│   ├── tensorboard/               # TensorBoard log files
│   └── wandb/                      # W&B log files
│
└── results_*/                      # Model checkpoints (generated)
    ├── results_tensorboard/        # TensorBoard run checkpoints
    └── results_wandb/              # W&B run checkpoints
```

## References

- **DistilBERT Paper**: [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
- **SST-2 Dataset**: [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)
- **GLUE Benchmark**: [GLUE: A Multi-Task Benchmark and Analysis Platform](https://gluebenchmark.com/)
- **Hugging Face Transformers**: [Documentation](https://huggingface.co/docs/transformers/)
- **TensorBoard**: [Official Documentation](https://www.tensorflow.org/tensorboard)
- **Weights & Biases**: [Official Documentation](https://docs.wandb.ai/)

## License

This project is created for educational purposes. Feel free to use and modify as needed.

## Author

**Sai Karyekar**

---

Happy Learning!

If you find this project helpful, please consider giving it a star ⭐

