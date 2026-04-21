# Federated Learning on Breast Cancer Classification

## Overview

This project demonstrates a complete federated learning pipeline using the **Flower** framework for distributed machine learning. It compares the performance of a centralized model with a federated learning model trained on non-IID (Non-Identical and Independent) distributed data.

## Dataset

- **Source**: Breast Cancer Wisconsin Dataset (sklearn)
- **Samples**: 569 instances
- **Features**: 30 numerical features
- **Classes**: 2 (Binary Classification)
  - 0: Malignant (cancerous)
  - 1: Benign (non-cancerous)

## Project Workflow

### 1. **Environment Setup**
- Imports essential libraries: PyTorch, scikit-learn, Pandas, NumPy, Flower
- Sets random seeds for reproducibility

### 2. **Data Loading & Exploration**
- Loads the breast cancer dataset
- Displays dataset shape and class distribution
- Prepares features (X) and labels (y)

### 3. **Data Preprocessing**
- **Train-Test Split**: 80-20 split with stratification
- **Normalization**: StandardScaler applied (fitted on training data only)
- **Tensor Conversion**: Converts NumPy arrays to PyTorch tensors

### 4. **Centralized Model Training**

#### Model Architecture (BreastCancerMLP)
```
Input (30) → FC1 (64) → ReLU → FC2 (128) → ReLU → Dropout(0.3)
→ FC3 (64) → ReLU → Dropout(0.3) → FC4 (2) → Output
```

#### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 50
- **Batch Size**: 32
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

### 5. **Non-IID Data Partitioning**

The dataset is partitioned across 5 clients using the **Dirichlet distribution** with alpha=0.5, creating a realistic non-IID scenario where:
- Different clients have different class distributions
- Data is heterogeneous across clients
- Simulates real-world federated environments

**Function**: `partition_non_iid()`
- Uses Dirichlet distribution for realistic data imbalance
- Handles edge cases (empty clients)
- Returns client-indexed data partitions

### 6. **Federated Learning Implementation**

#### Flower Client (FlowerClient)
- Implements NumPyClient interface
- **Local Training**: 3 local epochs per round
- **Parameter Exchange**: Get/Set model parameters
- **Evaluation**: Client-side evaluation on local data

#### Federated Training Configuration
- **Number of Clients**: 5
- **Communication Rounds**: 20
- **Local Epochs**: 3 per client per round
- **Aggregation Strategy**: FedAvg (Federated Averaging)
- **Min Fit Clients**: 3
- **Min Available Clients**: 5

#### FedAvg Strategy
- Aggregates model parameters from all participating clients
- Centralized evaluation on held-out test set
- Tracks accuracy across rounds

### 7. **Evaluation & Comparison**

The notebook performs detailed evaluation including:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity (true positive rate)
- **F1-Score**: Harmonic mean of precision and recall

**Comparison**:
- Centralized Model: Baseline trained on full dataset
- Federated Model: Aggregated from distributed clients

## Key Features

✅ **Non-IID Data Distribution**: Realistic federated scenario with heterogeneous data  
✅ **Privacy-Preserving**: Clients only share model parameters, not raw data  
✅ **Scalable Architecture**: Easily extends to more clients  
✅ **Comprehensive Evaluation**: Multiple metrics for performance assessment  
✅ **Error Handling**: Graceful handling of Ray/Flower compatibility issues  

## Dependencies

```
torch
numpy
pandas
scikit-learn
flwr (Flower)
ray (for simulation)
```

## How to Run

1. **Activate the virtual environment**:
   ```bash
   source fl_env/bin/activate
   ```

2. **Run the notebook**:
   ```bash
   jupyter notebook main.ipynb
   ```

3. **Execute cells sequentially** to:
   - Load and preprocess data
   - Train centralized model
   - Setup federated learning
   - Run federated simulation
   - Compare results

## Expected Output

- Centralized model training loss per epoch
- Non-IID data distribution across clients
- Federated learning accuracy per round
- Performance comparison metrics (accuracy, precision, recall, F1)
- Federated model test accuracy

## Troubleshooting

### Flower Simulation Issues
If the simulation fails with Ray compatibility errors:
```
Error: Incompatible Ray/Python version
Solution: Install Ray 2.20.0+ with: pip install 'ray[default]>=2.20.0'
Alternative: Use start_server/start_numpy_client with separate processes
```

## Project Structure

```
Code/
├── main.ipynb                 # Main federated learning notebook
├── README.md                  # This file
├── fl_env/                    # Python 3.12 virtual environment
└── flwr_env/                  # Python 3.10 virtual environment
```

## Learning Objectives

This project demonstrates:
1. **Federated Learning Concepts**: Distributed model training without centralized data
2. **Non-IID Data Handling**: Working with heterogeneous data distributions
3. **Model Aggregation**: Parameter averaging across clients
4. **Privacy-Preserving ML**: Collaborative learning while maintaining data privacy
5. **PyTorch & Flower Integration**: Building production-ready FL systems

## Results Interpretation

- **If FL ≈ Centralized**: Model converges well despite data heterogeneity
- **If FL < Centralized**: Non-IID distribution challenges (expected)
- **Key Insight**: Trade-off between privacy and model performance in FL

## References

- [Flower Documentation](https://flower.dev/)
- [Federated Learning Overview](https://arxiv.org/abs/1602.05629)
- [Non-IID Data Distribution](https://arxiv.org/abs/1909.06335)
- [PyTorch Documentation](https://pytorch.org/)
