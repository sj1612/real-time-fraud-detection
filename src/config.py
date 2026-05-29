import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Datasets
REAL_DATA_PATH = os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv")
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "synthetic_transactions.csv")

# Device setting (use GPU if available, safe from DLL import failures)
DEVICE = "cpu"
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except (ImportError, OSError):
    pass

# Feature Pipeline Configuration
# PaySim columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
RAW_NUMERIC_FEATURES = [
    "amount", 
    "oldbalanceOrg", 
    "newbalanceOrig", 
    "oldbalanceDest", 
    "newbalanceDest",
    "is_transfer",
    "is_cash_out"
]

# Stateful engineered features added by feature_store.py
ENGINEERED_FEATURES = [
    "balance_error_orig",   # oldbalanceOrg - newbalanceOrig - amount
    "balance_error_dest",   # oldbalanceDest + amount - newbalanceDest
    "velocity_10m",         # Number of transactions by this origin in last 10m
    "velocity_1h",          # Number of transactions by this origin in last 1h
    "monetary_deviation",   # Current amount / running mean amount
    "impossible_travel"     # Binary flag for sudden geographical speed (> 1000 km/h)
]

ALL_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES

# Sequence configuration for LSTM model
SEQUENCE_LENGTH = 5  # We look at historical window of 5 transactions per sender

# Model Hyperparameters - Isolation Forest
IFOREST_PARAMS = {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": 0.01,  # Assume 1% anomalies on normal-ish training calibration
    "random_state": 42,
    "n_jobs": -1
}

# Model Hyperparameters - PyTorch LSTM Autoencoder
LSTM_PARAMS = {
    "hidden_dim": 32,
    "latent_dim": 16,
    "num_layers": 2,
    "lr": 0.001,
    "batch_size": 256,
    "epochs": 10,
    "contamination": 0.01
}

# Initial Ensemble configuration
DEFAULT_ALPHA = 0.5  # Blending weight for Consensus Score: alpha * P_IF + (1-alpha) * P_LSTM

