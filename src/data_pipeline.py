import os
import random
import time
import csv
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import (
    REAL_DATA_PATH,
    SYNTHETIC_DATA_PATH,
    ALL_FEATURES,
    SEQUENCE_LENGTH,
    MODEL_DIR
)
from src.feature_store import StatefulFeatureStore

def generate_synthetic_data(n_records: int = 50000, seed: int = None) -> List[Dict[str, Any]]:
    """
    Generate highly realistic, stateful simulated banking transactions 
    mimicking the PaySim schema, injecting temporal and structural fraud patterns.
    Uses pure-python standard csv package (fully pandas-free).
    """
    print(f"Generating {n_records} synthetic transactions...")
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Pre-define some popular accounts
    senders = [f"C{random.randint(100000, 199999)}" for _ in range(500)]
    merchants = [f"M{random.randint(200000, 299999)}" for _ in range(300)]
    receivers = [f"C{random.randint(300000, 399999)}" for _ in range(200)]
    
    # Track origin balances
    balances = {sender: float(np.random.exponential(50000.0) + 1000.0) for sender in senders}
    dest_balances = {rcv: float(np.random.exponential(10000.0) + 10.0) for rcv in receivers}
    dest_balances.update({m: 0.0 for m in merchants}) # merchants start at 0
    
    # Geographic locations for accounts
    sender_coords = {sender: (random.uniform(40.5, 40.9), random.uniform(-74.3, -73.7)) for sender in senders} # NYC area
    receiver_coords = {rcv: (random.uniform(40.5, 40.9), random.uniform(-74.3, -73.7)) for rcv in receivers}
    receiver_coords.update({m: (random.uniform(40.5, 40.9), random.uniform(-74.3, -73.7)) for m in merchants})
    
    records = []
    current_time = time.time()
    
    for i in range(n_records):
        step = i // 100  # 100 transactions per hour
        tx_type = random.choices(
            ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
            weights=[0.40, 0.25, 0.25, 0.08, 0.02]
        )[0]
        
        sender = random.choice(senders)
        old_bal = balances[sender]
        
        # Decide recipient
        if tx_type == "PAYMENT":
            receiver = random.choice(merchants)
        else:
            receiver = random.choice(receivers)
            
        old_bal_dest = dest_balances.get(receiver, 0.0)
        
        # Default normal transaction amount
        amount = float(np.random.exponential(500.0) + 1.0)
        is_fraud = 0
        
        # Determine if this transaction will be fraudulent (approx 0.5% rate)
        if random.random() < 0.005:
            is_fraud = 1
            fraud_type = random.choice(["large_cashout", "sequence_burst", "impossible_travel", "balance_exploit"])
            
            if fraud_type == "large_cashout":
                # Wipe out entire account balance
                amount = old_bal
                tx_type = "CASH_OUT"
            elif fraud_type == "sequence_burst":
                # Micro-transactions
                amount = random.uniform(1.0, 10.0)
                tx_type = "TRANSFER"
            elif fraud_type == "impossible_travel":
                amount = random.uniform(500.0, 5000.0)
                tx_type = "TRANSFER"
            elif fraud_type == "balance_exploit":
                # Generates severe balance discrepancy
                amount = random.uniform(100.0, 500.0)
                tx_type = "CASH_OUT"
        
        # Calculate new balances
        new_bal = old_bal
        new_bal_dest = old_bal_dest
        
        if not is_fraud or fraud_type != "balance_exploit":
            if tx_type in ["CASH_OUT", "TRANSFER"]:
                amount = min(amount, old_bal)
                new_bal = old_bal - amount
                new_bal_dest = old_bal_dest + amount
            elif tx_type == "CASH_IN":
                new_bal = old_bal + amount
                new_bal_dest = max(0.0, old_bal_dest - amount)
            elif tx_type == "PAYMENT":
                amount = min(amount, old_bal)
                new_bal = old_bal - amount
                new_bal_dest = old_bal_dest + amount
        else:
            # Under balance exploit, balances don't update correctly to create "balance_error" features
            new_bal = old_bal - (amount * 0.1) # origin balance is docked
            new_bal_dest = old_bal_dest + amount

        # Update balances
        balances[sender] = new_bal
        dest_balances[receiver] = new_bal_dest
        
        # Geographic coordinates
        sender_lat, sender_lon = sender_coords[sender]
        
        # In case of geographic anomaly, sender coordinates "teleport" far away
        if is_fraud and fraud_type == "impossible_travel":
            sender_lat = random.uniform(20.0, 30.0) # Teleport to India / London
            sender_lon = random.uniform(-10.0, 10.0)
            
        # Timestamp
        time_elapsed = random.uniform(10.0, 300.0)
        if is_fraud and fraud_type == "sequence_burst":
            time_elapsed = random.uniform(0.1, 1.0) # immediate burst
            
        current_time += time_elapsed
        
        records.append({
            "step": int(step),
            "type": tx_type,
            "amount": amount,
            "nameOrig": sender,
            "oldbalanceOrg": old_bal,
            "newbalanceOrig": new_bal,
            "nameDest": receiver,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": 0,
            "latitude": sender_lat,
            "longitude": sender_lon,
            "timestamp": current_time
        })
        
    # Write to CSV
    keys = records[0].keys()
    with open(SYNTHETIC_DATA_PATH, "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(records)
        
    print(f"Synthetic transaction dataset cached at {SYNTHETIC_DATA_PATH}")
    return records

def load_raw_data() -> List[Dict[str, Any]]:
    """
    Check if the real PaySim dataset is present. 
    If yes, read it using python standard csv. If not, generate synthetic data.
    """
    records = []
    
    if os.path.exists(REAL_DATA_PATH):
        print(f"Found real PaySim dataset at {REAL_DATA_PATH}. Loading via CSV reader...")
        np.random.seed(42)
        with open(REAL_DATA_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Add coordinate and timestamp grids dynamically
                step = int(row["step"])
                timestamp = time.time() + (step * 3600.0) + random.uniform(0, 3600)
                records.append({
                    "step": step,
                    "type": row["type"],
                    "amount": float(row["amount"]),
                    "nameOrig": row["nameOrig"],
                    "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "newbalanceOrig": float(row["newbalanceOrig"]),
                    "nameDest": row["nameDest"],
                    "oldbalanceDest": float(row["oldbalanceDest"]),
                    "newbalanceDest": float(row["newbalanceDest"]),
                    "isFraud": int(row["isFraud"]),
                    "isFlaggedFraud": int(row["isFlaggedFraud"]),
                    "latitude": float(np.random.uniform(40.5, 40.9)),
                    "longitude": float(np.random.uniform(-74.3, -73.7)),
                    "timestamp": timestamp
                })
                if len(records) >= 150000:
                    break
        return records
        
    elif os.path.exists(SYNTHETIC_DATA_PATH):
        print(f"Loading cached synthetic dataset from {SYNTHETIC_DATA_PATH}...")
        with open(SYNTHETIC_DATA_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "step": int(row["step"]),
                    "type": row["type"],
                    "amount": float(row["amount"]),
                    "nameOrig": row["nameOrig"],
                    "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "newbalanceOrig": float(row["newbalanceOrig"]),
                    "nameDest": row["nameDest"],
                    "oldbalanceDest": float(row["oldbalanceDest"]),
                    "newbalanceDest": float(row["newbalanceDest"]),
                    "isFraud": int(row["isFraud"]),
                    "isFlaggedFraud": int(row["isFlaggedFraud"]),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "timestamp": float(row["timestamp"])
                })
        return records
    else:
        return generate_synthetic_data(seed=42)

def preprocess_and_enrich(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], StatefulFeatureStore]:
    """
    Pass raw transaction rows through our StatefulFeatureStore in temporal order.
    """
    print("Enriching transactions with rolling stateful features...")
    # Sort chronologically
    records.sort(key=lambda x: x["timestamp"])
    
    store = StatefulFeatureStore()
    enriched_records = []
    
    for r in records:
        enriched_records.append(store.enrich(r))
        
    return enriched_records, store

def get_train_test_pipelines(enriched_records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare data splits, train scikit-learn standard scaling, and return:
    X_train_scaled, X_test_scaled, y_train, y_test, and the scaler.
    """
    print("Preparing train-test splits and scaling pipeline...")
    
    split_idx = int(len(enriched_records) * 0.8)
    train_records = enriched_records[:split_idx]
    test_records = enriched_records[split_idx:]
    
    # Training: Unsupervised models are calibrated strictly on NORMAL transactions
    train_normal = [r for r in train_records if r["isFraud"] == 0]
    
    # Extract features
    X_train_raw = np.array([[r[f] for f in ALL_FEATURES] for r in train_normal])
    X_test_raw = np.array([[r[f] for f in ALL_FEATURES] for r in test_records])
    
    y_train = np.array([r["isFraud"] for r in train_normal])
    y_test = np.array([r["isFraud"] for r in test_records])
    
    # Fit Scaler on normal training distribution
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    
    # Transform
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Save the fitted scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Fitted standard scaler saved at {scaler_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def create_sequences_for_lstm(
    enriched_records: List[Dict[str, Any]], 
    scaler: StandardScaler, 
    seq_length: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group transactions by sender (nameOrig) and construct sliding transaction sequences 
    of length sequence_length for PyTorch LSTM training.
    """
    print("Constructing sequence database for LSTM training...")
    feature_dim = len(ALL_FEATURES)
    sequences = []
    labels = []
    
    # Extract raw features and scale
    raw_features_matrix = np.array([[r[f] for f in ALL_FEATURES] for r in enriched_records])
    scaled_features = scaler.transform(raw_features_matrix)
    
    # Track running historical index per sender
    sender_history = defaultdict = {}
    
    for i in range(len(enriched_records)):
        row = enriched_records[i]
        sender = row["nameOrig"]
        is_fraud = int(row["isFraud"])
        feat = scaled_features[i].tolist()
        
        if sender not in sender_history:
            sender_history[sender] = []
            
        history = sender_history[sender]
        history.append(feat)
        
        if len(history) > seq_length:
            history.pop(0)
            
        # Get sequence: if shorter than seq_length, zero-pad at the beginning
        seq = history.copy()
        if len(seq) < seq_length:
            padding_size = seq_length - len(seq)
            padding = [[0.0] * feature_dim] * padding_size
            seq = padding + seq
            
        sequences.append(seq)
        labels.append(is_fraud)
        
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int32)
