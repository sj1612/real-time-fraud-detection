import os
import time
import numpy as np
import joblib
import torch

from src.data_pipeline import generate_synthetic_data
from src.feature_store import StatefulFeatureStore
from src.models.lstm_autoencoder import AnomalyLSTMAutoencoder
from src.config import MODEL_DIR, SEQUENCE_LENGTH, ALL_FEATURES, DEFAULT_ALPHA

def run_benchmark():
    print("========================================================")
    # 1. Load ML Assets
    print("[Benchmark] Loading pre-trained model checkpoints...")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    iforest_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
    lstm_path = os.path.join(MODEL_DIR, "lstm_autoencoder.pth")
    ensemble_path = os.path.join(MODEL_DIR, "ensemble_meta.joblib")
    
    if not (os.path.exists(scaler_path) and os.path.exists(iforest_path) and os.path.exists(lstm_path)):
        raise FileNotFoundError("Pre-trained models missing! Please run 'py -m src.train' first.")
        
    scaler = joblib.load(scaler_path)
    iforest = joblib.load(iforest_path)
    
    lstm_ae = AnomalyLSTMAutoencoder()
    lstm_ae.load(lstm_path)
    
    alpha = DEFAULT_ALPHA
    if os.path.exists(ensemble_path):
        meta = joblib.load(ensemble_path)
        alpha = meta.get("alpha", DEFAULT_ALPHA)
        
    # 2. Iniciting Stateful Store and Generator
    store = StatefulFeatureStore()
    
    # 3. Synthesizing 10,000 transactions for benchmark
    print("\n[Benchmark] Synthesizing 10,000 transactions...")
    txs = generate_synthetic_data(n_records=10000)
    
    feature_dim = len(ALL_FEATURES)
    latencies = []
    
    print("\n[Benchmark] Commencing real-time streaming inference benchmark...")
    
    # Warmup pass (JIT compiler and torch CUDA allocations)
    print("[Benchmark] Running 50 warmup iterations...")
    for i in range(50):
        tx = txs[i]
        enriched = store.enrich(tx)
        sender = enriched["nameOrig"]
        latest_tx_list = store.history[sender]
        raw_vector = np.array([latest_tx_list[-1]["feature_vector"]])
        scaled_vector = scaler.transform(raw_vector)
        _ = iforest.predict_score(scaled_vector)[0]
        seq_raw = store.get_recent_sequence(sender, SEQUENCE_LENGTH, feature_dim)
        seq_scaled = scaler.transform(np.array(seq_raw))
        seq_scaled_batch = np.expand_dims(seq_scaled, axis=0)
        _ = lstm_ae.predict_score(seq_scaled_batch)[0]
        
    store.clear()  # reset history for clean benchmark state
    
    # Start Benchmark
    total_start_time = time.perf_counter()
    
    for i in range(len(txs)):
        # Start high-resolution per-transaction timer
        tx_start = time.perf_counter()
        
        tx = txs[i]
        
        # A. Stateful counter counters, geo-speeds, monetary deviations
        enriched = store.enrich(tx)
        sender = enriched["nameOrig"]
        
        # B. Unscaled vectors extraction and transform
        latest_tx_list = store.history[sender]
        raw_vector = np.array([latest_tx_list[-1]["feature_vector"]])
        scaled_vector = scaler.transform(raw_vector)
        
        # C. Model A: Isolation Forest Anomaly Inference
        iforest_score_raw = float(iforest.predict_score(scaled_vector)[0])
        
        # D. Model B: PyTorch LSTM Autoencoder Inference
        seq_raw = store.get_recent_sequence(sender, SEQUENCE_LENGTH, feature_dim)
        seq_scaled = scaler.transform(np.array(seq_raw))
        seq_scaled_batch = np.expand_dims(seq_scaled, axis=0)
        lstm_score_raw = float(lstm_ae.predict_score(seq_scaled_batch)[0])
        
        # E. Apply Domain Heuristics & Rules
        tx_type = enriched.get("type")
        is_eligible = 1.0 if tx_type in ["TRANSFER", "CASH_OUT"] else 0.0
        
        rule_boost = 0.0
        if tx_type in ["TRANSFER", "CASH_OUT"]:
            if float(enriched.get("newbalanceOrig", 0.0)) == 0.0 and float(enriched.get("oldbalanceOrg", 0.0)) > 0.0 and abs(float(enriched.get("oldbalanceOrg", 0.0)) - float(enriched.get("amount", 0.0))) < 1.0:
                rule_boost = 0.95
            elif tx_type == "TRANSFER" and float(enriched.get("amount", 0.0)) > 1000.0 and abs(float(enriched.get("oldbalanceDest", 0.0)) + float(enriched.get("amount", 0.0)) - float(enriched.get("newbalanceDest", 0.0))) > 10.0:
                rule_boost = 0.90
                
        # Overdraft discrepancy Penalty
        penalty = 1.0
        if float(enriched.get("balance_error_orig", 0.0)) < -1.0:
            penalty = 0.0
            
        iforest_score = max(iforest_score_raw * is_eligible, rule_boost) * penalty
        lstm_score = max(lstm_score_raw * is_eligible, rule_boost) * penalty
        
        # F. Consensus Aggregation Blending
        _ = alpha * iforest_score + (1.0 - alpha) * lstm_score
        
        # Stop high-resolution per-transaction timer
        tx_end = time.perf_counter()
        
        # Convert to milliseconds
        elapsed_ms = (tx_end - tx_start) * 1000.0
        latencies.append(elapsed_ms)
        
    total_end_time = time.perf_counter()
    total_time_seconds = total_end_time - total_start_time
    
    # 4. Compute Metrics
    tps = len(txs) / total_time_seconds
    mean_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50.0)
    p95_lat = np.percentile(latencies, 95.0)
    p99_lat = np.percentile(latencies, 99.0)
    max_lat = np.max(latencies)
    
    print("\n========================================================")
    print("          SENTRY-AI PERFORMANCE BENCHMARK REPORT")
    print("========================================================")
    print(f"Total Transactions Processed : {len(txs):,}")
    print(f"Total Benchmark Time         : {total_time_seconds:.4f} seconds")
    print(f"Throughput (TPS)             : {tps:.2f} Transactions / Sec")
    print("--------------------------------------------------------")
    print("            HIGH-RESOLUTION LATENCY TELEMETRY")
    print("--------------------------------------------------------")
    print(f"Average Latency              : {mean_lat:.4f} ms")
    print(f"Median (P50) Latency         : {p50_lat:.4f} ms")
    print(f"P95 Inference Latency        : {p95_lat:.4f} ms")
    print(f"P99 Inference Latency        : {p99_lat:.4f} ms")
    print(f"Maximum Latency              : {max_lat:.4f} ms")
    print("========================================================\n")

if __name__ == "__main__":
    run_benchmark()
