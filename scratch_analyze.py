import os
import numpy as np
import joblib
import torch

from src.data_pipeline import (
    load_raw_data, 
    preprocess_and_enrich, 
    get_train_test_pipelines, 
    create_sequences_for_lstm
)
from src.models.isolation_forest import AnomalyIsolationForest
from src.models.lstm_autoencoder import AnomalyLSTMAutoencoder
from src.config import MODEL_DIR, DEFAULT_ALPHA

def main():
    print("Loading pre-trained models...")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    iforest_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
    lstm_path = os.path.join(MODEL_DIR, "lstm_autoencoder.pth")
    
    scaler = joblib.load(scaler_path)
    iforest = joblib.load(iforest_path)
    lstm_ae = AnomalyLSTMAutoencoder()
    lstm_ae.load(lstm_path)
    
    print("Loading data...")
    raw_records = load_raw_data()
    enriched_records, store = preprocess_and_enrich(raw_records)
    
    X_train_scaled, X_test_scaled, y_train, y_test, _ = get_train_test_pipelines(enriched_records)
    
    seq_features, seq_labels = create_sequences_for_lstm(enriched_records, scaler)
    split_idx = int(len(enriched_records) * 0.8)
    
    X_test_seq = seq_features[split_idx:]
    y_test_seq = seq_labels[split_idx:]
    
    # Recalculate robust mu and sigma for LSTM Autoencoder on training set
    print("Recalculating robust mu and sigma for LSTM...")
    X_train_seq_normal = seq_features[:split_idx][seq_labels[:split_idx] == 0]
    
    # Run LSTM on training set to get errors
    if lstm_ae.model is not None:
        import torch
        lstm_ae.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_train_seq_normal, dtype=torch.float32).to(torch.device("cpu" if not torch.cuda.is_available() else "cuda"))
            recon = lstm_ae.model(x_tensor)
            train_errors = torch.mean((recon - x_tensor) ** 2, dim=(1, 2)).cpu().numpy()
            
        # Robustly prune top 2% of training errors
        q = np.percentile(train_errors, 98.0)
        clean_train_errors = train_errors[train_errors <= q]
        robust_mu = float(np.mean(clean_train_errors))
        robust_sigma = float(np.std(clean_train_errors))
        if robust_sigma == 0.0:
            robust_sigma = 1.0
            
        print(f"Original LSTM parameters: mu={lstm_ae.mu:.4f}, sigma={lstm_ae.sigma:.4f}")
        print(f"Robust LSTM parameters: mu={robust_mu:.4f}, sigma={robust_sigma:.4f}")
        
        # Override LSTM parameters for evaluation
        lstm_ae.mu = robust_mu
        lstm_ae.sigma = robust_sigma
        
    # Get scores
    print("Running inference...")
    iforest_scores_raw = iforest.predict_score(X_test_scaled)
    lstm_scores_raw = lstm_ae.predict_score(X_test_seq)
    
    # Calculate eligibility filter
    test_records = enriched_records[split_idx:]
    is_eligible = np.array([1.0 if r["type"] in ["TRANSFER", "CASH_OUT"] else 0.0 for r in test_records])
    
    # Apply rule-based boost to prioritize actual fraud signatures:
    # Rule A: Emptying account on TRANSFER/CASH_OUT
    # Rule B: Destination balance discrepancy on TRANSFER
    rule_boosts = np.zeros(len(test_records))
    for i, r in enumerate(test_records):
        if r["type"] in ["TRANSFER", "CASH_OUT"]:
            # Empty account signature: oldbalanceOrg == amount and newbalanceOrig == 0
            if r["newbalanceOrig"] == 0.0 and r["oldbalanceOrg"] > 0.0 and abs(r["oldbalanceOrg"] - r["amount"]) < 1.0:
                rule_boosts[i] = 0.95
            # Destination discrepancy on TRANSFER: oldbalanceDest + amount != newbalanceDest
            elif r["type"] == "TRANSFER" and r["amount"] > 1000.0 and abs(r["oldbalanceDest"] + r["amount"] - r["newbalanceDest"]) > 10.0:
                rule_boosts[i] = 0.90
                
    # Apply negative balance error penalty (massive negative balance errors are normal transactions)
    penalty_mask = np.ones(len(test_records))
    for i, r in enumerate(test_records):
        if r["balance_error_orig"] < -1.0:
            penalty_mask[i] = 0.0
            
    iforest_scores = np.maximum(iforest_scores_raw * is_eligible, rule_boosts) * penalty_mask
    lstm_scores = np.maximum(lstm_scores_raw * is_eligible, rule_boosts) * penalty_mask
    
    consensus_scores = (DEFAULT_ALPHA * iforest_scores + (1 - DEFAULT_ALPHA) * lstm_scores)
    
    print(f"y_test shape: {y_test.shape}")
    print(f"y_test_seq shape: {y_test_seq.shape}")
    
    # Find fraud indices
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    for name, raw_s, filt_s in [("Isolation Forest", iforest_scores_raw, iforest_scores), 
                                 ("LSTM Autoencoder", lstm_scores_raw, lstm_scores),
                                 ("Consensus Ensemble", DEFAULT_ALPHA * iforest_scores_raw + (1 - DEFAULT_ALPHA) * lstm_scores_raw, consensus_scores)]:
        print(f"\n--- {name} ---")
        print(f"  Raw ROC-AUC: {roc_auc_score(y_test_seq, raw_s):.4f} | Raw PR-AUC: {average_precision_score(y_test_seq, raw_s):.4f}")
        print(f"  Filtered ROC-AUC: {roc_auc_score(y_test_seq, filt_s):.4f} | Filtered PR-AUC: {average_precision_score(y_test_seq, filt_s):.4f}")
        
        # Calculate Recall at Top K% Alert Rates for the Filtered Scores
        total_frauds = np.sum(y_test_seq == 1)
        if total_frauds > 0:
            print("  Recall at Top K% budgets (Alert Rates):")
            for K in [1.0, 0.5, 0.1]:
                n_alerts = max(1, int(len(filt_s) * (K / 100.0)))
                # Sort indices of filtered scores descending
                top_alert_indices = np.argsort(filt_s)[::-1][:n_alerts]
                captured_frauds = np.sum(y_test_seq[top_alert_indices] == 1)
                recall_at_k = (captured_frauds / total_frauds) * 100.0
                print(f"    Recall at Top {K}% ({n_alerts} alerts): {recall_at_k:.2f}% ({captured_frauds}/{total_frauds} frauds)")
        
    fraud_indices = np.where(y_test_seq == 1)[0]
    print(f"Fraud indices in test set: {fraud_indices}")
    
    print("\n--- SCORES FOR ACTUAL FRAUD TRANSACTIONS ---")
    for idx in fraud_indices:
        record = enriched_records[split_idx + idx]
        print(f"Idx: {idx} | Type: {record['type']} | Amount: {record['amount']:.2f} | isFraud: {record['isFraud']}")
        print(f"  Orig Bal Error: {record['balance_error_orig']:.2f} | Dest Bal Error: {record['balance_error_dest']:.2f}")
        print(f"  IForest Score: {iforest_scores[idx]:.6f} | LSTM Score: {lstm_scores[idx]:.6f} | Consensus Score: {consensus_scores[idx]:.6f}")
        
    print("\n--- TOP 20 HIGHEST CONSENSUS SCORING TRANSACTIONS ---")
    top_indices = np.argsort(consensus_scores)[::-1][:20]
    for idx in top_indices:
        record = enriched_records[split_idx + idx]
        print(f"Idx: {idx} | Type: {record['type']} | Amount: {record['amount']:.2f} | isFraud: {record['isFraud']}")
        print(f"  Orig Bal Error: {record['balance_error_orig']:.2f} | Dest Bal Error: {record['balance_error_dest']:.2f}")
        print(f"  IForest Score: {iforest_scores[idx]:.6f} | LSTM Score: {lstm_scores[idx]:.6f} | Consensus Score: {consensus_scores[idx]:.6f}")
        
    # Analyze threshold-based metrics
    print("\nConsensus score range:", np.min(consensus_scores), "to", np.max(consensus_scores))
    print("Normal test scores mean:", np.mean(consensus_scores[y_test_seq == 0]), "std:", np.std(consensus_scores[y_test_seq == 0]))
    if len(fraud_indices) > 0:
        print("Fraud test scores mean:", np.mean(consensus_scores[y_test_seq == 1]), "std:", np.std(consensus_scores[y_test_seq == 1]))
    
if __name__ == "__main__":
    main()
