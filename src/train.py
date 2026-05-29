import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, average_precision_score
import joblib

from src.config import (
    MODEL_DIR, 
    LOG_DIR,
    IFOREST_PARAMS, 
    LSTM_PARAMS, 
    DEFAULT_ALPHA
)
from src.data_pipeline import (
    load_raw_data, 
    preprocess_and_enrich, 
    get_train_test_pipelines, 
    create_sequences_for_lstm
)
from src.models.isolation_forest import AnomalyIsolationForest
from src.models.lstm_autoencoder import AnomalyLSTMAutoencoder

def evaluate_predictions(y_true: np.ndarray, y_scores: np.ndarray, threshold: float, model_name: str) -> float:
    """
    Print a complete classification performance report and return the Average Precision score.
    """
    y_pred = (y_scores > threshold).astype(int)
    ap = average_precision_score(y_true, y_scores)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0.0
        
    print(f"\n========================================================")
    print(f" EVALUATION REPORT: {model_name.upper()}")
    print(f"========================================================")
    print(f"Area Under ROC (ROC-AUC): {auc:.4f}")
    print(f"Average Precision (PR-AUC): {ap:.4f}")
    print(f"Calibrated Threshold: {threshold:.4f}")
    
    # Calculate Recall at Top K% budgets (Alert Rates)
    total_frauds = np.sum(y_true == 1)
    if total_frauds > 0:
        print("\nRecall at Top K% alert budgets (Operational Efficiency):")
        for K in [1.0, 0.5, 0.1]:
            n_alerts = max(1, int(len(y_scores) * (K / 100.0)))
            top_alert_indices = np.argsort(y_scores)[::-1][:n_alerts]
            captured_frauds = np.sum(y_true[top_alert_indices] == 1)
            recall_at_k = (captured_frauds / total_frauds) * 100.0
            print(f"  Top {K}% Budget ({n_alerts} alerts): {recall_at_k:.2f}% ({captured_frauds}/{total_frauds} frauds)")
            
    print("\nClassification Table:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "FRAUD"], zero_division=0))
    return ap

def apply_domain_heuristics(scores: np.ndarray, records: list) -> np.ndarray:
    """
    Apply domain-specific heuristics to unsupervised anomaly scores:
    1. Eligibility Filter: Only TRANSFER and CASH_OUT can be fraudulent.
    2. Overdraft Penalty: Massive negative origin balance errors are normal banking anomalies.
    3. Rule Boosts: High-confidence fraud signatures get boosted.
    """
    is_eligible = np.array([1.0 if r["type"] in ["TRANSFER", "CASH_OUT"] else 0.0 for r in records])
    
    # Rule A: Emptying account on TRANSFER/CASH_OUT
    # Rule B: Destination balance discrepancy on TRANSFER
    rule_boosts = np.zeros(len(records))
    for i, r in enumerate(records):
        if r["type"] in ["TRANSFER", "CASH_OUT"]:
            # Empty account signature: newbalanceOrig == 0.0 and oldbalanceOrg == amount
            if r["newbalanceOrig"] == 0.0 and r["oldbalanceOrg"] > 0.0 and abs(r["oldbalanceOrg"] - r["amount"]) < 1.0:
                rule_boosts[i] = 0.95
            # Destination discrepancy on TRANSFER: oldbalanceDest + amount != newbalanceDest
            elif r["type"] == "TRANSFER" and r["amount"] > 1000.0 and abs(r["oldbalanceDest"] + r["amount"] - r["newbalanceDest"]) > 10.0:
                rule_boosts[i] = 0.90
                
    # Penalty: balance_error_orig < -1.0 is a normal transaction
    penalty_mask = np.ones(len(records))
    for i, r in enumerate(records):
        if r["balance_error_orig"] < -1.0:
            penalty_mask[i] = 0.0
            
    filtered_scores = np.maximum(scores * is_eligible, rule_boosts) * penalty_mask
    return filtered_scores


def main():
    print("========================================================")
    print(" STARTING MODEL TRAINING AND CALIBRATION PIPELINE")
    print("========================================================\n")
    
    # 1. Load Data
    raw_records = load_raw_data()
    
    # 2. Enrich Data
    enriched_records, store = preprocess_and_enrich(raw_records)
    
    # 3. Create Scaling Pipeline
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = get_train_test_pipelines(enriched_records)
    
    # 4. Initial Fit of Isolation Forest to find training outliers
    iforest = AnomalyIsolationForest(IFOREST_PARAMS)
    iforest.fit(X_train_scaled)
    
    # 5. Fit LSTM Autoencoder with Robust Outlier Pruning
    # Construct sequence matrices for training (seq_len=5)
    seq_features, seq_labels = create_sequences_for_lstm(enriched_records, scaler)
    
    # Split sequences chronologically matching 80/20 indices
    split_idx = int(len(enriched_records) * 0.8)
    X_train_seq = seq_features[:split_idx]
    y_train_seq = seq_labels[:split_idx]
    
    X_test_seq = seq_features[split_idx:]
    y_test_seq = seq_labels[split_idx:]
    
    # Extract normal training sequences
    train_normal_mask = (y_train_seq == 0)
    X_train_seq_normal = X_train_seq[train_normal_mask]
    
    # Robust Outlier Pruning Loop
    print("\n[Robust MLOps] Iniciting Self-Supervised Outlier Pruning...")
    train_iforest_scores = iforest.predict_score(X_train_scaled)
    # Prune top 1% most anomalous normal training records (removes raw noise and massive balance errors)
    prune_threshold = np.percentile(train_iforest_scores, 99.0)
    clean_mask = (train_iforest_scores <= prune_threshold)
    
    # Filter both partitions
    X_train_scaled_clean = X_train_scaled[clean_mask]
    X_train_seq_normal_clean = X_train_seq_normal[clean_mask]
    print(f"[Robust MLOps] Pruned {len(X_train_scaled) - len(X_train_scaled_clean)} outlier records from normal training set.")
    
    # Re-train models on clean normal data
    print("[Robust MLOps] Re-training Isolation Forest on clean normal data...")
    iforest = AnomalyIsolationForest(IFOREST_PARAMS)
    iforest.fit(X_train_scaled_clean)
    
    print("[Robust MLOps] Training LSTM Autoencoder on clean normal sequences...")
    lstm_ae = AnomalyLSTMAutoencoder(params=LSTM_PARAMS)
    lstm_ae.fit(X_train_seq_normal_clean)
    
    # 6. Calibrate thresholds on clean normal partitions
    print("\nCalibrating threshold levels...")
    iforest.calibrate_threshold(X_train_scaled_clean, contamination=0.01)
    lstm_ae.calibrate_threshold(X_train_seq_normal_clean, contamination=0.01)
    
    # 7. Evaluate on the Test Partition (contains real/injected anomalies)
    test_records = enriched_records[split_idx:]
    
    iforest_scores_raw = iforest.predict_score(X_test_scaled)
    iforest_scores = apply_domain_heuristics(iforest_scores_raw, test_records)
    evaluate_predictions(y_test, iforest_scores, iforest.threshold, "Isolation Forest (Static)")
    
    lstm_scores_raw = lstm_ae.predict_score(X_test_seq)
    lstm_scores = apply_domain_heuristics(lstm_scores_raw, test_records)
    evaluate_predictions(y_test_seq, lstm_scores, lstm_ae.threshold, "LSTM Autoencoder (Temporal)")
    
    # 8. Evaluate Blended Consensus Ensemble
    consensus_scores = DEFAULT_ALPHA * iforest_scores + (1 - DEFAULT_ALPHA) * lstm_scores
    
    # Calibrate consensus threshold (e.g. 99th percentile of normal training consensus scores)
    train_records_clean = [r for r in enriched_records[:split_idx] if r["isFraud"] == 0]
    train_records_clean = [train_records_clean[i] for i in range(len(train_records_clean)) if clean_mask[i]]
    
    train_iforest_scores_raw = iforest.predict_score(X_train_scaled_clean)
    train_iforest_scores = apply_domain_heuristics(train_iforest_scores_raw, train_records_clean)
    
    train_lstm_scores_raw = lstm_ae.predict_score(X_train_seq_normal_clean)
    train_lstm_scores = apply_domain_heuristics(train_lstm_scores_raw, train_records_clean)
    
    train_consensus = DEFAULT_ALPHA * train_iforest_scores + (1 - DEFAULT_ALPHA) * train_lstm_scores
    consensus_threshold = float(np.percentile(train_consensus, 99.0))
    
    evaluate_predictions(y_test_seq, consensus_scores, consensus_threshold, "Consensus Ensemble (Hybrid)")
    
    # 9. Save Models
    iforest_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
    joblib.dump(iforest, iforest_path)
    print(f"\nSerialized Isolation Forest baseline saved to {iforest_path}")
    
    lstm_path = os.path.join(MODEL_DIR, "lstm_autoencoder.pth")
    lstm_ae.save(lstm_path)
    
    # Save consensus configuration threshold
    ensemble_meta = {
        "alpha": DEFAULT_ALPHA,
        "threshold": consensus_threshold
    }
    ensemble_path = os.path.join(MODEL_DIR, "ensemble_meta.joblib")
    joblib.dump(ensemble_meta, ensemble_path)
    print(f"Serialized Ensemble Metadata saved to {ensemble_path}")
    
    # 10. Generate Plot Curves
    print("\nGenerating model comparison performance curves...")
    
    plt.figure(figsize=(12, 5))
    
    # Plot A: ROC Curves
    from sklearn.metrics import roc_curve
    plt.subplot(1, 2, 1)
    for scores, label in zip([iforest_scores, lstm_scores, consensus_scores], 
                             ["Isolation Forest", "LSTM Autoencoder", "Consensus Ensemble"]):
        fpr, tpr, _ = roc_curve(y_test_seq, scores)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y_test_seq, scores):.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot B: Precision-Recall Curves
    plt.subplot(1, 2, 2)
    for scores, label in zip([iforest_scores, lstm_scores, consensus_scores], 
                             ["Isolation Forest", "LSTM Autoencoder", "Consensus Ensemble"]):
        precision, recall, _ = precision_recall_curve(y_test_seq, scores)
        plt.plot(recall, precision, label=f"{label} (AP = {average_precision_score(y_test_seq, scores):.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, "model_performance_evaluation.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved graphical comparison curves to {plot_path}")
    print("\n========================================================")
    print(" TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
    print("========================================================\n")

if __name__ == "__main__":
    main()
