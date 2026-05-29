import os
import time
import numpy as np
import joblib
import streamlit as st

from src.feature_store import StatefulFeatureStore
from src.models.lstm_autoencoder import AnomalyLSTMAutoencoder
from src.data_pipeline import generate_synthetic_data, load_raw_data
from src.config import MODEL_DIR, SEQUENCE_LENGTH, ALL_FEATURES, REAL_DATA_PATH, SYNTHETIC_DATA_PATH

# Set Page Config
st.set_page_config(page_title="Sentry-AI Lite Dashboard", layout="wide")

# 1. Load ML Models (Cached so they only load once)
@st.cache_resource
def load_ml_assets():
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    iforest_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
    lstm_path = os.path.join(MODEL_DIR, "lstm_autoencoder.pth")
    
    if not (os.path.exists(scaler_path) and os.path.exists(iforest_path) and os.path.exists(lstm_path)):
        return None, None, None
        
    scaler = joblib.load(scaler_path)
    iforest = joblib.load(iforest_path)
    lstm_ae = AnomalyLSTMAutoencoder()
    lstm_ae.load(lstm_path)
    return scaler, iforest, lstm_ae

scaler, iforest, lstm_ae = load_ml_assets()

# --- HEADER SECTION ---
st.title("Real time Anamoly Detection System for Financial Transactions")
st.write("An lightweight web portal for Real-Time Unsupervised ML Financial Stream Guardian.")

if scaler is None:
    st.warning("⚠️ **Pre-trained models missing!** Please run the training pipeline first: `py -m src.train` in your local directory to serialize scaler and models.")
    st.stop()

# Initialize Feature Store inside Streamlit Session State (maintains state across clicks/reruns)
if "store" not in st.session_state:
    st.session_state.store = StatefulFeatureStore()
if "history_txs" not in st.session_state:
    st.session_state.history_txs = []
if "anomaly_count" not in st.session_state:
    st.session_state.anomaly_count = 0
if "chart_amounts" not in st.session_state:
    st.session_state.chart_amounts = []

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ Consensus Controls")
st.sidebar.markdown("---")

alpha = st.sidebar.slider("Ensemble Weight Blending (α)", 0.0, 1.0, 0.50, step=0.05)
st.sidebar.caption("⬅ Pure PyTorch LSTM  |  Pure Isolation Forest ➡")

threshold = st.sidebar.slider("Consensus Anomaly Threshold", 0.0, 1.0, 0.86, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Simulation Feed Source")

has_real = os.path.exists(REAL_DATA_PATH)
has_synth = os.path.exists(SYNTHETIC_DATA_PATH)

mode_options = ["🌐 Synthesize In-Memory (Web Safe)"]
if has_real:
    mode_options.append("💾 Local CSV: Kaggle PaySim database")
elif has_synth:
    mode_options.append("💾 Local CSV: Cached synthetic transactions")
else:
    mode_options.append("⚠️ CSV Playback (No local database found)")

sim_mode = st.sidebar.selectbox("Feed Ingest Mode", mode_options)

if "csv_records" not in st.session_state:
    st.session_state.csv_records = None
if "csv_index" not in st.session_state:
    st.session_state.csv_index = 0

if "Local CSV:" in sim_mode and st.session_state.csv_records is None:
    with st.spinner("Loading records from local CSV file..."):
        try:
            st.session_state.csv_records = load_raw_data()
            st.sidebar.success(f"Successfully loaded {len(st.session_state.csv_records):,} transaction records!")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV database: {e}")
            sim_mode = "🌐 Synthesize In-Memory (Web Safe)"

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Simulation Gateway")
start_sim = st.sidebar.button("▶️ Start Live Stream Simulation")
reset_sim = st.sidebar.button("⏸️ Reset/Clear Local Feed")

if reset_sim:
    st.session_state.store.clear()
    st.session_state.history_txs = []
    st.session_state.anomaly_count = 0
    st.session_state.chart_amounts = []
    st.sidebar.success("Cleared stream session history!")
    st.rerun()

# --- KPI METRICS ROW ---
kpi_placeholder = st.empty()

# Create standard metric row function
def update_kpi_row(total, anomalies):
    ratio = (anomalies / total * 100) if total > 0 else 0.0
    with kpi_placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions Scored", total)
        col2.metric("Anomalies Flagged", anomalies, delta=f"{ratio:.2f}% Fraud Ratio", delta_color="inverse")
        col3.metric("Avg Score Blended (α)", f"{alpha:.2f}")

update_kpi_row(len(st.session_state.history_txs), st.session_state.anomaly_count)

st.markdown("---")

# Dynamic Placeholders for Simulation Charts and Tables
status_placeholder = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

# Function to run inference on a single transaction and update state
def score_and_record_transaction(tx):
    # Run Feature Enrichment
    enriched = st.session_state.store.enrich(tx)
    sender = enriched["nameOrig"]
    
    # Prepare Feature Vector
    latest_tx_list = st.session_state.store.history[sender]
    raw_vector = np.array([latest_tx_list[-1]["feature_vector"]])
    scaled_vector = scaler.transform(raw_vector)
    
    # Run Model Inferences
    iforest_score_raw = float(iforest.predict_score(scaled_vector)[0])
    
    seq_raw = st.session_state.store.get_recent_sequence(sender, SEQUENCE_LENGTH, len(ALL_FEATURES))
    seq_scaled = scaler.transform(np.array(seq_raw))
    seq_scaled_batch = np.expand_dims(seq_scaled, axis=0)
    lstm_score_raw = float(lstm_ae.predict_score(seq_scaled_batch)[0])
    
    # Apply Domain Heuristics & Rules
    tx_type = enriched.get("type")
    is_eligible = 1.0 if tx_type in ["TRANSFER", "CASH_OUT"] else 0.0
    
    rule_boost = 0.0
    if tx_type in ["TRANSFER", "CASH_OUT"]:
        # Empty account signature
        if float(enriched.get("newbalanceOrig", 0.0)) == 0.0 and float(enriched.get("oldbalanceOrg", 0.0)) > 0.0 and abs(float(enriched.get("oldbalanceOrg", 0.0)) - float(enriched.get("amount", 0.0))) < 1.0:
            rule_boost = 0.95
        # Destination discrepancy
        elif tx_type == "TRANSFER" and float(enriched.get("amount", 0.0)) > 1000.0 and abs(float(enriched.get("oldbalanceDest", 0.0)) + float(enriched.get("amount", 0.0)) - float(enriched.get("newbalanceDest", 0.0))) > 10.0:
            rule_boost = 0.90
            
    # Overdraft Penalty
    penalty = 1.0
    if float(enriched.get("balance_error_orig", 0.0)) < -1.0:
        penalty = 0.0
        
    iforest_score = max(iforest_score_raw * is_eligible, rule_boost) * penalty
    lstm_score = max(lstm_score_raw * is_eligible, rule_boost) * penalty
    
    # Consensus Blending
    consensus_score = alpha * iforest_score + (1.0 - alpha) * lstm_score
    is_anomaly = int(consensus_score > threshold)
    
    # Record to local history
    enriched["consensus_score"] = consensus_score
    enriched["is_anomaly"] = is_anomaly
    
    st.session_state.history_txs.append(enriched)
    st.session_state.chart_amounts.append(tx["amount"])
    if len(st.session_state.chart_amounts) > 50:
        st.session_state.chart_amounts.pop(0)
        
    if is_anomaly:
        st.session_state.anomaly_count += 1
        
    return enriched, is_anomaly

# Function to render chart and table dynamically
def render_dynamic_visuals():
    # 1. Update Chart
    if st.session_state.chart_amounts:
        with chart_placeholder.container():
            st.subheader("📈 Transaction Risk Vector (Real-Time Rolling Amount)")
            st.line_chart(st.session_state.chart_amounts)
            
    # 2. Update Table
    if st.session_state.history_txs:
        with table_placeholder.container():
            st.subheader("📋 Live Dynamic Event Ticker")
            logs_to_show = st.session_state.history_txs[-10:][::-1]
            table_data = []
            for tx in logs_to_show:
                table_data.append({
                    "Time": time.strftime("%H:%M:%S", time.localtime(tx["timestamp"])),
                    "Sender": tx["nameOrig"],
                    "Recipient": tx["nameDest"],
                    "Type": tx["type"],
                    "Amount": f"${tx['amount']:.2f}",
                    "Balance Error": f"{tx['balance_error_orig']:.2f}",
                    "Consensus Score": f"{tx['consensus_score']:.3f}",
                    "Status": "🚨 FRAUD RISK" if tx["is_anomaly"] == 1 else "✔️ PASSED"
                })
            st.table(table_data)

# --- RUN ANIMATED SIMULATION ---
if start_sim:
    if "Local CSV:" in sim_mode and st.session_state.csv_records is None:
        status_placeholder.error("⚠️ Local CSV records could not be loaded. Please verify your data files or switch to the Synthetic Generator.")
    else:
        status_placeholder.info(f"⚡ **Live Stream Simulation Ticking... Ingesting from {sim_mode}**")
        
        # We run a bounded stream simulation of 100 transaction ticks
        for i in range(100):
            # Generate or retrieve 1 transaction (simulate a card swipe)
            if "Local CSV:" in sim_mode and st.session_state.csv_records:
                tx_idx = (st.session_state.csv_index + i) % len(st.session_state.csv_records)
                tx = st.session_state.csv_records[tx_idx]
            else:
                tx = generate_synthetic_data(n_records=1)[0]
            
            # Run ML Core & Rule Engine
            enriched, is_anomaly = score_and_record_transaction(tx)
            
            # Update Metric Row
            update_kpi_row(len(st.session_state.history_txs), st.session_state.anomaly_count)
            
            # Render dynamic visual charts and tables
            render_dynamic_visuals()
            
            # Sleep to simulate live tick frequency (150ms per transaction)
            time.sleep(0.15)
            
        if "Local CSV:" in sim_mode and st.session_state.csv_records:
            st.session_state.csv_index = (st.session_state.csv_index + 100) % len(st.session_state.csv_records)
            
        status_placeholder.success("✔️ **Simulation loop completed.** Click 'Start Live Stream Simulation' to stream another 100 swipes!")
else:
    # If simulation is not running, show static visuals and provide manual trigger
    render_dynamic_visuals()
    
    st.subheader("⚡ Manual Transaction Trigger")
    st.write("Click this button to manually swipe a single credit card transaction and inspect its details.")
    if st.button("Generate & Score Single Transaction"):
        if "Local CSV:" in sim_mode and st.session_state.csv_records:
            tx_idx = st.session_state.csv_index % len(st.session_state.csv_records)
            tx = st.session_state.csv_records[tx_idx]
            st.session_state.csv_index = (tx_idx + 1) % len(st.session_state.csv_records)
        else:
            tx = generate_synthetic_data(n_records=1)[0]
            
        enriched, is_anomaly = score_and_record_transaction(tx)
        
        # Update UI
        update_kpi_row(len(st.session_state.history_txs), st.session_state.anomaly_count)
        st.rerun()
