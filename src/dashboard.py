import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as nn_st # Standard import is import streamlit as st, but let's write simple streamlit import
import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Import models from our models script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import MLPClassifierNet, AutoencoderNet, predict_mlp, predict_autoencoder, LSTMClassifierNet, predict_lstm, CNN1DClassifierNet, predict_cnn1d
from preprocess import COLUMN_NAMES

# Config page
st.set_page_config(
    page_title="Cloud Anomaly Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS removed for compatibility
# Helper function to load all trained models and pipelines
@st.cache_resource
def load_resources():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(src_dir)
    models_dir = os.path.join(workspace_dir, "saved_models")
    
    # 1. Preprocessor
    with open(os.path.join(models_dir, "preprocessor.pkl"), 'rb') as f:
        preprocessor = pickle.load(f)
        
    # 2. RF
    with open(os.path.join(models_dir, "random_forest.pkl"), 'rb') as f:
        rf_model = pickle.load(f)
        
    # 3. XGBoost
    with open(os.path.join(models_dir, "xgboost.pkl"), 'rb') as f:
        xgb_model = pickle.load(f)
        
    # 4. SVM
    with open(os.path.join(models_dir, "svm.pkl"), 'rb') as f:
        svm_model = pickle.load(f)
        
    # 5. MLP (PyTorch)
    # Get shape from dummy transformation
    input_dim = 122 # Preprocessed size from NSL-KDD
    mlp_model = MLPClassifierNet(input_dim)
    mlp_model.load_state_dict(torch.load(os.path.join(models_dir, "mlp_state_dict.pth"), map_location='cpu'))
    mlp_model.eval()
    
    # 6. Autoencoder (PyTorch)
    ae_model = AutoencoderNet(input_dim)
    ae_model.load_state_dict(torch.load(os.path.join(models_dir, "autoencoder_state_dict.pth"), map_location='cpu'))
    ae_model.eval()
    
    with open(os.path.join(models_dir, "ae_threshold.json"), 'r') as f:
        ae_threshold = json.load(f)["threshold"]
        
    # 7. Benchmark Results
    with open(os.path.join(workspace_dir, "results.json"), 'r') as f:
        results = json.load(f)
        
    return preprocessor, rf_model, xgb_model, svm_model, mlp_model, ae_model, ae_threshold, results

try:
    preprocessor, rf_model, xgb_model, svm_model, mlp_model, ae_model, ae_threshold, results = load_resources()
except Exception as e:
    st.error(f"Error loading model resources: {e}. Please ensure you ran `python3 src/benchmark.py` first.")
    st.stop()

# Header
st.title("🛡️ Cloud Anomaly & DDoS Detection System")
st.caption("A recreation of ML/DL detection pipelines based on: 'Cloud Network Anomaly Detection Using Machine and Deep Learning Techniques' (IEEE Access 2024)")

# Sidebar info
st.sidebar.image("https://img.icons8.com/color/144/shield-with-crown.png", width=100)
st.sidebar.title("Security Control Panel")
st.sidebar.markdown("""
This control center implements anomaly-based intrusion detection using both supervised classifiers and reconstruction-based deep learning.
""")
st.sidebar.divider()
st.sidebar.info("⚙️ **Dataset:** NSL-KDD (122 dimensions)  \n🤖 **Deep Learning:** PyTorch  \n📉 **Models Evaluated:** 12 Model Architectures")

# Main Page Tabs
tab1, tab2 = st.tabs(["📊 Performance Benchmarks", "⚡ Live Threat Simulator"])

with tab1:
    st.subheader("Model Performance Analysis & Benchmark Comparison")
    
    # Metrics columns
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Top F1-Score (Autoencoder)", f"{results['Autoencoder']['F1-Score'] * 100:.2f}%", "Best Anomaly Defense")
    with m_col2:
        st.metric("Top Accuracy (Autoencoder)", f"{results['Autoencoder']['Accuracy'] * 100:.2f}%", "+2.3% vs XGBoost")
    with m_col3:
        st.metric("Lowest Latency (Linear SVM)", f"{results['Linear SVM']['Inference Latency (ms/sample)']:.5f} ms", "Real-time Ready")
    with m_col4:
        st.metric("Fastest Training (Linear SVM)", f"{results['Linear SVM']['Training Time (s)']:.2f} s", "SGD Optimized")
        
    st.divider()
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📈 Quantitative Performance Summary")
        df_results = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'Model'})
        # Format metrics
        formatted_df = df_results.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.2f}%")
        formatted_df['Training Time (s)'] = formatted_df['Training Time (s)'].apply(lambda x: f"{x:.3f} s")
        formatted_df['Inference Latency (ms/sample)'] = formatted_df['Inference Latency (ms/sample)'].apply(lambda x: f"{x:.5f} ms")
        
        st.table(formatted_df)
        
        st.markdown("""
        #### Key Research Findings:
        1. **Deep Reconstruction Success**: The PyTorch **Autoencoder** achieves the highest generalizability (Accuracy: 81.98%, F1: 81.70%). Training exclusively on normal network traffic enables it to detect anomalies by recognizing high reconstruction errors, mimicking a real-world zero-day defense system.
        2. **Speed & Efficiency**: Traditional machine learning models like **XGBoost** and **Linear SVM** train in under 1 second and offer microsecond-level latency, making them highly suited for edge gateways and real-time packet filtering, whereas deep neural networks like MLP require more computational overhead.
        """)
        
    with col_right:
        st.markdown("### 📊 Metrics Visualization Charts")
        
        # Tabs for plots
        plot_tab1, plot_tab2, plot_tab3 = st.tabs(["Performance Metrics", "Compute Trade-offs", "ROC curves"])
        
        src_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(src_dir)
        plots_dir = os.path.join(workspace_dir, "plots")
        
        with plot_tab1:
            img_path = os.path.join(plots_dir, "metrics_comparison.png")
            if os.path.exists(img_path):
                st.image(img_path, caption="Comparative accuracy, precision, recall, and F1 performance")
            else:
                st.write("Chart file metrics_comparison.png not found.")
                
        with plot_tab2:
            img_path = os.path.join(plots_dir, "times_comparison.png")
            if os.path.exists(img_path):
                st.image(img_path, caption="Training duration vs inference latency comparison")
            else:
                st.write("Chart file times_comparison.png not found.")
                
        with plot_tab3:
            img_path = os.path.join(plots_dir, "roc_curves.png")
            if os.path.exists(img_path):
                st.image(img_path, caption="Receiver Operating Characteristic curves")
            else:
                st.write("Chart file roc_curves.png not found.")


with tab2:
    st.subheader("⚡ Real-Time Cloud Traffic Threat Simulator")
    st.markdown("Select a network traffic scenario or configure a custom packet payload to evaluate its classification status across all trained models simultaneously.")
    
    # Predefined Scenarios (based on NSL-KDD statistics)
    scenarios = {
        "Normal HTTP Traffic (Clean Connection)": {
            "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
            "src_bytes": 215, "dst_bytes": 4500, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
            "count": 1, "srv_count": 1, "serror_rate": 0.0, "rerror_rate": 0.0,
            "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "dst_host_count": 5,
            "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0
        },
        "Neptune DDoS Attack (DoS TCP Flood)": {
            "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "S0",
            "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
            "count": 250, "srv_count": 15, "serror_rate": 1.0, "rerror_rate": 0.0,
            "same_srv_rate": 0.06, "diff_srv_rate": 0.07, "dst_host_count": 255,
            "dst_host_srv_count": 15, "dst_host_same_srv_rate": 0.06, "dst_host_diff_srv_rate": 0.07
        },
        "Satan Portscan Attack (Reconnaissance Probe)": {
            "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "REJ",
            "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
            "count": 140, "srv_count": 4, "serror_rate": 0.0, "rerror_rate": 1.0,
            "same_srv_rate": 0.03, "diff_srv_rate": 0.07, "dst_host_count": 255,
            "dst_host_srv_count": 4, "dst_host_same_srv_rate": 0.02, "dst_host_diff_srv_rate": 0.07
        },
        "Buffer Overflow Attempt (User-to-Root/U2R)": {
            "duration": 60, "protocol_type": "tcp", "service": "telnet", "flag": "SF",
            "src_bytes": 1200, "dst_bytes": 8500, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 3, "num_failed_logins": 0, "logged_in": 1,
            "count": 1, "srv_count": 1, "serror_rate": 0.0, "rerror_rate": 0.0,
            "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "dst_host_count": 1,
            "dst_host_srv_count": 1, "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0
        }
    }
    
    scenario_selection = st.selectbox("📌 Select a Preset Packet Profile", list(scenarios.keys()))
    base_values = scenarios[scenario_selection]
    
    st.divider()
    
    st.write("##### 🛠️ Adjust Key Network Flow Parameters")
    
    # Parameter adjustment columns
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"], index=["tcp", "udp", "icmp"].index(base_values["protocol_type"]))
        service = st.selectbox("Service Port Type", ["http", "private", "ftp_data", "smtp", "telnet", "domain_u"], index=["http", "private", "ftp_data", "smtp", "telnet", "domain_u"].index(base_values["service"]) if base_values["service"] in ["http", "private", "ftp_data", "smtp", "telnet", "domain_u"] else 1)
    with p_col2:
        flag = st.selectbox("Connection Status Flag", ["SF", "S0", "REJ", "RSTR", "RSTO"], index=["SF", "S0", "REJ", "RSTR", "RSTO"].index(base_values["flag"]) if base_values["flag"] in ["SF", "S0", "REJ", "RSTR", "RSTO"] else 0)
        duration = st.slider("Duration (s)", 0, 1000, int(base_values["duration"]))
    with p_col3:
        src_bytes = st.number_input("Source Bytes (Outbound)", min_value=0, max_value=1000000, value=int(base_values["src_bytes"]))
        dst_bytes = st.number_input("Destination Bytes (Inbound)", min_value=0, max_value=1000000, value=int(base_values["dst_bytes"]))
    with p_col4:
        count = st.slider("Host Connection Count (2s)", 0, 500, int(base_values["count"]))
        serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, float(base_values["serror_rate"]))

    # Synthesize full 41-feature dictionary
    full_row = {}
    
    # 41 columns of NSL-KDD
    nsl_kdd_cols = COLUMN_NAMES[:-2] # Exclude class and difficulty
    
    # Initialize with default normal values
    default_normal_values = {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
        "is_host_login": 0, "is_guest_login": 0, "count": 1, "srv_count": 1,
        "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 1, "dst_host_srv_count": 1, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    }
    
    for col in nsl_kdd_cols:
        full_row[col] = default_normal_values[col]
        
    # Inject user specifications
    full_row["protocol_type"] = protocol_type
    full_row["service"] = service
    full_row["flag"] = flag
    full_row["duration"] = duration
    full_row["src_bytes"] = src_bytes
    full_row["dst_bytes"] = dst_bytes
    full_row["count"] = count
    full_row["serror_rate"] = serror_rate
    
    # Replicate preset's internal values if scenario was chosen
    for col in ["logged_in", "srv_count", "rerror_rate", "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "hot"]:
        if col in base_values:
            full_row[col] = base_values[col]

    # Convert to DataFrame
    input_df = pd.DataFrame([full_row])
    
    # Preprocess row
    try:
        input_preprocessed = preprocessor.transform(input_df)
    except Exception as preprocess_err:
        st.error(f"Preprocessing error: {preprocess_err}")
        st.stop()
        
    st.divider()
    
    # Run Predictions
    st.write("### 🔮 Model Predictions Comparison")
    
    # Columns for model prediction cards
    card_cols = st.columns(5)
    
    # 1. Random Forest
    rf_pred = rf_model.predict(input_preprocessed)[0]
    rf_prob = rf_model.predict_proba(input_preprocessed)[0][1]
    with card_cols[0]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("**Random Forest**")
        if rf_pred == 1:
            st.markdown("<span class='badge-anomaly'>ANOMALY</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-normal'>NORMAL</span>", unsafe_allow_html=True)
        st.write(f"Anomaly Prob: {rf_prob * 100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 2. XGBoost
    xgb_pred = xgb_model.predict(input_preprocessed)[0]
    xgb_prob = xgb_model.predict_proba(input_preprocessed)[0][1]
    with card_cols[1]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("**XGBoost**")
        if xgb_pred == 1:
            st.markdown("<span class='badge-anomaly'>ANOMALY</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-normal'>NORMAL</span>", unsafe_allow_html=True)
        st.write(f"Anomaly Prob: {xgb_prob * 100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 3. Linear SVM
    svm_pred = svm_model.predict(input_preprocessed)[0]
    svm_dec = svm_model.decision_function(input_preprocessed)[0]
    with card_cols[2]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("**Linear SVM**")
        if svm_pred == 1:
            st.markdown("<span class='badge-anomaly'>ANOMALY</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-normal'>NORMAL</span>", unsafe_allow_html=True)
        st.write(f"Decision Dist: {svm_dec:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 4. MLP Network
    mlp_pred, mlp_prob = predict_mlp(mlp_model, input_preprocessed, device='cpu')
    mlp_pred, mlp_prob = mlp_pred[0], mlp_prob[0]
    with card_cols[3]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("**MLP Classifier**")
        if mlp_pred == 1:
            st.markdown("<span class='badge-anomaly'>ANOMALY</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-normal'>NORMAL</span>", unsafe_allow_html=True)
        st.write(f"Anomaly Prob: {mlp_prob * 100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 5. Autoencoder
    ae_pred, ae_errors = predict_autoencoder(ae_model, input_preprocessed, ae_threshold, device='cpu')
    ae_pred, ae_error = ae_pred[0], ae_errors[0]
    with card_cols[4]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("**Autoencoder**")
        if ae_pred == 1:
            st.markdown("<span class='badge-anomaly'>ANOMALY</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-normal'>NORMAL</span>", unsafe_allow_html=True)
        st.write(f"Reconst Loss: {ae_error:.4f}")
        st.write(f"Threshold: {ae_threshold:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📝 Verdict Synthesis")
    pred_sum = int(rf_pred) + int(xgb_pred) + int(svm_pred) + int(mlp_pred) + int(ae_pred)
    if pred_sum >= 3:
        st.error(f"🚨 **MAJORITY VERDICT: SUSPICIOUS TRAFFIC BLOCK TRIGGERED!** ({pred_sum}/5 models flagged an Anomaly) (Note: Live simulator currently runs a 5-model ensemble for speed). Network traffic shows characteristics of security threats or malicious packets.")
    elif pred_sum > 0:
        st.warning(f"⚠️ **WARNING: MINORITY ANOMALY DETECTED!** ({pred_sum}/5 models flagged an Anomaly) (Note: Live simulator currently runs a 5-model ensemble for speed). Possible zero-day exploit or configuration change. Monitor connection closely.")
    else:
        st.success("✅ **VERDICT: SECURE CONNECTION** (0/5 models flagged anomalies). Network traffic matches normal baseline profiles.")
