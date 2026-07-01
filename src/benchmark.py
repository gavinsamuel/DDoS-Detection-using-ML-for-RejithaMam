import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import torch

# Import custom scripts
from preprocess import preprocess_nsl_kdd
from models import (
    get_rf_model, get_xgb_model, get_svm_model,
    get_dt_model, get_knn_model, get_nb_model, get_lr_model, get_lgbm_model,
    train_mlp, predict_mlp,
    train_autoencoder, find_ae_threshold, predict_autoencoder,
    train_lstm, predict_lstm, train_cnn1d, predict_cnn1d
)

def plot_confusion_matrices(cms, model_names, save_dir):
    """Plots and saves confusion matrices for all models side-by-side."""
    n_models = len(cms)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]
        
    for i, (name, cm) in enumerate(zip(model_names, cms)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        axes[i].set_title(f"{name}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices.png"), dpi=150)
    plt.close()

def plot_roc_curves(roc_data, save_dir):
    """Plots and saves ROC curves for the models."""
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc_score) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=150)
    plt.close()

def plot_metrics_comparison(metrics_df, save_dir):
    """Generates a bar chart comparing performance metrics across all models."""
    melted_df = pd.melt(metrics_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='value', hue='variable', data=melted_df, palette='viridis')
    plt.ylim(0.7, 1.02)
    plt.xlabel("Model")
    plt.ylabel("Metric Score")
    plt.title("Model Performance Comparison (Binary Classification)")
    plt.legend(title="Metrics", loc="lower left")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_comparison.png"), dpi=150)
    plt.close()

def plot_times_comparison(metrics_df, save_dir):
    """Generates bar charts comparing training and inference times."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training Time
    sns.barplot(x='Model', y='Training Time (s)', data=metrics_df, ax=ax1, palette='magma')
    ax1.set_title("Training Time (Seconds) - Lower is Better")
    ax1.set_ylabel("Seconds")
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Inference Latency
    sns.barplot(x='Model', y='Inference Latency (ms/sample)', data=metrics_df, ax=ax2, palette='magma')
    ax2.set_title("Inference Latency (ms per Sample) - Lower is Better")
    ax2.set_ylabel("Milliseconds")
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "times_comparison.png"), dpi=150)
    plt.close()

def main():
    # Setup folders
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(workspace_dir, "data")
    models_dir = os.path.join(workspace_dir, "saved_models")
    plots_dir = os.path.join(workspace_dir, "plots")
    artifact_dir = "/Users/gavin/.gemini/antigravity-ide/brain/f8205597-8eae-4563-9fbb-a5f62d278182"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    train_file = os.path.join(data_dir, "KDDTrain+.txt")
    test_file = os.path.join(data_dir, "KDDTest+.txt")
    
    print("=== Step 1: Loading & Preprocessing Dataset ===")
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi, feature_names, preprocessor = preprocess_nsl_kdd(train_file, test_file)
    
    # Save the preprocessor pipeline
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Fitted preprocessor pipeline saved to {preprocessor_path}")
    
    # Save feature names for reference in dashboard
    feature_names_path = os.path.join(models_dir, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    
    # Autoencoder is trained only on Normal traffic in train set
    normal_indices = (y_train_bin == 0)
    X_train_normal = X_train[normal_indices]
    print(f"Training set has {X_train_normal.shape[0]} normal traffic samples out of {X_train.shape[0]}.")
    
    # Setup Device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using Deep Learning training device: {device}")
    
    results = {}
    cms = []
    roc_data = {}
    model_names = []
    
    # ==========================================
    # Model Training and Evaluation Benchmark Loop
    # ==========================================
    
    # 1. Random Forest
    print("\n=== Model 1: Training Random Forest ===")
    rf_model = get_rf_model()
    t0 = time.time()
    rf_model.fit(X_train, y_train_bin)
    t_train = time.time() - t0
    
    # Evaluate RF
    t0 = time.time()
    rf_preds = rf_model.predict(X_test)
    t_infer = (time.time() - t0) / len(X_test) * 1000 # ms per sample
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    # Save Model
    with open(os.path.join(models_dir, "random_forest.pkl"), 'wb') as f:
        pickle.dump(rf_model, f)
        
    results["Random Forest"] = {
        "Accuracy": accuracy_score(y_test_bin, rf_preds),
        "Precision": precision_score(y_test_bin, rf_preds),
        "Recall": recall_score(y_test_bin, rf_preds),
        "F1-Score": f1_score(y_test_bin, rf_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, rf_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, rf_probs)
    roc_data["Random Forest"] = (fpr, tpr, roc_auc_score(y_test_bin, rf_probs))
    model_names.append("Random Forest")
    print(f"RF F1: {results['Random Forest']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")
    
    # 2. XGBoost
    print("\n=== Model 2: Training XGBoost ===")
    xgb_model = get_xgb_model()
    t0 = time.time()
    xgb_model.fit(X_train, y_train_bin)
    t_train = time.time() - t0
    
    # Evaluate XGB
    t0 = time.time()
    xgb_preds = xgb_model.predict(X_test)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Save Model
    with open(os.path.join(models_dir, "xgboost.pkl"), 'wb') as f:
        pickle.dump(xgb_model, f)
        
    results["XGBoost"] = {
        "Accuracy": accuracy_score(y_test_bin, xgb_preds),
        "Precision": precision_score(y_test_bin, xgb_preds),
        "Recall": recall_score(y_test_bin, xgb_preds),
        "F1-Score": f1_score(y_test_bin, xgb_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, xgb_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, xgb_probs)
    roc_data["XGBoost"] = (fpr, tpr, roc_auc_score(y_test_bin, xgb_probs))
    model_names.append("XGBoost")
    print(f"XGB F1: {results['XGBoost']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")
    
    # 3. SVM (Stochastic Gradient Descent Linear SVM)
    print("\n=== Model 3: Training Linear SVM ===")
    svm_model = get_svm_model()
    t0 = time.time()
    svm_model.fit(X_train, y_train_bin)
    t_train = time.time() - t0
    
    # Evaluate SVM
    t0 = time.time()
    svm_preds = svm_model.predict(X_test)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    svm_dec = svm_model.decision_function(X_test)
    # Min-max scale decision values to act as probabilities for ROC curve
    svm_probs = (svm_dec - svm_dec.min()) / (svm_dec.max() - svm_dec.min())
    
    # Save Model
    with open(os.path.join(models_dir, "svm.pkl"), 'wb') as f:
        pickle.dump(svm_model, f)
        
    results["Linear SVM"] = {
        "Accuracy": accuracy_score(y_test_bin, svm_preds),
        "Precision": precision_score(y_test_bin, svm_preds),
        "Recall": recall_score(y_test_bin, svm_preds),
        "F1-Score": f1_score(y_test_bin, svm_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, svm_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, svm_probs)
    roc_data["Linear SVM"] = (fpr, tpr, roc_auc_score(y_test_bin, svm_probs))
    model_names.append("Linear SVM")
    print(f"SVM F1: {results['Linear SVM']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")
    

    # Decision Tree
    print("\n=== Training Decision Tree ===")
    dt_model = get_dt_model()
    t0 = time.time()
    dt_model.fit(X_train, y_train_bin)
    t_train = time.time() - t0
    t0 = time.time()
    dt_preds = dt_model.predict(X_test)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    dt_probs = dt_model.predict_proba(X_test)[:, 1]
    results["Decision Tree"] = {
        "Accuracy": accuracy_score(y_test_bin, dt_preds),
        "Precision": precision_score(y_test_bin, dt_preds),
        "Recall": recall_score(y_test_bin, dt_preds),
        "F1-Score": f1_score(y_test_bin, dt_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, dt_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, dt_probs)
    roc_data["Decision Tree"] = (fpr, tpr, roc_auc_score(y_test_bin, dt_probs))
    model_names.append("Decision Tree")
    print(f"DT F1: {results['Decision Tree']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")

    # KNN
    print("\n=== Training KNN ===")
    knn_model = get_knn_model()
    t0 = time.time()
    knn_model.fit(X_train, y_train_bin)
    t_train = time.time() - t0
    t0 = time.time()
    knn_preds = knn_model.predict(X_test)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    knn_probs = knn_model.predict_proba(X_test)[:, 1]
    results["KNN"] = {
        "Accuracy": accuracy_score(y_test_bin, knn_preds),
        "Precision": precision_score(y_test_bin, knn_preds),
        "Recall": recall_score(y_test_bin, knn_preds),
        "F1-Score": f1_score(y_test_bin, knn_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, knn_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, knn_probs)
    roc_data["KNN"] = (fpr, tpr, roc_auc_score(y_test_bin, knn_probs))
    model_names.append("KNN")
    print(f"KNN F1: {results['KNN']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")

    # Naive Bayes
    print("\n=== Training Naive Bayes ===")
    nb_model = get_nb_model()
    t0 = time.time()
    nb_model.fit(X_train, y_train_bin)
    t_train = time.time() - t0
    t0 = time.time()
    nb_preds = nb_model.predict(X_test)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    nb_probs = nb_model.predict_proba(X_test)[:, 1]
    results["Naive Bayes"] = {
        "Accuracy": accuracy_score(y_test_bin, nb_preds),
        "Precision": precision_score(y_test_bin, nb_preds),
        "Recall": recall_score(y_test_bin, nb_preds),
        "F1-Score": f1_score(y_test_bin, nb_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, nb_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, nb_probs)
    roc_data["Naive Bayes"] = (fpr, tpr, roc_auc_score(y_test_bin, nb_probs))
    model_names.append("Naive Bayes")
    print(f"NB F1: {results['Naive Bayes']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")

    # Logistic Regression
    print("\n=== Training Logistic Regression ===")
    lr_model = get_lr_model()
    t0 = time.time()
    lr_model.fit(X_train, y_train_bin)
    t_train = time.time() - t0
    t0 = time.time()
    lr_preds = lr_model.predict(X_test)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    results["Logistic Regression"] = {
        "Accuracy": accuracy_score(y_test_bin, lr_preds),
        "Precision": precision_score(y_test_bin, lr_preds),
        "Recall": recall_score(y_test_bin, lr_preds),
        "F1-Score": f1_score(y_test_bin, lr_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, lr_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, lr_probs)
    roc_data["Logistic Regression"] = (fpr, tpr, roc_auc_score(y_test_bin, lr_probs))
    model_names.append("Logistic Regression")
    print(f"LR F1: {results['Logistic Regression']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")

    # LightGBM
    print("\n=== Training LightGBM ===")
    lgbm_model = get_lgbm_model()
    t0 = time.time()
    lgbm_model.fit(X_train, y_train_bin)
    t_train = time.time() - t0
    t0 = time.time()
    lgbm_preds = lgbm_model.predict(X_test)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
    results["LightGBM"] = {
        "Accuracy": accuracy_score(y_test_bin, lgbm_preds),
        "Precision": precision_score(y_test_bin, lgbm_preds),
        "Recall": recall_score(y_test_bin, lgbm_preds),
        "F1-Score": f1_score(y_test_bin, lgbm_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, lgbm_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, lgbm_probs)
    roc_data["LightGBM"] = (fpr, tpr, roc_auc_score(y_test_bin, lgbm_probs))
    model_names.append("LightGBM")
    print(f"LGBM F1: {results['LightGBM']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")

    # 4. Deep Learning: MLP
    print("\n=== Model 4: Training Multi-Layer Perceptron (PyTorch) ===")
    mlp_model, t_train = train_mlp(X_train, y_train_bin, epochs=8, batch_size=128, lr=0.001, device=device)
    
    # Evaluate MLP
    t0 = time.time()
    mlp_preds, mlp_probs = predict_mlp(mlp_model, X_test, device=device)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    
    # Save Model
    torch.save(mlp_model.state_dict(), os.path.join(models_dir, "mlp_state_dict.pth"))
    
    results["MLP Network"] = {
        "Accuracy": accuracy_score(y_test_bin, mlp_preds),
        "Precision": precision_score(y_test_bin, mlp_preds),
        "Recall": recall_score(y_test_bin, mlp_preds),
        "F1-Score": f1_score(y_test_bin, mlp_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, mlp_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, mlp_probs)
    roc_data["MLP Network"] = (fpr, tpr, roc_auc_score(y_test_bin, mlp_probs))
    model_names.append("MLP Network")
    print(f"MLP F1: {results['MLP Network']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")
    
    # 5. Deep Learning: Autoencoder
    print("\n=== Model 5: Training Autoencoder (PyTorch) ===")
    ae_model, t_train = train_autoencoder(X_train_normal, epochs=8, batch_size=128, lr=0.001, device=device)
    ae_threshold = find_ae_threshold(ae_model, X_train_normal, percentile=97.0, device=device)
    
    # Evaluate Autoencoder
    t0 = time.time()
    ae_preds, ae_errors = predict_autoencoder(ae_model, X_test, ae_threshold, device=device)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    
    # Save Model & Threshold
    torch.save(ae_model.state_dict(), os.path.join(models_dir, "autoencoder_state_dict.pth"))
    with open(os.path.join(models_dir, "ae_threshold.json"), 'w') as f:
        json.dump({"threshold": float(ae_threshold)}, f)
        
    # Min-max scale reconstruction errors to act as scores for ROC-AUC
    ae_probs = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min())
    
    results["Autoencoder"] = {
        "Accuracy": accuracy_score(y_test_bin, ae_preds),
        "Precision": precision_score(y_test_bin, ae_preds),
        "Recall": recall_score(y_test_bin, ae_preds),
        "F1-Score": f1_score(y_test_bin, ae_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, ae_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, ae_probs)
    roc_data["Autoencoder"] = (fpr, tpr, roc_auc_score(y_test_bin, ae_probs))
    model_names.append("Autoencoder")
    print(f"AE F1: {results['Autoencoder']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")
    

    # Deep Learning: LSTM
    print("\n=== Training LSTM (PyTorch) ===")
    lstm_model, t_train = train_lstm(X_train, y_train_bin, epochs=8, batch_size=128, lr=0.001, device=device)
    t0 = time.time()
    lstm_preds, lstm_probs = predict_lstm(lstm_model, X_test, device=device)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    results["LSTM"] = {
        "Accuracy": accuracy_score(y_test_bin, lstm_preds),
        "Precision": precision_score(y_test_bin, lstm_preds),
        "Recall": recall_score(y_test_bin, lstm_preds),
        "F1-Score": f1_score(y_test_bin, lstm_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, lstm_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, lstm_probs)
    roc_data["LSTM"] = (fpr, tpr, roc_auc_score(y_test_bin, lstm_probs))
    model_names.append("LSTM")
    print(f"LSTM F1: {results['LSTM']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")

    # Deep Learning: CNN1D
    print("\n=== Training CNN1D (PyTorch) ===")
    cnn_model, t_train = train_cnn1d(X_train, y_train_bin, epochs=8, batch_size=128, lr=0.001, device=device)
    t0 = time.time()
    cnn_preds, cnn_probs = predict_cnn1d(cnn_model, X_test, device=device)
    t_infer = (time.time() - t0) / len(X_test) * 1000
    results["CNN1D"] = {
        "Accuracy": accuracy_score(y_test_bin, cnn_preds),
        "Precision": precision_score(y_test_bin, cnn_preds),
        "Recall": recall_score(y_test_bin, cnn_preds),
        "F1-Score": f1_score(y_test_bin, cnn_preds),
        "Training Time (s)": t_train,
        "Inference Latency (ms/sample)": t_infer
    }
    cms.append(confusion_matrix(y_test_bin, cnn_preds))
    fpr, tpr, _ = roc_curve(y_test_bin, cnn_probs)
    roc_data["CNN1D"] = (fpr, tpr, roc_auc_score(y_test_bin, cnn_probs))
    model_names.append("CNN1D")
    print(f"CNN1D F1: {results['CNN1D']['F1-Score']:.4f} | Train Time: {t_train:.2f}s")

    # ==========================================
    # Reporting and Visualization
    # ==========================================
    print("\n=== Step 3: Generating Performance Reports ===")
    
    # Convert results to DataFrame
    metrics_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'Model'})
    print("\nBenchmark Results Summary Table:")
    print(metrics_df.to_string(index=False))
    
    # Save metrics JSON
    with open(os.path.join(workspace_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(artifact_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Metrics JSON saved successfully.")
    
    # Generate Plots
    print("Generating visual plots...")
    plot_confusion_matrices(cms, model_names, plots_dir)
    plot_roc_curves(roc_data, plots_dir)
    plot_metrics_comparison(metrics_df, plots_dir)
    plot_times_comparison(metrics_df, plots_dir)
    
    # Copy plots to artifact directory for markdown display
    import shutil
    for filename in ["confusion_matrices.png", "roc_curves.png", "metrics_comparison.png", "times_comparison.png"]:
        src_plot = os.path.join(plots_dir, filename)
        dest_plot = os.path.join(artifact_dir, filename)
        if os.path.exists(src_plot):
            shutil.copy2(src_plot, dest_plot)
            
    print(f"Visual plots generated and saved to {plots_dir} and artifact directory.")

if __name__ == "__main__":
    main()
