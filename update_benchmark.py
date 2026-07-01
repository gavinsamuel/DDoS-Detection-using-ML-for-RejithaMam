import re

with open("src/benchmark.py", "r") as f:
    content = f.read()

# Update imports
old_import = """from models import (
    get_rf_model, get_xgb_model, get_svm_model,
    train_mlp, predict_mlp,
    train_autoencoder, find_ae_threshold, predict_autoencoder
)"""
new_import = """from models import (
    get_rf_model, get_xgb_model, get_svm_model,
    get_dt_model, get_knn_model, get_nb_model, get_lr_model, get_lgbm_model,
    train_mlp, predict_mlp,
    train_autoencoder, find_ae_threshold, predict_autoencoder,
    train_lstm, predict_lstm, train_cnn1d, predict_cnn1d
)"""
content = content.replace(old_import, new_import)

# Define new ML models code
new_ml_models = """
    # Decision Tree
    print("\\n=== Training Decision Tree ===")
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
    print("\\n=== Training KNN ===")
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
    print("\\n=== Training Naive Bayes ===")
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
    print("\\n=== Training Logistic Regression ===")
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
    print("\\n=== Training LightGBM ===")
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
"""

# Insert new ML models before MLP
insert_idx_ml = content.find("    # 4. Deep Learning: MLP")
content = content[:insert_idx_ml] + new_ml_models + "\\n" + content[insert_idx_ml:]

# Define new DL models code
new_dl_models = """
    # Deep Learning: LSTM
    print("\\n=== Training LSTM (PyTorch) ===")
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
    print("\\n=== Training CNN1D (PyTorch) ===")
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
"""

# Insert new DL models before reporting
insert_idx_dl = content.find("    # ==========================================\n    # Reporting and Visualization")
content = content[:insert_idx_dl] + new_dl_models + "\\n" + content[insert_idx_dl:]

with open("src/benchmark.py", "w") as f:
    f.write(content)

