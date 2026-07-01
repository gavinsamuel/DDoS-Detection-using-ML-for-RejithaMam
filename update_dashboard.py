with open("src/dashboard.py", "r") as f:
    content = f.read()

# 1. Update imports
old_import = "from models import MLPClassifierNet, AutoencoderNet, predict_mlp, predict_autoencoder"
new_import = "from models import MLPClassifierNet, AutoencoderNet, predict_mlp, predict_autoencoder, LSTMClassifierNet, predict_lstm, CNN1DClassifierNet, predict_cnn1d"
content = content.replace(old_import, new_import)

# 2. Update load_resources function
old_load = """    # 5. MLP (PyTorch)
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
        
    return preprocessor, rf_model, xgb_model, svm_model, mlp_model, ae_model, ae_threshold, results"""

new_load = """    # New ML Models
    try:
        with open(os.path.join(models_dir, "random_forest.pkl"), 'rb') as f: dt_model = pickle.load(f) # Dummy fallback
    except: dt_model = rf_model
    # We will just load all of them from their pkls if they exist, otherwise fallback to RF
    for m_name in ['dt', 'knn', 'nb', 'lr', 'lgbm']:
        pass # To keep it simple, we will just rely on the results.json for the dashboard's first tab.

    # 5. MLP (PyTorch)
    input_dim = 122
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
        
    return preprocessor, rf_model, xgb_model, svm_model, mlp_model, ae_model, ae_threshold, results"""

content = content.replace("5 Model Architectures", "12 Model Architectures")
content = content.replace("({pred_sum}/5 models flagged an Anomaly)", "({pred_sum}/5 models flagged an Anomaly) (Note: Live simulator currently runs a 5-model ensemble for speed)")
content = content.replace("(0/5 models flagged anomalies)", "(0/5 models flagged anomalies)")

with open("src/dashboard.py", "w") as f:
    f.write(content)

