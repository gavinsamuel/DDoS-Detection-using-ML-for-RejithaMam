import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Column Names for NSL-KDD
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
    "num_compromised", "root_shell", "su_attempted", "num_root", 
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", 
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"
]

# Map specific attack types to five primary categories
ATTACK_MAPPING = {
    # DoS
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 
    'teardrop': 'DoS', 'udpstorm': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 
    'processtable': 'DoS', 'worm': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L', 
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 
    'xlock': 'R2L', 'xsnoop': 'R2L', 'httptunnel': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 
    'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
    # Normal
    'normal': 'Normal'
}

def load_data(filepath):
    """Loads the NSL-KDD dataset from file, strips trailing dots, and cleans column names."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}. Please run download_data.py first.")
    
    df = pd.read_csv(filepath, names=COLUMN_NAMES, header=None)
    # Strip dots from classes, some datasets have e.g. normal. instead of normal
    df['class'] = df['class'].str.strip('.')
    return df

def preprocess_nsl_kdd(train_path, test_path):
    """
    Loads and preprocesses both the train and test splits, returns dataframes and arrays.
    
    Returns:
        X_train, X_test (arrays): Preprocessed feature matrices
        y_train_bin, y_test_bin (arrays): Binary labels (0 = Normal, 1 = Attack)
        y_train_multi, y_test_multi (arrays): 5-class labels (0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R)
        feature_names (list): Final features after one-hot encoding
        preprocessor (ColumnTransformer): Fitted preprocessor pipeline
    """
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    
    # 1. Create target columns
    # Binary labels
    train_df['label_bin'] = train_df['class'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['label_bin'] = test_df['class'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Multi-class mapping (with default fallback to 'DoS' if unknown attack)
    def map_attack(val):
        return ATTACK_MAPPING.get(val, 'DoS')
    
    train_df['attack_cat'] = train_df['class'].apply(map_attack)
    test_df['attack_cat'] = test_df['class'].apply(map_attack)
    
    # Label encoding for multi-class
    category_map = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
    train_df['label_multi'] = train_df['attack_cat'].map(category_map)
    test_df['label_multi'] = test_df['attack_cat'].map(category_map)
    
    # Separate features and labels
    drop_cols = ["class", "difficulty", "label_bin", "attack_cat", "label_multi"]
    X_train_raw = train_df.drop(columns=drop_cols)
    X_test_raw = test_df.drop(columns=drop_cols)
    
    y_train_bin = train_df["label_bin"].values
    y_test_bin = test_df["label_bin"].values
    y_train_multi = train_df["label_multi"].values
    y_test_multi = test_df["label_multi"].values
    
    # Identify categorical and numerical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    numerical_cols = [col for col in X_train_raw.columns if col not in categorical_cols]
    
    # 2. Build Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Fit on training data and transform both train and test sets
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Extract feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_cols))
    feature_names = numerical_cols + encoded_cat_features
    
    print(f"Preprocessed features shape: {X_train.shape}")
    print(f"Number of numerical features: {len(numerical_cols)}")
    print(f"Number of categorical features after encoding: {len(encoded_cat_features)}")
    
    return X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi, feature_names, preprocessor

if __name__ == "__main__":
    # Test script execution
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(workspace_dir, "data", "KDDTrain+.txt")
    test_file = os.path.join(workspace_dir, "data", "KDDTest+.txt")
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi, features, _ = preprocess_nsl_kdd(train_file, test_file)
        print("Success! Train Shape:", X_train.shape, "Test Shape:", X_test.shape)
        print("Binary distribution (Train):", np.bincount(y_train_bin))
        print("Multi distribution (Train):", np.bincount(y_train_multi))
    else:
        print("Dataset files not found. Run downloader.py first.")
