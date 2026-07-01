# Cloud Network Anomaly Detection Using AI

## 📖 Project Overview
This project serves as a comprehensive testing environment for evaluating Artificial Intelligence algorithms in the context of cloud security. It is directly based on the research paper: 
**"Cloud Network Anomaly Detection Using Machine and Deep Learning Techniques — Recent Research Advancements"**.

The primary goal of this system is to identify "anomalies" in cloud networks. Specifically, it distinguishes between normal, benign network traffic and malicious activities such as **Distributed Denial of Service (DDoS) attacks** and other intrusions.

## 📊 The Dataset
The project utilizes the **NSL-KDD Dataset**, a highly recognized benchmark dataset in the field of cybersecurity and Intrusion Detection Systems (IDS).

- **Source:** The dataset is automatically downloaded from a public GitHub repository.
- **Content:** It contains thousands of records of internet traffic.
- **Labels:** Each record in the dataset is labeled as either **Normal** or categorized into one of four major attack types:
  1. **DoS (Denial of Service):** Flooding a system to disrupt service.
  2. **Probe:** Scanning a network to gather information or find vulnerabilities.
  3. **R2L (Remote to Local):** Unauthorized access from a remote machine.
  4. **U2R (User to Root):** Unauthorized access to local superuser (root) privileges.

## 🏗️ Project Architecture & Pipeline
The codebase is structured into a streamlined pipeline with 5 main components:

### 1. Data Downloading (`src/downloader.py`)
This script automatically fetches the raw NSL-KDD training and testing datasets (`KDDTrain+.txt` and `KDDTest+.txt`) from the internet and saves them to the local `data/` directory.

### 2. Data Preprocessing (`src/preprocess.py`)
AI models require numerical data to function. The preprocessor cleans the raw dataset by:
- Grouping all the specific attack signatures into the 5 primary categories (Normal, DoS, Probe, R2L, U2R).
- Converting categorical text data (like network protocols) into numerical formats using techniques like One-Hot Encoding.
- Scaling and standardizing the numerical features so the models can train efficiently.

### 3. Model Definitions (`src/models.py`)
This file acts as a library containing the blueprints for all the AI models proposed in the base paper. 

**Implemented Machine Learning (ML) Models:**
- Random Forest
- XGBoost
- Linear Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes (Gaussian)
- Logistic Regression
- LightGBM

**Implemented Deep Learning (DL) Models (Built with PyTorch):**
- Multi-Layer Perceptron (MLP)
- Autoencoder (Anomaly Detection)
- Long Short-Term Memory (LSTM)
- 1D Convolutional Neural Network (CNN)

### 4. Benchmarking Engine (`src/benchmark.py`)
This is the core execution script. It loads the preprocessed data, trains every single model defined in `models.py`, and pits them against the test dataset. 
- It tracks crucial performance metrics such as **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Training Time**, and **Inference Latency** (how fast it predicts an attack).
- It generates comparative visualizations (Confusion Matrices, ROC Curves, and bar charts) and saves them in the `plots/` directory.
- It outputs a compiled `results.json` file containing all the performance data.

### 5. Interactive Dashboard (`src/dashboard.py`)
A user interface built with **Streamlit**. It reads the models, the generated metrics, and the plots, presenting them in a premium, visually appealing web dashboard so you can easily explore and compare the results without looking at terminal outputs.

## 🚀 How to Run the Project

1. **Download the Data:**
   ```bash
   python src/downloader.py
   ```
2. **Run the Benchmark (Train all models):**
   *(Note: This might take a few minutes as Deep Learning models require time to train).*
   ```bash
   python src/benchmark.py
   ```
3. **Launch the UI Dashboard:**
   ```bash
   streamlit run src/dashboard.py
   ```
