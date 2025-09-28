# Blood Group Detection from Fingerprint Images

An AI-powered system that predicts blood groups from fingerprint images using deep learning.

## 🔍 Overview

This project uses a ResNet50-based deep learning model to analyze fingerprint patterns and predict blood groups. The system includes a web interface for easy interaction and real-time predictions.

## 🛠️ Prerequisites

- Python 3.10 or higher
- Windows 10/11
- 8GB RAM minimum
- NVIDIA GPU (optional, but recommended)

## 📦 Installation

1. **Clone or Download Project**
```batch
# Install Git LFS first
git lfs install

# Clone repository with LFS support
git clone https://github.com/Nithin3302/BloodGroup-Detection.git
cd BloodGroup-Detection

# Pull LFS files
git lfs pull
```

2. **Set Up Virtual Environment**
```batch
# Navigate to project directory (adjust path as needed)
cd C:\Users\YourUsername\Projects\bloodgroup_detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate
```

3. **Install Dependencies**
```batch
# Ensure you're in the project directory
cd C:\Users\YourUsername\Projects\bloodgroup_detection

# Install specific numpy version first
pip install numpy==1.23.5

# Install remaining dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
bloodgroup_detection/
├── data/
│   └── dataset_blood_group/    # Dataset folder
│       ├── A+/
│       ├── A-/
│       ├── B+/
│       ├── B-/
│       ├── AB+/
│       ├── AB-/
│       ├── O+/
│       └── O-/
├── models/
│   └── blood_group_model.h5    # Trained model
├── src/
│   ├── app.py                  # Streamlit interface
│   ├── predict.py              # Prediction logic
│   ├── train.py               # Training script
│   ├── data_processing.py     # Data utilities
│   ├── model.py               # Model architecture
│   └── utils.py               # Helper functions
├── static/
│   └── model_performance.png   # Performance graphs
├── requirements.txt
└── README.md
```

## 🚀 Running the Application

1. **Start the Web Interface**
```batch
streamlit run src/app.py
```

2. **Exit the Application**
```batch
# Press Ctrl + C in terminal to exit
```

## 💡 Usage

1. Upload a fingerprint image (supported formats: BMP, JPG, PNG)
2. View the prediction results and confidence scores
3. Check detailed probability distribution for all blood groups

## ⚙️ System Requirements

- CPU Mode: Will work but slower
- GPU Mode: Automatically used if NVIDIA GPU available
- Memory Usage: ~2GB minimum during prediction
- Storage: ~500MB for model and dependencies

For any issues or questions, please open an issue in the repository.
