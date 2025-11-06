# Brain Tumor Detector Web App
FastAPI-based web application for detecting brain tumors from MRI scan photos.

## Local Development ##
1. Clone the repo
2. Create python virtual environment
```bash
python -m venv venv
```
3. Acitvate venv
```bash
venv\Scripts\activate
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```
5. Run the App:
```bash
uvicorn api.index:app --reload
```

# Brain Tumor Detector Training & Testing

A CNN-based deep learning model to classify brain MRI scans into 4 categories: No Tumor, Glioma, Meningioma, and Pituitary tumor.

## Dataset

Download the dataset from Kaggle - [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset should be organized as follows:
```
project_folder/
├── Training/
│   ├── glioma/*.jpg
│   ├── meningioma/*.jpg
│   ├── pituitary/*.jpg
│   └── notumor/*.jpg
└── Testing/
    ├── glioma/*.jpg
    ├── meningioma/*.jpg
    ├── pituitary/*.jpg
    └── notumor/*.jpg
```

## Project Structure

```
brain-tumor-detector/
├── requirements.txt
├── README.md
├── tumor_detector.ipynb        # Main notebook
├── tumor_detector.pth          # Saved model weights (after training)
├── Training/                   # Training dataset
└── Testing/                    # Testing dataset
```

## Installation

### 1. Create a Virtual Environment

#### Using venv (Python built-in)
```bash
# Create virtual environment
python -m venv tumor_env

# Activate virtual environment
# On Windows:
tumor_env\Scripts\activate
# On macOS/Linux:
source tumor_env/bin/activate
```

#### Using conda (Recommended for GPU support)
```bash
# Create conda environment
conda create -n tumor_env python=3.11

# Activate environment
conda activate tumor_env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. GPU Support (Optional but Recommended)

If you have an NVIDIA GPU, install PyTorch with CUDA support:

```bash
# CUDA 12.4 (Recommended to check compatible CUDA for your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Check PyTorch installation:
```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

## Troubleshooting

### CUDA Errors
If you get "CUDA error: device-side assert triggered":
- Ensure labels are of type `.long()` not `.float()`
- Check that loss function matches task (CrossEntropyLoss for multi-class)

### Memory Issues
If you run out of GPU memory:
- Reduce batch size from 32 to 16 or 8
- Reduce number of epochs

### Module Not Found Errors
```bash
pip install -r requirements.txt
```

