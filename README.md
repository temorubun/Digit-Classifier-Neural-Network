# Neural Network from Scratch

Simple neural network implementation for MNIST digit classification, built from scratch with NumPy.

*Based on Michael Nielsen's "[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)".*

## Quick Start

### Option 1: Using Docker (Recommended)

1. **Build and run with Docker:**
   ```bash
   # Build the development image
   docker build -t neural-network-app-dev .

   # Run with volume mount for development
   docker run -p 8501:8501 -v ${PWD}:/app neural-network-app-dev
   ```
   This will:
   - Install all Python and system dependencies (including OpenCV)
   - Automatically train the model if not exists (see `train.py`)
   - Start the Streamlit web interface
   - Mount your local code for live development
   - Access the app at http://localhost:8501

### Option 2: Local Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the network:**
   ```bash
   python train.py
   ```
   Ini akan melatih model dan menyimpan hasilnya ke `trained_network.pkl`.

3. **Test with web app:**
   ```bash
   streamlit run app.py
   ```
   Buka http://localhost:8501 di browser Anda, gambar digit, dan lihat prediksi.

## Files

- `network.py` - Neural network implementation
- `mnist_loader.py` - MNIST data loader
- `train.py` - Script untuk melatih dan menyimpan model
- `train.ipynb` - (Opsional) Notebook untuk eksplorasi/training manual
- `app.py` - Streamlit web interface
- `data/mnist.pkl.gz` - MNIST dataset
- `Dockerfile` - Docker configuration for development
- `trained_network.pkl` - Model hasil training

## Network Architecture Example

- Input: 784 neurons (28×28 pixels)
- Hidden: 16 → 16 neurons
- Output: 10 neurons (digits 0-9)
- Activation: Sigmoid
