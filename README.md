# Neural Network from Scratch

Simple neural network implementation for MNIST digit classification, built from scratch with NumPy.

*Based on Michael Nielsen's "[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)".*

## Quick Start (Manual Docker Workflow)

Jalankan langkah-langkah berikut untuk menjalankan aplikasi di dalam container Docker berbasis python:3.9:

1. **Tarik image python:3.9 (jika belum ada):**
   ```bash
   docker pull python:3.9
   ```

2. **Cek image yang tersedia:**
   ```bash
   docker images
   ```

3. **Jalankan container dengan volume mount ke direktori kerja:**
   ```bash
   docker run -it --network host --name app_neural_network -v ${PWD}:/app -w /app python:3.9 bash
   ```

4. **Install dependencies Python:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install dependency sistem untuk OpenCV:**
   ```bash
   apt-get update
   apt-get install -y libgl1-mesa-glx
   ```

6. **Latih model neural network:**
   ```bash
   python train.py
   ```

7. **Jalankan aplikasi web Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Buka http://localhost:8501 di browser Anda, gambar digit, dan lihat prediksi.

---

## Files

- `network.py` - Neural network implementation
- `mnist_loader.py` - MNIST data loader
- `train.py` - Script untuk melatih dan menyimpan model
- `train.ipynb` - (Opsional) Notebook untuk eksplorasi/training manual
- `app.py` - Streamlit web interface
- `data/mnist.pkl.gz` - MNIST dataset
- `trained_network.pkl` - Model hasil training

## Network Architecture Example

- Input: 784 neurons (28×28 pixels)
- Hidden: 16 → 16 neurons
- Output: 10 neurons (digits 0-9)
- Activation: Sigmoid

---

## Cara Kerja Neural Network di Project Ini

1. **Arsitektur Jaringan**
   - Jaringan terdiri dari 4 layer: input (784 neuron untuk 28x28 piksel), dua hidden layer (masing-masing 16 neuron), dan output (10 neuron untuk digit 0-9).
   - Setiap neuron dihubungkan ke neuron di layer berikutnya dengan bobot (weight) dan bias yang diinisialisasi secara acak.

2. **Training (Pelatihan)**
   - Data MNIST (gambar digit tulisan tangan) dimuat dan diproses menjadi vektor.
   - Proses training menggunakan algoritma backpropagation dan stochastic gradient descent:
     - Data dibagi menjadi mini-batch.
     - Untuk setiap mini-batch, jaringan melakukan feedforward (menghitung output dari input) dan backpropagation (menghitung error dan memperbarui bobot/bias).
     - Proses ini diulang selama beberapa epoch (putaran) untuk meminimalkan error.
   - Setelah training selesai, model disimpan ke file `trained_network.pkl`.

3. **Prediksi (Inference)**
   - Model yang sudah dilatih dapat memprediksi digit dari gambar baru.
   - Pada aplikasi web, pengguna menggambar digit di kanvas.
   - Gambar diproses (crop, resize, normalisasi) agar sesuai format MNIST.
   - Gambar diubah menjadi vektor dan dimasukkan ke jaringan.
   - Output jaringan adalah vektor probabilitas untuk setiap digit (0-9); digit dengan probabilitas tertinggi dipilih sebagai prediksi.

4. **Aktivasi**
   - Fungsi aktivasi yang digunakan adalah sigmoid, yang membatasi output neuron antara 0 dan 1.

---
