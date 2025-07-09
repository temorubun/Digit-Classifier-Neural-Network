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
   - **Setiap neuron dihubungkan ke neuron di layer berikutnya dengan bobot (weight) dan bias yang diinisialisasi secara acak.**
     
     **Penjelasan detail:**
     - Setiap koneksi antar neuron diwakili oleh sebuah bobot (weight) \( w_{jk} \), di mana \( w_{jk} \) adalah bobot dari neuron ke-\( k \) di layer sebelumnya ke neuron ke-\( j \) di layer saat ini.
     - Setiap neuron (kecuali layer input) juga memiliki bias \( b_j \) yang diinisialisasi secara acak.
     - Pada inisialisasi, semua bobot dan bias diambil dari distribusi normal (Gaussian) dengan rata-rata 0 dan standar deviasi 1:
       
       \[
       w_{jk} \sim \mathcal{N}(0, 1) \qquad b_j \sim \mathcal{N}(0, 1)
       \]
     - Untuk setiap layer \( l \) (selain input), vektor bias \( \mathbf{b}^l \) dan matriks bobot \( \mathbf{W}^l \) diinisialisasi dengan ukuran sesuai jumlah neuron di layer tersebut dan layer sebelumnya.
     
     **Proses Feedforward:**
     - Input dari layer sebelumnya (atau input gambar) adalah vektor \( \mathbf{a}^{l-1} \).
     - Setiap neuron di layer \( l \) menghitung nilai input total (z) sebagai:
       
       \[
       z^l_j = \sum_k w^l_{jk} a^{l-1}_k + b^l_j
       \]
       atau dalam bentuk vektor:
       \[
       \mathbf{z}^l = \mathbf{W}^l \mathbf{a}^{l-1} + \mathbf{b}^l
       \]
     - Nilai \( z^l_j \) kemudian dilewatkan ke fungsi aktivasi sigmoid:
       \[
       a^l_j = \sigma(z^l_j) = \frac{1}{1 + e^{-z^l_j}}
       \]
     - Proses ini diulang untuk setiap layer hingga output.
     
     **Implementasi di kode:**
     - Inisialisasi bobot dan bias: lihat konstruktor `__init__` di `network.py`.
     - Proses feedforward: lihat fungsi `feedforward` di `network.py`.

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
