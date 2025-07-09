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

### **Alur Kerja: Input → Proses → Output**

#### 1. **Input**
- **Sumber data:**
  - Saat training: gambar digit dari dataset MNIST (format 28x28 piksel, grayscale).
  - Saat prediksi: gambar digit yang Anda gambar di aplikasi web (kanvas Streamlit).
- **Representasi data:**
  - Setiap gambar diubah menjadi vektor kolom berukuran 784x1 (karena 28x28 = 784 piksel).
  - Nilai piksel dinormalisasi ke rentang 0-1 (0 = hitam, 1 = putih).

**Kode (dari `mnist_loader.py` dan `app.py`):**
```python
# Untuk training (mnist_loader.py)
training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]

# Untuk prediksi di web app (app.py)
processed = crop_and_center(gray)  # hasil 28x28
input_vector = processed.reshape(784, 1)  # jadi 784x1
```

#### 2. **Proses (Feedforward di Neural Network)**
- **Langkah-langkah utama:**
  1. **Input vektor** (784x1) masuk ke layer pertama (input layer).
  2. **Setiap layer** (kecuali input) melakukan:
     - Mengalikan input dengan bobot, menambahkan bias:
       \[
       \mathbf{z} = \mathbf{W} \mathbf{a} + \mathbf{b}
       \]
     - Menerapkan fungsi aktivasi sigmoid ke setiap elemen z:
       \[
       \mathbf{a}_{\text{baru}} = \sigma(\mathbf{z}) = \frac{1}{1 + e^{-\mathbf{z}}}
       \]
     - Output dari layer ini menjadi input untuk layer berikutnya.
  3. Proses diulang hingga layer output.

**Kode (dari `network.py`):**
```python
def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, a) + b  # z = W*a + b
        a = sigmoid(z)        # a = sigmoid(z)
    return a

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
```

**Contoh kode sederhana:**
```python
import numpy as np

# Bobot dan bias acak untuk 1 layer
W = np.array([[0.2, -0.5],
              [1.5,  0.3],
              [-1.2, 0.7]])
b = np.array([[0.1], [0.2], [-0.3]])
a_prev = np.array([[0.6], [0.9]])  # input dari layer sebelumnya

z = np.dot(W, a_prev) + b  # z = W*a + b
sigmoid = lambda x: 1 / (1 + np.exp(-x))
a = sigmoid(z)
print(a)
```

#### 3. **Output**
- **Bentuk output:**
  - Vektor berukuran 10x1, setiap elemen mewakili probabilitas prediksi untuk digit 0-9.
  - Nilai tertinggi menunjukkan digit yang diprediksi jaringan.
- **Interpretasi:**
  - Misal output: `[0.01, 0.02, 0.95, 0.01, ...]` → prediksi = 2 (karena 0.95 paling besar di indeks ke-2).

**Kode (dari `app.py`):**
```python
output = net.feedforward(input_vector)
prediction = int(np.argmax(output))
```

**Rumus:**
\[
prediksi = \text{argmax}(\mathbf{a}_{\text{output}})
\]

---

### 1. **Arsitektur Jaringan dan Koneksi Antar Neuron**
Jaringan terdiri dari beberapa layer:
- **Input layer:** 784 neuron (untuk 28x28 piksel gambar)
- **Hidden layer:** 2 layer, masing-masing 16 neuron
- **Output layer:** 10 neuron (untuk digit 0-9)

#### **Bobot (Weight) dan Bias**
Setiap neuron di layer (selain input) menerima input dari semua neuron di layer sebelumnya. Setiap koneksi memiliki **bobot** \( w_{jk} \), dan setiap neuron memiliki **bias** \( b_j \).

- **Bobot**: Mengatur seberapa besar pengaruh input dari neuron sebelumnya.
- **Bias**: Nilai tambahan yang memungkinkan model lebih fleksibel.

**Inisialisasi:**
- Semua bobot dan bias diinisialisasi secara acak dari distribusi normal (mean=0, std=1):
  \[
  w_{jk} \sim \mathcal{N}(0, 1) \qquad b_j \sim \mathcal{N}(0, 1)
  \]

**Kode Python (dari `network.py`):**
```python
self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # Bias untuk setiap neuron (kecuali input)
self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # Bobot antar layer
```

#### **Proses Feedforward (Perhitungan Output)**
Feedforward adalah proses menghitung output jaringan dari input hingga output layer.

**Langkah-langkah:**
1. Input gambar diubah menjadi vektor kolom (784x1).
2. Untuk setiap layer:
   - Hitung jumlah input ke setiap neuron:
     \[
     z_j = \sum_k w_{jk} a_k + b_j
     \]
     atau dalam bentuk vektor:
     \[
     \mathbf{z} = \mathbf{W} \mathbf{a} + \mathbf{b}
     \]
   - Terapkan fungsi aktivasi sigmoid ke setiap nilai z:
     \[
     a_j = \sigma(z_j) = \frac{1}{1 + e^{-z_j}}
     \]
3. Output akhir adalah prediksi jaringan.

**Kode Python (dari `network.py`):**
```python
def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, a) + b  # z = W*a + b
        a = sigmoid(z)        # a = sigmoid(z)
    return a

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
```

#### **Contoh Sederhana (dengan angka kecil)**
Misal layer sebelumnya punya 2 neuron, layer sekarang 3 neuron:
- Bobot: matriks 3x2 (3 neuron, masing-masing terhubung ke 2 input)
- Bias: vektor 3x1

```python
import numpy as np

# Bobot dan bias acak
W = np.array([[0.2, -0.5],
              [1.5,  0.3],
              [-1.2, 0.7]])
b = np.array([[0.1], [0.2], [-0.3]])
a_prev = np.array([[0.6], [0.9]])  # input dari layer sebelumnya

# Proses feedforward satu layer
z = np.dot(W, a_prev) + b  # z = W*a + b
# Fungsi aktivasi sigmoid
sigmoid = lambda x: 1 / (1 + np.exp(-x))
a = sigmoid(z)
print(a)
```
**Penjelasan kode:**
- `np.dot(W, a_prev)`: Mengalikan bobot dengan input.
- `+ b`: Menambahkan bias ke setiap neuron.
- `sigmoid(z)`: Mengubah hasil ke rentang 0-1.
- Output `a` adalah hasil aktivasi setiap neuron di layer tersebut.

---

### 2. **Training (Pelatihan)**
- Data MNIST (gambar digit tulisan tangan) dimuat dan diproses menjadi vektor.
- Proses training menggunakan algoritma backpropagation dan stochastic gradient descent:
  - Data dibagi menjadi mini-batch.
  - Untuk setiap mini-batch, jaringan melakukan feedforward (menghitung output dari input) dan backpropagation (menghitung error dan memperbarui bobot/bias).
  - Proses ini diulang selama beberapa epoch (putaran) untuk meminimalkan error.
- Setelah training selesai, model disimpan ke file `trained_network.pkl`.

### 3. **Prediksi (Inference)**
- Model yang sudah dilatih dapat memprediksi digit dari gambar baru.
- Pada aplikasi web, pengguna menggambar digit di kanvas.
- Gambar diproses (crop, resize, normalisasi) agar sesuai format MNIST.
- Gambar diubah menjadi vektor dan dimasukkan ke jaringan.
- Output jaringan adalah vektor probabilitas untuk setiap digit (0-9); digit dengan probabilitas tertinggi dipilih sebagai prediksi.

### 4. **Aktivasi**
- Fungsi aktivasi yang digunakan adalah sigmoid, yang membatasi output neuron antara 0 dan 1.

---
