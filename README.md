# Deteksi Berita Hoax - TheHack 2025

Proyek ini merupakan aplikasi web berbasis **Natural Language Processing (NLP)** yang bertujuan untuk **mendeteksi berita hoax berbahasa Indonesia** menggunakan model **IndoBERT**.  
Sistem dibangun dengan arsitektur **client-server** dan dijalankan secara **lokal (local environment)** selama tahap pengembangan dan pengujian.

---

## 🚀 Arsitektur Sistem

Aplikasi ini terdiri dari tiga komponen utama:

### 🔹 1. Frontend (Client Side)
- Dibangun menggunakan **Next.js** (React framework) dengan dukungan **JavaScript** dan **Tailwind CSS**.  
- Pengguna dapat memasukkan teks berita atau URL berita, yang kemudian dikirim ke server backend melalui **REST API endpoint**.  
- Hasil prediksi ditampilkan secara *real-time* dalam bentuk label:
  - **"Hoax"** atau **"Valid"**
  - Disertai *confidence score* (tingkat kepercayaan model).  
- Antarmuka juga menyediakan fitur pencarian berita menggunakan **SearXNG** untuk membantu pengguna melakukan verifikasi silang terhadap berita yang dimasukkan.

---

### 🔹 2. Backend (Server Side)
- Dikembangkan menggunakan **Node.js** dengan framework **Express.js**.  
- Berfungsi sebagai REST API server yang:
  - Menerima input teks dari frontend.
  - Mengirim teks ke model AI berbasis Python untuk diprediksi.
  - Mengembalikan hasil prediksi ke frontend dalam format JSON.  
- Backend juga menangani routing API, autentikasi dasar (jika diperlukan), dan koneksi dengan search engine eksternal.  
- Sistem ini tidak menggunakan database permanen — input dan hasil prediksi diproses langsung tanpa penyimpanan.

---

### 🔹 3. Model Kecerdasan Artifisial (AI Engine)
- Dikembangkan menggunakan **Python** dengan framework **PyTorch** dan **Hugging Face Transformers**.  
- Model utama adalah **IndoBERT**, sebuah model *transformer pre-trained* yang diadaptasi khusus untuk bahasa Indonesia.  
- Dataset pelatihan terdiri atas kumpulan berita yang telah dilabeli sebagai hoax dan non-hoax.  
- Proses pelatihan dan evaluasi model dilakukan dengan pustaka berikut:
  - **Pandas**, **NumPy**, dan **Regex (re)** untuk preprocessing dan normalisasi data.  
  - **Torch**, **Torch.nn**, dan **Torch.utils.data** untuk pembangunan arsitektur jaringan dan batch loader.  
  - **Transformers** (AutoTokenizer, AutoModel, Scheduler) untuk tokenisasi dan fine-tuning model IndoBERT.  
  - **Scikit-learn** untuk pembagian dataset dan evaluasi metrik (Accuracy, Precision, Recall, F1-score).  
  - **Matplotlib** dan **Seaborn** untuk visualisasi hasil.  
  - **TQDM** untuk menampilkan progres pelatihan.  
- Model berjalan secara **lokal (offline)** dan berkomunikasi dengan backend Node.js melalui **HTTP request lokal** untuk melakukan prediksi.

---

### 🔹 4. Search Engine (Eksternal)
- Sistem terhubung dengan **SearXNG**, sebuah *metasearch engine* open-source yang di-*deploy* pada **Microsoft Azure**.  
- Fitur ini membantu pengguna mencari berita dari sumber kredibel sebagai langkah tambahan dalam proses verifikasi.  
- Komunikasi antara sistem dan SearXNG dilakukan menggunakan **API HTTP** (JSON / query string).

---

## ⚙️ Deployment dan Eksekusi

Seluruh komponen sistem dijalankan secara **lokal** selama tahap pengembangan.  
### Infrastruktur yang digunakan:
- **Node.js runtime** untuk backend server.  
- **Next.js development server** untuk frontend.  
- **Python environment** (Anaconda atau venv) untuk model AI.  
- **SearXNG** berjalan secara *remote* di Azure.  

---

## 💻 Bahasa Pemrograman dan Framework

Aplikasi ini dikembangkan dengan kombinasi berbagai bahasa dan framework modern, yaitu:  

**Frontend:**  
- Dibangun menggunakan **JavaScript** dengan framework **Next.js** dan styling **Tailwind CSS**.  
- Menyediakan antarmuka yang cepat, responsif, dan ramah pengguna.  

**Backend:**  
- Dikembangkan menggunakan **Node.js** dan framework **Express.js**.  
- Berfungsi sebagai REST API server yang menjembatani komunikasi antara frontend dan model AI.  

**Model AI:**  
- Menggunakan **Python** sebagai bahasa utama dengan framework **PyTorch** dan **Hugging Face Transformers**.  
- Model berbasis **IndoBERT** digunakan untuk mendeteksi dan mengklasifikasikan berita hoax.  

**Data Processing:**  
- Library **Pandas**, **NumPy**, dan **Regex (re)** digunakan untuk preprocessing dan normalisasi teks berita.  

**Evaluasi Model:**  
- Library **Scikit-learn**, **Matplotlib**, dan **Seaborn** digunakan untuk evaluasi performa model serta visualisasi hasil.  

**Search Engine Integration:**  
- Menggunakan **SearXNG** (yang di-deploy di Microsoft Azure) sebagai metasearch engine eksternal untuk validasi silang berita.  

---

## 🧰 Tools dan Lingkungan Pengembangan

Selama pengembangan proyek, digunakan beberapa tools dan environment pendukung berikut:  

- **Visual Studio Code** → Sebagai code editor utama untuk menulis dan menguji kode.  
- **Node.js** dan **Python Runtime** → Untuk menjalankan server backend dan model AI.  
- **Git & GitHub** → Untuk version control dan kolaborasi pengembangan tim.  
- **npm** dan **pip** → Untuk mengelola dependensi package di Node.js dan Python.  
- **Anaconda / venv** → Untuk mengatur environment Python selama pelatihan model.  
- **Postman** → Untuk melakukan pengujian endpoint REST API antar komponen sistem.  
- **Microsoft Azure (SearXNG)** → Sebagai host untuk search engine eksternal.  
- **Google Chrome / Edge** → Untuk menjalankan dan menguji tampilan aplikasi web di sisi frontend.  

---

## 🧠 Model Machine Learning
- Model utama: **IndoBERT (indobenchmark/indobert-base-p1)**  
- Framework: **PyTorch**  
- Task: **Binary Classification (Hoax vs Valid)**  
- Input: Teks berita berbahasa Indonesia.  
- Output: Label prediksi (`1` = Hoax, `0` = Valid) dan *confidence score*.  

---

## 📊 Evaluasi Model
Evaluasi performa model dilakukan menggunakan metrik berikut:
- **Confusion Matrix**  
- **Accuracy, Precision, Recall, dan F1-Score**  
- **ROC Curve dan AUC**  

Hasil evaluasi menunjukkan bahwa model mampu mengklasifikasikan berita hoax dengan akurasi tinggi pada dataset berbahasa Indonesia.

---

## 💼 Kolaborasi dan Kontribusi

Proyek ini dikembangkan oleh **Kelompok 4** dalam rangka kompetisi **The Hack 2025 - NLP Track**.  

---

## 🧩 Lisensi
Proyek ini dikembangkan untuk kepentingan akademik dan kompetisi.  
Seluruh kode dan model dapat digunakan untuk pembelajaran dengan menyertakan atribusi kepada pengembang asli.

---
