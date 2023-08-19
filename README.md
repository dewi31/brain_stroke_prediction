# Laporan Proyek Machine Learning - Dewi Wahidatul Karimah

## Domain Proyek
Stroke atau Cerebrovascular disease menurut World  Health  Organization  (WHO) adalah "tanda-tanda  klinis  yang  berkembang  cepat  akibat  gangguan fungsi  otak  fokal  atau  global  karena  adanya  sumbatan atau  pecahnya  pembuluh  darah  di  otak  dengan  gejala-gejala  yang  berlangsung  selama  24  jam  atau  lebih". 

Beberapa cara diagnosis stroke (Gold Standart) diantaranya menggunakan Computed Tomography ( CT ) scan, Magnetic Resonance Imaging (MRI) dan Elektrokardiogram (EKG atau ECG). Penerapan gold standart tersebut masih memiliki kekurangan yakni pasien yang tidak memungkinkan untuk berpindah tempat, mahalnya biaya, tidak semua rumah sakit memiliki peralatan tersebut, memakan waktu lebih lama dan efek radiasi. Karena kekurangan tersebut,  maka diperlukan sebuah sistem prediksi diagnosis penyakit stroke berbasis kecerdasan buatan sebagai alat pendukung keputusan untuk memprediksi penyakit stroke secara akurat dan efisien. 

Referensi: [Klasifikasi Stroke Berdasarkan Kelainan Patologis dengan Learning Vector Quantization ](https://media.neliti.com/media/publications/69296-ID-klasifikasi-stroke-berdasarkan-kelainan.pdf) 

## Business Understanding

### Problem Statements

1. Bagaimana cara mengolah data agar data dapat dilatih dengan model neural network.
2. Bagaimana cara menentukan parameter terbaik untuk model neural network.
3. Berapa tingkat akurasi model neural network dengan parameter terbaik pada data latih dan data uji.

### Goals

1. Mengetahui cara mengolah data agar data dapat dilatih dengan model neural network.
2. Mengetahui metode untuk menentukan parameter terbaik untuk model neural network.
3. Mendapatkan tingkat akurasi model neural network dengan parameter terbaik pada data latih dan data uji.

### Solution statements

1. Melakukan melakukan metode one hot enoding, train test split, dan normalization pada data preparation untuk menyiapkan data agar dapat dilatih dengan model neural network.
2. Melakukan hyperparameter tuning menggunakan grid search untuk menemukan parameter jumlah unit layer dan fungsi optimizer terbaik.
3. membangun model neural network dengan parameter terbaik.

## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan data pasien stroke dan yang tidak. Dataset ini dapat diunduh di [Kaggle : Brain stroke prediction dataset](https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset).

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Dataset memiliki 4981 sample dengan 11 fitur.
+ Dataset memiliki 3 fitur bertipe int64, 3 fitur bertipe float64, dan 5 fitur bertipe object.
+ Tidak ada nilai yang hilang dalam dataset.

### Variabel-variabel pada dataset:

- gender: jenis kelamin dari pasien
- age: umur pasien
- hypertension: 0 jika pasien tidak hipertensi, 1 jika pasien hipertensi
- heart_disease:0 jika pasien tidak memiliki penyakit jantung, 1 jika pasien memiliki penyakit jantung
- ever_married: status pernikahan pasien
- work_type: jenis pekerjaan pasien
- Residence_type: tipe tempat tinggal
- avg_glucose_level: kadar glukosa rata-rata dalam darah
- bmi: indeks massa tubuh
- smoking_status: status merokok pasien
- stroke: 1 jika pasien mengalami stroke atau 0 jika tidak

### Univariate Analysis

Univariate Analysis digunakan untuk menganalisis tiap variabel dalam suatu data

#### Analisis data kontinu dari dataset

Kolom/fitur yang termasuk data kontinu dari dataset adalah kolum 'age', 'avg_glucose_level', dan 'bmi'
![image](https://github.com/dewi31/brain_stroke_prediction/assets/87901348/82851c08-f62d-4ab8-8d73-4ccc9f0748dd)

Dari gambar tersebut dapat dilihat bahwa umur terkecil 0.08 tahun dan terbesar 82 tahun, kadar glokosa rata-rata terkecil 55.12 dan terbesar 271.74, indeks massa tubuh terkecil 14 dan terbesar 48.9.

#### Analisis data kategorikal dari dataset

Jumlah persebaran dari data kategorikal dapat dilihat pada tabel berikut,

| Nama Kolom     | Tipe Data | Jumlah masing-masing kategori                                             |
|----------------|-----------|---------------------------------------------------------------------------|
| gender         | object    |   Female : 2907, Male : 2074                                              |
| hypertension   | int64     |   0 : 4502, 1 : 479                                                       |
| heart_disease  | int64     |   0 : 4706, 1 : 275                                                       |
| ever_married   | object    |   No : 1701, Yes : 3280                                                   |
| work_type      | object    |   Govt_job : 644, Private : 2860, Self-employed : 804, children : 673     |
| Residence_type | object    |   Rural : 2449, Urban : 2532                                              |
| smoking_status | object    |   Unknown : 1500, formerly smoked : 867, never smoked : 1838, smokes :776 |
| stroke         | int64     |   0 : 4733, 1 : 248                                                       |

Dapat dilihat dari tabel di atas, persebaran data tidak merata terutama fitur target 'stroke' yang tidak mengalami stroke berjumlah 4733 dan yang mengalami stroke berjumlah 248.

## Data Preparation

+ One Hot Encoding

  One hot encoding adalah teknik mengubah data kategorik dengan membuat kolom baru dengan nilai 0 dan 1 untuk setiap kategori. Fitur yang akan diubah menjadi numerik pada proyek ini adalah gender, hypertension, heart_disease, ever_married, work_type, Residence_type, dan smoking_status.
  
+ Train Test Split

  Train test split aja proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 4981 dibagi menjadi 3984 untuk data latih dan 997 untuk data uji.
  
+ Normalization

  Normalization adalah teknik mengubah nilai-nilai dari sebuah fitur ke dalam skala yang sama. Salah satu teknik normalisasi yang digunakan pada proyek ini adalah Standarisasi dengan sklearn.preprocessing.StandardScaler. Fitur yang akan di normalisasi pada proyek ini adalah age, avg_glucose_level, dan bmi.

## Modeling

### Neural Network

Neural Network adalah sebuah cabang dari kecerdasan buatan (artificial intelligence) yang cara kerjanya meniru cara kerja saraf neuron otak manusia. Neural network terbagi ke dalam 3 bagian yakni input layer yang bertugas menerima input secara mentah, hidden layer bertugas memproses input, dan output layer menghasilkan nilai keluaran. Parameter yang digunakan untuk membangun model neural network adalah :
- Jumlah neuron pada hidden layer
- Fungsi aktivasi : merupakan fungsi yang digunakan pada jaringan syaraf untuk mengaktifkan atau tidak mengaktifkan neuron. Pada proyek ini, fungsi aktivasi ReLu digunakan pada hidden layer sedangkan fungsi aktivasi sigmoid untuk output layer.
- Loss function : fungsi yang mengukur seberapa bagus performa yang dihasilkan oleh model dalam melakukan prediksi terhadap target. Loss function yang digunakan pada proyek ini adalah binary cross-entropy karena klasifikasi biner.
- Optimizer : metode yang digunakan untuk mengubah atribut neural network seperti bobot pada layer untuk mengurangi nilai loss function.
- Epoch : jumlah pelatihan model terhadap data latih.

### Hyperparameter Tuning (Grid Search)

Hyperparameter tuning adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan adalah grid search. Grid search pada proyek ini digunakan untuk menentukan jumlah neuron dan optimizer terbaik. Berikut adalah hasil dari Grid Search pada proyek ini :

![image](https://github.com/dewi31/brain_stroke_prediction/assets/87901348/c750b359-f982-470d-9a0f-0a144b39fba7)

## Evaluation

Metrik evaluasi yang digunakan pada proyek ini adalah akurasi dan loss/error. Akurasi dan loss menggambarkan seberapa baik model neural network yang dibuat dapat memprediksi nilai target yang tepat sesuai dengan sasaran. Semakin tinggi nilai akurasi dan semakin kecil nilai loss maka semakin baik model dalam mengklasifikasikan data. Hasil akurasi dan loss dalam 10 epoch untuk data uji dan latih adalah sebagai berikut,
![download](https://github.com/dewi31/brain_stroke_prediction/assets/87901348/a97ece0b-6ec2-44f4-980b-9fdb69a4bf04)

![download (1)](https://github.com/dewi31/brain_stroke_prediction/assets/87901348/22940642-698d-44da-8b2b-86c8daadf3de)

dari gambar grafik diatas dapat dilihat bahwa model tidak overfitting meskipun data tidak seimbang. Nilai akhir akurasi dan loss untuk data latih dan uji adalah sebagai berikut,
|            | akurasi  | loss |
|------------|----------|------|
| data latih |     0.95 | 0.17 |
| data uji   |     0.96 | 0.15 |
