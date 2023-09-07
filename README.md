# Brain Stroke Prediction - Dewi Wahidatul Karimah

## Domain Proyek
Stroke atau *Cerebrovascular disease* menurut *World  Health  Organization*  (WHO) adalah "tanda-tanda  klinis  yang  berkembang  cepat  akibat  gangguan fungsi  otak  fokal  atau  global  karena  adanya  sumbatan atau  pecahnya  pembuluh  darah  di  otak  dengan  gejala-gejala  yang  berlangsung  selama  24  jam  atau  lebih" [1]. Menurut Data *World Stroke Organization* setiap tahunnya ada 13,7 kasus baru stroke dan sekitar 5,5 juta kematian akibat penyakit stroke. Sedangkan kejadian stroke di Indonesia menurut Data Riskesdas 2013 prevalensi stroke nasional 12,1 per mil, sedangkan pada Riskesdas 2018 prevalensi stroke 10,9 per mil, tertinggi di Provinsi Kalimantan Timur (14,7 per mil), terendah di Provinsi Papua (4,1 per mil) [2].

Beberapa cara diagnosis stroke (*Gold Standart*) telah dilakukan diantaranya *Computed Tomography* ( CT ) *scan*, *Magnetic Resonance Imaging* (MRI) dan *Elektrokardiogram* (EKG atau ECG). Penerapan *gold standart* tersebut masih memiliki kekurangan yakni pasien yang tidak memungkinkan untuk berpindah tempat, mahalnya biaya, tidak semua rumah sakit memiliki peralatan tersebut, memakan waktu lebih lama dan efek radiasi. Karena kekurangan tersebut,  maka diperlukan sebuah sistem prediksi diagnosis penyakit stroke berbasis kecerdasan buatan sebagai alat pendukung keputusan untuk memprediksi penyakit stroke secara akurat dan efisien [1]. 

## Business Understanding

### Problem Statements

1. Bagaimana cara mengolah dataset yang memiliki berbeda tipe agar data dapat dilatih dengan model *neural network*.
2. Bagaimana cara menentukan parameter terbaik untuk model *neural network*.
3. Berapa tingkat akurasi, *loss*, dan *performance matrix* model *neural network* dengan parameter terbaik.

### Goals

1. Mengetahui cara mengolah dataset agar dapat memiliki tipe numerik dan dapat dilatih dengan model *neural network*.
2. Mengetahui metode untuk menentukan parameter terbaik untuk model *neural network*.
3. Mendapatkan tingkat akurasi *loss*, dan *performance matrix*, model *neural network* dengan parameter terbaik.

### Solution statements

1. Melakukan melakukan metode *one hot enoding*, *train test split*, SMOTE, dan *normalization* pada *data preparation* untuk menyiapkan data agar dapat dilatih dengan model *neural network*.
2. Melakukan *hyperparameter tuning* menggunakan *grid search* untuk menemukan parameter jumlah unit layer dan fungsi optimizer terbaik.
3. Membangun model *neural network* dengan parameter terbaik. model *neural network* dipilih karena memiliki kemampuan untuk mengekstrak pola dan mendeteksi tren yang terlalu kompleks dari sekumpulan dataset.

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
![Proyek-Predictive-Analysis-ipynb-Colaboratory (1)](https://github.com/dewi31/brain_stroke_prediction/assets/87901348/66e02cb7-d24c-4a4a-936c-848ff917a2a1)

Gambar 1. Deskripsi statistik data kontinu

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

Tabel 1. Persebaran data kategorikal

Dapat dilihat dari tabel di atas, persebaran data tidak merata terutama fitur target 'stroke' yang tidak mengalami stroke berjumlah 4733 dan yang mengalami stroke berjumlah 248.

## Data Preparation

+ One Hot Encoding

  One hot encoding adalah teknik mengubah data kategorik dengan membuat kolom baru dengan nilai 0 dan 1 untuk setiap kategori. Teknik ini dimaksudkan agar model yang digunakan tidak mengira data kategorikal merupakan jenis data diskrit biasa yang memiliki keterkaitan. Seperti contoh, pada label 'hypertension' yang harusnya kategorikal (0 jika pasien tidak hipertensi, 1 jika pasien hipertensi) menjadi diskrit (level dari penyakit 'hypertension'). Fitur yang akan diubah menjadi numerik pada proyek ini adalah gender, hypertension, heart_disease, ever_married, work_type, Residence_type, dan smoking_status.

+ SMOTE

  *Synthetic Minority Over-sampling Technique* (SMOTE) merupakan metode umum diterapkan dalam rangka menangani ketidak seimbangan kelas. Metode ini bekerja dengan membuat replikasi dari data minoritas. Penggunaan SMOTE ini dimaksud agar model dapat mengklasifikasikan dengan tepat kasus stroke pada dataset yang merupakan kasus minoritas. Dengan menggunakan metode ini jumlah label positif stroke (1) yang awalnya hanya 248 menjadi 4733 sama dengan jumlah label negatif stroke (0).
  
+ Normalization

  Normalization adalah teknik mengubah nilai-nilai dari sebuah fitur ke dalam skala yang sama. Penggunaan teknik ini dimaksudkan untuk memastikan data tetap konsisten dan membuat model machine learning bekerja lebih baik. Salah satu teknik normalisasi yang digunakan pada proyek ini adalah Standarisasi dengan sklearn.preprocessing.StandardScaler. Fitur yang akan di normalisasi pada proyek ini adalah age, avg_glucose_level, dan bmi.

+ Train Test Split

  Train test split aja proses membagi data menjadi data latih dan data uji. Penggunaan teknik ini dimaksudkan untuk untuk mengetahui performa model terhadap data baru selain data latih. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 4981 dibagi menjadi 7572 untuk data latih dan 1894 untuk data uji.

## Modeling

### *Neural Network*

*Neural Network* adalah sebuah cabang dari kecerdasan buatan (*artificial intelligence*) yang cara kerjanya meniru cara kerja saraf neuron otak manusia. *Neural network* terbagi ke dalam 3 bagian yakni input layer yang bertugas menerima input secara mentah, hidden layer bertugas memproses input, dan output layer menghasilkan nilai keluaran. Parameter yang digunakan untuk membangun model neural network adalah :
- Jumlah neuron pada *hidden layer*. Pemilihan jumlah neuron pada setiap *hidden layer* dan jumlah hidden layer pada *neural network* itu penting karena terlalu sedikit neuron atau *hidden layer* dapat menghasilkan model yang kurang akurat, sedangkan terlalu banyak dapat menyebabkan *overfitting*.
- Fungsi aktivasi : merupakan fungsi yang digunakan pada jaringan syaraf untuk mengaktifkan atau tidak mengaktifkan neuran. Fungsi aktivasi dapat memutuskan apakah masukan neuron ke jaringan penting atau tidak dalam proses prediksi menggunakan operasi matematika yang lebih sederhana. Pada proyek ini, fungsi aktivasi ReLu digunakan pada hidden layer sedangkan fungsi aktivasi *sigmoid* untuk output layer karena fungsi aktivasi tersebut bekerja sangat baik untuk kasus klasifikasi biner.
- Loss function : fungsi yang mengukur seberapa bagus performa yang dihasilkan oleh model dalam melakukan prediksi terhadap target. Loss function yang digunakan pada proyek ini adalah binary cross-entropy karena klasifikasi biner.
- Optimizer : metode yang digunakan untuk mengubah atribut neural network seperti bobot pada layer untuk mengurangi nilai loss function.
- Epoch : jumlah pelatihan model terhadap data latih.

### *Hyperparameter Tuning (Grid Search)*

*Hyperparameter tuning* adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan adalah *grid search*. *Grid search* pada proyek ini digunakan untuk menentukan jumlah neuron dan optimizer terbaik. Berikut adalah hasil dari *Grid Search* pada proyek ini :

![image](https://github.com/dewi31/brain_stroke_prediction/assets/87901348/26294a46-0d49-4cf2-bbf3-6979d66df660)

Gambar 2. Hasil metode *Grid Search*

## Evaluation

Metrik evaluasi yang digunakan pada proyek ini adalah akurasi, *loss/error*, dan *confusion matrix*. 
+ Akurasi dan loss
  Akurasi dan *loss* menggambarkan seberapa baik model neural network yang dibuat dapat memprediksi nilai target yang tepat sesuai dengan sasaran. Semakin tinggi nilai akurasi dan semakin kecil nilai loss maka semakin baik model dalam mengklasifikasikan data. Hasil akurasi dan loss dalam 10 epoch untuk data uji dan latih adalah sebagai berikut,

![image](https://github.com/dewi31/brain_stroke_prediction/assets/87901348/46dfa654-5dbb-4f34-9765-b5bfa9edb6d0)

Gambar 3. Grafik akurasi *training* dan *validation* 

![image](https://github.com/dewi31/brain_stroke_prediction/assets/87901348/daf0a600-d6f6-4eb5-8fbc-3a9768b41fe3)

Gambar 4. Grafik *loss training* dan *validation*

dari gambar grafik diatas dapat dilihat bahwa model sedikit *overfitting* meskipun data telah diubh menjadi seimbang menggunakan metode SMOTE. Nilai akhir akurasi dan *los*s untuk data latih dan uji adalah sebagai berikut,

|            | akurasi  | loss |
|------------|----------|------|
| data latih |     0.97 | 0.11 |
| data uji   |     0.96 | 0.13 |

Tabel 2. Hasil akhir akurasi dan *loss* pada data latih dan uji

+ *Confusion Matrix*
  Confusion Matrix digunakan untuk menghitung berbagai *performance metrics* untuk mengukur kinerja model yang telah dibuat. Pada penelitian ini performance metrics yang digunakan adalah presisi, *recall*, dan *F1 score*. Presisi merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positf. *Recall* merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif. *F1 score* merupakan perbandingan rata-rata presisi dan *recall* yang dibobotkan. Persamaan *performance metrics* tersebut dapat ditulis sebgai berikut,

$$ presisi = {TP \over (TP + FP)} $$

$$ recall = {TP \over (TP + FN)} $$

$$ f1 score = {2 * (Recall*Precission) \over (Recall + Precission)} $$

Hasil *confusion matrix* pada data uji dengan menggunakan parameter terbaik adalah sebagai berikut,

![image](https://github.com/dewi31/brain_stroke_prediction/assets/87901348/20471811-722f-41dd-b92e-cbe65a8f17b8)

Gambar 5. Hasil  *Confusion Matrix* 

Dari gambar 5. dapat dilihat bahwa data yang diprediksi benar negatif stroke sebesar 929, data yang diprediksi benar positif stroke sebesar 887, data yang seharusnya negatif stroke tetapi diprediksi positif stroke sebesar 12, dan data yng seharusnya positif stroke tetapi diprediksi negatif stroke sebesar 66. Setelah didapatkan *confusion matrix* dihitung *performance metrics* dan dihasilkan nilai sebagai berikut,

|  Presisi   |  recall  | f1 score |
|------------|----------|----------|
|       0.99 |     0.93 |     0.96 |

Tabel 3. Hasil *performance metrics* 

## REFERENCES

[1] Arifianto, A. S., Sarosa, M., & Setyawati, O. (2014). Klasifikasi stroke berdasarkan kelainan patologis dengan learning vector quantization. Jurnal EECCIS (Electrics, Electronics, Communications, Controls, Informatics, Systems), 8(2), 117-122.

[2] Byna, A., & Basit, M. (2020). Penerapan Metode Adaboost Untuk Mengoptimasi Prediksi Penyakit Stroke Dengan Algoritma Naïve Bayes. Jurnal Sisfokom (Sistem Informasi Dan Komputer), 9(3), 407-411.
