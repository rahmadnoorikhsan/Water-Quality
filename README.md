# LAPORAN PROYEK MACHINE LEARNING - RAHMAD NOOR IKHSAN

## Domain Proyek
**Domain:** Kesehatan dan Lingkungan

**Judul:** Prediksi Kelayakan Air Minum Berdasarkan Parameter Kimia dan Fisika

### Latar Belakang

Akses terhadap air minum yang bersih dan layak merupakan salah satu kebutuhan dasar manusia yang mempengaruhi kesehatan secara langsung. Menurut WHO (World Health Organization), kontaminasi air minum dapat menyebabkan berbagai penyakit seperti diare, kolera, dan gangguan sistem pencernaan lainnya. Sayangnya, tidak semua wilayah memiliki fasilitas pengujian laboratorium yang memadai untuk menguji kualitas air secara langsung. Oleh karena itu, penerapan teknologi machine learning untuk memprediksi kelayakan air berdasarkan parameter kimia dan fisika air menjadi solusi potensial yang efisien dan praktis.

Berdasarkan studi yang dilakukan oleh Misra et al. (2021) dalam jurnal Water Quality Assessment Using Machine Learning, pendekatan prediktif berbasis data telah menunjukkan akurasi tinggi dalam menentukan status potabilitas air dengan hanya menggunakan parameter seperti pH, turbiditas, total dissolved solids, dan lainnya. Ini menjadi dasar kuat untuk eksplorasi model machine learning dalam proyek ini.

## Business Understanding

### Problem Statement
- Bagaimana membangun model machine learning yang mampu memprediksi apakah air layak minum atau tidak berdasarkan parameter kimia dan fisika seperti pH, hardness, solids, sulfate, dan lainnya?

- Algoritma machine learning apa yang memberikan performa terbaik untuk klasifikasi potabilitas air?

### Goals
- Mengembangkan model klasifikasi untuk memprediksi kelayakan air minum menggunakan dataset kualitas air.

- Membandingkan performa beberapa algoritma machine learning seperti Decision Tree, Random Forest, dan XGBoost.

- Menentukan model terbaik yang dapat digunakan untuk membantu proses evaluasi kualitas air secara otomatis.

### Solution Statements

- Melakukan eksplorasi dan pembersihan data termasuk penanganan missing value, outlier, dan ketidakseimbangan kelas.

- Melakukan proses normalisasi dan pembagian data latih-uji.

- Membangun beberapa model klasifikasi dan melakukan evaluasi berdasarkan metrik seperti accuracy, precision, recall, dan F1-score.

## Data Understanding

### Informasi Datasets
| Jenis | Keterangan |
| ------ | ------ |
| Title | Water Quality |
| Source | [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability/data) |
| Maintainer | [Aditya Kadiwal](https://www.kaggle.com/adityakadiwal)  |
| License | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| Visibility | Public |
| Tags | Earth and Nature, Beginner, Energy, Public Health, Environment, Binary Classification, Water Bodies |
| Usability | 10.00 |

### EDA - Deskripsi Variabel
Bagian ini menyajikan gambaran awal dari beberapa data dalam dataset kualitas air minum. Dataset ini berformat CSV (Comma-Separated Values) dan terdiri dari 3.276 data dengan 10 fitur yang merepresentasikan berbagai parameter kimia dan fisika air, serta satu label target (Potability) yang menunjukkan kelayakan air untuk dikonsumsi.

**Tabel Sampel Data**

| ph |	Hardness | Solids | Chloramines | Sulfate | Conductivity | Organic_carbon | Trihalomethanes | Turbidity | Potability |
| -------- | ---------- | ------------ | -------- | ---------- | ---------- | --------- | --------- | -------- | - |  
| 3.716080 | 129.422921 | 18630.057858 | 6.635246 | 359.948574 | 592.885359 | 15.180013 | 56.329076 | 4.500656 | 0 |
| 8.099124 | 224.236259 | 19909.541732 | 9.275884 | 356.886136 | 418.606213 | 16.868637 | 66.420093 | 3.055934 | 0 |
| 8.316766 | 214.373394 | 22018.417441 | 8.059332 | 356.886136 | 363.266516 | 18.436524 | 100.34167 | 4.628771 | 0 |
| 9.092223 | 181.101509 | 17978.986339 | 6.546600 | 310.135738 | 398.410813 | 11.558279 | 31.997993 | 4.075075 | 0 |

**Penjelasan Variabel**

- pH: Menunjukkan tingkat keasaman atau kebasaan air (skala 0–14). Nilai ideal air minum berkisar antara 6.5 hingga 8.5.

- Hardness (Kekerasan): Kemampuan air dalam mengendapkan sabun, mengindikasikan kandungan kalsium dan magnesium (mg/L).

- Solids: Total padatan terlarut dalam air (ppm). Jumlah padatan terlarut yang terlalu tinggi dapat memengaruhi rasa dan kejernihan air.

- Chloramines: Kandungan kloramin dalam air (ppm), senyawa hasil desinfeksi yang digunakan untuk membunuh mikroorganisme.

- Sulfate: Jumlah sulfat terlarut dalam air (mg/L). Dapat menimbulkan rasa pahit dan efek samping jika berlebihan.

- Conductivity: Konduktivitas listrik air dalam μS/cm, menggambarkan seberapa banyak ion terlarut di dalam air.

- Organic Carbon: Kandungan karbon organik dalam air (ppm), yang bisa menjadi indikator pencemaran dari bahan organik alami atau buatan.

- Trihalomethanes: Senyawa kimia hasil samping dari proses klorinasi air (μg/L), beberapa di antaranya berpotensi karsinogenik.

- Turbidity (Kekeruhan): Ukuran kejernihan air dalam NTU (Nephelometric Turbidity Units). Kekeruhan tinggi dapat mengindikasikan keberadaan partikel tersuspensi yang berbahaya.

- Potability: Label target yang menunjukkan apakah air layak dikonsumsi manusia:

  - 1: Layak minum

  - 0: Tidak layak minum

### EDA - Menangani Missing Values dan Outliers

#### Pemeriksaan Missing Values

Langkah awal dalam eksplorasi data adalah mengidentifikasi nilai yang hilang (missing values) pada setiap fitur. Berdasarkan hasil pengecekan, ditemukan bahwa beberapa kolom memiliki data yang tidak lengkap (bernilai null), seperti ditunjukkan pada tabel berikut:

| Kolom | Missing Values |
| ----- | -------------- | 
| ph | 491 | 
| Hardness | 0 | 
| Solids | 0 |
| Chloramines | 0 |
| Sulfate | 781 | 
| Conductivity | 0 |
| Organic_carbon | 0 |
| Trihalomethanes | 162 |
| Turbidity | 0 |
| Potability | 0 |

Dapat diperhatikan dari tabel di atas, terdapat 3 kolom yang memiliki Missing Values, yaitu:

- 491 nilai kosong pada kolom `pH`

- 781 nilai kosong pada kolom `Sulfate`

- 162 nilai kosong pada kolom `Trihalomethanes`

#### Strategi Penanganan Missing Values

Untuk menentukan metode imputasi yang sesuai, kita perlu memahami bentuk distribusi dari fitur-fitur tersebut. Berikut visualisasi distribusi pada fitur yang memiliki missing values.

![Distribusi kolom yang memiliki missing values](https://raw.githubusercontent.com/rahmadnoorikhsan/Water-Quality/main/resource/missing_values_columns.png)

Berdasarkan hasil visualisasi distribusi, dapat kita tentukan strategi yang sesuai untuk menangani missing values seperti berikut:

- Kolom `pH` dan `Sulfate` menunjukkan pola distribusi yang cenderung **simetris**, sehingga pengisian missing values menggunakan **nilai rata-rata (Mean)** merupakan pendekatan yang sesuai.

- Kolom `Trihalomethanes` tampak memiliki distribusi yang sedikit **skewed (miring)**, sehingga lebih tepat menggunakan **Median** untuk mengisi nilai yang hilang karena median lebih tahan terhadap outlier.

#### Pemeriksaan Outliers
Outlier adalah nilai-nilai ekstrim yang secara signifikan berbeda dari mayoritas data lainnya. Kehadiran outlier dapat memengaruhi distribusi data dan kinerja model prediktif. Untuk itu, dilakukan analisis visual menggunakan boxplot pada setiap fitur numerik.

Gambar berikut menunjukkan boxplot dari masing-masing fitur:
![Boxplot Outliers](https://raw.githubusercontent.com/rahmadnoorikhsan/Water-Quality/main/resource/boxplot_outliers.png)

Dari hasil visualisasi boxplot di atas, dapat disimpulkan bahwa seluruh fitur numerik memiliki outliers, yang terlihat sebagai titik-titik di luar whisker (garis vertikal batas atas dan bawah).

Beberapa catatan penting:

- Solids, Conductivity, Trihalomethanes, dan Sulfate menunjukkan jumlah outlier yang cukup signifikan, terutama di sisi nilai tinggi.

- Organic_carbon dan Chloramines memiliki outliers di kedua sisi distribusi (nilai rendah dan tinggi).

- pH, Turbidity, dan Hardness juga memiliki beberapa outlier, namun distribusinya masih relatif simetris dan terkontrol.

#### Strategi Penanganan Outliers

Setelah dilakukan pemeriksaan visual melalui boxplot, ditemukan bahwa seluruh fitur numerik dalam dataset memiliki outliers. Kehadiran outlier dapat memengaruhi analisis data dan mengganggu performa model klasifikasi, karena model bisa terjebak pada pola yang tidak representatif dari mayoritas data.

Untuk menangani permasalahan ini, digunakan pendekatan berbasis K-Means Clustering sebagai metode deteksi dan penanganan outlier.

#### Alasan Pemilihan K-Means Clustering
K-Means Clustering dipilih karena beberapa keunggulan yang dimilikinya dalam mendeteksi outliers:

- Multivariat: mampu menganalisis hubungan antar fitur sekaligus, bukan hanya berdasarkan satu kolom saja.

- Bebas asumsi distribusi: tidak mengharuskan data mengikuti distribusi tertentu seperti normal.

- Efisien dan skalabel: dapat digunakan pada dataset berukuran besar dan kompleks.

- Adaptif terhadap struktur data: dapat mengenali kelompok mayoritas yang merepresentasikan pola umum data.

### EDA - Univariate Analysis

#### Fitur Kategorikal
Pada tahap ini, dilakukan analisis univariat terhadap fitur kategorikal dalam dataset, yaitu kolom Potability. Kolom ini bersifat biner dan merepresentasikan apakah sampel air layak untuk dikonsumsi atau tidak. Nilai kategorikalnya terbagi sebagai berikut:

- 0 → Air tidak layak untuk diminum

- 1 → Air layak untuk diminum

![Visualisasi Univariate Categorical](https://raw.githubusercontent.com/rahmadnoorikhsan/Water-Quality/main/resource/univariate_categorical.png)

Hasil visualisasi diatas menunjukkan adanya ketidakseimbangan kelas (imbalanced class) antara dua kategori tersebut.

 - Kategori 0 (tidak layak minum) memiliki frekuensi yang lebih dominan

 - Kategori 1 (layak minum) berjumlah lebih sedikit

Distribusi ini menunjukkan bahwa mayoritas sampel air dalam dataset termasuk kategori tidak layak minum, yang dapat memengaruhi performa model klasifikasi apabila tidak ditangani dengan tepat.

#### Fitur Numerikal
Analisis univariat terhadap fitur numerikal bertujuan untuk memahami karakteristik distribusi data dari masing-masing variabel numerik yang tersedia dalam dataset. Visualisasi yang digunakan adalah histogram, yang memberikan gambaran frekuensi kemunculan nilai-nilai dalam rentang tertentu.

![Visualisasi Univariate Numerical](https://raw.githubusercontent.com/rahmadnoorikhsan/Water-Quality/main/resource/univariate_numerical.png)

Berdasarkan visualisasi distribusi univariat fitur numerikal, dapat disimpulkan:

- pH: Distribusi cenderung simetris dengan puncak di sekitar nilai 7 hingga 8.

- Hardness: Distribusi membentuk pola normal (bell-shaped) dengan konsentrasi nilai pada kisaran 150 hingga 250.

- Solids: Distribusi right-skewed atau miring ke kanan. Sebagian besar nilai terkonsentrasi pada kisaran 25.000 hingga 35.000.

- Chloramines: Distribusi terlihat mendekati normal, dengan puncak pada kisaran 6 hingga 8.

- Sulfate: Distribusi relatif simetris, meskipun terdapat sedikit kecondongan ke arah kanan.

- Conductivity: Distribusi berbentuk simetris, mirip dengan distribusi normal, dengan puncak pada kisaran 400–500.

- Organic_carbon: Distribusi sedikit right-skewed, dengan mayoritas nilai berada di kisaran 10 hingga 20.

- Trihalomethanes: Distribusi juga menunjukkan right-skewed dengan konsentrasi nilai terbanyak berada di bawah 80.

- Turbidity: Distribusi relatif simetris dengan mayoritas nilai berkisar antara 2 hingga 5.

### EDA - Multivariate Analysis
Analisis multivariat dalam tahap eksplorasi data (EDA) ini bertujuan untuk memahami bagaimana berbagai fitur numerik berinteraksi dan berhubungan dengan variabel target, yaitu Potability.

![Visualisasi Multivariate Barplot](https://raw.githubusercontent.com/rahmadnoorikhsan/Water-Quality/main/resource/multivariate_barplot.png)
![Visualisasi Multivariate Barplot](https://raw.githubusercontent.com/rahmadnoorikhsan/Water-Quality/main/resource/multivariate_kdeplot.png)

Berdasarkan dari visualisasi Rata-Rata dan Distribusi Fitur Numerikal terhadap terhadap kolom Potability, dapat disimpulkan sebagai berikut:
- pH

  - Rata-rata terhadap Potability:
    - Air tidak layak minum (0): Rata-rata pH sekitar 6.85.
    - Air layak minum (1): Rata-rata pH sekitar 7.18. Secara rata-rata, air yang layak minum cenderung memiliki pH sedikit lebih tinggi.
  - Distribusi berdasarkan Potability: Distribusi pH untuk kedua kelas (layak dan tidak layak minum) menunjukkan tumpang tindih yang signifikan. Namun, puncak distribusi untuk air layak minum (Potability 1) tampak sedikit bergeser ke kanan (nilai pH lebih tinggi) dibandingkan air tidak layak minum (Potability 0).

- Hardness

  - Rata-rata terhadap Potability:
    - Air tidak layak minum (0): Rata-rata Hardness sekitar 196.47.
    - Air layak minum (1): Rata-rata Hardness sekitar 193.96. Rata-rata kekerasan air tidak layak minum sedikit lebih tinggi, meskipun perbedaannya kecil.
  - Distribusi berdasarkan Potability: Distribusi kekerasan untuk kedua kelas sangat tumpang tindih. Distribusi untuk air tidak layak minum (Potability 0) sedikit bergeser ke kanan (nilai kekerasan lebih tinggi) dibandingkan air layak minum.

- Solids

  - Rata-rata terhadap Potability:
    - Air tidak layak minum (0): Rata-rata Solids sekitar 30639.58.
    - Air layak minum (1): Rata-rata Solids sekitar 31537.32. Air yang layak minum cenderung memiliki rata-rata total padatan terlarut yang sedikit lebih tinggi.
  - Distribusi berdasarkan Potability: Distribusi Solids menunjukkan tumpang tindih yang besar antara kedua kelas, dengan kedua distribusi cenderung miring ke kanan (right-skewed). Distribusi untuk air layak minum (Potability 1) tampaknya memiliki konsentrasi yang sedikit lebih tinggi pada nilai solids yang lebih tinggi.

- Chloramines

  - Rata-rata terhadap Potability:
    - Air tidak layak minum (0): Rata-rata Chloramines sekitar 7.05.
    - Air layak minum (1): Rata-rata Chloramines sekitar 7.00. Perbedaan rata-rata Chloramines antara kedua kelas sangat kecil, dengan air tidak layak minum memiliki rata-rata sedikit lebih tinggi.
  - Distribusi berdasarkan Potability: Distribusi Chloramines untuk air layak dan tidak layak minum hampir identik dan sangat tumpang tindih, menunjukkan bahwa fitur ini mungkin memiliki daya pembeda yang rendah secara individual.

- Sulfate

  - Rata-rata terhadap Potability:
    - Air tidak layak minum (0): Rata-rata Sulfate sekitar 334.61.
    - Air layak minum (1): Rata-rata Sulfate sekitar 320.38. Air tidak layak minum cenderung memiliki rata-rata kadar sulfat yang lebih tinggi.
  - Distribusi berdasarkan Potability: Distribusi sulfat untuk kedua kelas menunjukkan tumpang tindih yang cukup besar. Puncak distribusi untuk air tidak layak minum (Potability 0) sedikit bergeser ke kanan (nilai sulfat lebih tinggi) dibandingkan air layak minum.

- Conductivity

  - Rata-rata terhadap Potability:
    - Air tidak layak minum (0): Rata-rata Conductivity sekitar 429.19.
    - Air layak minum (1): Rata-rata Conductivity sekitar 431.02. Rata-rata konduktivitas sedikit lebih tinggi pada air yang layak minum, namun perbedaannya sangat tipis.
  - Distribusi berdasarkan Potability: Distribusi konduktivitas untuk kedua kelas sangat mirip dan tumpang tindih secara signifikan. Ada sedikit indikasi bahwa puncak distribusi air layak minum (Potability 1) berada pada nilai konduktivitas yang sedikit lebih tinggi.

- Organic Carbon (Karbon Organik)

  - Rata-rata terhadap Potability:
      - Air tidak layak minum (0): Rata-rata Organic Carbon sekitar 14.46.
      - Air layak minum (1): Rata-rata Organic Carbon sekitar 14.21. Rata-rata karbon organik sedikit lebih tinggi pada air yang tidak layak minum.
  - Distribusi berdasarkan Potability:
      Distribusi karbon organik untuk kedua kelas menunjukkan tumpang tindih yang besar. Puncak distribusi untuk air tidak layak minum (Potability 0) sedikit bergeser ke kanan (nilai karbon organik lebih tinggi).

- Trihalomethanes

  - Rata-rata terhadap Potability:
    - Air tidak layak minum (0): Rata-rata Trihalomethanes sekitar 65.73.
    - Air layak minum (1): Rata-rata Trihalomethanes sekitar 66.57. Air layak minum memiliki rata-rata Trihalomethanes yang sedikit lebih tinggi.
  - Distribusi berdasarkan Potability:
    Distribusi Trihalomethanes untuk kedua kelas sangat tumpang tindih. Puncak distribusi untuk air layak minum (Potability 1) sedikit bergeser ke kanan (nilai Trihalomethanes lebih tinggi).

- Turbidity

  - Rata-rata terhadap Potability:
    - Air tidak layak minum (0): Rata-rata Turbidity sekitar 3.98.
    - Air layak minum (1): Rata-rata Turbidity sekitar 3.98. Rata-rata kekeruhan hampir identik untuk kedua kelas.
  - Distribusi berdasarkan Potability: Distribusi kekeruhan untuk air layak dan tidak layak minum hampir sepenuhnya tumpang tindih, menunjukkan bahwa kekeruhan saja mungkin bukan merupakan prediktor yang kuat untuk kelayakan air minum.

## Data Preparation

Data Preparation adalah proses menyiapkan data agar siap digunakan oleh algoritma machine learning. Ini mencakup pembersihan data, imputasi nilai yang hilang, transformasi fitur, normalisasi, encoding data kategorikal, dan pembagian data menjadi set pelatihan dan pengujian. Tahap ini penting untuk meningkatkan kualitas dan performa model yang akan dibangun.

Pada proyek ini, tahapan data preparation yang dilakukan difokuskan pada dua aspek utama:

- Penanganan Missing Values
- Penanganan Outliers
- Standardisasi.
- Pembagian dataset menjadi data latih dan data uji.

### Penanganan Missing Values
Pada tahap Exploratory Data Analysis (EDA) telah mengidentifikasi adanya nilai yang hilang (missing values) pada beberapa fitur dalam dataset. Keberadaan nilai yang hilang ini, jika tidak ditangani, dapat mengganggu proses analisis dan mengurangi akurasi model machine learning.

#### Ringkasan Temuan EDA dan Strategi Imputasi
Berdasarkan hasil pemeriksaan missing values pada tahap EDA, ditemukan bahwa tiga kolom memiliki data yang tidak lengkap:

- Kolom pH memiliki 491 nilai kosong.
- Kolom Sulfate memiliki 781 nilai kosong.
- Kolom Trihalomethanes memiliki 162 nilai kosong.

Analisis distribusi pada tahap EDA menunjukkan:

- Kolom pH dan Sulfate memiliki pola distribusi yang cenderung simetris. Berdasarkan temuan ini, strategi yang dipilih pada EDA untuk mengisi missing values pada kedua kolom ini adalah menggunakan nilai rata-rata (Mean).
- Kolom Trihalomethanes menunjukkan distribusi yang sedikit skewed (miring). Oleh karena itu, strategi yang ditetapkan pada EDA adalah menggunakan Median untuk mengisi nilai yang hilang, karena median lebih robust terhadap outlier.

#### Implementasi Penanganan Missing Values
Berdasarkan strategi yang telah ditentukan pada tahap EDA, langkah-langkah implementasi imputasi data yang hilang pada tahap Data Preparation adalah sebagai berikut:

- Menghitung nilai mean untuk kolom `ph` dan `Sulfate` dari data yang tersedia.
- Menghitung nilai median untuk kolom `Trihalomethanes` dari data yang tersedia.
- Mengisi nilai yang hilang pada masing-masing kolom menggunakan nilai statistik yang relevan yang telah dihitung.

```python
# Menangani missing values menggunakan mean
df['ph'] = df['ph'].fillna(df['ph'].mean())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())

# Menangani missing values menggunakan median
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())
```

### Penanganan Outliers
Outlier, atau nilai-nilai ekstrim, dapat secara signifikan memengaruhi distribusi data dan kinerja model prediktif. Identifikasi dan penanganan outlier merupakan langkah penting dalam persiapan data.

#### Ringkasan Temuan EDA dan Strategi Penanganan Outliers
Analisis visual menggunakan boxplot pada setiap fitur numerik selama tahap EDA menyimpulkan bahwa seluruh fitur numerik memiliki outliers. Kehadiran outlier ini berpotensi memengaruhi analisis data dan mengganggu performa model klasifikasi.

Untuk menangani permasalahan ini, pada tahap EDA telah diputuskan untuk menggunakan pendekatan berbasis K-Means Clustering sebagai metode deteksi dan penanganan outlier. Alasan pemilihan K-Means Clustering pada EDA meliputi kemampuannya dalam analisis multivariat, tidak adanya asumsi distribusi tertentu, efisiensi, skalabilitas, dan adaptabilitas terhadap struktur data.

#### Implementasi Penanganan Outliers
Mengikuti strategi yang dirumuskan pada tahap EDA, K-Means Clustering diterapkan pada tahap Data Preparation untuk mengidentifikasi dan menangani outliers. Proses ini melibatkan pengelompokan data, di mana data poin yang berada sangat jauh dari centroid cluster mana pun dianggap sebagai outlier dan kemudian dihapus dari dataset.

```python
from sklearn.cluster import KMeans

# KMeansClustering - menghapus outlier
kmeans = KMeans(n_clusters = 2, random_state=42)
kmeans.fit(df)
labels = kmeans.labels_
df = df[labels == labels.max()]
df.reset_index(inplace = True)
df.shape
```

Setelah menerapkan K-Means Clustering untuk menangani outliers, jumlah dataset yang siap untuk tahap selanjutnya berkurang menjadi 1248 baris data.

### Standarisasi
Standardisasi merupakan salah satu teknik transformasi fitur yang umum digunakan dalam persiapan pemodelan machine learning. Tujuan utama dari standardisasi adalah untuk mengubah skala fitur-fitur numerik sehingga memiliki properti statistik yang seragam, yaitu rata-rata (mean) mendekati 0 dan standar deviasi mendekati 1.

Dalam proyek ini, teknik standardisasi yang digunakan adalah StandardScaler dari pustaka Scikit-learn. StandardScaler bekerja dengan cara mengurangkan nilai rata-rata (mean) dari setiap fitur dan kemudian membaginya dengan standar deviasi fitur tersebut. Secara matematis, untuk setiap nilai fitur x, nilai yang distandardisasi z dihitung sebagai:

$$z = \frac{(x - \mu)}{\sigma}$$

dimana μ adalah rata-rata (mean) dari fitur dan σ adalah standar deviasi dari fitur tersebut. Hasilnya adalah distribusi fitur dengan standar deviasi sama dengan 1 dan mean sama dengan 0, di mana sekitar 68% dari nilai akan berada dalam rentang -1 hingga 1.

Langkah-langkah implementasi standardisasi pada proyek ini adalah sebagai berikut:

- Pemisahan Fitur dan Target: Dataset dipisahkan menjadi matriks fitur (X), yang berisi semua variabel independen, dan vektor target (y), yang berisi variabel dependen (Potability).
- StandardScaler: Objek StandardScaler diinisialisasi dan kemudian metode fit_transform() diterapkan pada matriks fitur (X). Metode fit() menghitung mean dan standar deviasi dari data pelatihan, sementara metode transform() menerapkan transformasi standardisasi menggunakan parameter yang telah dihitung.

```python
from sklearn.preprocessing import StandardScaler

# Memisahkan fitur dan target
X = df.drop(['Potability'], axis=1)
y = df['Potability']

# Inisialisasi dan aplikasi Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Pembagian dataset menjadi data latih dan data uji
Setelah fitur-fitur distandardisasi, langkah selanjutnya adalah membagi dataset menjadi dua subset terpisah: data latih (training set) dan data uji (testing set). Data latih digunakan untuk melatih model agar dapat mengenali pola dalam data. Sementara itu, data uji, yang tidak pernah dilihat oleh model selama proses pelatihan, digunakan untuk menguji seberapa baik model dapat membuat prediksi pada data baru yang belum pernah ditemui sebelumnya.

Pada proyek ini, pembagian dataset dilakukan menggunakan fungsi train_test_split dari pustaka Scikit-learn. Dataset akan dibagi dengan rasio 70% untuk data latih dan 30% untuk data uji.

```python
from sklearn.model_selection import train_test_split

# Pembagian dataset menjadi data latih dan data uji dengan rasio 70%-30%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
```

## Model Development
Tahap Model Development merupakan inti dari proses machine learning di mana algoritma pembelajaran mesin diterapkan untuk menganalisis data dan menjawab problem statement yang telah dirumuskan pada tahap Business Understanding. Proses ini melibatkan pemilihan algoritma yang sesuai dan pelatihan model menggunakan data yang telah melalui tahap Data Preparation. Tujuan utama dari tahap ini adalah untuk membangun dan mengidentifikasi model prediktif yang mampu memberikan performa terbaik berdasarkan metrik evaluasi yang relevan dengan kasus penggunaan.

Pada proyek ini, akan dikembangkan beberapa model klasifikasi untuk memprediksi kelayakan air minum (Potability). Tiga algoritma klasifikasi yang berbeda akan diimplementasikan dan dievaluasi performanya. Pemilihan beberapa algoritma memungkinkan perbandingan dan penentuan pendekatan mana yang paling efektif untuk dataset yang digunakan. Ketiga algoritma tersebut adalah:

- Decision Tree Classifier
- Random Forest Classifier
- XGBoost (Extreme Gradient Boosting) Classifier

### Decision Tree Classifier
Decision Tree Classifier adalah algoritma supervised learning yang populer dan intuitif, dapat digunakan baik untuk tugas klasifikasi maupun regresi. Algoritma ini bekerja dengan membangun model prediktif dalam bentuk struktur pohon. Setiap node internal pada pohon merepresentasikan sebuah "tes" pada suatu fitur, setiap cabang merepresentasikan hasil dari tes tersebut (sebuah aturan keputusan), dan setiap leaf node (daun) merepresentasikan label kelas atau prediksi akhir. Keunggulan Decision Tree antara lain kemudahannya untuk diinterpretasikan dan divisualisasikan.

Proses inisialisasi dan pelatihan model Decision Tree Classifier dilakukan sebagai berikut:

```python
from sklearn.tree import DecisionTreeClassifier

# Model Development menggunakan Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
```

### Random Forest Classifier
Random Forest adalah algoritma ensemble learning yang bekerja dengan membangun sejumlah besar decision tree secara acak pada berbagai sub-sampel dari dataset dan menggunakan rata-rata atau voting mayoritas dari prediksi masing-masing pohon untuk menghasilkan prediksi akhir yang lebih akurat dan stabil..

Proses inisialisasi dan pelatihan model Random Forest Classifier adalah sebagai berikut:
```python
from sklearn.ensemble import RandomForestClassifier

# Model Development menggunakan Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

### XGBoost (Extreme Gradient Boosting) Classifier
XGBoost (Extreme Gradient Boosting) adalah implementasi yang sangat dioptimalkan dan efisien dari algoritma Gradient Boosting Decision Trees (GBDT). Gradient Boosting bekerja dengan membangun model secara aditif dan sekuensial: pohon keputusan baru dibangun untuk memperbaiki kesalahan atau residu dari pohon-pohon sebelumnya. XGBoost dikenal karena kecepatan komputasinya yang tinggi, performa prediktif yang sangat baik, dan kemampuannya untuk menangani berbagai jenis data serta fitur regularisasi untuk mencegah overfitting.

Proses inisialisasi dan pelatihan model XGBoost Classifier dapat dilakukan sebagai berikut:
```python
from xgboost import XGBClassifier

# Model Development menggunakan XGBoost Classifier
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
```

## Evaluation
Tahap evaluasi merupakan langkah krusial untuk mengukur dan memvalidasi performa model machine learning yang telah dikembangkan. Pada tahap ini, model-model yang telah dilatih akan diuji menggunakan data pengujian (test set) yang belum pernah dilihat sebelumnya oleh model selama proses pelatihan. Tujuan utama dari evaluasi adalah untuk mendapatkan estimasi yang objektif mengenai seberapa baik model dapat melakukan generalisasi pada data baru, mengidentifikasi potensi masalah seperti overfitting (model terlalu baik pada data latih tetapi buruk pada data baru) atau underfitting (model gagal menangkap pola dalam data)

### Metrik Evaluasi yang Digunakan

Sebelum masuk ke hasil evaluasi masing-masing model, penting untuk memahami metrik yang digunakan. Dalam konteks klasifikasi biner ini (air layak minum vs. tidak layak minum), kita akan mengacu pada:

* **True Positive (TP)**: Jumlah sampel Positif yang diprediksi dengan benar sebagai Positif.
* **False Positive (FP)**: Jumlah sampel Negatif yang salah diprediksi sebagai Positif (Type I Error).
* **True Negative (TN)**: Jumlah sampel Negatif yang diprediksi dengan benar sebagai Negatif.
* **False Negative (FN)**: Jumlah sampel Positif yang salah diprediksi sebagai Negatif (Type II Error).

Berikut adalah metrik-metrik yang digunakan:

* **Accuracy (Akurasi)**

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$
        
Akurasi mengukur proporsi total prediksi yang benar (baik layak maupun tidak layak minum) dari keseluruhan jumlah sampel dalam data uji. Meskipun merupakan metrik yang umum, akurasi bisa menjadi kurang informatif jika terdapat ketidakseimbangan kelas dalam dataset.

* **Precision (Presisi)**
      
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
      
Presisi mengukur proporsi sampel yang benar-benar layak minum dari semua sampel yang diprediksi sebagai layak minum oleh model. Presisi yang tinggi menunjukkan bahwa model memiliki tingkat kesalahan positif (FP) yang rendah. Dalam konteks ini, presisi tinggi sangat penting untuk meminimalkan risiko mengklasifikasikan air yang tidak aman sebagai aman untuk diminum.

* **Recall (Sensitivitas atau True Positive Rate)**
      
$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$
      
Recall mengukur proporsi sampel yang benar-benar layak minum yang berhasil diidentifikasi dengan benar oleh model. Recall yang tinggi menunjukkan bahwa model memiliki tingkat kesalahan negatif (FN) yang rendah, artinya model mampu "menemukan" sebagian besar air yang memang layak minum.

* **F1-Score**

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$
      
F1-Score adalah rata-rata harmonik dari Presisi dan Recall. Metrik ini memberikan keseimbangan antara kedua metrik tersebut dan sangat berguna ketika terdapat _trade-off_ antara Presisi dan Recall, atau ketika terdapat ketidakseimbangan kelas. Nilai F1-Score yang tinggi menunjukkan bahwa model memiliki keseimbangan yang baik antara presisi dan recall.

### Fungsi Evaluasi Model
Untuk mempermudah proses evaluasi dan memastikan konsistensi, sebuah fungsi Python akan digunakan untuk menghitung dan menampilkan metrik-metrik di atas serta visualisasi confusion matrix.

```python

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, model_name="Model"):
    print(f"Evaluasi Model: {model_name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred)) # Asumsi kelas positif adalah 1 (layak minum)
    print("Recall:", recall_score(y_true, y_pred))       # Asumsi kelas positif adalah 1
    print("F1 Score:", f1_score(y_true, y_pred))         # Asumsi kelas positif adalah 1
    print("Classification Report:\n", classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
```

### Evaluasi Decision Tree Classifier
Model Decision Tree Classifier yang telah dilatih kemudian dievaluasi menggunakan data uji.

```python
# Menggunakan prediksi dari data uji untuk Decision Tree Classifier
y_pred_dt = dt_model.predict(X_test)

# Mengevaluasi performa model Decision Tree Classifier
evaluate_model(y_test, y_pred_dt, "Decision Tree Classifier")
```

**Hasil Evaluasi Decision Tree Classifier:**
Model Decision Tree Classifier menunjukkan performa yang sangat baik dalam melakukan klasifikasi. Model ini mencapai accuracy sebesar 92.8%, yang berarti sebagian besar prediksi yang dihasilkan sudah tepat. Dengan precision sebesar 92.95%, model memiliki kemampuan yang tinggi dalam memberikan prediksi yang benar untuk kelas air layak minum. Recall sebesar 90.06% menunjukkan bahwa model mampu mengenali sebagian besar data air yang benar-benar layak. Hal ini menghasilkan F1 Score sebesar 91.48%, yang menunjukkan keseimbangan yang solid antara ketepatan dan cakupan prediksi.

### Evaluasi Random Forest Classifier
Selanjutnya, model Random Forest Classifier dievaluasi.

```python
# Menggunakan prediksi dari data uji untuk Random Forest
y_pred_rf = rf_model.predict(X_test)

# Mengevaluasi performa model Random Forest
evaluate_model(y_test, y_pred_rf, "Random Forest")
```

**Hasil Evaluasi Random Forest Classifier:**
Model Random Forest menunjukkan performa yang cukup baik namun belum optimal. Model ini menghasilkan accuracy sebesar 79.2%, yang berarti sebagian besar prediksi sudah benar, namun masih terdapat cukup banyak kesalahan. Dengan precision sebesar 84.87%, model cukup baik dalam memastikan bahwa prediksi air layak minum memang benar. Namun, recall-nya hanya 62.73%, menandakan bahwa masih banyak data air yang benar-benar layak minum tidak berhasil terdeteksi oleh model. Hal ini menyebabkan F1 Score sebesar 72.14%, yang menunjukkan bahwa keseimbangan antara precision dan recall belum ideal.

### Evaluasi XGBoost Classifier
Terakhir, evaluasi dilakukan untuk model XGBoost Classifier.

```python
# Menggunakan prediksi dari data uji untuk XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# Mengevaluasi performa model XGBoost
evaluate_model(y_test, y_pred_xgb, "XGBoost")
```

**Hasil Evaluasi XGBoost Classifier:**
Model XGBoost menunjukkan performa yang sangat baik. Model ini berhasil mencapai accuracy sebesar 97.33%, yang menandakan bahwa sebagian besar prediksi pada data uji sudah tepat. Selain itu, model mencatat precision sebesar 98.09%, artinya prediksi air layak minum sangat jarang salah. Recall-nya sebesar 95.65%, menunjukkan kemampuan model dalam mengenali hampir semua data air yang benar-benar layak minum. Kombinasi ini menghasilkan F1 Score sebesar 96.85%, mencerminkan keseimbangan yang sangat baik antara ketepatan dan kelengkapan prediksi.

### Pemilihan Model Terbaik
Berdasarkan hasil evaluasi dari ketiga model yang telah diuji, dapat disimpulkan performa masing-masing sebagai berikut (mengambil nilai untuk kelas positif/layak minum):

| Model	| Accuracy | Precision | Recall | F1-Score |
| ----- | -------- | --------- | ------ | -------- |
| Decision Tree Classifier | 92.80% | 92.95% | 90.06% | 91.48% |
| Random Forest Classifier	| 79.20% |	84.87% |	62.73% | 72.14% |
| XGBoost Classifier |	97.33% | 98.09% | 95.65%	| 96.85% |

Dari tabel di atas, XGBoost Classifier secara konsisten menunjukkan performa tertinggi di semua metrik evaluasi utama dibandingkan dengan Decision Tree Classifier dan Random Forest Classifier.

## Referensi
- WHO. (2023). Drinking-Water. https://www.who.int/news-room/fact-sheets/detail/drinking-water
- Misra, S., Sinha, A., & Ghosh, R. (2021). Water Quality Assessment Using Machine Learning: A Comparative Study. Journal of Environmental Informatics.
