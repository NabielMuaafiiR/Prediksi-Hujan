# Laporan Proyek Machine Learning - Nabiel Muaafii Rahman

## Domain Proyek
Curah hujan merupakan salah satu komponen kunci dalam sistem hidrologi yang berperan penting dalam berbagai aspek kehidupan, seperti pertanian, pengelolaan sumber daya air, serta mitigasi bencana hidrometeorologi. Di wilayah perkotaan padat seperti Beijing, perubahan pola curah hujan dapat berdampak signifikan terhadap infrastruktur kota, sistem drainase, dan kesejahteraan masyarakat.

Beberapa studi sebelumnya menunjukkan bahwa peristiwa hujan ekstrem selalu membawa dampak dan kerugian besar bagi kehidupan manusia, dan frekuensi kejadiannya telah meningkat secara signifikan dalam beberapa dekade terakhir di China (Jiang _et al_. 2020).  Di kota-kota besar seperti Beijing, perubahan curah hujan yang dipicu oleh urbanisasi dan perubahan iklim global telah menimbulkan tantangan baru, seperti banjir perkotaan dan tekanan terhadap sistem drainase. Selain faktor antropogenik, kondisi sirkulasi atmosfer dan karakteristik topografi lokal turut mempengaruhi pola spasial dan temporal curah hujan di wilayah ini.

Analisis data curah hujan dari berbagai stasiun pengamatan selama periode 2013–2017 dapat memberikan wawasan lebih mendalam mengenai dinamika iklim lokal dan pola distribusi curah hujan di Beijing. Namun, tantangan utama dalam analisis data ini adalah adanya ketidakseimbangan kelas, terutama ketika memodelkan kejadian curah hujan ekstrem yang relatif jarang terjadi dibandingkan kondisi normal. Data yang tidak seimbang menjadi masalah saat membuat model prediksi menggunakan machine learning (Erlin _et al_. 2022).

Untuk mengatasi tantangan tersebut, penelitian ini akan menggunakan dua pendekatan pemodelan: pertama, algoritma Random Forest yang dikombinasikan dengan teknik Synthetic Minority Over-sampling Technique (SMOTE) untuk mengatasi ketidakseimbangan data; dan kedua, model Long Short-Term Memory (LSTM) yang mampu menangkap pola temporal dalam data deret waktu, meskipun dengan ketidakseimbangan kelas yang ada. Kedua pendekatan ini diharapkan dapat memberikan hasil yang akurat dan andal dalam memprediksi pola curah hujan di wilayah Beijing, serta mendukung upaya perencanaan dan mitigasi bencana yang lebih baik.

## Business Understanding

### Problem Statements
- Bagaimana pola curah hujan di wilayah Beijing selama periode 2013–2017 berdasarkan data pengamatan stasiun cuaca?
- Bagaimana dampak ketidakseimbangan data terhadap performa model prediksi curah hujan, khususnya dalam mengidentifikasi kejadian curah hujan ekstrem?
- Sejauh mana algoritma Random Forest yang dikombinasikan dengan SMOTE mampu meningkatkan akurasi dalam klasifikasi kejadian curah hujan ekstrem?
- Bagaimana performa model Long Short-Term Memory (LSTM) dalam memprediksi curah hujan menggunakan data deret waktu yang tidak seimbang?
- Metode manakah yang memberikan performa prediktif yang lebih baik dalam konteks prediksi curah hujan ekstrem di Beijing: Random Forest dengan SMOTE atau LSTM dengan data asli yang tidak seimbang?

### Goals
- Menganalisis pola curah hujan di Beijing selama periode 2013–2017 untuk memahami dinamika iklim lokal.
- Mengidentifikasi pengaruh ketidakseimbangan data terhadap akurasi prediksi curah hujan ekstrem.
- Menerapkan dan mengevaluasi model Random Forest yang dioptimalkan menggunakan teknik SMOTE untuk menangani ketidakseimbangan data.
- Membangun dan mengevaluasi model LSTM untuk prediksi curah hujan berdasarkan data deret waktu tanpa penyeimbangan kelas.
- Membandingkan performa kedua pendekatan model tersebut dalam konteks prediksi curah hujan ekstrem untuk mendukung perencanaan dan mitigasi risiko hidrometeorologi di wilayah perkotaan.

## Data Understanding
Dataset yang diambil pada projek ini berasal dari github https://github.com/marceloreis/HTI/tree/master/PRSA_Data_20130301-20170228. Data yang digunakan hanya satu yaitu daerah Aotizhongxin. Data terdiri dari 35064 baris dan 17 kolom: 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station'

### Tipe Data
```
<class 'pandas.core.frame.DataFrame'>
Index: 35064 entries, 1 to 35064
Data columns (total 17 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   year     35064 non-null  int64  
 1   month    35064 non-null  int64  
 2   day      35064 non-null  int64  
 3   hour     35064 non-null  int64  
 4   PM2.5    34139 non-null  float64
 5   PM10     34346 non-null  float64
 6   SO2      34129 non-null  float64
 7   NO2      34041 non-null  float64
 8   CO       33288 non-null  float64
 9   O3       33345 non-null  float64
 10  TEMP     35044 non-null  float64
 11  PRES     35044 non-null  float64
 12  DEWP     35044 non-null  float64
 13  RAIN     35044 non-null  float64
 14  wd       34983 non-null  object 
 15  WSPM     35050 non-null  float64
 16  station  35064 non-null  object 
dtypes: float64(11), int64(4), object(2)
memory usage: 4.8+ MB
```

### Deskripsi Variabel
Nama Fitur                                     | Deskripsi
----------------------------------------------|------------------------------------------------------------
No                                             | Nomor urut atau indeks baris data
year                                           | Tahun pengambilan data
month                                          | Bulan pengambilan data (1–12)
day                                            | Hari pengambilan data (1–31)
hour                                           | Jam pengambilan data (0–23)
PM2.5                                          | Konsentrasi partikel udara berukuran ≤ 2.5 µm (µg/m³)
PM10                                           | Konsentrasi partikel udara berukuran ≤ 10 µm (µg/m³)
SO2                                            | Konsentrasi gas sulfur dioksida di udara (µg/m³)
NO2                                            | Konsentrasi gas nitrogen dioksida di udara (µg/m³)
CO                                             | Konsentrasi gas karbon monoksida di udara (mg/m³)
O3                                             | Konsentrasi ozon di permukaan (µg/m³)
TEMP                                           | Suhu udara (°C) saat pengamatan
PRES                                           | Tekanan udara (hPa) saat pengamatan
DEWP                                           | Titik embun (°C) saat pengamatan
RAIN                                           | Jumlah curah hujan (mm) dalam satu jam
wd                                             | Arah angin saat pengamatan (misalnya: NW, SE, dll.)
WSPM                                           | Kecepatan angin (m/s) saat pengamatan
station                                        | Nama atau kode stasiun pengamatan


### Melihat Missing Value dan Duplikasi Data
```
df.isna().sum()
```
**Output:**
```
year	0
month	0
day	0
hour	0
PM2.5	925
PM10	718
SO2	935
NO2	1023
CO	1776
O3	1719
TEMP	20
PRES	20
DEWP	20
RAIN	20
wd	81
WSPM	14
```
Terdapat banyak sekali missing value. Terutama pada kolom CO dengan total 1776.

```
df.duplicated().sum()
```
**Output:**
```
np.int64(0)
```

Tidak terdapat baris duplikat.

### Melihat Outlier

![boxplot sebelum](https://github.com/user-attachments/assets/352c60e4-a23c-4083-a546-9be12e6bfb56)

Untuk penanganan outlier, awalnya diterapkan metode imputation pada salinan data, yaitu dengan mengganti nilai yang berada di bawah batas bawah (lower bound) dengan nilai batas bawah tersebut, dan sebaliknya, nilai di atas batas atas diganti dengan nilai batas atas. Pendekatan ini bertujuan untuk menjaga konsistensi data numerik agar tidak terpengaruh ekstremitas yang tidak wajar.


### Visualisasi Data EDA
1. Bagaimana curah hujan tiap bulannya dalam kurun waktu satu tahun?
   
![heatmap](https://github.com/user-attachments/assets/6c60cd4c-4e5d-4bec-85a3-cf48c2b757bc)

  Pada sumbu x menunjukkan bulan Januari hingga bulan Desember, sumbu y menunjukkan tahun dari tahun 2013 hingga tahun 2017 dengan angka dalam tiap sel merepresentasikan proporsi rata-rata curah hujan pada bulan tersebut di tahun tersebut. Heatmap rata-rata curah hujan dari tahun 2013 hingga 2017 menunjukkan pola musiman yang konsisten, di mana curah hujan cenderung memuncak pada pertengahan tahun, khususnya di bulan Juni hingga Agustus. Puncak tertinggi terjadi pada Juli 2016 dengan intensitas curah hujan yang relatif paling besar dibandingkan bulan dan tahun lainnya. Sebaliknya, bulan-bulan seperti Januari, Februari, November, dan Desember secara umum menunjukkan intensitas curah hujan yang rendah, menandakan musim kering. Tahun 2016 dan 2015 terlihat memiliki curah hujan yang lebih tinggi secara umum dibandingkan tahun-tahun lainnya, sementara tahun 2013 dan 2017 tampak lebih kering dengan lebih banyak bulan yang memiliki curah hujan rendah.

![linechart](https://github.com/user-attachments/assets/b5d7442f-0a53-4bfc-8810-ae94d96056e5)

  Selain heatmap, terdapat line chart yang dibuat dan bisa dilihat sama dengan heatmap bahwa puncak tertinggi curah hujan terjadi pada bulan Juli tahun 2016 . Grafik ini sangat penting dalam membangun model. Ia membantu memahami data, memvalidasi asumsi, dan memilih strategi modeling yang tepat.

2. Faktor apa yang menyebabkan curah hujan?

![scatter](https://github.com/user-attachments/assets/fd97fbb7-0576-4a5f-a9c3-c6f9d56814a6)

   Pada Gambar diatas kita bisa melihat scatter plot dan tabel numerik yang menunjukkan bahwa RAIN tidak ada hubungan yang berarti dengan fitur lainnya ditunjukkan bahwa nilai tertinggi pada tabel numerik hanya 0.080789 yang mana sangat kecil untuk menunjukkan bahwa fitur ini berkorelasi dengan RAIN.
## Data Preparation

### Membuang Kolom
```
df = df.drop(columns = ['station'])
```
Kolom station dibuang dikarenakan dalam data ini hanya menggunakan satu stasiun saja. Sehingga kolom station tidak perlu untuk dimasukkan kedalam prediksi.

### Penanganan Missing Value
```
# Mengisi missing value kolom numerik dengan rata-rata
for col in ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']:
    df[col] = df[col].fillna(value=df[col].mean())

# Mengisi missing value kolom kategorik dengan modus
df['wd'] = df['wd'].fillna(df['wd'].mode()[0])

#Melihat kembali data apakah masih ada missing value
print(df.isna().sum())
```
**Output:**
```
year     0
month    0
day      0
hour     0
PM2.5    0
PM10     0
SO2      0
NO2      0
CO       0
O3       0
TEMP     0
PRES     0
DEWP     0
RAIN     0
wd       0
WSPM     0
dtype: int64
```
Missing value yang terdapat pada kolom numerik diatasi dengan cara imputasi menggunakan nilai rata-rata (mean), sementara missing value pada kolom kategorik diisi dengan nilai yang paling sering muncul (modus).

### Menangani Outlier
![boxplot sesudah](https://github.com/user-attachments/assets/21dd2ca0-ca32-4d00-a94c-dbf80bc9d4fb)

Namun, setelah metode ini diterapkan, ditemukan anomali penting: fitur “RAIN” menghilang dari dataset. Hal ini mengindikasikan bahwa perlakuan terhadap outlier justru menyebabkan informasi penting hilang, terutama karena data menunjukkan tidak ada kejadian hujan selama 2013–2017 dalam dataset asli, yang berisiko mengacaukan makna dari fitur tersebut.Sebagai solusi, diputuskan untuk tidak melakukan imputasi pada outlier secara langsung. Sebagai gantinya, ditambahkan kolom baru bernama “RAIN_Category” yang mengelompokkan data ke dalam kategori hujan dan tidak hujan, agar analisis dapat diarahkan pada pendekatan klasifikasi berdasarkan keberadaan hujan, tanpa merusak integritas data asli.

| RAIN_Category |  Count  |
|-----|-----|
|  Tidak Hujan  | 33664  | 
|  Hujan  | 1400 |

dilakukan kembali imputation pada data asli (df):
```
# Lakukan hal yang sama untuk kolom lain yang mengandung outlier
for col in ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM', 'PM2.5']:
    impute_outliers_iqr(df, col)
```
Dikarenakan sebelumnya hanya melihat bagaimana jika diberlakukan outlier. Semua outlier ditangani kecuali pada kolom RAIN.

### Memisahkan fitur dan label
```
X = df.drop(['RAIN', 'RAIN_Category', 'year', 'hour'], axis=1)
y = df['RAIN_Category']
```
Memisahkan fitur dan label sekaligus menghapus kolom yang tidak dibutuhkan. Kolom year dan hour dihapus karena agar model lebih fokus menangkap pola pada bulan dan hari saja.
### Melakukan encoding label:
```
df = pd.get_dummies(df, columns=['wd'], prefix='wd')
```

```
y_encoded = y.map({'Hujan': 1, 'Tidak Hujan': 0})
```
Encoding dilakukan sebagai langkah praproses untuk mengubah data kategorikal menjadi format numerik, karena model pembelajaran mesin hanya dapat memproses data dalam bentuk angka, bukan string.

### Scaling data untuk model LSTM
```
# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
Scaling dilakukan pada model LSTM saja, dikarenakan fitur satu sama lain saling berkaitan, yang menyebabkan perbedaan skala antar fitur dapat mempengaruhi proses pembelajaran model secara signifikan. LSTM (Long Short-Term Memory) sangat sensitif terhadap skala data karena menggunakan fungsi aktivasi seperti sigmoid dan tanh, yang kinerjanya optimal pada nilai input yang terstandarisasi. Tanpa proses scaling, fitur dengan nilai yang besar dapat mendominasi perhitungan dan menyebabkan konvergensi yang lambat atau bahkan tidak stabil. Berbeda dengan model Random Forest, yang bersifat non-parametrik dan tidak terpengaruh oleh skala fitur, sehingga tidak memerlukan proses normalisasi atau standardisasi sebagai prasyarat pelatihan model.

### SMOTE data untuk model Random Forest
```
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Shape before SMOTE:", X.shape)
print("Shape after SMOTE:", X_resampled.shape)
print("\nClass distribution before SMOTE:\n", y.value_counts())
print("\nClass distribution after SMOTE:\n", y_resampled.value_counts())
```
SMOTE dilakukan untuk menangani ketidak seimbangan data yang ada. Hal ini bertujuan untuk membantu model Random Forest memprediksi kelas minoritas, yang dalam kasus ini adalah kelas tidak hujan yang memiliki perbandingan 0.0399% dari keseluruhan data.


### Sliding Window untuk LSTM
```
# Fungsi untuk membuat sequence (misal 3 waktu sebelumnya)
def create_sequences(X, y, time_steps=3):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_encoded, time_steps=3)
```
Sliding window dilakukan untuk Menangkap Ketergantungan Waktu. Banyak fenomena, seperti curah hujan atau polusi udara, bergantung pada data sebelumnya (misalnya, jam/tanggal sebelumnya). Dengan window, model dapat mempelajari pola perubahan dari waktu ke waktu. Tak hanya itu windowing merupakan syarat untuk Model LSTM/RNN. Model berbasis memori seperti LSTM membutuhkan urutan sebagai input, bukan data tabular biasa.

### Splitting Data
```
#Split data Random Forest
X_train,X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Split data LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42) 
```
Sebelum data dimasukkan kedalam model, data dibagi menjadi dua bagian, yaitu data latih dan data uji, dengan proporsi 80% untuk data latih dan 20% untuk data uji. Pembagian ini bertujuan agar model dapat melakukan analisis dan pembelajaran tanpa mengalami kebocoran data (data leakage), sehingga hasil evaluasi model menjadi lebih objektif dan dapat dipercaya.

## Modeling
1. Model Random Forest
```
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
```

Random forest merupakan algoritma pengembangan dari algoritma decision tree. Random forest merupakan prinsip umum pengklasifikasi yang menggabungkan L tree-structured based classifier (Ali J, et al. 2012). Tujuan utama Random Forest Classifier adalah meningkatkan akurasi prediksi dan mengurangi risiko overfitting yang sering terjadi pada pohon keputusan tunggal.

Proses kerja Random Forest Classifier dimulai dengan membentuk beberapa pohon keputusan (decision tree) berdasarkan subset acak dari data latih dan fitur (proses ini disebut bootstrap sampling dan feature randomization). Setiap pohon akan melakukan klasifikasi secara independen, dan hasil akhir ditentukan berdasarkan  mayoritas suara dari seluruh pohon.

Pada pengujian ini model Random Forest diinisialisasi dengan parameter default dan dilatih menggunakan data latih melalui model.fit(), kemudian digunakan untuk memprediksi kategori hujan pada data uji dengan model.predict().

2. Model LSTM
```
# Build model
lstm = Sequential()
lstm.add(LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2])))
lstm.add(Dense(1, activation='sigmoid'))

lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Buat callback
early_stop = EarlyStopping(
    monitor='val_loss',     # Pantau loss pada validation set
    patience=5,             # Hentikan setelah 5 epoch tanpa perbaikan
    restore_best_weights=True # Gunakan bobot terbaik (bukan dari epoch terakhir)
)
```

  Model deep learning yang digunakan dalam kode di atas adalah jaringan saraf Long Short-Term Memory (LSTM) yang dirancang khusus untuk memproses dan mempelajari pola dari data deret waktu. Arsitektur model dibangun secara berurutan menggunakan Keras Sequential, dengan satu layer LSTM yang terdiri dari 64 unit tersembunyi (hidden units) dan diikuti oleh sebuah layer Dense dengan 1 neuron serta fungsi aktivasi sigmoid. Fungsi sigmoid digunakan karena tugas model adalah klasifikasi biner—dalam hal ini, untuk memprediksi apakah akan terjadi hujan (1) atau tidak (0) pada waktu tertentu.

LSTM merupakan jenis Recurrent Neural Network (RNN) yang mengatasi kelemahan RNN standar dalam mengingat informasi jangka panjang akibat masalah vanishing gradient. Dengan struktur internal yang terdiri dari tiga gerbang utama—forget gate, input gate, dan output gate—LSTM dapat mengatur aliran informasi secara selektif: menentukan informasi mana yang perlu dilupakan, disimpan, atau diteruskan ke waktu berikutnya. Hal ini sangat penting dalam memodelkan fenomena seperti curah hujan, yang memiliki pola musiman dan ketergantungan temporal kuat, baik dalam skala harian, mingguan, maupun bulanan.

Dalam implementasi ini, model juga dilengkapi dengan callback EarlyStopping, yaitu sebuah strategi pelatihan yang secara otomatis menghentikan proses training ketika performa model pada data validasi tidak membaik setelah sejumlah epoch tertentu. Dalam hal ini, parameter patience=5 berarti model akan berhenti jika selama lima epoch berturut-turut tidak ada penurunan nilai val_loss. Selain itu, dengan restore_best_weights=True, bobot terbaik model akan dipulihkan secara otomatis untuk menghindari overfitting akibat pelatihan yang terlalu lama.

Kombinasi antara arsitektur LSTM, normalisasi fitur (melalui MinMaxScaler), dan penggunaan teknik EarlyStopping membuat model ini cukup tangguh dalam mengenali pola hujan dari data historis, sekaligus mampu menghindari overfitting dan meningkatkan generalisasi terhadap data yang belum pernah dilihat sebelumnya. Model seperti ini sangat cocok digunakan dalam aplikasi prediksi cuaca, khususnya dalam mengantisipasi hujan ekstrem atau perubahan iklim lokal yang sulit diprediksi dengan pendekatan konvensional.
  
```
history = lstm.fit(
    X_train_lstm, y_train_lstm,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop]
)
```
  Lalu dilakukan fitting data terhadap model yang sudah dibangun, dengan epoch sebanyak 100, batch size digunakan 16. Setelah cell di run didapatkan hasil berikut:

 ```
Epoch 1/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 9s 4ms/step - accuracy: 0.9618 - loss: 0.1752 - val_accuracy: 0.9551 - val_loss: 0.1494
Epoch 2/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.9597 - loss: 0.1340 - val_accuracy: 0.9551 - val_loss: 0.1481
Epoch 3/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 11s 5ms/step - accuracy: 0.9593 - loss: 0.1231 - val_accuracy: 0.9565 - val_loss: 0.1275
Epoch 4/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9616 - loss: 0.1134 - val_accuracy: 0.9569 - val_loss: 0.1278
Epoch 5/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 9s 4ms/step - accuracy: 0.9643 - loss: 0.1072 - val_accuracy: 0.9565 - val_loss: 0.1211
Epoch 6/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9633 - loss: 0.1061 - val_accuracy: 0.9569 - val_loss: 0.1244
Epoch 7/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9648 - loss: 0.1001 - val_accuracy: 0.9565 - val_loss: 0.1181
Epoch 8/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.9636 - loss: 0.1033 - val_accuracy: 0.9569 - val_loss: 0.1282
Epoch 9/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9636 - loss: 0.1027 - val_accuracy: 0.9565 - val_loss: 0.1162
Epoch 10/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9648 - loss: 0.0984 - val_accuracy: 0.9590 - val_loss: 0.1126
Epoch 11/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 11s 5ms/step - accuracy: 0.9658 - loss: 0.0965 - val_accuracy: 0.9597 - val_loss: 0.1086
Epoch 12/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9663 - loss: 0.0913 - val_accuracy: 0.9604 - val_loss: 0.1118
Epoch 13/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.9665 - loss: 0.0961 - val_accuracy: 0.9619 - val_loss: 0.1060
Epoch 14/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9678 - loss: 0.0867 - val_accuracy: 0.9601 - val_loss: 0.1053
Epoch 15/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9663 - loss: 0.0892 - val_accuracy: 0.9590 - val_loss: 0.1083
Epoch 16/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 11s 5ms/step - accuracy: 0.9681 - loss: 0.0852 - val_accuracy: 0.9583 - val_loss: 0.1062
Epoch 17/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9679 - loss: 0.0848 - val_accuracy: 0.9608 - val_loss: 0.1030
Epoch 18/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.9683 - loss: 0.0824 - val_accuracy: 0.9608 - val_loss: 0.1017
Epoch 19/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.9720 - loss: 0.0764 - val_accuracy: 0.9590 - val_loss: 0.1109
Epoch 20/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.9699 - loss: 0.0787 - val_accuracy: 0.9597 - val_loss: 0.1059
Epoch 21/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - accuracy: 0.9716 - loss: 0.0756 - val_accuracy: 0.9601 - val_loss: 0.1059
Epoch 22/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9730 - loss: 0.0719 - val_accuracy: 0.9572 - val_loss: 0.1069
Epoch 23/100
1578/1578 ━━━━━━━━━━━━━━━━━━━━ 11s 5ms/step - accuracy: 0.9733 - loss: 0.0716 - val_accuracy: 0.9590 - val_loss: 0.1026
```

## Evaluation

### Penjelasan Matriks
Matriks evaluasi yang digunakan antara lain **akurasi, precision, recall, dan F1-score**, tetapi pada projek ini lebih difokuskan pada akurasi dan F1-score.

#### 1. Akurasi
   
Akurasi adalah metrik evaluasi yang mengukur proporsi prediksi yang benar terhadap seluruh jumlah data.

Rumus:

![rumus akurasi](https://github.com/user-attachments/assets/3a640902-3148-406f-905b-eb68e89f12da)

Keterangan:
TP: True Positive (positif yang diprediksi benar)

TN: True Negative (negatif yang diprediksi benar)

FP: False Positive (negatif yang diprediksi positif)

FN: False Negative (positif yang diprediksi negatif)

Kelemahan:
Akurasi tidak cocok untuk dataset yang tidak seimbang, karena bisa memberikan skor tinggi hanya dengan memprediksi mayoritas kelas.

#### 2. F1-Score
   
F1-score adalah metrik harmonis antara precision dan recall. F1 digunakan untuk mengevaluasi model terutama ketika data tidak seimbang, karena mempertimbangkan kesalahan pada prediksi kelas minoritas.

Rumus:
F1-score

![F1 score](https://github.com/user-attachments/assets/a7d5f333-d92d-4363-949c-f8e97e8da769)

Precision:

![precision](https://github.com/user-attachments/assets/52a40c27-3abd-40d0-a995-ad336814bc3c)
 
Mengukur berapa banyak prediksi positif yang benar.

Recall:

![recall](https://github.com/user-attachments/assets/6311ce4e-bd19-4d2d-8e3e-8f5202c5b017)
 
Mengukur berapa banyak dari total data positif yang berhasil dikenali dengan benar.

Keunggulan:
- F1-score tidak terpengaruh oleh distribusi kelas.

Sangat cocok jika:

- False positive dan false negative sama-sama penting.

- Kelas target (misal: positif) jumlahnya jauh lebih sedikit daripada kelas lain.

### Hasil Evaluasi

#### 1. Random Forest dengan SMOTE
Pada pengujian pertama dengan parameter default didapatkan akurasi sebesar 98% dan F1-score sebesar 0.9845 dengan detail confusion matriks sebagai berikut:

Confusion Matrix:
|     |  Tidak Hujan  |  Hujan  |  
|-----|-----|-----|
|  Tidak Hujan  | 6692  | 41  | 
|  Hujan  | 170  | 6563 | 


Classification Report:
```
              precision    recall  f1-score   support

       Hujan       0.98      0.99      0.98      6733
 Tidak Hujan       0.99      0.97      0.98      6733

    accuracy                           0.98     13466
   macro avg       0.98      0.98      0.98     13466
weighted avg       0.98      0.98      0.98     13466
```


#### 2. LSTM tanpa SMOTE
Hasil training dari model LSTM amat bagus dapat dilihat pada grafik berikut:

![image](https://github.com/user-attachments/assets/dedb802a-2e6f-49a9-88e7-15205aad267f)

Dilakukan testing menggunakan model LSTM hasil evaluasi model mengalami penurunan dimana akurasi menjadi 96.%, dan F1-score sebesar 0.4875, untuk detail confusion matriks sebagai berikut:

Confusion Matrix:
|     |  Tidak Hujan  |  Hujan  |  
|-----|-----|-----|
|  Tidak Hujan  | 6619  | 114  | 
|  Hujan  | 153  | 127 | 


Classification Report:
```
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      6714
           1       0.64      0.31      0.42       299

    accuracy                           0.96      7013
   macro avg       0.81      0.65      0.70      7013
weighted avg       0.96      0.96      0.96      7013
```
Keterangan: 0=Tidak Hujan, 1=Hujan
## Inference
Untuk inference diambil satu data secara acak pada df dengan label 'Hujan', dengan data sebagai berikut:
```
data = {
        'month': [3],
        'day': [12],
        'PM2.5': [117.0],
        'PM10': [127.0],
        'SO2': [73.0],
        'NO2': [81.0],
        'CO': [1262.945145],
        'O3': [47.0],
        'TEMP': [6.4],
        'PRES': [1005.0],
        'DEWP': [-1],
        'WSPM': [2.2],
        'wd_E': [False],
        'wd_ENE': [False],
        'wd_ESE': [False],
        'wd_N': [True],
        'wd_NE': [False],
        'wd_NNE': [False],
        'wd_NNW': [False],
        'wd_NW': [False],
        'wd_S': [False],
        'wd_SE': [False],
        'wd_SSE': [False],
        'wd_SSW': [False],
        'wd_SW': [False],
        'wd_W': [False],
        'wd_WNW': [False],
        'wd_WSW': [False]
        }
data_df = pd.DataFrame(data)
```
### Random Forest dengan SMOTE
```
pred = rfc.predict(data_df)
print(pred)
```
**Output:**
```
['Hujan']
```
Hasil prediksi oleh random forest sesuai dengan label yang asli yaitu Hujan.

### LSTM tanpa SMOTE
```
# Scaling (pakai scaler dari training)
data_baru_scaled = scaler.transform(data_df)

# Bentuk ke [samples, time_steps, features]
X_input = np.expand_dims(data_baru_scaled, axis=0)  # (1, 3, jumlah_fitur)

# Prediksi probabilitas hujan
prediksi = lstm.predict(X_input)

# Output probabilitas (karena sigmoid)
print("Probabilitas hujan:", prediksi[0][0])

# Konversi ke label 0 atau 1 jika perlu
label = 1 if prediksi[0][0] > 0.3 else 0
print("Apakah akan hujan?", "Ya" if label == 1 else "Tidak")
```
**Output:**
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 250ms/step
Probabilitas hujan: 0.11804846
Apakah akan hujan? Tidak
```
Hasil yang didapat Tidak Hujan. Tidak sesuai dengan label yang sebenarnya.
## Kesimpulan
Berdasarkan hasil evaluasi, model Random Forest yang dilatih dengan teknik SMOTE menunjukkan performa yang sangat baik dalam memprediksi apakah akan terjadi hujan atau tidak, dengan akurasi sebesar 98% dan F1-score sebesar 0.9846. Model ini mampu mengenali kedua kelas dengan seimbang dan menghasilkan prediksi yang sesuai dengan label sebenarnya pada saat inferensi. Sebaliknya, model LSTM yang dilatih tanpa penyeimbangan data menghasilkan akurasi yang tampak tinggi (96%) namun memiliki F1-score yang sangat rendah (0.4189), terutama karena gagal mengenali kelas minoritas (hujan). Hal ini terlihat jelas dari confusion matrix dan hasil inferensi yang tidak sesuai. Dengan demikian, dapat disimpulkan bahwa pada dataset yang tidak seimbang, model klasik seperti Random Forest yang dikombinasikan dengan teknik penyeimbangan data seperti SMOTE jauh lebih efektif dibandingkan model deep learning seperti LSTM yang tidak dibarengi dengan penanganan ketidakseimbangan data.

## Saran
1. Pada model LSTM dilakukan juga teknik oversampling pada kelas minoritas, seperti menggunakan teknik SMOTE atau metode oversampling lainnya, untuk mengatasi ketidakseimbangan data dan meningkatkan kemampuan model dalam mengenali semua kelas secara adil.
2. Menerapkan teknik evaluasi yang lebih beragam, seperti confusion matrix per kelas, macro/micro averaging, dan visualisasi performa model, untuk memberikan pemahaman lebih menyeluruh terhadap hasil klasifikasi.

## Refrensi
Ali J, Khan R, Ahmad N, Maqsood I. 2012. Random forest and decision trees. International Journal of Computer Science Issue (IJCSI). 9(5): 272.

Erlin E, Desnelita Y, Nasution N, Suryati L, Zoromi F. 2022. Dampak SMOTE terhadap Kinerja Random Forest Classifier berdasarkan Data Tidak seimbang. MATRIK: Jurnal Manajemen, Teknik Informatika dan Rekayasa Komputer. 21(3): 677-690.

Jiang X, Luo Y, Zhang DL, Wu M. 2020. Urbanization enhanced summertime extreme hourly precipitation over the Yangtze River Delta. Journal of Climate. 33(13): 5809-5826.

Dicoding. (2024). Machine Learning Terapan. Diakses pada 25 Mei 2025 dari https://www.dicoding.com/academies/319-machine-learning-terapan.
