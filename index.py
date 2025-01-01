import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz

import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.ops import unary_union

from sklearn.metrics import silhouette_score

# Sidebar dengan tombol
st.sidebar.header("Fuzzy C-Means")
if st.sidebar.button("Beranda"):
    st.session_state.page = "Beranda"
if st.sidebar.button("Library"):
    st.session_state.page = "Library"
if st.sidebar.button("Data"):
    st.session_state.page = "Data"
if st.sidebar.button("Data Preparation"):
    st.session_state.page = "Data Preparation"
if st.sidebar.button("Klasterisasi"):
    st.session_state.page = "Klasterisasi"
if st.sidebar.button("Visualisasi"):
    st.session_state.page = "Visualisasi"
if st.sidebar.button("Evaluasi"):
    st.session_state.page = "Evaluasi"

df_miskin = pd.read_csv("data/Jumlah Penduduk Miskin Provinsi Sulawesi Tenggara Menurut Kabupaten_Kota, 2024.csv")
df_miskin.name = "Data Jumlah Penduduk Miskin Berdasarkan Kabupaten/Kota di Sulawesi Tenggara Tahun 2024"

df_total = pd.read_csv("data/Jumlah Penduduk Provinsi Sulawesi Tenggara Menurut Kabupaten_Kota, 2024.csv")
df_total.name = "data/Data Jumlah Penduduk Berdasarkan Kabupaten/Kota di Sulawesi Tenggara Tahun 2024"

df_miskin = df_miskin[:17]
df_miskin['Jiwa (Ribu)'] = (df_miskin['Jiwa (Ribu)'] * 1000).astype(int)

name_mapping = {
    'Kota Kendari': 'Kota Kendari',
    'Kab. Konawe Selatan': 'Konawe Selatan',
    'Kab. Konawe': 'Konawe',
    'Kab. Kolaka': 'Kolaka',
    'Kab. Muna': 'Muna',
    'Kab. Bombana': 'Bombana',
    'Kota Bau Bau': 'Kota Baubau',
    'Kab. Kolaka Utara': 'Kolaka Utara',
    'Kab. Kolaka Timur': 'Kolaka Timur',
    'Kab. Buton Tengah': 'Buton Tengah',
    'Kab. Buton': 'Buton',
    'Kab. Wakatobi': 'Wakatobi',
    'Kab. Buton Selatan': 'Buton Selatan',
    'Kab. Muna Barat': 'Muna Barat',
    'Kab. Konawe Utara': 'Konawe Utara',
    'Kab. Buton Utara': 'Buton Utara',
    'Kab. Kep. Konawe': 'Konawe Kepulauan'
}

df_total['Nama Data'] = df_total['Nama Data'].replace(name_mapping)

df_merged = pd.merge(df_total, df_miskin, left_on='Nama Data', right_on='Wilayah', how='inner')

df_merged = df_merged[['Nama Data', 'Nilai', 'Jiwa (Ribu)']]
df_merged.columns = ['Nama Kabupaten/Kota', 'Jumlah Penduduk', 'Jumlah Penduduk Miskin']
df_merged['Angka Kemiskinan'] = (df_merged['Jumlah Penduduk Miskin'] / df_merged['Jumlah Penduduk']) * 100

data = df_merged[["Nama Kabupaten/Kota", "Angka Kemiskinan"]]

# Membuat objek scaler
scaler = MinMaxScaler()

# Melakukan Min-Max normalization pada kolom 'Angka Kemiskinan'
data['Angka Kemiskinan'] = scaler.fit_transform(data[['Angka Kemiskinan']])

# Memilih hanya kolom kedua (misalnya 'Feature2') untuk klastering
X = data[['Angka Kemiskinan']].values.T # Mengambil kolom kedua dan mengubahnya menjadi array 2D

n_clusters = 3
m = 2
max_iter = 100
error_threshold = 0.0001

# Melakukan klastering Fuzzy C-Means
cntr, u, _, _, _, _, _ = fuzz.cmeans(X, n_clusters, m, error=error_threshold, maxiter=max_iter, init=None)

# Menentukan klaster berdasarkan derajat keanggotaan tertinggi
cluster_labels = np.argmax(u, axis=0)

# Menambahkan label klaster ke DataFrame
data['Cluster'] = cluster_labels

# Menampilkan data yang telah dikelompokkan
print(data)


# Menampilkan halaman berdasarkan state
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

if st.session_state.page == "Beranda":
    st.header("Klasterisasi Tingkat Kemiskinan tiap Kabupaten di Provinsi Sulawesi Tenggara")
    st.write("Penelitian ini bertujuan untuk dapat menghasilkan kluster tingkat kemiskinan tiap kabupaten di provinsi sulawesi tenggara dengan menerapkan algoritma Fuzzy C-Means.")

    st.image("image/flowchart_klasterisasi_fuzzy_cmeans.svg")
elif st.session_state.page == "Library":
    st.header("Library")
    images = ["image/pandas_logo.svg", "image/numpy_logo.svg", "image/scikit_learn_logo.svg", "image/logo_scikit_fuzzy.png", "image/matplotlib_logo.svg", "image/seaborn_logo.svg", "image/geopandas_logo.svg", "image/logo_shapely.svg"]
    image_width = 200
    images_per_row = 3
    col = st.columns(len(images))
    for i in range(0, len(images), images_per_row):
        row_images = images[i:i + images_per_row]  # Ambil subset gambar untuk baris ini
        cols = st.columns(len(row_images))  # Buat kolom sesuai jumlah gambar dalam baris
        for col, img_path in zip(cols, row_images):
            with col:
                st.image(img_path, width=image_width)

elif st.session_state.page == "Data":
    st.header("Data")
    st.subheader("Berikut ini merupakan data yang digunakan")
    st.subheader("1. Data Jumlah Penduduk Miskin Berdasarkan Kabupaten/Kota di Sulawesi Tenggara Tahun 2024")
    st.table(df_miskin)
    st.subheader("2. Data Jumlah Penduduk Berdasarkan Kabupaten/Kota di Sulawesi Tenggara Tahun 2024")
    st.table(df_total)
    
elif st.session_state.page == "Data Preparation":
    st.header("Tahapan Data Preparation")
    st.subheader("1. Cleansing Data")
    st.write("Ketika dilakukan pengecekan data tidak ditemukan data yang kosong sehingga tidak perlu dilakukan cleansing data")
    st.subheader("2. Data Preprocessing")
    st.write("""
             Tahap ini diperlukan karena alasan-alasan berikut:

- Data yang akan digunakan hanya mencakup kabupaten/kota, sehingga data **Sulawesi Tenggara** akan dihapus.
- Data pada `df_miskin` kolom kedua akan diubah formatnya menjadi satuan (bukan ribu jiwa).
- Penamaan kabupaten/kota pada kedua dataframe tidak konsisten.
- Kedua dataframe akan digabungkan, dengan kolom kedua menyimpan data jumlah penduduk miskin, dan kolom ketiga berisi jumlah total penduduk.
- Diperlukannya angka kemiskinan, dengan menghitung persentase jumlah penduduk miskin dari keseluruhannya.
             """)
    
elif st.session_state.page == "Klasterisasi":
    st.header("Klasterisasi")
    st.subheader("1. proses klasterisasi dengan Fuzzy C-Means")
    st.write("""
             - X = data[['Angka Kemiskinan']].values.T (Mengambil kolom angka kemiskinan dan mengubahnya menjadi array 2D)
             - n_cluster = 3
             - m = 2
             - error treshold = 0.0001
             """)
    st.subheader("2. Menentukan klaster berdasarkan derajat keanggotaan tertinggi")
    st.subheader("3. Menambahkan label klaster ke DataFrame")
    st.write("")
    st.write("Hasil Klasterisasi")
    st.table(data)
    
elif st.session_state.page == "Visualisasi":
    st.header("Hasil visualisasi")
    # Visualisasi hasil klastering
    
    st.subheader("1. Scatter Plot")
    fig, ax = plt.subplots()
    ax.scatter(data.index, data['Angka Kemiskinan'], c=data['Cluster'], cmap='viridis')
    ax.set_title('Fuzzy C-Means Clustering')
    ax.set_xlabel('Index')
    ax.set_ylabel('Tingkat Kemiskinan')
# other plotting actions...
    st.pyplot(fig)
    
    st.subheader("2. Diagram Batang")
    # Urutkan df_merged berdasarkan 'Jumlah Penduduk Miskin' dari yang terbesar
    df_sorted = df_merged[['Nama Kabupaten/Kota', 'Jumlah Penduduk Miskin']].sort_values(by='Jumlah Penduduk Miskin', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))  # Menentukan ukuran gambar

    sns.barplot(x='Jumlah Penduduk Miskin', y='Nama Kabupaten/Kota', data=df_sorted, palette='rocket', ax=ax)

    # Menambahkan label dan judul
    ax.set_title('Jumlah Penduduk Miskin di Kabupaten/Kota Sulawesi Tenggara')
    ax.set_xlabel('Jumlah Penduduk Miskin (Ribu Jiwa)')
    ax.set_ylabel('Nama Kabupaten/Kota')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Menampilkan plot di Streamlit
    st.pyplot(fig)
    
    st.subheader("3. Peta Sebaran")
    
    shp_path = 'shapefile'
    df_gdf = gpd.read_file(shp_path)

    # Memproses data geometris
    data_gdf = df_gdf[['NAMOBJ', 'geometry']]
    data_gdf.columns = ["Nama Kabupaten/Kota", 'geometry']
    grouped = data_gdf.groupby('Nama Kabupaten/Kota')['geometry'].apply(unary_union).reset_index()

    # Membuat GeoDataFrame baru dengan geometri yang sudah digabungkan
    data_gdf = gpd.GeoDataFrame(grouped, geometry='geometry')
    data_gdf['Nama Kabupaten/Kota'] = data_gdf['Nama Kabupaten/Kota'].replace({'Kota Bau Bau': 'Kota Baubau'})

    # Mengurutkan data berdasarkan angka kemiskinan
    df_sorted = df_merged[['Nama Kabupaten/Kota', 'Angka Kemiskinan']].sort_values(by='Angka Kemiskinan', ascending=False)

    # Menggabungkan data geometri dan angka kemiskinan
    data_gdf_ordered = pd.merge(df_sorted, data_gdf, on='Nama Kabupaten/Kota', how='left')
    data_gdf_ordered = gpd.GeoDataFrame(data_gdf_ordered, geometry=data_gdf_ordered['geometry'])

    # Membuat plot
    fig, ax = plt.subplots(figsize=(20, 20))
    data_gdf_ordered.plot(ax=ax, column="Angka Kemiskinan", legend=False, cmap='rocket_r')

    # Menambahkan label di dalam area geometrinya menggunakan representative_point()
    for x, y, label in zip(data_gdf_ordered.geometry.centroid.x, 
                        data_gdf_ordered.geometry.centroid.y, 
                        data_gdf_ordered['Nama Kabupaten/Kota']):
        # Menggunakan representative_point untuk memastikan label berada di dalam geometri
        point = data_gdf_ordered.loc[data_gdf_ordered['Nama Kabupaten/Kota'] == label, 'geometry'].values[0].representative_point()

        # Menambahkan label dengan latar belakang dan pengaturan warna kontras
        ax.text(
            point.x, point.y, label,
            fontsize=12, ha='center', color='white',
            bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.3')
        )

    # Menambahkan judul
    plt.title('Visualisasi Peta Berdasarkan Angka Kemiskinan')

    # Menampilkan plot di Streamlit
    st.pyplot(fig)

elif st.session_state.page == "Evaluasi":
    st.header("Evaluasi")
    st.write(
        """
        ## Evaluasi Hasil Klastering Fuzzy C-Means

### 1. **Fuzzy Partition Coefficient (FPC):** `0.8561`
- **FPC** mengukur sejauh mana klaster yang terbentuk memiliki pemisahan yang jelas. Nilai FPC berkisar antara 0 dan 1:
  - **Nilai mendekati 1** menunjukkan pemisahan klaster yang baik, di mana keanggotaan data terhadap lebih dari satu klaster adalah kecil dan jelas.
  - **Nilai mendekati 0** menunjukkan bahwa klaster sangat tumpang tindih.
  
  **Interpretasi:**
  Nilai **0.8561** menunjukkan bahwa klaster yang terbentuk memiliki pemisahan yang cukup baik, dengan sebagian besar data terdistribusi dengan jelas ke dalam klaster yang berbeda. Meskipun ada kemungkinan sedikit tumpang tindih, pemisahan antar klaster sudah cukup jelas.

### 2. **Sum of Squared Errors (SSE):** `0.1001`
- **SSE** mengukur sejauh mana data dalam klaster terpisah dari pusat klaster. Semakin kecil nilai SSE, semakin baik pemisahan data dalam klaster tersebut.
  - **Nilai rendah** menunjukkan data berada dekat dengan pusat klaster.
  - **Nilai tinggi** menunjukkan bahwa data lebih tersebar atau lebih jauh dari pusat klaster.

  **Interpretasi:**
  Nilai **0.1001** menunjukkan bahwa kesalahan (jarak antara data dan pusat klaster) relatif kecil, yang berarti klaster terbentuk dengan baik dan data tidak terlalu tersebar.

### 3. **Silhouette Score:** `0.6807`
- **Silhouette Score** mengukur kualitas klaster berdasarkan dua aspek: kedekatan antar data dalam klaster yang sama dan jarak data terhadap klaster lainnya. Nilai ini berkisar antara **-1 hingga 1**:
  - **Nilai mendekati 1** menunjukkan klaster yang sangat baik.
  - **Nilai mendekati -1** menunjukkan data mungkin dikelompokkan ke dalam klaster yang salah.
  - **Nilai mendekati 0** menunjukkan data berada di perbatasan antara dua klaster.

  **Interpretasi:**
  Nilai **0.6807** menunjukkan bahwa klaster yang terbentuk cukup baik, dengan data dalam klaster cenderung lebih mirip satu sama lain dan terpisah dengan cukup jelas dari klaster lainnya.

---

### Kesimpulan:
- **Fuzzy Partition Coefficient (FPC) = 0.8561**: Klaster cukup terpisah, meskipun mungkin ada sedikit tumpang tindih.
- **Sum of Squared Errors (SSE) = 0.1001**: Klaster cukup homogen, dengan data berada dekat dengan pusat klaster.
- **Silhouette Score = 0.6807**: Klaster terbentuk dengan baik, dengan pemisahan yang jelas antara klaster.

Secara keseluruhan, hasil ini menunjukkan bahwa **klaster yang terbentuk cukup baik** dengan pemisahan yang jelas dan kesalahan yang relatif kecil. Namun, masih ada potensi untuk perbaikan, terutama dalam hal pemisahan klaster yang lebih jelas dan pengurangan tumpang tindih antara klaster.

        """
    )