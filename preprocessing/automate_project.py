"""
automate_Nama-siswa.py
Script otomatisasi preprocessing dataset Diabetes (Pima Indians)
untuk submission Dicoding - Membangun Sistem Machine Learning

Author: Nama-siswa
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import sys


def preprocess_diabetes(raw_path: str, output_path: str) -> pd.DataFrame:
    """
    Fungsi otomatis preprocessing dataset Diabetes.
    
    Parameters:
    -----------
    raw_path : str
        Path ke file dataset mentah (CSV)
    output_path : str
        Path untuk menyimpan dataset hasil preprocessing
    
    Returns:
    --------
    pd.DataFrame
        DataFrame yang sudah siap digunakan untuk training model
    """
    print(f"🔄 Memuat dataset dari: {raw_path}")
    
    # 1. Load dataset
    df = pd.read_csv(raw_path)
    
    # Validasi kolom wajib
    required_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    if not all(col in df.columns for col in required_cols):
        # Jika file tanpa header, tambahkan nama kolom
        df = pd.read_csv(raw_path, names=required_cols)
    
    print(f"✅ Dataset dimuat. Shape: {df.shape}")
    
    # 2. Identifikasi fitur medis yang mengandung nilai 0 invalid
    medical_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # 3. Handle Missing Values (0 pada fitur medis dianggap missing)
    df[medical_cols] = df[medical_cols].replace(0, np.nan)
    missing_before = df[medical_cols].isnull().sum().sum()
    print(f"✅ Nilai 0 pada fitur medis diubah menjadi NaN (total: {missing_before})")
    
    # 4. Imputasi Missing Values dengan Median
    imputer = SimpleImputer(strategy='median')
    df[medical_cols] = imputer.fit_transform(df[medical_cols])
    print("✅ Missing values diimputasi menggunakan Median")
    
    # 5. Hapus Data Duplikat
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"✅ {dup_count} data duplikat dihapus")
    else:
        print("✅ Tidak ditemukan data duplikat")
    
    # 6. Deteksi & Penanganan Outlier (IQR Method - Capping)
    def cap_outliers(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Handle outlier menggunakan metode IQR capping"""
        df_out = dataframe.copy()
        for col in columns:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)
        return df_out
    
    df = cap_outliers(df, medical_cols)
    print("✅ Outlier ditangani dengan metode IQR Capping")
    
    # 7. Encoding Data Kategorikal (jika ada)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols and 'Outcome' not in categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"✅ Encoding dilakukan untuk kolom: {categorical_cols}")
    else:
        print("✅ Tidak ada fitur kategorikal → Encoding dilewati")
    
    # 8. Standarisasi Fitur Numerik (kecuali target 'Outcome')
    feature_cols = [col for col in df.columns if col != 'Outcome']
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("✅ Fitur dinormalisasi menggunakan StandardScaler")
    
    # 9. Simpan hasil preprocessing
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"📦 Dataset hasil preprocessing disimpan: {output_path}")
    
    return df


def main():
    """Entry point untuk menjalankan script secara standalone"""
    
    # 🔥 PENTING: Dapatkan path absolut dari lokasi script ini
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Naik satu level ke root project
    
    # Path default (otomatis menyesuaikan struktur folder)
    default_raw = os.path.join(project_root, "diabetes_raw", "diabetes.csv")
    default_output = os.path.join(script_dir, "diabetes_preprocessing", "diabetes_preprocessed.csv")
    
    # Allow override via command line arguments
    raw_path = sys.argv[1] if len(sys.argv) > 1 else default_raw
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    # Validasi file input
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"❌ Dataset mentah tidak ditemukan: {raw_path}\n"
                               f"Path yang dicari: {os.path.abspath(raw_path)}")
    
    print(f"📂 Script location: {script_dir}")
    print(f"📂 Project root: {project_root}")
    print(f"📂 Input path: {os.path.abspath(raw_path)}")
    print(f"📂 Output path: {os.path.abspath(output_path)}")
    print("-" * 60)
    
    # Jalankan preprocessing
    try:
        result_df = preprocess_diabetes(raw_path, output_path)
        print(f"\n🎉 Preprocessing selesai! Baris hasil: {len(result_df)}")
        print(f"📊 Kolom: {result_df.columns.tolist()}")
        return 0
    except Exception as e:
        print(f"❌ Error saat preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)