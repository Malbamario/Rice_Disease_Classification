import os
import time
import pickle
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

def normalize_image(image):
    nmax = 255 #New maximum
    nmin = 0 #New minimum
    return cv2.normalize(image,None,alpha = nmin,beta = nmax,norm_type = cv2.NORM_MINMAX)

def pre_process_image(image_np_array):
    img_resized = cv2.resize(image_np_array, IMG_SIZE)
    return img_resized
    # return normalize_image(img_resized)

def test_model_performance(test_data_dir, model_path, scaler_path, classes):
    """
    Menguji model SVM yang telah disimpan dan menghitung performa serta durasi.

    Args:
        test_data_dir (str): Path ke folder yang berisi gambar untuk diuji.
        model_path (str): Path ke file model SVM yang disimpan.
        scaler_path (str): Path ke file scaler (StandardScaler) yang disimpan.
        classes (list): Daftar nama kelas (string).
    """
    print("--- Memulai Pengujian Performa Model ---")
    start_total_time = time.time()

    # --- 1. Memuat Model dan Scaler ---
    print(f"\n[{time.strftime('%H:%M:%S', time.localtime())}] Memuat feature extractor, model dan scaler...")
    try:
        feature_extractor = tf.keras.models.load_model(EXTRACTOR_PATH, compile=False)
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Pastikan path model dan scaler sudah benar.")
        return
    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] Model dan scaler berhasil dimuat.")

    # --- 2. Mengumpulkan Path dan Label Gambar Uji ---
    X_test_paths = []
    y_true_labels = []
    for class_name in classes:
        class_path = os.path.join(test_data_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"Peringatan: Folder '{class_path}' tidak ditemukan. Lewati.")
            continue
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                X_test_paths.append(os.path.join(class_path, img_name))
                y_true_labels.append(class_name)
    
    if not X_test_paths:
        print("Error: Tidak ada gambar yang ditemukan di direktori uji.")
        return

    # --- 3. Melakukan Prediksi dan Mengukur Waktu per Tahap ---
    y_pred_labels = []
    
    total_time_load_img = 0
    total_time_extract_feat = 0
    total_time_scale_feat = 0
    total_time_predict = 0

    print(f"\n[{time.strftime('%H:%M:%S', time.localtime())}] Memulai loop prediksi untuk {len(X_test_paths)} gambar...")

    for i, img_path in enumerate(X_test_paths):
        # Memuat gambar
        start_stage = time.time()
        img = cv2.imread(img_path)
        end_stage = time.time()
        total_time_load_img += (end_stage - start_stage)
        
        # Preprocessing dan ekstraksi fitur
        start_stage = time.time()
        preprossed_img = pre_process_image(img)
        # print(preprossed_img.shape)
        img_processed = np.expand_dims(preprossed_img, axis=0)
        # img_processed = tf.keras.applications.mobilenet_v2.preprocess_input(img_processed)
        features = feature_extractor.predict(img_processed, verbose=0)
        end_stage = time.time()
        total_time_extract_feat += (end_stage - start_stage)
        
        # Scaling fitur
        start_stage = time.time()
        # Perhatikan: reshape(-1, 1) jika hanya 1 fitur, atau [features] jika banyak fitur
        # scaler.transform membutuhkan input 2D
        scaled_features = scaler.transform(features)
        end_stage = time.time()
        total_time_scale_feat += (end_stage - start_stage)
        
        # Prediksi
        start_stage = time.time()
        prediction = model.predict(scaled_features)
        y_pred_labels.append(prediction[0])
        end_stage = time.time()
        total_time_predict += (end_stage - start_stage)

        if (i + 1) % 100 == 0 or (i + 1) == len(X_test_paths):
            print(f"[{time.strftime('%H:%M:%S', time.localtime())}] Selesai memproses {i + 1}/{len(X_test_paths)} gambar.")

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time

    # --- 4. Menampilkan Hasil ---
    print("\n--- Hasil Evaluasi ---")
    print(type(y_pred_labels[0]))
    print(type(y_true_labels[0]))
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    print(f"Akurasi Prediksi Keseluruhan: {accuracy:.4f}")

    print("\n--- Ringkasan Durasi Waktu ---")
    print(f"Total Waktu Keseluruhan: {total_duration:.4f} detik")
    print(f"Total Gambar Dites: {len(X_test_paths)}")
    print(f"\nDurasi Rata-rata per Gambar:")
    print(f"  - Memuat Gambar: {total_time_load_img / len(X_test_paths):.6f} detik")
    print(f"  - Ekstraksi Fitur: {total_time_extract_feat / len(X_test_paths):.6f} detik")
    print(f"  - Scaling Fitur: {total_time_scale_feat / len(X_test_paths):.6f} detik")
    print(f"  - Prediksi Model: {total_time_predict / len(X_test_paths):.6f} detik")
    
    # print("\n*Botttleneck* terbesar kemungkinan berada pada tahap Ekstraksi Fitur atau Memuat Gambar.")

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    IMG_SIZE = (227, 227)
    TEST_DATA_DIR = f"E:/Backup Mario/Coding Room/Rice_Disease/datasets/result/test_images"
    EXTRACTOR_PATH = 'E:/Backup Mario/Coding Room/Rice_Disease/Results/250826 2156 Mixed On Field No Duplicate Normal Indo/mobilenet_feature_extractor.h5'
    MODEL_PATH = f"E:/Backup Mario/Coding Room/Rice_Disease/Results/250826 2156 Mixed On Field No Duplicate Normal Indo/svm_model.pkl"
    SCALER_PATH = f"E:/Backup Mario/Coding Room/Rice_Disease/Results/250826 2156 Mixed On Field No Duplicate Normal Indo/scaler.pkl"
    CLASSES = ['bacterial_leaf_blight', 'blast', 'brown_spot', 'normal', 'tungro']
    
    test_model_performance(TEST_DATA_DIR, MODEL_PATH, SCALER_PATH, CLASSES)
