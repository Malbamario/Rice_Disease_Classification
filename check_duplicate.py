import os
import hashlib
import cv2 # Digunakan untuk membaca gambar dan memastikan itu gambar yang valid (opsional, tapi bagus)
import numpy as np # Digunakan oleh cv2

def calculate_image_hash(filepath):
    """
    Menghitung hash MD5 dari konten biner sebuah file gambar.
    Ini efektif untuk mendeteksi duplikat byte-per-byte.
    """
    hasher = hashlib.md5()
    try:
        # Membaca file dalam mode biner
        with open(filepath, 'rb') as f:
            # Membaca file dalam potongan-potongan untuk efisiensi memori
            while chunk := f.read(8192): # Membaca 8KB sekaligus
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error membaca atau menghitung hash untuk {filepath}: {e}")
        return None

def checking(image1, image2):

    try:
        img1 = cv2.imread(image1)
        img2 = cv2.imread(image2)
        if img1 is None:
            print(f"Warning: File {image1} bukan gambar yang valid atau rusak. Lewati.")
        if img2 is None:
            print(f"Warning: File {image2} bukan gambar yang valid atau rusak. Lewati.")
    except Exception as e:
        print(f"Warning: Gagal memverifikasi dengan OpenCV: {e}. Lewati.")
    
    hash1 = calculate_image_hash(image1)
    hash2 = calculate_image_hash(image2)

    if hash1 == hash2:
        print(f"Hash sama: {image1} dan {image2}")
    else:
        print(f"Hash tidak sama: {image1} dan {image2}")

# --- Cara Penggunaan ---
if __name__ == "__main__":
    image1 = f"datasets/local_dataset_no_duplicate/Bacterialblight/BACTERAILBLIGHT3_001.jpg" # Ganti dengan path folder dataset hasil augmentasi
    image2 = f"datasets/local_dataset_no_duplicate/Bacterialblight/BACTERAILBLIGHT3_015.jpg" # Ganti dengan path folder dataset hasil augmentasi
    
    checking(image1, image2)