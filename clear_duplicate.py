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

def find_and_remove_duplicates_in_folder(folder_path):
    """
    Mencari dan menghapus gambar duplikat di dalam folder yang sama.
    Akan menyimpan satu salinan asli dan menghapus sisanya.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' tidak ditemukan.")
        return

    print(f"Memulai pencarian duplikat di: {folder_path}")

    hashes_seen = {}  # Dictionary untuk menyimpan hash: [list_of_filepaths]
    total_files_scanned = 0
    duplicates_found_count = 0
    files_deleted_count = 0

    # Mendukung format gambar umum
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    # Tahap 1: Memindai semua file dan menghitung hash
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                filepath = os.path.join(root, filename)
                total_files_scanned += 1
                
                # Opsional: Memverifikasi apakah file adalah gambar yang valid menggunakan OpenCV
                # Ini bisa membantu menghindari error jika ada file non-gambar dengan ekstensi gambar
                try:
                    img = cv2.imread(filepath)
                    if img is None:
                        print(f"Warning: File {filepath} bukan gambar yang valid atau rusak. Lewati.")
                        continue
                except Exception as e:
                    print(f"Warning: Gagal memverifikasi {filepath} dengan OpenCV: {e}. Lewati.")
                    continue

                file_hash = calculate_image_hash(filepath)

                if file_hash:
                    if file_hash in hashes_seen:
                        hashes_seen[file_hash].append(filepath)
                        duplicates_found_count += 1
                    else:
                        hashes_seen[file_hash] = [filepath]
    
    print(f"\nSelesai memindai {total_files_scanned} file.")
    print(f"Ditemukan {duplicates_found_count} potensi duplikat (termasuk salinan asli).")

    # Tahap 2: Menghapus file duplikat
    print("\nMemulai penghapusan duplikat...")
    for file_list in hashes_seen.values():
        if len(file_list) > 1:
            # Jika ada lebih dari satu file dengan hash yang sama, berarti ada duplikat
            original_file = file_list[0] # Simpan file pertama sebagai 'original'
            print(f"\nDuplikat untuk hash {calculate_image_hash(original_file)} (Original: {original_file}):")
            for i in range(1, len(file_list)):
                duplicate_file = file_list[i]
                try:
                    os.remove(duplicate_file)
                    print(f"  Dihapus: {duplicate_file}")
                    files_deleted_count += 1
                except OSError as e:
                    print(f"  Error menghapus {duplicate_file}: {e}")
    
    print("\n--- Proses Selesai ---")
    print(f"Total file yang dipindai: {total_files_scanned}")
    print(f"Total duplikat yang ditemukan: {duplicates_found_count}")
    print(f"Total file duplikat yang dihapus: {files_deleted_count}")

# --- Cara Penggunaan ---
if __name__ == "__main__":
    # Ganti 'path/to/your/dataset_folder' dengan path folder dataset kamu
    # Misalnya: 'E:/Backup Mario/Coding Room/Rice_Disease/train/brownspot'
    # Atau jika ingin memindai seluruh dataset_padi_augmented, berikan path root-nya
    
    # Contoh penggunaan untuk satu folder kelas:
    # folder_to_clean = r"E:\Backup Mario\Coding Room\Rice_Disease\brownspot" # Ganti dengan path folder yang ingin kamu bersihkan
    # find_and_remove_duplicates_in_folder(folder_to_clean)

    # Contoh penggunaan untuk semua folder kelas di dalam train, val, test
    # Ini akan mengiterasi melalui setiap subfolder kelas
    dataset_dir = f"datasets\local_dataset_no_duplicate" # Ganti dengan path folder dataset hasil augmentasi
    
    print(f"Memindai dan menghapus duplikat di seluruh struktur: {dataset_dir}")
    
    if os.path.isdir(dataset_dir):
        for class_name in os.listdir(dataset_dir):
            class_folder_path = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_folder_path):
                print(f"\n--- Memproses folder: {class_folder_path} ---")
                find_and_remove_duplicates_in_folder(class_folder_path)
    else:
        print(f"Folder dataset tidak ditemukan di {dataset_dir}. Lewati.")