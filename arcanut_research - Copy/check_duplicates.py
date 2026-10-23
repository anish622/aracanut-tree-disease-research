import os
import hashlib

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

train_path = r"C:\Users\ASUS\Downloads\archive (5)\dataset\train"
test_path = r"C:\Users\ASUS\Downloads\archive (5)\dataset\test"

train_hashes = {}
for class_folder in os.listdir(train_path):
    folder = os.path.join(train_path, class_folder)
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            train_hashes[get_file_hash(fpath)] = fname

duplicates_found = 0
for class_folder in os.listdir(test_path):
    folder = os.path.join(test_path, class_folder)
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            h = get_file_hash(fpath)
            if h in train_hashes:
                print(f"⚠️ EXACT DUPLICATE: {fname} (test) == {train_hashes[h]} (train)")
                duplicates_found += 1

print(f"\nTotal exact duplicates between train/test: {duplicates_found}")