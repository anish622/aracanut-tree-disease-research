import os
import shutil
import hashlib
from collections import defaultdict
import random


original_train = r"C:\Users\ASUS\Downloads\archive (5)\dataset\train"
original_test = r"C:\Users\ASUS\Downloads\archive (5)\dataset\test"
output_path = r"C:\Users\ASUS\Downloads\archive (5)\dataset_clean"  

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_source_id(filename):
    
    if "_original_" in filename:
        return filename.split("_original_")[1].split(".jpg")[0][:15]
    return filename


all_classes = sorted(os.listdir(original_train))
class_to_sources = {}  

seen_hashes = set()

for class_name in all_classes:
    source_groups = defaultdict(list)
    for base_folder in [original_train, original_test]:
        class_folder = os.path.join(base_folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            fpath = os.path.join(class_folder, fname)
            file_hash = get_file_hash(fpath)
            if file_hash in seen_hashes:
                continue  # skip exact duplicate
            seen_hashes.add(file_hash)
            source_id = get_source_id(fname)
            source_groups[source_id].append(fpath)
    class_to_sources[class_name] = source_groups


for split in ['train', 'val', 'test']:
    for class_name in all_classes:
        os.makedirs(os.path.join(output_path, split, class_name), exist_ok=True)

total_copied = {'train': 0, 'val': 0, 'test': 0}


for class_name, source_groups in class_to_sources.items():
    source_ids = list(source_groups.keys())
    random.shuffle(source_ids)
    source_ids.sort(key=lambda sid: len(source_groups[sid]), reverse=True)

    total_images = sum(len(v) for v in source_groups.values())
    target = {
        'train': total_images * TRAIN_RATIO,
        'val': total_images * VAL_RATIO,
        'test': total_images * TEST_RATIO
    }
    assigned_count = {'train': 0, 'val': 0, 'test': 0}
    split_sources = {'train': [], 'val': [], 'test': []}

    for sid in source_ids:
        cnt = len(source_groups[sid])
        deficits = {s: target[s] - assigned_count[s] for s in ['train', 'val', 'test']}
        best_split = max(deficits, key=deficits.get)
        split_sources[best_split].append(sid)
        assigned_count[best_split] += cnt

    for split in ['train', 'val', 'test']:
        for source_id in split_sources[split]:
            for fpath in source_groups[source_id]:
                dest = os.path.join(output_path, split, class_name, os.path.basename(fpath))
                shutil.copy(fpath, dest)
                total_copied[split] += 1

print(" Done rebuilding dataset with no leakage.")
print(f"Train images: {total_copied['train']}")
print(f"Val images: {total_copied['val']}")
print(f"Test images: {total_copied['test']}")