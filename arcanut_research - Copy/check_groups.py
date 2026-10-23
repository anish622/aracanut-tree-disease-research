import os
import hashlib
import sys
import numpy as np
from collections import defaultdict

original_train = r"C:\Users\ASUS\Downloads\archive (5)\dataset\train"
original_test = r"C:\Users\ASUS\Downloads\archive (5)\dataset\test"

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_source_id(filename):
    if "_original_" in filename:
        return filename.split("_original_")[1].split(".jpg")[0][:15]
    return filename

all_classes = sorted(os.listdir(original_train))
class_to_idx = {name: i for i, name in enumerate(all_classes)}

filepaths, labels, groups = [], [], []
seen_hashes = set()

for class_name in all_classes:
    for base_folder in [original_train, original_test]:
        class_folder = os.path.join(base_folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            fpath = os.path.join(class_folder, fname)
            h = get_file_hash(fpath)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            filepaths.append(fpath)
            labels.append(class_to_idx[class_name])
            groups.append(get_source_id(fname))

filepaths = np.array(filepaths)
labels = np.array(labels)
groups = np.array(groups)

print(f"Total unique images: {len(filepaths)}")
print(f"Classes: {all_classes}")


class_group_counts = defaultdict(set)
for lbl, grp in zip(labels, groups):
    class_group_counts[all_classes[lbl]].add(grp)

print("\nUnique source-photo groups per class:")
for cname in all_classes:
    n_groups = len(class_group_counts[cname])
    n_images = int((labels == class_to_idx[cname]).sum())
    print(f"  {cname}: {n_groups} unique groups, {n_images} total images")

sys.exit("Diagnostic only")