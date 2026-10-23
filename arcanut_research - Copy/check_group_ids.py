
import os

TRAIN = r"C:\Users\ASUS\Downloads\archive (5)\dataset\train"
TEST = r"C:\Users\ASUS\Downloads\archive (5)\dataset\test"


def raw_source(fn):
    if "_original_" in fn:
        return fn.split("_original_")[1].split(".jpg")[0]
    return fn


for c in sorted(os.listdir(TRAIN)):
    names = []
    for base in (TRAIN, TEST):
        d = os.path.join(base, c)
        if os.path.isdir(d):
            names += sorted(os.listdir(d))
    ids = [raw_source(n) for n in names]
    print(f"\nClass {c}: {len(names)} files")
    for n in names[:3]:
        print("   example file:", n)
    for L in (15, 25, 40, None):
        groups = set(i[:L] if L else i for i in ids)
        print(f"   groups if first {L if L else 'ALL'} chars are used: {len(groups)}")