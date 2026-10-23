

import os
import sys
import time
import hashlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import (
    ResNet50, Xception, InceptionV3, DenseNet121, MobileNetV2, EfficientNetB0,
)
from tensorflow.keras.applications import (
    resnet50, xception, inception_v3, densenet, mobilenet_v2, efficientnet,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report


ORIGINAL_TRAIN = r"C:\Users\ASUS\Downloads\archive (5)\dataset\train"
ORIGINAL_TEST = r"C:\Users\ASUS\Downloads\archive (5)\dataset\test"
OUT_DIR = "grouped_results_v2"   

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
PATIENCE = 5
N_FOLDS = 5          
TEST_SPLITS = 5      
SEED = 42
SAVE_MODELS = "--save-models" in sys.argv

MODELS = {
    "ResNet50":       (ResNet50,       resnet50.preprocess_input),
    "Xception":       (Xception,       xception.preprocess_input),
    "InceptionV3":    (InceptionV3,    inception_v3.preprocess_input),
    "DenseNet121":    (DenseNet121,    densenet.preprocess_input),
    "MobileNetV2":    (MobileNetV2,    mobilenet_v2.preprocess_input),
    "EfficientNetB0": (EfficientNetB0, efficientnet.preprocess_input),
}

if "--models" in sys.argv:
    i = sys.argv.index("--models")
    names = []
    for a in sys.argv[i + 1:]:
        if a.startswith("--"):
            break
        names.append(a)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"Unknown model(s) {unknown}. Choose from {list(MODELS)}")
    MODELS = {n: MODELS[n] for n in names}

if "--quick" in sys.argv:
    if "--models" not in sys.argv:
        MODELS = {"MobileNetV2": MODELS["MobileNetV2"]}
    N_FOLDS = 2
    EPOCHS = 2
    OUT_DIR = "grouped_results_quick_v2"
    print(">>> QUICK TEST MODE (2 folds, 2 epochs) - numbers are NOT final <<<")

AUTOTUNE = tf.data.AUTOTUNE
os.makedirs(OUT_DIR, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)



def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_source_id(filename):
   
    if "_original_" in filename:
        return filename.split("_original_")[1].split(".jpg")[0]   
    return filename


all_classes = sorted(os.listdir(ORIGINAL_TRAIN))
class_to_idx = {name: i for i, name in enumerate(all_classes)}
num_classes = len(all_classes)

paths, labels, groups = [], [], []
seen_hashes = set()
for class_name in all_classes:
    for base in (ORIGINAL_TRAIN, ORIGINAL_TEST):
        folder = os.path.join(base, class_name)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            h = get_file_hash(fpath)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            paths.append(fpath)
            labels.append(class_to_idx[class_name])
            groups.append(f"{class_name}::{get_source_id(fname)}")

paths = np.array(paths)
labels = np.array(labels)
groups = np.array(groups)

print(f"Unique images: {len(paths)} | classes: {all_classes}")
print(f"Unique source-photo groups: {len(set(groups))}")
for i, c in enumerate(all_classes):
    print(f"  class {c}: {int((labels == i).sum())} images, "
          f"{len(set(groups[labels == i]))} source groups")

with open(os.path.join(OUT_DIR, "class_names.txt"), "w") as f:
    for i, c in enumerate(all_classes):
        f.write(f"{i}\t{c}\n")


outer = StratifiedGroupKFold(n_splits=TEST_SPLITS, shuffle=True, random_state=SEED)
dev_idx, test_idx = next(outer.split(paths, labels, groups))
print(f"\nTrain+val pool: {len(dev_idx)} images | held-out test: {len(test_idx)} images")

cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(cv.split(dev_idx, labels[dev_idx], groups[dev_idx]))


def assert_no_group_overlap(**named_idx):
    names = list(named_idx)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = set(groups[named_idx[names[i]]])
            b = set(groups[named_idx[names[j]]])
            assert not (a & b), f"LEAK: {names[i]} and {names[j]} share source photos!"



def make_ds(p, l, preprocess, training=False):
    ds = tf.data.Dataset.from_tensor_slices((p, l))

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return preprocess(img), label

    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(500, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_model(ctor):
    base = ctor(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def evaluate(model, ds, y_true):
    probs = model.predict(ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = float((y_pred == y_true).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return acc, f1, y_pred


def save_training_curves(history, model_name, out_png):
    h = history.history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h["accuracy"], label="Train Accuracy")
    plt.plot(h["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch"); plt.legend(); plt.title(f"{model_name} Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(h["loss"], label="Train Loss")
    plt.plot(h["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.legend(); plt.title(f"{model_name} Loss")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_confusion_matrix(y_true, y_pred, model_name, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=all_classes, yticklabels=all_classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix (held-out test set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_gradcam(model, preprocess, model_name, out_png):
    """Grad-CAM on the last feature map of the base network.
    Uses one held-out TEST image per class."""
    base = model.layers[0]                       
    x = base.output                             
    for layer in model.layers[1:]:               
        x = layer(x)
    grad_model = tf.keras.Model(base.input, [base.output, x])

    sample = []
    for c in range(num_classes):
        ids = test_idx[labels[test_idx] == c]
        if len(ids):
            sample.append(ids[0])

    cols = 3
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.3 * rows))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for ax, i in zip(axes, sample):
        raw = tf.image.decode_jpeg(tf.io.read_file(paths[i]), channels=3)
        raw = tf.image.resize(raw, IMG_SIZE)
        arr = tf.expand_dims(preprocess(tf.identity(raw)), 0)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(arr)
            pred_index = tf.argmax(preds[0])
            score = preds[:, pred_index]
        grads = tape.gradient(score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heat = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
        heat = tf.maximum(heat, 0) / (tf.math.reduce_max(heat) + 1e-8)
        heat = cv2.resize(heat.numpy(), IMG_SIZE)
        heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        original = raw.numpy().astype("uint8")
        overlay = np.uint8(original * 0.6 + heat * 0.4)
        ax.imshow(overlay)
        ax.set_title(f"True: {all_classes[labels[i]]} | Pred: {all_classes[int(pred_index)]}",
                     fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{model_name} Grad-CAM (held-out test images)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()



per_fold_csv = os.path.join(OUT_DIR, "results_per_fold.csv")
results = pd.read_csv(per_fold_csv).to_dict("records") if os.path.exists(per_fold_csv) else []
done = {(r["model"], int(r["fold"])) for r in results}

final_csv = os.path.join(OUT_DIR, "final_models_test.csv")
final_rows = pd.read_csv(final_csv).to_dict("records") if os.path.exists(final_csv) else []
final_done = {r["model"] for r in final_rows}

for model_name, (ctor, preprocess) in MODELS.items():
    mdir = os.path.join(OUT_DIR, model_name)
    os.makedirs(mdir, exist_ok=True)

  
    for fold, (tr_rel, ho_rel) in enumerate(folds, start=1):
        if (model_name, fold) in done:
            print(f"Skipping {model_name} fold {fold} (already done)")
            continue

        tr_idx = dev_idx[tr_rel]
        ho_idx = dev_idx[ho_rel]

        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        fit_rel, val_rel = next(inner.split(tr_idx, labels[tr_idx], groups[tr_idx]))
        fit_idx, val_idx = tr_idx[fit_rel], tr_idx[val_rel]

        assert_no_group_overlap(fit=fit_idx, val=val_idx, fold=ho_idx, test=test_idx)

        print(f"\n===== {model_name} | CV fold {fold}/{N_FOLDS} | "
              f"fit {len(fit_idx)} val {len(val_idx)} fold-test {len(ho_idx)} =====")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + fold)

        fit_ds = make_ds(paths[fit_idx], labels[fit_idx], preprocess, training=True)
        val_ds = make_ds(paths[val_idx], labels[val_idx], preprocess)
        ho_ds = make_ds(paths[ho_idx], labels[ho_idx], preprocess)
        test_ds = make_ds(paths[test_idx], labels[test_idx], preprocess)

        model = build_model(ctor)
        t0 = time.time()
        model.fit(fit_ds, validation_data=val_ds, epochs=EPOCHS,
                  callbacks=[EarlyStopping(monitor="val_accuracy", patience=PATIENCE,
                                           restore_best_weights=True)],
                  verbose=2)
        secs = time.time() - t0

        cv_acc, cv_f1, _ = evaluate(model, ho_ds, labels[ho_idx])
        test_acc, test_f1, _ = evaluate(model, test_ds, labels[test_idx])
        print(f"{model_name} fold {fold}: CV acc {cv_acc:.4f} | "
              f"held-out test acc {test_acc:.4f} | {secs/60:.1f} min")

        results.append({"model": model_name, "fold": fold,
                        "cv_acc": cv_acc, "cv_macro_f1": cv_f1,
                        "test_acc": test_acc, "test_macro_f1": test_f1,
                        "train_minutes": round(secs / 60, 2)})
        pd.DataFrame(results).to_csv(per_fold_csv, index=False)


    if model_name in final_done:
        print(f"Skipping final model for {model_name} (already done)")
        continue

    print(f"\n===== {model_name} | FINAL model + curves / confusion matrix / Grad-CAM =====")
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_rel, val_rel = next(inner.split(dev_idx, labels[dev_idx], groups[dev_idx]))
    fit_idx, val_idx = dev_idx[fit_rel], dev_idx[val_rel]
    assert_no_group_overlap(fit=fit_idx, val=val_idx, test=test_idx)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    fit_ds = make_ds(paths[fit_idx], labels[fit_idx], preprocess, training=True)
    val_ds = make_ds(paths[val_idx], labels[val_idx], preprocess)
    test_ds = make_ds(paths[test_idx], labels[test_idx], preprocess)

    model = build_model(ctor)
    history = model.fit(fit_ds, validation_data=val_ds, epochs=EPOCHS,
                        callbacks=[EarlyStopping(monitor="val_accuracy", patience=PATIENCE,
                                                 restore_best_weights=True)],
                        verbose=2)

    y_true = labels[test_idx]
    acc, f1, y_pred = evaluate(model, test_ds, y_true)
    print(f"{model_name} FINAL held-out test accuracy: {acc:.4f} | macro-F1: {f1:.4f}")

    save_training_curves(history, model_name, os.path.join(mdir, "training_curves.png"))
    save_confusion_matrix(y_true, y_pred, model_name, os.path.join(mdir, "confusion_matrix.png"))
    with open(os.path.join(mdir, "classification_report.txt"), "w") as f:
        f.write(f"{model_name} - held-out test set ({len(y_true)} images)\n")
        f.write(f"Accuracy: {acc*100:.2f}%   Macro-F1: {f1*100:.2f}%\n\n")
        f.write(classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                      target_names=all_classes, zero_division=0))
    save_gradcam(model, preprocess, model_name, os.path.join(mdir, "gradcam.png"))
    if SAVE_MODELS:
        model.save(os.path.join(mdir, f"{model_name}_final.keras"))

    final_rows.append({"model": model_name, "final_test_acc": acc, "final_test_macro_f1": f1})
    pd.DataFrame(final_rows).to_csv(final_csv, index=False)
    print(f"Saved figures and report to {os.path.abspath(mdir)}")


df = pd.DataFrame(results)
rows = []
for name, g in df.groupby("model"):
    rows.append({
        "Model": name,
        "Folds": len(g),
        "CV Acc (mean±std %)": f"{g.cv_acc.mean()*100:.2f} ± {g.cv_acc.std(ddof=0)*100:.2f}",
        "CV Macro-F1 (%)": f"{g.cv_macro_f1.mean()*100:.2f} ± {g.cv_macro_f1.std(ddof=0)*100:.2f}",
        "Held-out Test Acc (%)": f"{g.test_acc.mean()*100:.2f} ± {g.test_acc.std(ddof=0)*100:.2f}",
        "Held-out Test Macro-F1 (%)": f"{g.test_macro_f1.mean()*100:.2f} ± {g.test_macro_f1.std(ddof=0)*100:.2f}",
        "_sort": g.cv_acc.mean(),
    })
summary = (pd.DataFrame(rows).sort_values("_sort", ascending=False)
           .drop(columns="_sort").reset_index(drop=True))
summary.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)

print("\n" + "=" * 100)
print("FINAL RESULTS (grouped split - no source photo shared between train/val/test)")
print("=" * 100)
print(summary.to_string(index=False))
print(f"\nSaved to: {os.path.abspath(OUT_DIR)}")
