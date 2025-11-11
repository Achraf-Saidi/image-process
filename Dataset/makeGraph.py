import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def plot_curves(history, out_png):
    plt.figure()
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=180); plt.close()

def plot_confusion(cm, class_names, out_png, normalize=False):
    """
    cm: (C,C) confusion matrix (rows = GT, cols = Pred)
    normalize: if True, normalize by GT row
    """
    cm = cm.astype(np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True) + 1e-9
        cm_disp = cm / row_sum
    else:
        cm_disp = cm

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm_disp, interpolation='nearest')
    plt.title("Confusion matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    fmt = ".2f" if normalize else "d"

    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            txt = f"{cm_disp[i,j]:.2f}" if normalize else f"{int(cm_disp[i,j])}"
            plt.text(j, i, txt, ha="center", va="center", color="white" if cm_disp[i,j] > cm_disp.max()/2 else "black")

    plt.ylabel('GT'); plt.xlabel('Pred')
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=180); plt.close()

def plot_bars(values, labels, title, ylabel, out_png):
    plt.figure(figsize=(6.5, 4))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20, ha='right')
    plt.title(title); plt.ylabel(ylabel)
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=180); plt.close()

def plot_double_bars(a, b, labels, title, ylabel, legends, out_png):
    plt.figure(figsize=(6.5, 4))
    x = np.arange(len(labels))
    w = 0.38
    plt.bar(x - w/2, a, width=w, label=legends[0])
    plt.bar(x + w/2, b, width=w, label=legends[1])
    plt.xticks(x, labels, rotation=20, ha='right')
    plt.title(title); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=180); plt.close()
