"""
=============================================================================
  DEEP LEARNING WITH CNNs & TRANSFER LEARNING — COMPLETE SOLUTION
  Dataset : CIFAR-10  (10 classes, 60 000 images, 32×32 → resized 224×224)
  Models  : Custom CNN | ResNet-50 | DenseNet-121 | EfficientNet-B0
  Author  : <Your Name>
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, time, copy, json, warnings
import numpy  as np
import matplotlib
matplotlib.use("Agg")                      # headless rendering
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import seaborn           as sns

from PIL import Image

import torch
import torch.nn          as nn
import torch.optim       as optim
import torch.nn.functional as F
from torch.utils.data   import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models     as models
from torchvision.datasets     import CIFAR10

from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, top_k_accuracy_score,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    device      = "cuda" if torch.cuda.is_available() else "cpu",
    num_classes = 10,
    batch_size  = 64,
    epochs      = 10,           # increase for better accuracy
    lr          = 1e-3,
    img_size    = 224,
    seed        = 42,
    data_root   = "./data",
    out_dir     = "./results",
    freeze_epochs = 5,          # epochs to freeze backbone in transfer models
)

os.makedirs(CFG["out_dir"], exist_ok=True)
DEVICE = torch.device(CFG["device"])

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck",
]

print(f"[INFO]  Running on : {DEVICE}")
print(f"[INFO]  PyTorch    : {torch.__version__}")
print(f"[INFO]  Output dir : {CFG['out_dir']}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING  (CIFAR-10 → resized to 224×224)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tfm = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tfm = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Build datasets  ─  apply different transforms without duplicating downloads
full_train_aug  = CIFAR10(CFG["data_root"], train=True,  download=True,  transform=train_tfm)
full_train_eval = CIFAR10(CFG["data_root"], train=True,  download=False, transform=eval_tfm)
test_ds         = CIFAR10(CFG["data_root"], train=False, download=True,  transform=eval_tfm)

# Reproducible 85 / 15 split (train / val) of the 50 000-image training set
rng    = torch.Generator().manual_seed(CFG["seed"])
all_idx = list(range(len(full_train_aug)))
np.random.shuffle(all_idx)
val_size   = int(0.15 * len(all_idx))
train_idx  = all_idx[val_size:]
val_idx    = all_idx[:val_size]

train_ds = Subset(full_train_aug,  train_idx)   # augmented
val_ds   = Subset(full_train_eval, val_idx)     # no augmentation

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=0, pin_memory=False)

print(f"\n[DATA]  Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SAMPLE-GRID VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def show_samples(loader, save_path):
    imgs, lbls = next(iter(loader))
    imgs = imgs[:16]
    lbls = lbls[:16]
    inv  = transforms.Normalize(
        mean=[-m/s for m,s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std =[1/s   for s   in IMAGENET_STD])
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for ax, img, lbl in zip(axes.flat, imgs, lbls):
        ax.imshow(np.clip(inv(img).permute(1,2,0).numpy(), 0, 1))
        ax.set_title(CLASS_NAMES[lbl], fontsize=7)
        ax.axis("off")
    plt.suptitle("Sample Training Images", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  Saved → {save_path}")

show_samples(train_loader, f"{CFG['out_dir']}/sample_images.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BASELINE CNN FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    """
    Custom 6-block CNN with BatchNorm, Dropout, and Global Average Pooling.
    Architecture overview (input 224×224×3):
      Block-1 : Conv 32  → BN → ReLU → MaxPool  → 112×112
      Block-2 : Conv 64  → BN → ReLU → MaxPool  →  56×56
      Block-3 : Conv 128 → BN → ReLU → MaxPool  →  28×28
      Block-4 : Conv 256 → BN → ReLU → MaxPool  →  14×14
      Block-5 : Conv 512 → BN → ReLU → MaxPool  →   7×7
      Block-6 : Conv 512 → BN → ReLU → GAP      →   1×1
      Head    : FC 512 → Dropout(0.5) → FC num_classes
    """
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            self._block(  3,  32),   # 112
            self._block( 32,  64),   #  56
            self._block( 64, 128),   #  28
            self._block(128, 256),   #  14
            self._block(256, 512),   #   7
        )
        self.extra = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.extra(x)
        x = self.gap(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRANSFER-LEARNING MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_transfer_model(name: str, num_classes: int, freeze=True):
    """
    Returns (model, feature_params, head_params) for two-stage training.
    freeze=True : only the head is trainable initially.
    """
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze:
            for p in m.parameters(): p.requires_grad = False
        m.fc = nn.Sequential(
            nn.Linear(m.fc.in_features, 512),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
        head_params    = list(m.fc.parameters())
        feature_params = [p for n,p in m.named_parameters() if "fc" not in n]

    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        if freeze:
            for p in m.parameters(): p.requires_grad = False
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(
            nn.Linear(in_f, 512),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
        head_params    = list(m.classifier.parameters())
        feature_params = [p for n,p in m.named_parameters() if "classifier" not in n]

    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze:
            for p in m.parameters(): p.requires_grad = False
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
        head_params    = list(m.classifier.parameters())
        feature_params = [p for n,p in m.named_parameters() if "classifier" not in n]

    else:
        raise ValueError(f"Unknown model: {name}")

    return m, feature_params, head_params


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    """One forward pass over loader. Returns avg_loss, accuracy."""
    is_train = (phase == "train")
    model.train(is_train)
    total_loss, correct, total = 0., 0, 0

    with torch.set_grad_enabled(is_train):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, lbls)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == lbls).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def train_model(model, name, feature_params=None, head_params=None,
                is_transfer=False):
    """
    Full training loop with:
     - Two-stage transfer learning (freeze backbone → unfreeze)
     - CosineAnnealingLR scheduler
     - Best-model checkpointing
    Returns history dict.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {name.upper()}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if is_transfer:
        # Stage 1: only head
        optimizer = optim.AdamW(head_params, lr=CFG["lr"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"],
                                weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6)

    history   = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_acc  = 0.
    best_wts  = copy.deepcopy(model.state_dict())
    epoch_times = []

    for epoch in range(1, CFG["epochs"]+1):
        t0 = time.time()

        # ── Unfreeze backbone after freeze_epochs  ──────────────────────────
        if is_transfer and epoch == CFG["freeze_epochs"] + 1:
            print(f"  [Stage-2] Unfreezing backbone at epoch {epoch}")
            for p in model.parameters(): p.requires_grad = True
            optimizer = optim.AdamW(
                [{"params": feature_params, "lr": CFG["lr"]*0.1},
                 {"params": head_params,    "lr": CFG["lr"]}],
                weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CFG["epochs"] - CFG["freeze_epochs"], eta_min=1e-6)

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, "train")
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      "val")
        scheduler.step()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc:
            best_acc = vl_acc
            best_wts = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch:02d}/{CFG['epochs']}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.4f}/{vl_acc:.4f}  "
              f"t={elapsed:.1f}s")

    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), f"{CFG['out_dir']}/{name}_best.pth")
    history["avg_epoch_time"] = float(np.mean(epoch_times))
    history["best_val_acc"]   = best_acc
    print(f"\n  ✓ Best val acc: {best_acc:.4f}  avg epoch: {np.mean(epoch_times):.1f}s")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FULL EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, loader):
    """Returns y_true, y_pred, y_proba, inference_time_per_image."""
    model.eval()
    all_labels, all_preds, all_proba = [], [], []
    t0 = time.time()
    n  = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_labels.extend(lbls.numpy())
            all_preds.extend(preds)
            all_proba.extend(probs)
            n += len(imgs)

    inf_time = (time.time() - t0) / n * 1000   # ms/image
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_proba),
            inf_time)


def compute_metrics(y_true, y_pred, y_proba, name):
    """Compute & print all required metrics. Returns dict."""
    acc   = (y_true == y_pred).mean()
    prec  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec   = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    f1    = f1_score       (y_true, y_pred, average="macro", zero_division=0)
    top3  = top_k_accuracy_score(y_true, y_proba, k=3)
    top5  = top_k_accuracy_score(y_true, y_proba, k=5)
    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_NAMES, zero_division=0)

    print(f"\n── {name.upper()} ──────────────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  Top-3 Acc : {top3:.4f}")
    print(f"  Top-5 Acc : {top5:.4f}")
    print(f"\nPer-class report:\n{report}")

    return dict(name=name, accuracy=acc, precision=prec, recall=rec,
                f1=f1, top3=top3, top5=top5)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def plot_curves(history, name):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["train_loss"])+1)

    a1.plot(ep, history["train_loss"], label="Train", linewidth=2)
    a1.plot(ep, history["val_loss"],   label="Val",   linewidth=2)
    a1.set(title=f"{name} — Loss", xlabel="Epoch", ylabel="Loss")
    a1.legend(); a1.grid(alpha=.3)

    a2.plot(ep, history["train_acc"], label="Train", linewidth=2)
    a2.plot(ep, history["val_acc"],   label="Val",   linewidth=2)
    a2.set(title=f"{name} — Accuracy", xlabel="Epoch", ylabel="Accuracy")
    a2.legend(); a2.grid(alpha=.3)

    plt.suptitle(f"Training Curves — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{CFG['out_dir']}/curves_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  Curves saved → {path}")


def plot_confusion(y_true, y_pred, name):
    cm_arr = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=.5, ax=ax)
    ax.set(title=f"Confusion Matrix — {name}",
           xlabel="Predicted", ylabel="True")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    path = f"{CFG['out_dir']}/cm_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  CM saved → {path}")


def show_misclassified(model, loader, name, n_show=6):
    """Display n_show misclassified examples with true vs predicted labels."""
    model.eval()
    wrong_imgs, wrong_true, wrong_pred = [], [], []
    inv = transforms.Normalize(
        mean=[-m/s for m,s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std =[1/s   for s   in IMAGENET_STD])

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs_d = imgs.to(DEVICE)
            preds  = model(imgs_d).argmax(1).cpu()
            mask   = (preds != lbls)
            for i in mask.nonzero(as_tuple=True)[0]:
                wrong_imgs.append(inv(imgs[i]).permute(1,2,0).numpy().clip(0,1))
                wrong_true.append(lbls[i].item())
                wrong_pred.append(preds[i].item())
            if len(wrong_imgs) >= n_show:
                break

    n  = min(n_show, len(wrong_imgs))
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    if n == 1: axes = [axes]
    for ax, img, t, p in zip(axes, wrong_imgs[:n], wrong_true[:n], wrong_pred[:n]):
        ax.imshow(img)
        ax.set_title(f"True: {CLASS_NAMES[t]}\nPred: {CLASS_NAMES[p]}",
                     fontsize=8, color="red")
        ax.axis("off")
    plt.suptitle(f"Misclassified Examples — {name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = f"{CFG['out_dir']}/misclassified_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  Misclassified saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  GRAD-CAM  (applied to best transfer-learning model)
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM implementation for CNN models.
    target_layer: the last convolutional layer whose gradients are hooked.
    """
    def __init__(self, model, target_layer):
        self.model   = model
        self.grads   = None
        self.acts    = None
        self._hooks  = []
        self._hooks.append(
            target_layer.register_forward_hook(
                lambda m, inp, out: setattr(self, "acts", out.detach())))
        self._hooks.append(
            target_layer.register_full_backward_hook(
                lambda m, gi, go:  setattr(self, "grads", go[0].detach())))

    def __call__(self, img_tensor, class_idx=None):
        self.model.eval()
        logits = self.model(img_tensor.unsqueeze(0).to(DEVICE))
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=(2,3), keepdim=True)
        cam     = (weights * self.acts).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam    -= cam.min()
        if cam.max() > 0: cam /= cam.max()
        return cam, class_idx

    def remove_hooks(self):
        for h in self._hooks: h.remove()


def plot_gradcam(model, target_layer, loader, model_name, n=6):
    """Show Grad-CAM overlays for n correctly AND n misclassified images."""
    gcam    = GradCAM(model, target_layer)
    inv = transforms.Normalize(
        mean=[-m/s for m,s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std =[1/s   for s   in IMAGENET_STD])

    correct_set,   miss_set = [], []

    model.eval()
    for imgs, lbls in loader:
        for i in range(imgs.size(0)):
            if len(correct_set) >= n and len(miss_set) >= n:
                break
            img   = imgs[i]
            lbl   = lbls[i].item()
            with torch.no_grad():
                pred = model(img.unsqueeze(0).to(DEVICE)).argmax(1).item()

            cam, _ = gcam(img, pred)
            rgb    = inv(img).permute(1,2,0).numpy().clip(0,1)
            entry  = (rgb, cam, lbl, pred)

            if pred == lbl and len(correct_set) < n:
                correct_set.append(entry)
            elif pred != lbl and len(miss_set) < n:
                miss_set.append(entry)
        if len(correct_set) >= n and len(miss_set) >= n:
            break

    gcam.remove_hooks()

    for tag, dataset in [("correct", correct_set), ("misclassified", miss_set)]:
        n_plot = len(dataset)
        if n_plot == 0: continue
        fig, axes = plt.subplots(n_plot, 3, figsize=(9, 3*n_plot))
        if n_plot == 1: axes = axes[np.newaxis, :]

        for row, (rgb, cam, t, p) in enumerate(dataset):
            h, w   = rgb.shape[:2]
            heatmap = np.array(Image.fromarray(
                np.uint8(cm.jet(cam)*255)[:,:,:3]).resize((w,h)))/255.

            axes[row,0].imshow(rgb)
            axes[row,0].set_title(f"T:{CLASS_NAMES[t]}  P:{CLASS_NAMES[p]}",
                                  fontsize=8,
                                  color=("green" if t==p else "red"))
            axes[row,0].axis("off")

            axes[row,1].imshow(cam, cmap="jet")
            axes[row,1].set_title("CAM heatmap", fontsize=8)
            axes[row,1].axis("off")

            axes[row,2].imshow(np.clip(0.5*rgb + 0.5*heatmap, 0, 1))
            axes[row,2].set_title("Overlay", fontsize=8)
            axes[row,2].axis("off")

        plt.suptitle(f"Grad-CAM — {model_name.upper()} ({tag})",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = f"{CFG['out_dir']}/gradcam_{model_name}_{tag}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT]  Grad-CAM ({tag}) saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. COUNT PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# 11. COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(all_metrics):
    names  = [m["name"] for m in all_metrics]
    accs   = [m["accuracy"]  * 100 for m in all_metrics]
    f1s    = [m["f1"]        * 100 for m in all_metrics]
    top3s  = [m["top3"]      * 100 for m in all_metrics]

    x     = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - width, accs,  width, label="Top-1 Acc (%)", color="#4C72B0")
    b2 = ax.bar(x,         f1s,   width, label="Macro F1  (%)", color="#DD8452")
    b3 = ax.bar(x + width, top3s, width, label="Top-3 Acc (%)", color="#55A868")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — CIFAR-10 Test Set",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    path = f"{CFG['out_dir']}/comparison_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  Comparison chart saved → {path}")


def plot_efficiency(eff_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names  = [d["name"]       for d in eff_data]
    params = [d["params_M"]   for d in eff_data]
    times  = [d["epoch_time"] for d in eff_data]
    infs   = [d["inf_ms"]     for d in eff_data]

    colors = ["#4C72B0","#DD8452","#55A868","#C44E52"]

    for ax, vals, title, ylabel in zip(
            axes,
            [params, times, infs],
            ["Parameters (M)","Avg Epoch Time (s)","Inference (ms/img)"],
            ["Millions","Seconds","Milliseconds"]):
        bars = ax.bar(names, vals, color=colors[:len(names)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02*max(vals),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    plt.suptitle("Efficiency Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{CFG['out_dir']}/efficiency.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  Efficiency chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN  ──  BUILD → TRAIN → EVALUATE  ──  ALL 4 MODELS
# ─────────────────────────────────────────────────────────────────────────────
all_histories = {}
all_metrics   = []
all_efficiency= []

# ── 12-A  Baseline CNN ───────────────────────────────────────────────────────
print("\n" + "█"*60)
print("  MODEL 1/4 : Baseline CNN from Scratch")
print("█"*60)

cnn = BaselineCNN(CFG["num_classes"]).to(DEVICE)
p_tot, p_tr = count_params(cnn)
print(f"  Parameters — Total: {p_tot/1e6:.2f}M  Trainable: {p_tr/1e6:.2f}M")

hist_cnn = train_model(cnn, "baseline_cnn")
all_histories["baseline_cnn"] = hist_cnn
plot_curves(hist_cnn, "Baseline CNN")

y_true, y_pred, y_proba, inf_ms = evaluate_model(cnn, test_loader)
met = compute_metrics(y_true, y_pred, y_proba, "Baseline CNN")
all_metrics.append(met)
plot_confusion(y_true, y_pred, "baseline_cnn")
show_misclassified(cnn, test_loader, "baseline_cnn")
all_efficiency.append(dict(name="Baseline CNN", params_M=p_tot/1e6,
                            epoch_time=hist_cnn["avg_epoch_time"], inf_ms=inf_ms))

# ── 12-B  ResNet-50 ──────────────────────────────────────────────────────────
print("\n" + "█"*60)
print("  MODEL 2/4 : ResNet-50 (Transfer Learning)")
print("█"*60)

rn50, feat_rn, head_rn = build_transfer_model("resnet50", CFG["num_classes"])
rn50 = rn50.to(DEVICE)
p_tot, p_tr = count_params(rn50)
print(f"  Parameters — Total: {p_tot/1e6:.2f}M  Trainable: {p_tr/1e6:.2f}M")

hist_rn = train_model(rn50, "resnet50", feat_rn, head_rn, is_transfer=True)
all_histories["resnet50"] = hist_rn
plot_curves(hist_rn, "ResNet-50")

y_true, y_pred, y_proba, inf_ms = evaluate_model(rn50, test_loader)
met = compute_metrics(y_true, y_pred, y_proba, "ResNet-50")
all_metrics.append(met)
plot_confusion(y_true, y_pred, "resnet50")
show_misclassified(rn50, test_loader, "resnet50")
all_efficiency.append(dict(name="ResNet-50", params_M=p_tot/1e6,
                            epoch_time=hist_rn["avg_epoch_time"], inf_ms=inf_ms))

# ── 12-C  DenseNet-121 ───────────────────────────────────────────────────────
print("\n" + "█"*60)
print("  MODEL 3/4 : DenseNet-121 (Transfer Learning)")
print("█"*60)

dn121, feat_dn, head_dn = build_transfer_model("densenet121", CFG["num_classes"])
dn121 = dn121.to(DEVICE)
p_tot, p_tr = count_params(dn121)
print(f"  Parameters — Total: {p_tot/1e6:.2f}M  Trainable: {p_tr/1e6:.2f}M")

hist_dn = train_model(dn121, "densenet121", feat_dn, head_dn, is_transfer=True)
all_histories["densenet121"] = hist_dn
plot_curves(hist_dn, "DenseNet-121")

y_true, y_pred, y_proba, inf_ms = evaluate_model(dn121, test_loader)
met = compute_metrics(y_true, y_pred, y_proba, "DenseNet-121")
all_metrics.append(met)
plot_confusion(y_true, y_pred, "densenet121")
show_misclassified(dn121, test_loader, "densenet121")
all_efficiency.append(dict(name="DenseNet-121", params_M=p_tot/1e6,
                            epoch_time=hist_dn["avg_epoch_time"], inf_ms=inf_ms))

# ── 12-D  EfficientNet-B0 ────────────────────────────────────────────────────
print("\n" + "█"*60)
print("  MODEL 4/4 : EfficientNet-B0 (Transfer Learning)")
print("█"*60)

effb0, feat_ef, head_ef = build_transfer_model("efficientnet_b0", CFG["num_classes"])
effb0 = effb0.to(DEVICE)
p_tot, p_tr = count_params(effb0)
print(f"  Parameters — Total: {p_tot/1e6:.2f}M  Trainable: {p_tr/1e6:.2f}M")

hist_ef = train_model(effb0, "efficientnet_b0", feat_ef, head_ef, is_transfer=True)
all_histories["efficientnet_b0"] = hist_ef
plot_curves(hist_ef, "EfficientNet-B0")

y_true, y_pred, y_proba, inf_ms = evaluate_model(effb0, test_loader)
met = compute_metrics(y_true, y_pred, y_proba, "EfficientNet-B0")
all_metrics.append(met)
plot_confusion(y_true, y_pred, "efficientnet_b0")
show_misclassified(effb0, test_loader, "efficientnet_b0")
all_efficiency.append(dict(name="EfficientNet-B0", params_M=p_tot/1e6,
                            epoch_time=hist_ef["avg_epoch_time"], inf_ms=inf_ms))

# ─────────────────────────────────────────────────────────────────────────────
# 13. GRAD-CAM  on best transfer model  (pick highest accuracy)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("  Grad-CAM visualisation")
print("─"*60)

best_name = max(
    [("resnet50", rn50, hist_rn["best_val_acc"]),
     ("densenet121", dn121, hist_dn["best_val_acc"]),
     ("efficientnet_b0", effb0, hist_ef["best_val_acc"])],
    key=lambda x: x[2]
)
best_model_name, best_model_obj = best_name[0], best_name[1]

# identify target conv layer per architecture
if best_model_name == "resnet50":
    target_layer = best_model_obj.layer4[-1].conv3
elif best_model_name == "densenet121":
    target_layer = best_model_obj.features.denseblock4.denselayer16.conv2
else:   # efficientnet_b0
    target_layer = best_model_obj.features[-1][0]

print(f"  → Using {best_model_name.upper()} for Grad-CAM")
plot_gradcam(best_model_obj, target_layer, test_loader, best_model_name, n=6)

# ─────────────────────────────────────────────────────────────────────────────
# 14. FINAL COMPARISON PLOTS & JSON SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
plot_comparison(all_metrics)
plot_efficiency(all_efficiency)

# Save summary JSON
summary = {
    "config"    : CFG,
    "metrics"   : all_metrics,
    "efficiency": all_efficiency,
}
# make serialisable
for m in summary["metrics"]:
    for k, v in m.items():
        if isinstance(v, (np.floating, np.integer, float)):
            m[k] = float(v)
for e in summary["efficiency"]:
    for k, v in e.items():
        if isinstance(v, float):
            e[k] = float(v)

with open(f"{CFG['out_dir']}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 15. PRINT FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"  {'Model':<20}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  "
      f"{'F1':>6}  {'Top-3':>6}  {'Params':>8}  {'Epoch(s)':>8}  {'Inf(ms)':>7}")
print("="*75)
for m, e in zip(all_metrics, all_efficiency):
    print(f"  {m['name']:<20}  {m['accuracy']*100:>5.1f}%"
          f"  {m['precision']*100:>5.1f}%  {m['recall']*100:>5.1f}%"
          f"  {m['f1']*100:>5.1f}%  {m['top3']*100:>5.1f}%"
          f"  {e['params_M']:>6.1f}M  {e['epoch_time']:>8.1f}  {e['inf_ms']:>7.2f}")
print("="*75)

print(f"\n[DONE]  All outputs saved to → ./{CFG['out_dir']}/")
print("        Files generated:")
for f in sorted(os.listdir(CFG["out_dir"])):
    print(f"          {f}")
