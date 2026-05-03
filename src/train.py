import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import KFold
import numpy as np

from model import PointNet, DGCNN
from Lima3D import PoleDataset


# -------------------------
# 🧠 Train one fold
# -------------------------
def train_fold(model, train_loader, val_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):

        # ---------------- TRAIN ----------------
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            points = points.permute(0, 2, 1)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_losses.append(total_loss / len(train_loader))
        train_accs.append(correct / total)

        # ---------------- VAL ----------------
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        # Per-class tracking
        class_correct = {0: 0, 1: 0}
        class_total   = {0: 0, 1: 0}

        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                points = points.permute(0, 2, 1)

                outputs = model(points)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for cls in [0, 1]:
                    mask = labels == cls
                    class_correct[cls] += (preds[mask] == labels[mask]).sum().item()
                    class_total[cls]   += mask.sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(correct / total)
        
        mono_acc  = class_correct[0] / class_total[0] if class_total[0] else 0
        bi_acc    = class_correct[1] / class_total[1] if class_total[1] else 0


        print(
            f"Epoch {epoch+1}: "
            f"Train Loss {train_losses[-1]:.4f} | Train Acc {train_accs[-1]:.4f} | "
            f"Val Loss {val_losses[-1]:.4f} | Val Acc {val_accs[-1]:.4f} | "
            f"mono {mono_acc:.2f} | bi {bi_acc:.2f}"   # ← key metric
        )

    return train_losses, val_losses, train_accs, val_accs

def print_loader_class_balance(loader, name="Loader"):
    class_counts = {}

    for _, labels in loader:
        for l in labels.numpy():
            class_counts[int(l)] = class_counts.get(int(l), 0) + 1

    total = sum(class_counts.values())

    print(f"\n📊 {name} class balance:")
    for k in sorted(class_counts.keys()):
        print(f"  Class {k}: {class_counts[k]}")

    print("\n📊 Ratios:")
    for k in sorted(class_counts.keys()):
        print(f"  Class {k}: {class_counts[k] / total:.4f}")

# -------------------------
# 🔁 K-Fold Training
# -------------------------
def kfold_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = PoleDataset(
        "data/pole_dataset/output",
        num_points=312,
        use_cache=True
    )

    labels_array = np.array(dataset.labels)

    k = 5
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    all_fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n================ Fold {fold+1}/{k} ================")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)



        # -------------------------
        # 🎯 Weighted Sampler (TRAIN ONLY)
        # -------------------------
        train_labels = labels_array[train_idx]

        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts

        sample_weights = class_weights[train_labels]

        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=32,
            sampler=sampler
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=32,
            shuffle=False
        )

        # print_loader_class_balance(train_loader, "Train Loader (WeightedSampler)")
        # print_loader_class_balance(val_loader, "Val Loader")

        # -------------------------
        # Model
        # -------------------------
        model = DGCNN().to(device)

        # -------------------------
        # Train fold
        # -------------------------
        _, _, _, val_accs = train_fold(
            model, train_loader, val_loader, device, epochs=20
        )

        fold_acc = val_accs[-1]
        all_fold_accs.append(fold_acc)

        print(f"Fold {fold+1} Final Acc: {fold_acc:.4f}")

        torch.save(model.state_dict(), f"pointnet_fold_{fold+1}.pth")

    print("\n================ FINAL RESULT ================")
    print("Mean Accuracy:", np.mean(all_fold_accs))
    print("Std Accuracy:", np.std(all_fold_accs))


if __name__ == "__main__":
    kfold_train()
