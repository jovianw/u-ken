import os
import numpy as np
import nibabel as nib
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from timeit import default_timer as timer
from copy import deepcopy
from monai.transforms import Rand3DElasticd
from models.uken_small import UNet3DWithKEM


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, labels):
        """
        Compute the Dice loss between predictions and labels.
        preds: Tensor of shape (batch_size, 1, 128, 256, 256)
        labels: Tensor of shape (batch_size, 128, 256, 256)
        """
        preds = torch.sigmoid(preds)
        preds = preds.squeeze(1)

        intersection = (preds * labels).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3))

        dice_coeff = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice_coeff

        return dice_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Assume inputs are raw logits for binary classification
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        targets = targets.type(torch.float32)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss


def save_checkpoint(snapshot_dir, epoch_num, history, best_model, curr_model):
    # Save history
    save_history_path = os.path.join(
        snapshot_dir, f"small_epoch_{epoch_num}_history.npz"
    )
    np.savez_compressed(save_history_path, history=history)
    # Save best model
    best_model_path = os.path.join(snapshot_dir, f"small_epoch_{epoch_num}_best.pth")
    torch.save(best_model, best_model_path)
    # Save current model
    save_model_path = os.path.join(snapshot_dir, f"small_epoch_{epoch_num}.pth")
    torch.save(curr_model, save_model_path)
    print(f"Saved model to {save_model_path}, {best_model_path}")


class DeepLesion_dataset(Dataset):
    """
    Assumes that labels are already unzipped
    """

    def __init__(self, image_dir, label_dir, list_dir, split, clip=-1, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = [
            line.strip("\n")
            for line in open(
                os.path.join(list_dir, "deeplesion_" + self.split + ".txt")
            )
        ]
        if clip > 0:
            self.sample_list = self.sample_list[:clip]
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        filename = self.sample_list[idx]
        image_path = os.path.join(self.image_dir, filename + "_lesion_01.nii.gz")
        label_path = os.path.join(self.label_dir, filename + "_lesion_01.nii.gz")
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        image = np.expand_dims(np.moveaxis(np.squeeze(image), 2, 0), axis=0)
        label = np.expand_dims(np.moveaxis(np.squeeze(label), 2, 0), axis=0)
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = filename
        return sample


# Setup our own args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_dir",
    type=str,
    default="datasets/ULS23_DeepLesion3D/images",
    help="dir for images",
)
parser.add_argument(
    "--label_dir",
    type=str,
    default="datasets/ULS23_DeepLesion3D/labels",
    help="dir for validation slices",
)
parser.add_argument(
    "--list_dir",
    type=str,
    default="datasets",
    help="dir for validation slices",
)
parser.add_argument(
    "--snapshot_dir",
    type=str,
    default="snapshots",
    help="dir for model and data snapshots",
)
parser.add_argument(
    "--base_lr", type=float, default=0.05, help="segmentation network learning rate"
)
parser.add_argument(
    "--max_epoch", type=int, default=300, help="maximum epoch number to train"
)
parser.add_argument(
    "--patience", type=int, default=50, help="num epochs before early stopping"
)
parser.add_argument("--clip", type=int, default=-1, help="number of slices to clip")
parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("--ncodes", type=int, default=8, help="number of codes for KEM")
args = parser.parse_args()

image_dir = args.image_dir
label_dir = args.label_dir
list_dir = args.list_dir
snapshot_dir = args.snapshot_dir
base_lr = args.base_lr
max_epoch = args.max_epoch
patience = args.patience
batch_size = args.batch_size
ncodes = args.ncodes

model = UNet3DWithKEM(base_n_filter=16, ncodes=ncodes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
model = nn.DataParallel(model)
model.to(device)

transform = Rand3DElasticd(
    keys=["image", "label"],
    sigma_range=(5, 8),
    magnitude_range=(50, 100),
    prob=0.5,
    translate_range=5,
    rotate_range=np.pi / 12,
    scale_range=(0.1, 0.1, 0.1),
    padding_mode="border",
)

db_train = DeepLesion_dataset(
    image_dir, label_dir, list_dir, "train", transform=transform
)
db_val = DeepLesion_dataset(image_dir, label_dir, list_dir, "val", transform=transform)
db_test = DeepLesion_dataset(
    image_dir, label_dir, list_dir, "test", transform=transform
)
print("The length of train set is: {}".format(len(db_train)))
print("The length of val set is: {}".format(len(db_val)))

trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True)
valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=base_lr)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=4, verbose=True
)
dice_loss = DiceLoss()
# focal_loss = FocalLoss()

# Make snapshots dir
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)


max_iterations = max_epoch * len(trainloader)
best_val_loss = float("inf")
overall_start = timer()
history = []
wait = 0
best_model = None
total_epochs = 0
# For each epoch
iterator = tqdm(range(max_epoch), ncols=100)
for epoch_num in iterator:
    start = timer()
    total_train_loss = 0.0
    total_train_dice = 0.0

    # Training loop
    model.train()
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        outputs = model(image_batch)
        loss_dice = dice_loss(outputs, label_batch)
        # loss_focal = focal_loss(outputs.squeeze(1), label_batch.squeeze(1).float())
        # loss = 0.3 * loss_focal + 0.7 * loss_dice
        loss = loss_dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        total_train_dice += loss_dice.item()

        # Track training progress
        print(
            f"Epoch: {epoch_num}\t{100 * (i_batch + 1) / len(trainloader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.",
            end="\r",
        )

    avg_train_loss = total_train_loss / len(trainloader)
    avg_train_dice = total_train_dice / len(trainloader)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    total_val_dice = 0.0
    with torch.no_grad():  # No need to track gradients during validation
        for i_batch, sampled_batch in enumerate(valloader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)
            loss_dice = dice_loss(outputs, label_batch)
            # loss_focal = focal_loss(outputs.squeeze(1), label_batch.squeeze(1).float())
            # loss = 0.3 * loss_focal + 0.7 * loss_dice
            loss = loss_dice
            total_val_loss += loss.item()
            total_val_dice += loss_dice.item()

    avg_val_loss = total_val_loss / len(valloader)
    avg_val_dice = total_val_dice / len(valloader)
    history.append([avg_train_loss, avg_val_loss, avg_train_dice, avg_val_dice])

    scheduler.step(avg_val_loss)

    total_epochs += 1

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model = deepcopy(model.state_dict())
        wait = 0  # Reset wait counter
        print(f"Validation loss improved to {avg_val_loss:.4f}. Saving model...")
    else:
        wait += 1
        if wait >= patience:
            print("Stopping early due to lack of improvement in validation loss.")
            save_checkpoint(
                snapshot_dir, epoch_num, history, best_model, model.state_dict()
            )
            break

    # Save occasionally
    if (epoch_num + 1) % 1 == 0:
        save_checkpoint(
            snapshot_dir, epoch_num, history, best_model, model.state_dict()
        )

iterator.close()
total_time = timer() - overall_start
print(
    f"{total_time:.2f} total seconds elapsed. {total_time / (total_epochs+1):.2f} seconds per epoch."
)
