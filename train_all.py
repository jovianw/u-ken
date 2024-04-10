import sys

sys.path.insert(0, ".")
import os
import numpy as np
import nibabel as nib
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from timeit import default_timer as timer
from copy import deepcopy
from monai.transforms import Rand3DElasticd
import argparse
from models.uken_small import UNet3DWithKEM
import time


def dice_loss(preds, labels):
    """
    Compute the Dice loss between predictions and labels.
    preds: Tensor of shape (batch_size, 1, 128, 256, 256)
    labels: Tensor of shape (batch_size, 128, 256, 256)
    """
    # Ensure the predictions are in [0,1] by applying sigmoid
    preds = torch.sigmoid(preds)

    # Remove the channel dimension from preds to match labels' shape
    preds = preds.squeeze(1)

    # Calculate intersection and union
    intersection = (preds * labels).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3))

    # Compute Dice coefficient and Dice loss
    dice_coeff = (2.0 * intersection + 1e-6) / (
        union + 1e-6
    )  # Adding a small epsilon to avoid division by zero
    dice_loss = 1 - dice_coeff

    # Return the average Dice loss over the batch
    return dice_loss.mean()


def save_checkpoint(snapshot_dir, epoch_num, history, best_model, curr_model):
    timestr = time.strftime("%m%d-%H%M")
    # Save history
    save_history_path = os.path.join(
        snapshot_dir, f"epoch_{epoch_num}_history_{timestr}.npz"
    )
    np.savez_compressed(save_history_path, history=history)
    # Save best model
    best_model_path = os.path.join(
        snapshot_dir, f"epoch_{epoch_num}_best_{timestr}.pth"
    )
    torch.save(best_model, best_model_path)
    # Save current model
    save_model_path = os.path.join(snapshot_dir, f"epoch_{epoch_num}_{timestr}.pth")
    torch.save(curr_model, save_model_path)
    print(f"Saved model to {save_model_path}, {best_model_path}")


class All_dataset(Dataset):
    """
    Assumes that labels are already unzipped
    """

    def __init__(self, datasets_dir, list_dir, split, clip=-1, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = [
            line.strip("\n").split(" ")
            for line in open(os.path.join(list_dir, "all_" + self.split + ".txt"))
        ]
        if clip > 0:
            self.sample_list = self.sample_list[:clip]
        self.datasets_dir = datasets_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        subfolder, filename = self.sample_list[idx][0], self.sample_list[idx][1]
        image_path = os.path.join(
            self.datasets_dir, subfolder, "images", filename + ".nii.gz"
        )
        label_path = os.path.join(
            self.datasets_dir, subfolder, "labels", filename + ".nii.gz"
        )
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
    "--datasets_dir",
    type=str,
    default="datasets",
    help="dir for ULS23 datasets",
)
parser.add_argument(
    "--list_dir",
    type=str,
    default="datasets",
    help="dir for split list text files",
)
parser.add_argument(
    "--snapshot_dir",
    type=str,
    default="snapshots",
    help="dir for model and data snapshots",
)
parser.add_argument(
    "--base_lr", type=float, default=0.005, help="segmentation network learning rate"
)
parser.add_argument(
    "--max_epoch", type=int, default=300, help="maximum epoch number to train"
)
parser.add_argument(
    "--patience", type=int, default=10, help="num epochs before early stopping"
)
parser.add_argument("--clip", type=int, default=-1, help="number of slices to clip")
parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("--ncodes", type=int, default=8, help="number of codes for KEM")
args = parser.parse_args()

datasets_dir = args.datasets_dir
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

db_train = All_dataset(datasets_dir, list_dir, "train", transform=transform)
db_val = All_dataset(datasets_dir, list_dir, "val", transform=transform)
db_test = All_dataset(datasets_dir, list_dir, "test", transform=transform)
print("The length of train set is: {}".format(len(db_train)))
print("The length of val set is: {}".format(len(db_val)))
print("The length of test set is: {}".format(len(db_test)))

trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True)
valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=base_lr)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3, verbose=True
)

# Make snapshots dir
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

bce_loss = BCEWithLogitsLoss()
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

    # Training loop
    model.train()
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        outputs = model(image_batch)
        loss_bce = bce_loss(outputs.squeeze(1), label_batch)
        loss_dice = dice_loss(outputs, label_batch)
        loss = 0.4 * loss_bce + 0.6 * loss_dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Track training progress
        print(
            f"Epoch: {epoch_num}\t{100 * (i_batch + 1) / len(trainloader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.",
            end="\r",
        )

    avg_train_loss = total_train_loss / len(trainloader)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    with torch.no_grad():  # No need to track gradients during validation
        for i_batch, sampled_batch in enumerate(valloader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)
            loss_bce = bce_loss(outputs.squeeze(1), label_batch)
            loss_dice = dice_loss(outputs, label_batch)
            loss = 0.4 * loss_bce + 0.6 * loss_dice
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(valloader)
    history.append([avg_train_loss, avg_val_loss])

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
