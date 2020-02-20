"""Trains a vgg13 model on the FER+ dataset"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import vgg13
from torch.utils.data import dataset
from torchvision import transforms
import pandas as pd
import os
import time
import copy
from PIL import Image
import numpy as np
import random
from skimage import io, transform
from sklearn.metrics import confusion_matrix

# Paths for data, model, an experiment name etc
DATA_ROOT = "/home/rex/data/FERPlus/images/"
MODEL_DIR = "/home/rex/models/FERPlus/"
exp_name = "exp9_pretrained_e35_lr001_224_cmat_8classes"
MODEL_PATH = MODEL_DIR + exp_name
LEARNING_RATE = 0.001
NUM_EPOCHS = 35
device = "cuda:0"  #cpu

for _dir in [DATA_ROOT, MODEL_PATH]:
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def _process_row(row):
    """
    Process a single dataframe row, returns the argmax label
    :param row:
    :return:
    """
    return np.argmax(row)


class FERPlusDataset(dataset.Dataset):
    """
    Creats a PyTorch custom Dataset for batch iteration
    """
    def __init__(self, fer_data_dir, mode="train", transforms=None):
        self.fer_data_dir = fer_data_dir
        self.transforms = transforms
        self.mode = mode
        if self.mode == "train":
            self.img_dir = os.path.join(self.fer_data_dir, "FER2013Train")
        elif self.mode == "val":
            self.img_dir = os.path.join(self.fer_data_dir, "FER2013Valid")
        elif self.mode == "test":
            self.img_dir = os.path.join(self.fer_data_dir, "FER2013Test")
        self.label_file = os.path.join(self.img_dir, "label.csv")

        self.label_data_df = pd.read_csv(self.label_file, header=None)
        self.label_data_df.columns = [
            "img_name", "dims", "0", "1", "2", "3", "4", "5", "6", "7",
            "Unknown", "NF"
        ]

        # The arg-max label is the selected as the actual label for Majority Voting
        self.label_data_df['actual_label'] = self.label_data_df[[
            '0', '1', '2', '3', '4', '5', '6', '7'
        ]].apply(lambda x: _process_row(x), axis=1)

        # get all ilocs with actual label 0
        self.label_data_df.sort_values(by=['img_name'])

        if mode == "train":
            locs0 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '0'].index.values)

            # Sampling can be turned off otherwise selects only 40% of neutral ~ 4k images
            sample_indices0 = random.Random(1).sample(locs0,
                                                      int(len(locs0) * 0.6))
            locs1 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '1'].index.values)

            # Select only 50% of neutral ~ 4k images
            sample_indices1 = random.Random(1).sample(locs1,
                                                      int(len(locs1) * 0.5))

            locs5 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '5'].index.values)
            locs6 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '6'].index.values)
            locs7 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '7'].index.values)
            self.label_data_df = self.label_data_df.drop(sample_indices0 +
                                                         sample_indices1 +
                                                         locs5 + locs6 + locs7)

        elif mode in ["val", "test"]:
            locs5 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '5'].index.values)
            locs6 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '6'].index.values)
            locs7 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '7'].index.values)
            self.label_data_df = self.label_data_df.drop(locs5 + locs6 + locs7)

        self.image_file_names = self.label_data_df['img_name'].values

    def __getitem__(self, idx):
        img_file_name = self.image_file_names[idx]
        img_file = os.path.join(self.img_dir, img_file_name)
        img = Image.open(img_file).convert('RGB')
        img_class = self.get_class(img_file_name)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, torch.tensor(img_class).to(torch.long)

    def get_class(self, file_name):
        """
        Returns the label for a corresponding file
        :param file_name: Image file name
        :return:
        """
        row_df = self.label_data_df[self.label_data_df["img_name"] ==
                                    file_name]
        init_val = -1
        init_idx = -1
        for x in range(2, 10):
            max_val = max(init_val, row_df.iloc[0].values[x])
            if max_val > init_val:
                init_val = max_val
                init_idx = int(
                    x - 2
                )  # Labels indices must start at 0, -2 if all else -4!!!!!!
        return init_idx

    def __len__(self):
        return len(self.image_file_names)


# Create train, val and test transforms
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = FERPlusDataset(DATA_ROOT, mode="train", transforms=train_transform)
val_ds = FERPlusDataset(DATA_ROOT, mode="val", transforms=val_transform)
test_ds = FERPlusDataset(DATA_ROOT, mode="test", transforms=test_transform)

dataloaders = {
    'train':
    torch.utils.data.DataLoader(train_ds,
                                batch_size=64,
                                shuffle=True,
                                num_workers=8),
    'val':
    torch.utils.data.DataLoader(val_ds,
                                batch_size=64,
                                shuffle=True,
                                num_workers=8),
    'test':
    torch.utils.data.DataLoader(test_ds,
                                batch_size=64,
                                shuffle=True,
                                num_workers=8)
}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    phases = ["train", "val"]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for idx, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch != 0 and epoch % 1 == 0:
            # checkpoint model
            torch.save(model,
                       os.path.join(MODEL_PATH, f"model_{str(epoch)}.pth"))
            print("Running test mode")
            running_loss_test = 0
            running_corrects_test = 0

            running_preds = []
            running_labels = []

            for idx, data in enumerate(dataloaders["test"]):
                test_inputs, test_labels = data
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)
                outputs = model(test_inputs)
                loss = criterion(outputs, test_labels)
                _, preds = torch.max(outputs, 1)

                running_preds.append(preds.squeeze(0).cpu().numpy())
                running_labels.append(test_labels.squeeze(0).cpu().numpy())

                running_loss_test += loss.item()
                running_corrects_test += torch.sum(preds == test_labels.data)

            epoch_loss = running_loss_test / len(dataloaders["test"].dataset)
            epoch_acc = running_corrects_test.double() / len(
                dataloaders["test"].dataset)
            print('Test Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))
            cmat = confusion_matrix(
                [label for batch in running_labels for label in batch],
                [label for batch in running_preds for label in batch])
            print(f"Test confusion matrix: \n{cmat}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# PRE TRAINED
model = vgg13(pretrained=True)
num_ftrs = model.classifier[3].out_features

model.classifier = nn.Sequential(nn.Linear(7 * 7 * 512, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(1024, 8))

for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)

# Loss weights created from an analysis of the distribution
loss_weights = torch.tensor([
    0.01637922728586569, 0.022555393392596525, 0.047446274072006696,
    0.04776622646810901, 0.06865912762520193, 0.25796661608497723,
    0.8717948717948717, 1.0
])

loss_weights = loss_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=loss_weights)

# # Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# # Decay LR by a factor of 0.1 every 20 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

model = train_model(model,
                    criterion,
                    optimizer,
                    exp_lr_scheduler,
                    num_epochs=NUM_EPOCHS)

# Save model
torch.save(model, os.path.join(MODEL_PATH, "model.pth"))
