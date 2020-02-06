"""Inferencer for test images"""

import torch
from torch.utils.data import dataset, dataloader
from torchvision import transforms
import os
from PIL import Image
from matplotlib import pyplot as plt

BATCH_SIZE = 8
MODEL_DIR = "/home/rex/models/emotion-models/"
MODEL_NAME = "FERPlus_models_exp9_pretrained_e35_lr001_224_cmat_4classes_model_16.pth"  # trained model
SAVE_FOLDER = f"/home/rex/datasets/{MODEL_NAME}/"  # Path to prediction results

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

class_dict = {
    0: 'neutral',
    1: 'happiness',
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear',
    7: 'contempt'
}


class FERPlusInferenceDataset(dataset.Dataset):
    def __init__(self, fer_data_dir, transforms=None):
        self.img_dir = fer_data_dir
        self.transforms = transforms
        self.image_file_names = sorted(
            [x for x in os.listdir(self.img_dir) if x.endswith('.jpg')])

    def __getitem__(self, idx):
        img_file_name = self.image_file_names[idx]
        img_file = os.path.join(self.img_dir, img_file_name)
        img = Image.open(img_file).convert('LA').convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img, img_file

    def __len__(self):
        return len(self.image_file_names)


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main():
    print("Loading model...")
    model = torch.load(f'{MODEL_DIR}/{MODEL_NAME}',
                       map_location=torch.device('cpu'))
    print("model loaded...")
    test_ds = FERPlusInferenceDataset(
        "/home/rex/datasets/emotion-test-images/", transforms=test_transform)
    test_dataloader = dataloader.DataLoader(dataset=test_ds,
                                            batch_size=8,
                                            num_workers=4)

    for idx, data in enumerate(test_dataloader):
        img, img_name_batch = data
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        print("pred", pred)
        pred = pred.cpu().numpy()
        for pred_idx, _pred in enumerate(pred):
            img_file_name = img_name_batch[pred_idx]
            img = Image.open(img_file_name)
            plt.imshow(img)
            plt.text(0, 0, class_dict[_pred], fontsize=12)
            plt.savefig(os.path.join(SAVE_FOLDER,
                                     img_file_name.split("/")[-1]))
            plt.close()


if __name__ == "__main__":
    main()
