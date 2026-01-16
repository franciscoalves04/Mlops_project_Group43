from pathlib import Path
import typer
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np


EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (256, 256)
MAX_IMAGES = 1000
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


class MyDataset(Dataset):
    """Custom dataset for eye disease images."""

    def __init__(self, data_path: Path, transform=None):
        self.data_path = Path(data_path)
        self.samples = []
        self.transform = transform

        # Gather all image paths and labels
        for label, class_dir in enumerate(sorted(self.data_path.iterdir())):
            if not class_dir.is_dir():
                continue
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in EXTENSIONS]
            images = images[:MAX_IMAGES]
            for img_path in images:
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.BILINEAR)

        # PIL -> numpy -> torch tensor
        img = np.array(img, dtype=np.float32) / 255.0   # (H, W, C)
        img = torch.from_numpy(img).permute(2, 0, 1)    # (C, H, W)

        if self.transform:
            img = self.transform(img)

        return img, label


    def preprocess(self, output_folder: Path):
        """Preprocess raw data into train/val/test splits."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        for split in SPLIT_RATIOS:
            for class_dir in self.data_path.iterdir():
                if class_dir.is_dir():
                    (output_folder / split / class_dir.name).mkdir(parents=True, exist_ok=True)

        for class_dir in self.data_path.iterdir():
            if not class_dir.is_dir():
                continue
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in EXTENSIONS][:MAX_IMAGES]
            n_total = len(images)
            n_train = int(n_total * SPLIT_RATIOS["train"])
            n_val = int(n_total * SPLIT_RATIOS["val"])

            splits = {
                "train": images[:n_train],
                "val": images[n_train:n_train + n_val],
                "test": images[n_train + n_val:],
            }

            for split, img_list in splits.items():
                for img_path in tqdm(img_list, desc=f"{class_dir.name} ({split})"):
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
                    out_path = output_folder / split / class_dir.name / img_path.name
                    img.save(out_path)

            print(f"{class_dir.name}: {n_total} images processed")


def preprocess(data_path: Path = Path("data/raw"), output_folder: Path = Path("data/processed")):
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)
    print("Preprocessing completed.")


if __name__ == "__main__":
    typer.run(preprocess)
