from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from eye_deseases_classification.model import ResNet
from eye_deseases_classification.data import MyDataset

def evaluate():

    checkpoint_dir = Path("models")
    ckpts = list(checkpoint_dir.glob("*.ckpt"))

    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoint found in models/")

    best_ckpt = ckpts[0]  # only exist one (maybe change this)

    model = ResNet.load_from_checkpoint(best_ckpt)

    test_dataset = MyDataset("data/processed/test")
    test_loader = DataLoader(test_dataset, batch_size=16)

    trainer = pl.Trainer()
    trainer.test(model, test_loader)

if __name__ == "__main__":
    evaluate()
