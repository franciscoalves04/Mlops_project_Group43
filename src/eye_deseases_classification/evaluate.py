from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from eye_deseases_classification.model import ResNet
from eye_deseases_classification.data import MyDataset
from eye_deseases_classification.logger import logger 

def evaluate():
    logger.info("Starting evaluation script")

    checkpoint_dir = Path("models")
    ckpts = list(checkpoint_dir.glob("*.ckpt"))

    if len(ckpts) == 0:
        logger.error("No checkpoint found in models/")
        raise FileNotFoundError("No checkpoint found in models/")

    best_ckpt = ckpts[0]  # only one checkpoint (chage this)
    logger.info(f"Loading checkpoint: {best_ckpt}")

    model = ResNet.load_from_checkpoint(best_ckpt)

    test_dataset = MyDataset("data/processed/test")
    test_loader = DataLoader(test_dataset, batch_size=16)
    logger.info(f"Loaded test dataset with {len(test_dataset)} samples")

    trainer = pl.Trainer()
    logger.info("Starting model testing")
    trainer.test(model, test_loader)
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    evaluate()
