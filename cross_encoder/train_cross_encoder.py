import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path("../").resolve()))
from dataset_module import create_tensor_dataset, DataModule
from model import TextClassificationModel
from utils import load_obj

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    batch_size = 32
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data = load_obj("/home/sander/code/thesis/hover/leon/cross_encoder/cross_encoder_claim_sentence_label_pairs_train.json")

    logger = TensorBoardLogger("tb_logs", name=f"cross_enc_{model_name}_{timestamp}")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=10, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{timestamp}/",
        filename="bestckpt_{epoch}_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    model = TextClassificationModel(model_name=model_name)
    dataset, labels = create_tensor_dataset(model_name, data)
    data_module = DataModule(dataset, labels, batch_size)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    trainer = Trainer(logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=200,
        devices=1, 
        accelerator="gpu")

    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    best_model = TextClassificationModel.load_from_checkpoint(checkpoint_path=best_model_path, model_name=model_name)
    trainer.test(model=best_model, dataloaders=test_loader)
