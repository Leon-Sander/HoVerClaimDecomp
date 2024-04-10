import os
import sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append(str(Path("../").resolve()))
from dataset_module import create_tensor_dataset
from model import TextClassificationModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from utils import load_obj

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    #best_model_path = "/home/sander/code/thesis/hover/leon/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt" # with title
    best_model_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-16_23-01-03/bestckpt_epoch=1_val_loss=0.18.ckpt" # without title
    batch_size = 32
    #data = load_obj("/home/sander/code/thesis/hover/leon/cross_encoder/cross_encoder_claim_title_sentence_label_pairs_dev.json")
    data = load_obj("/home/sander/code/thesis/hover/leon/cross_encoder/cross_encoder_claim_sentence_label_pairs_dev.json")
    dataset, labels = create_tensor_dataset(model_name, data)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    trainer = Trainer()
    best_model = TextClassificationModel.load_from_checkpoint(checkpoint_path=best_model_path, model_name=model_name)
    trainer.test(model=best_model, dataloaders=test_loader)