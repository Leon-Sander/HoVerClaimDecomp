import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import torch
from utils import load_obj
import random
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import Subset
import torchmetrics

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=64):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.setup()

    def setup(self, stage=None):
        dataset_size = len(self.dataset)
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        #self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        indices = torch.randperm(dataset_size).tolist()  # Zufällige Permutation aller Indizes

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=20)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=20)


class TextClassificationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.test_accuracy = torchmetrics.classification.BinaryAccuracy()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        return output
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask, token_type_ids, labels=labels)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        self.train_accuracy.update(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask, token_type_ids, labels=labels)
        val_loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        self.val_accuracy.update(preds, labels)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask, token_type_ids, labels=labels)
        test_loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        self.test_accuracy.update(preds, labels)
        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_accuracy', self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)# 
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, verbose=True
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]


def create_tensor_dataset():
    data = load_obj("cross_encoder_claim_title_sentence_label_pairs.json")
    #random.shuffle(data)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("creating data")
    input_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []

    for claim, sentence, label in tqdm(data):
        encoded_dict = tokenizer.encode_plus(
            text=claim,
            text_pair=sentence,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        labels.append(label)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    return dataset


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logger = TensorBoardLogger("tb_logs", name=f"cross_enc_{timestamp}")
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


    dataset = create_tensor_dataset()
    data_module = DataModule(dataset)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()



    model = TextClassificationModel()
    trainer = Trainer(logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=200,
        devices=1, 
        accelerator="gpu")

    trainer.fit(model, train_loader, val_loader)


    best_model_path = checkpoint_callback.best_model_path
    best_model = TextClassificationModel.load_from_checkpoint(checkpoint_path=best_model_path)
    trainer.test(model=best_model, dataloaders=test_loader)
