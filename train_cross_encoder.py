import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import RobertaTokenizer, RobertaForSequenceClassification,BertTokenizer, BertForSequenceClassification
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
from sklearn.model_selection import train_test_split
from collections import Counter


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, labels, batch_size):
        super().__init__()
        self.dataset = dataset
        self.labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        self.batch_size = batch_size
        self.setup()

    def print_dataset_info(self, dataset, name):
        labels = [label.item() for _, _, _, label in dataset]
        label_distribution = Counter(labels)
        print(f"{name} Datensatz: {len(dataset)} Beispiele, Labelverteilung: {label_distribution}")

    def setup(self, stage=None):
        #dataset_size = len(self.dataset)
        #train_size = int(0.8 * len(self.dataset))
        #val_size = int(0.1 * len(self.dataset))
        #test_size = len(self.dataset) - train_size - val_size
        #self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        train_indices, temp_indices, _, _ = train_test_split(range(len(dataset)), self.labels, stratify=self.labels, test_size=0.2)
        val_indices, test_indices, _, _ = train_test_split(temp_indices, self.labels[temp_indices], stratify=self.labels[temp_indices], test_size=0.5)

        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

        self.print_dataset_info(self.train_dataset, "Trainings")
        self.print_dataset_info(self.val_dataset, "Validierungs")
        self.print_dataset_info(self.test_dataset, "Test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=20)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=20)


class TextClassificationModel(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        if self.model_name.startswith('roberta'):
            self.model = RobertaForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.test_accuracy = torchmetrics.classification.BinaryAccuracy()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        if self.model_name.startswith('roberta'):
            output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        else:
            output = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        return output

    def process_batch(self, batch):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        return loss, preds, labels

    def log_metrics(self, step_type, loss, preds, labels):
        accuracy_metric = getattr(self, f"{step_type}_accuracy")
        accuracy_metric.update(preds, labels)
        self.log(f"{step_type}_loss", loss, prog_bar=True)
        self.log(f"{step_type}_accuracy", accuracy_metric, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.process_batch(batch)
        self.log_metrics("train", loss, preds, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.process_batch(batch)
        self.log_metrics("val", loss, preds, labels)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.process_batch(batch)
        self.log_metrics("test", loss, preds, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
    data = load_obj("cross_encoder_claim_title_sentence_label_pairs2.json")
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
            max_length=320,
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
    return dataset, labels


if __name__ == "__main__":
    model_name = "roberta-large"
    #model_name='bert-base-uncased'
    batch_size = 64
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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


    dataset, labels = create_tensor_dataset()
    data_module = DataModule(dataset, labels, batch_size)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()



    model = TextClassificationModel(model_name=model_name)
    #model = TextClassificationModel(model_name=model_name)

    trainer = Trainer(logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=200,
        devices=1, 
        accelerator="gpu")

    trainer.fit(model, train_loader, val_loader)


    best_model_path = checkpoint_callback.best_model_path
    best_model = TextClassificationModel.load_from_checkpoint(checkpoint_path=best_model_path)
    trainer.test(model=best_model, dataloaders=test_loader)
