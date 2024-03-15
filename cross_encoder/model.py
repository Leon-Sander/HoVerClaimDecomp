from transformers import RobertaTokenizer, RobertaForSequenceClassification,BertTokenizer, BertForSequenceClassification
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import torch


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
            
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        
        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()
        self.test_precision = BinaryPrecision()
        
        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.test_recall = BinaryRecall()


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
        precision_metric = getattr(self, f"{step_type}_precision")
        recall_metric = getattr(self, f"{step_type}_recall")
        
        accuracy_metric(preds, labels)
        precision_metric(preds, labels)
        recall_metric(preds, labels)
        
        acc = accuracy_metric.compute()
        prec = precision_metric.compute()
        rec = recall_metric.compute()
        
        self.log(f"{step_type}_loss", loss, prog_bar=True)
        self.log(f"{step_type}_accuracy", acc, prog_bar=True)
        self.log(f"{step_type}_precision", prec, prog_bar=True)
        self.log(f"{step_type}_recall", rec, prog_bar=True)
        
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, verbose=True
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
    
    def predict(self, claim_text_pairs):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for claim, text in claim_text_pairs:
                inputs = self.tokenizer.encode_plus(
                    claim, text,
                    add_special_tokens=True,
                    max_length=320,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                token_type_ids = inputs.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)


                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                probs = torch.sigmoid(outputs.logits).squeeze()

                predicted_label = (probs > 0.5).long().cpu().numpy()
                predictions.append((claim, text, predicted_label))

        return predictions