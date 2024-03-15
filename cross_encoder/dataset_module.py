from transformers import RobertaTokenizer, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from collections import Counter
import pytorch_lightning as pl
from utils import load_obj
from tqdm import tqdm
import torch

def create_tensor_dataset(model_name, data):
    #data = load_obj("cross_encoder_claim_title_sentence_label_pairs_dev.json")
    #random.shuffle(data)
    if model_name.startswith('roberta'):
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    
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
        train_indices, temp_indices, _, _ = train_test_split(range(len(self.dataset)), self.labels, stratify=self.labels, test_size=0.2)
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