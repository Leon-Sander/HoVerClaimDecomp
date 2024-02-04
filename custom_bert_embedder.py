from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings

class CustomBertEmbedder(Embeddings):
    def __init__(self, gpu_count, batch_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_count = gpu_count
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.model.to(self.device)
        self.l2_normalize = True

        if self.gpu_count > 1:
            self.model = nn.DataParallel(self.model)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    @torch.no_grad()
    def _embed_text(self, text: str) -> List[float]:
        max_length = 512  # Bert's max length
        batch_dict = self.tokenizer([text], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)

        # Mean pooling
        input_mask_expanded = batch_dict['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        if self.l2_normalize:
            mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)

        return mean_embeddings.cpu().numpy()[0].tolist()

    @torch.no_grad()
    def batch_embed_text_tensor_multiple_gpus(self, texts: List[str]) -> List[List[float]]:
        max_length = 512  
        embeddings_list = []

        
        batch_size = self.batch_size * self.gpu_count
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)

            # Mean pooling
            input_mask_expanded = batch_dict['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            if self.l2_normalize:
                mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)

            embeddings_batch = mean_embeddings.cpu().numpy().tolist()
            embeddings_list.extend(embeddings_batch)

        return embeddings_list

    # Optionally implement the asynchronous methods
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)
