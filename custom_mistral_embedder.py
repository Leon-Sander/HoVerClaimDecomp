import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
import torch
import torch.nn.functional as F
from torch import nn
from typing import Mapping, List
from transformers import PreTrainedTokenizerFast, BatchEncoding
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List


class CustomMistralEmbedder(Embeddings):
    def __init__(self, gpu_count, batch_size, device="cuda:0"):
        super().__init__()
        self.device = device#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gpu_count = gpu_count #torch.cuda.device_count()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        self.model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', torch_dtype=torch.float16)
        self.model.eval()
        self.model.to(self.device)
        self.task = 'Given a claim, retrieve documents that support or refute the claim'
        self.question_task = "Given a question, retrieve relevant documents that best answer the question"
        self.multi_hop_task = 'Given a multi-hop claim, retrieve documents that support or refute the claim'
        #'Given a multi-hop question, retrieve documents that can help answer the question'
        self.pool_type = "last"
        self.l2_normalize = True

        #if self.gpu_count > 1:
        #    self.model = nn.DataParallel(self.model)

    def last_token_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'


    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    @torch.no_grad()
    def _embed_text(self, text: str) -> List[float]:
        max_length = 4096
        batch_dict = create_batch_dict(self.tokenizer, [text], always_add_eos=(self.pool_type == 'last'), max_length=max_length)
        batch_dict = move_to_cuda(batch_dict, self.device)

        with torch.cuda.amp.autocast():
            outputs = self.model(**batch_dict)
            embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], self.pool_type)
            if self.l2_normalize:
                embeds = F.normalize(embeds, p=2, dim=-1)
        del batch_dict
        del outputs
        return embeds.cpu().numpy()[0].tolist()

    @torch.no_grad()
    def batch_embed_text_tensor_multiple_gpus(self, texts: List[str]) -> List[List[float]]:
        max_length = 4096
        embeddings_list = []

        # Process in batches
        batch_size = self.batch_size * self.gpu_count
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
            batch_dict = create_batch_dict(self.tokenizer, batch_texts, always_add_eos=(self.pool_type == 'last'), max_length=max_length)
            batch_dict = move_to_cuda(batch_dict, self.device)

            with torch.cuda.amp.autocast():
                outputs = self.model(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], self.pool_type)
                if self.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                embeddings_batch = embeds.cpu().numpy().tolist()
                embeddings_list.extend(embeddings_batch)
                del batch_dict
                del outputs

        return embeddings_list

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

def move_to_cuda(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device, non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def move_to_cuda_old(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def pool(last_hidden_states: Tensor,
         attention_mask: Tensor,
         pool_type: str) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weightedavg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
        attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
        s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        emb = s / d
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


def create_batch_dict(tokenizer: PreTrainedTokenizerFast, input_texts: List[str], always_add_eos: bool, max_length: int = 512) -> BatchEncoding:
    if not always_add_eos:
        return tokenizer(
            input_texts,
            max_length=max_length,
            padding=True,
            pad_to_multiple_of=8,
            return_token_type_ids=False,
            truncation=True,
            return_tensors='pt'
        )
    else:
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True
        )

        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]

        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

