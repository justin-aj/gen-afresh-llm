from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
import torch
from tokenizer import BaseTokenizer
from typing import List, Optional, Union


class HFTextDataset(TorchDataset):  # Changed from HFDataset to TorchDataset
    """Wrap a HuggingFace Dataset (text column) into the same chunked token dataset used by train.py"""
    def __init__(self, hf_dataset: HFDataset, text_column: str, tokenizer: BaseTokenizer, max_seq_len: int, stride: Optional[int] = None):
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len // 2
        print("Concatenating HF dataset texts...")
        # join examples (filter None)
        texts = [t for t in hf_dataset[text_column] if t is not None]
        joined = "\n".join(texts)
        self.tokens = tokenizer.encode(joined, add_bos=True, add_eos=True)
        self.num_samples = max(1, (len(self.tokens) - max_seq_len) // self.stride)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_seq_len + 1
        chunk = self.tokens[start:end]
        if len(chunk) < self.max_seq_len + 1:
            chunk = chunk + [0] * (self.max_seq_len + 1 - len(chunk))
        chunk = torch.tensor(chunk, dtype=torch.long)
        return chunk[:-1], chunk[1:]