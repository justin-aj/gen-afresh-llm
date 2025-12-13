import os
from pathlib import Path
from typing import List, Optional, Union
from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass
    
    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        pass
    
    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        pass
    
    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Padding token ID."""
        pass


class SentencePieceTokenizer(BaseTokenizer):
    """
    Tokenizer using SentencePiece (used by LLaMA, Mistral, etc.)
    
    SentencePiece uses BPE (Byte Pair Encoding) or Unigram model
    to break text into subword tokens.
    
    Install: pip install sentencepiece
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to .model file (e.g., tokenizer.model)
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("Please install sentencepiece: pip install sentencepiece")
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
        # Special tokens
        self._bos_id = self.sp.bos_id()
        self._eos_id = self.sp.eos_id()
        self._pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else self._eos_id
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token
        
        Returns:
            List of token IDs
        """
        tokens = self.sp.Encode(text)
        
        if add_bos:
            tokens = [self._bos_id] + tokens
        if add_eos:
            tokens = tokens + [self._eos_id]
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.sp.Decode(tokens)
    
    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()
    
    @property
    def bos_token_id(self) -> int:
        return self._bos_id
    
    @property
    def eos_token_id(self) -> int:
        return self._eos_id
    
    @property
    def pad_token_id(self) -> int:
        return self._pad_id


class TiktokenTokenizer(BaseTokenizer):
    """
    Tokenizer using tiktoken (used by OpenAI GPT models).
    
    Fast BPE tokenizer with good support for code and multiple languages.
    
    Install: pip install tiktoken
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Args:
            encoding_name: Name of the encoding to use
                - "cl100k_base": GPT-4, ChatGPT
                - "p50k_base": GPT-3
                - "r50k_base": GPT-2
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError("Please install tiktoken: pip install tiktoken")
        
        self.enc = tiktoken.get_encoding(encoding_name)
        
        # tiktoken doesn't have built-in special tokens like BOS/EOS
        # We'll use common conventions
        self._bos_id = self.enc.encode("<|startoftext|>", allowed_special="all")[0] \
            if "<|startoftext|>" in self.enc._special_tokens else 0
        self._eos_id = self.enc.encode("<|endoftext|>", allowed_special="all")[0] \
            if "<|endoftext|>" in self.enc._special_tokens else 0
        self._pad_id = self._eos_id
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.enc.encode(text)
        
        if add_bos and self._bos_id:
            tokens = [self._bos_id] + tokens
        if add_eos and self._eos_id:
            tokens = tokens + [self._eos_id]
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.enc.decode(tokens)
    
    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab
    
    @property
    def bos_token_id(self) -> int:
        return self._bos_id
    
    @property
    def eos_token_id(self) -> int:
        return self._eos_id
    
    @property
    def pad_token_id(self) -> int:
        return self._pad_id


class SimpleTokenizer(BaseTokenizer):
    """
    Simple character-level tokenizer for testing/learning.
    
    Not suitable for real use, but helps understand the concept.
    """
    
    def __init__(self, vocab: Optional[str] = None):
        """
        Args:
            vocab: String of characters to include in vocabulary.
                   If None, uses basic ASCII + common chars.
        """
        if vocab is None:
            # Basic vocabulary: letters, digits, punctuation, space
            vocab = (
                " \n\t"
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            )
        
        # Special tokens
        self._pad_token = "<PAD>"
        self._bos_token = "<BOS>"
        self._eos_token = "<EOS>"
        self._unk_token = "<UNK>"
        
        # Build vocabulary
        special_tokens = [self._pad_token, self._bos_token, self._eos_token, self._unk_token]
        self._id_to_token = special_tokens + list(vocab)
        self._token_to_id = {tok: i for i, tok in enumerate(self._id_to_token)}
        
        # Special token IDs
        self._pad_id = 0
        self._bos_id = 1
        self._eos_id = 2
        self._unk_id = 3
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs (character by character)."""
        tokens = []
        
        if add_bos:
            tokens.append(self._bos_id)
        
        for char in text:
            token_id = self._token_to_id.get(char, self._unk_id)
            tokens.append(token_id)
        
        if add_eos:
            tokens.append(self._eos_id)
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        chars = []
        for token_id in tokens:
            if token_id < len(self._id_to_token):
                token = self._id_to_token[token_id]
                # Skip special tokens in output
                if token not in [self._pad_token, self._bos_token, self._eos_token]:
                    chars.append(token if token != self._unk_token else "�")
        return "".join(chars)
    
    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)
    
    @property
    def bos_token_id(self) -> int:
        return self._bos_id
    
    @property
    def eos_token_id(self) -> int:
        return self._eos_id
    
    @property
    def pad_token_id(self) -> int:
        return self._pad_id


def get_tokenizer(name_or_path: str) -> BaseTokenizer:
    """
    Factory function to get the appropriate tokenizer.
    
    Args:
        name_or_path: Either:
            - Path to a .model file (SentencePiece)
            - "tiktoken:encoding_name" (e.g., "tiktoken:cl100k_base")
            - "simple" for character-level tokenizer
    
    Returns:
        Tokenizer instance
    """
    if name_or_path == "simple":
        return SimpleTokenizer()
    
    if name_or_path.startswith("tiktoken:"):
        encoding_name = name_or_path.split(":")[1]
        return TiktokenTokenizer(encoding_name)
    
    if Path(name_or_path).exists():
        return SentencePieceTokenizer(name_or_path)
    
    raise ValueError(f"Unknown tokenizer: {name_or_path}")


if __name__ == "__main__":
    # Test SimpleTokenizer (always available)
    print("=== Simple Tokenizer ===")
    tokenizer = SimpleTokenizer()
    
    text = "Hello, World!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original:    '{text}'")
    print(f"Tokens:      {tokens}")
    print(f"Decoded:     '{decoded}'")
    print(f"Vocab size:  {tokenizer.vocab_size}")
    print(f"BOS ID:      {tokenizer.bos_token_id}")
    print(f"EOS ID:      {tokenizer.eos_token_id}")
    
    # Test with tiktoken (if installed)
    print("\n=== Tiktoken (if installed) ===")
    try:
        tokenizer = TiktokenTokenizer("cl100k_base")
        
        text = "Hello, World! The quick brown fox jumps over the lazy dog."
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        print(f"Original:    '{text}'")
        print(f"Tokens:      {tokens}")
        print(f"Num tokens:  {len(tokens)}")
        print(f"Decoded:     '{decoded}'")
        print(f"Vocab size:  {tokenizer.vocab_size}")
        
        # Show how subwords work
        print("\n--- Subword examples ---")
        examples = ["unhappiness", "tokenization", "ChatGPT", "🎉"]
        for word in examples:
            toks = tokenizer.encode(word)
            pieces = [tokenizer.decode([t]) for t in toks]
            print(f"'{word}' → {toks} → {pieces}")
            
    except ImportError:
        print("tiktoken not installed. Run: pip install tiktoken")
    
    # Test factory function
    print("\n=== Factory function ===")
    tok = get_tokenizer("simple")
    print(f"Got tokenizer: {type(tok).__name__}")
