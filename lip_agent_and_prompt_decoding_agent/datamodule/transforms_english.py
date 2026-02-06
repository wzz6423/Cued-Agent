#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
English character-level text transform for lip reading.
"""

import torch


class TextTransform_English:
    """Character-level tokenizer for English lip reading."""

    def __init__(self):
        # Character vocabulary: space + A-Z + apostrophe (matching dataset format)
        # Token mapping: 0=blank, 1=space, 2-27=A-Z, 28=apostrophe, 29=unk, 30=eos
        chars = " ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
        self.hashmap = {c: i + 1 for i, c in enumerate(chars)}  # 1-indexed
        self.hashmap["<unk>"] = len(chars) + 1

        # Token list: blank (0) + chars (1-28) + unk (29) + eos (30)
        self.token_list = ["<blank>"] + list(chars) + ["<unk>", "<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        """Convert text to token IDs."""
        text = text.upper()
        token_ids = []
        for char in text:
            if char in self.hashmap:
                token_ids.append(self.hashmap[char])
            else:
                token_ids.append(self.hashmap["<unk>"])
        return torch.tensor(token_ids)

    def post_process(self, token_ids):
        """Convert token IDs back to text."""
        token_ids = token_ids[token_ids != self.ignore_id]
        text = self._ids_to_str(token_ids, self.token_list)
        return text.strip()

    def _ids_to_str(self, token_ids, char_list):
        """Convert token ID tensor to string."""
        result = []
        for idx in token_ids:
            idx = int(idx)
            if 0 <= idx < len(char_list):
                char = char_list[idx]
                if char not in ["<blank>", "<eos>", "<unk>"]:
                    result.append(char)
        return "".join(result)


if __name__ == '__main__':
    # Test the tokenizer
    transform = TextTransform_English()
    print(f"Vocabulary size: {len(transform.token_list)}")
    print(f"Token list: {transform.token_list}")

    test_text = "HELLO WORLD"
    tokens = transform.tokenize(test_text)
    print(f"\nTest: '{test_text}'")
    print(f"Token IDs: {tokens.tolist()}")

    recovered = transform.post_process(tokens)
    print(f"Recovered: '{recovered}'")
