import torch

class Tokenizer:
    def __init__(self, vocab, ignore_case=False, oov_token=' '):
        """
        Args:
            vocab (str): vocabulary
            ignore_case (bool): should uppercase letters be treated as lowercase letters
            oov_token (str): out-of-vocabulary token
        """
        self.vocab = vocab
        self.ignore_case = ignore_case
        self.oov_token = oov_token
        self.tokenizer = {}
        for i, char in enumerate(vocab):
            if self.ignore_case:
                char = char.lower()
            if char not in self.tokenizer:
                self.tokenizer[char] = i + 1 # add 1 because 0 is reserved for 'blank'
        if self.oov_token not in self.tokenizer:
            self.tokenizer[self.oov_token] = len(vocab) + 1

    def encode(self, texts):
        """
        Convert text to a list of indexes.

        Args:
            texts (str or List[str]): text to be encoded
        
        Returns:
            indexed_tokens (IntTensor): encoded text
            lengths (IntTensor): length of each text
        """
        if isinstance(texts, str):
            texts = [texts]
        lengths = [len(text) for text in texts]
        texts = ''.join(texts)
        indexed_tokens = []
        for text in texts:
            for char in text:
                if char in self.tokenizer:
                    indexed_tokens.append(self.tokenizer[char])
                else:
                    indexed_tokens.append(self.tokenizer[self.oov_token])
        return torch.IntTensor(indexed_tokens), torch.IntTensor(lengths)


    def decode(self, indexed_tokens, lengths):
        """
        Convert a list of indexes tokens back to text.

        Args:
            indexed_tokens (IntTensor): encoded texts.
            lengths (IntTensor): length of each encoded text.

        Returns:
            texts (List): list of decoded texts.
        """
        if lengths.numel() == 1: # if the number of elements in lengths is 1
            length = lengths.item()
            assert indexed_tokens.numel() == length, f"text with length: {indexed_tokens.numel()} does not match declared length: {length}"
            text = []
            # merge repeated characters (unless they are separated by blank) and
            # drop blank characters e.g AABB -> AB, A A A -> AAA
            for i in range(length):
                if indexed_tokens[i] == 0:
                    continue
                if indexed_tokens[i] > 0 and indexed_tokens[i - 1] == indexed_tokens[i]:
                    continue
                char = self.vocab[indexed_tokens[i] - 1]
                text.append(char)
            return ''.join(text)
        else:
            assert indexed_tokens.numel() == lengths.sum(), f"texts with length: {indexed_tokens.numel()} does not match declared length: {lengths.sum()}"
            texts = []
            j = 0
            for length in lengths:
                text = self.decode(indexed_tokens[j:j+length], length)
                texts.append(text)
                j += length
            return texts
