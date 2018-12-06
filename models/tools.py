


import os
import torch
from torch import nn
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def lookup_word(self, idx):
        return self.idx2word[idx]

    def lookup_sequence(self, idx_list):
        result = []
        for i in idx_list:
          result.append(self.idx2word[i])
        return result

    @staticmethod
    def predict(some_string, test_model, corpus):
        """
        Example usage:
        --------------
        >>> corpus = Corpus('./gdrive/My Drive/nlp/data/raw/penn-treebank')
        >>> file_path = './gdrive/My Drive/nlp/model_wisse1'
        >>> test_model = torch.load(file_path).to(device)
        >>> corpus = Corpus('./gdrive/My Drive/nlp/data/raw/penn-treebank')
        >>> Dictionary.predict('new york stock', test_model = test_model, corpus=corpus)
        'exchange'
        >>> Dictionary.predict('las ', test_model = test_model, corpus=corpus)
        'vegas'
        """
        tokens = corpus.tokenize_string(some_string)
        tokens = tokens.unsqueeze(1)
        hidden = test_model.init_hidden(1)
        output, _ = test_model.forward(tokens, hidden)
        _, indices = torch.max(output, 2)
        indices = indices.data.cpu().numpy()
        word_list = corpus.dictionary.lookup_sequence(indices[:,0])
        output_words = " ".join(word_list)
        return output_words

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

    def tokenize_string(self, s):
        """Tokenizes a text file."""

        tokens = s.split()
        # Tokenize file content
        ids = torch.LongTensor(len(tokens))
        for i, word in enumerate(tokens):
            ids[i] = self.dictionary.word2idx[word]

        return ids

    def tokenize(self, path):
        """Tokenizes a text file."""
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:

            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


# Utility module to deal with lists of layers, see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
#     print('repackage hiddem', type(hidden))
    if isinstance(h, torch.Tensor):
      return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if model_type == 'LSTM':
      hidden = model.init_hidden(eval_batch_size)
    else:
      hidden, _ = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

