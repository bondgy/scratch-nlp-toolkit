import numpy as np
import torch
from torch import nn
from text_encoder import TextEncoder


class SelfAttention:
    def __init__(self, feature_dim):
        self.query = nn.Linear(feature_dim, feature_dim, bias=False, dtype=torch.double)
        self.key = nn.Linear(feature_dim, feature_dim, bias=False, dtype=torch.double)
        self.value = nn.Linear(feature_dim, feature_dim, bias=False, dtype=torch.double)

    def get_attention(self, x, heads_dim):
        example_dim, seq_dim, feature_dim = x.size()
        split_feature_dim = feature_dim // heads_dim
        query_x = self.query(x).view(example_dim, seq_dim, heads_dim, split_feature_dim).transpose(1, 2)
        key_x = self.key(x).view(example_dim, seq_dim, heads_dim, split_feature_dim).transpose(1, 2)
        value_x = self.value(x).view(example_dim, seq_dim, heads_dim, split_feature_dim).transpose(1, 2)
        query_x = query_x.reshape(heads_dim * example_dim, seq_dim, split_feature_dim)
        key_x = key_x.reshape(heads_dim * example_dim, seq_dim, split_feature_dim)
        value_x = value_x.reshape(heads_dim * example_dim, seq_dim, split_feature_dim)
        key_x = key_x / (feature_dim ** (1 / 4))
        query_x = query_x / (feature_dim ** (1 / 4))
        w = torch.bmm(query_x, key_x.transpose(1, 2))
        softmax = torch.softmax(w, 2, torch.double)
        y = torch.bmm(softmax, value_x).view((example_dim, heads_dim, split_feature_dim, seq_dim))
        y = y.transpose(1, 2).reshape(example_dim, seq_dim, feature_dim)
        return y

    @staticmethod
    def get_encoding(x, text):
        encoder = TextEncoder(text)
        sentence_dim, word_per_sentence_dim, embed_dim = x.shape
        positional_encoding = encoder.get_positional_encoding(sentence_dim, word_per_sentence_dim, embed_dim)
        encoding = x + positional_encoding
        return encoding

    @staticmethod
    def set_softmax(column):
        column = np.power(np.e, column)
        denominator = column.sum()
        return np.divide(column, denominator)


test = [[[3.0, 0.0, 5.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 5.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 13.0, 1.0]],
        [[2.0, 2.0, 2.0, 8.0, 2.0, 2.0], [2.0, 2.0, 6.0, 4.0, 2.0, 2.0], [3.0, 3.0, 10.0, 3.0, 3.0, 3.0]]]
test = np.array(test)
test = torch.tensor(test, dtype=torch.double)
norm1 = nn.LayerNorm(6, dtype=torch.double)
result = norm1(test)
print(test.shape)
print(result)
attention = SelfAttention(test.shape[2])
# print(attention.get_attention(test, 2))

