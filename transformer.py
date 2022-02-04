import numpy as np
# Currently a numpy implementation of pbloem's attention head


class SelfAttention:
    def __init__(self, feature_dim):
        self.query = np.random.rand(feature_dim, feature_dim)
        self.key = np.random.rand(feature_dim, feature_dim)
        self.value = np.random.rand(feature_dim, feature_dim)

    def get_attention(self, x, heads_dim):
        example_dim, seq_dim, feature_dim = x.shape
        split_feature_dim = feature_dim // heads_dim
        query_x = np.einsum('klj,vj', x, self.query).reshape((example_dim, seq_dim, heads_dim, split_feature_dim)).swapaxes(1, 2)
        key_x = np.einsum('klj,vj', x, self.key).reshape((example_dim, seq_dim, heads_dim, split_feature_dim)).swapaxes(1, 2)
        value_x = np.einsum('klj,vj', x, self.value).reshape((example_dim, seq_dim, heads_dim, split_feature_dim)).swapaxes(1, 2)
        query_x = query_x.reshape((heads_dim * example_dim, seq_dim, split_feature_dim))
        key_x = key_x.reshape((heads_dim * example_dim, seq_dim, split_feature_dim))
        value_x = value_x.reshape((heads_dim * example_dim, seq_dim, split_feature_dim))
        key_x = key_x / (feature_dim ** (1 / 4))
        query_x = query_x / (feature_dim ** (1 / 4))
        w = np.einsum('bnm,bmp->bnp', query_x, key_x.swapaxes(1, 2))
        softmax = np.apply_along_axis(self.set_softmax, 2, w.copy())
        y = np.einsum('bnm,bmp->bnp', softmax, value_x).reshape((example_dim, heads_dim, split_feature_dim, seq_dim))
        y = y.swapaxes(1, 2).reshape(example_dim, seq_dim, feature_dim)
        return y

    @staticmethod
    def set_softmax(column):
        column = np.power(np.e, column)
        denominator = column.sum()
        return np.divide(column, denominator)


