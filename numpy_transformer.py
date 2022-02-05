import numpy as np
from text_encoder import TextEncoder


class SelfAttention:
    """
    Currently a numpy implementation roughly based on pbloem's attention head
    """
    def __init__(self, feature_dim):
        """
        Initialization that sets the query, key, and value using the feature dimensions
        :param feature_dim: Number of features/embedding size per word
        """
        self.feature_dim = feature_dim
        self.query = np.random.rand(feature_dim, feature_dim)
        self.key = np.random.rand(feature_dim, feature_dim)
        self.value = np.random.rand(feature_dim, feature_dim)

    def get_attention(self, x, heads_dim, encoder_x=None, mask_amount=0):
        """
        Calculates the attention matrix for encoders and decoders
        :param x: The input from the embedding if present or the prior transformer unit if not
        :param heads_dim: The number of heads
        :param encoder_x: The output from the encoder layer to be added; used for an encoder-decoder unit
        :param mask_amount: The amount of sequences/words to hide
        :return: A matrix representation of the attention with the same size as x
        """
        example_dim, seq_dim, feature_dim = x.shape
        split_feature_dim = feature_dim // heads_dim
        query_x = np.einsum('klj,vj', x, self.query).reshape((example_dim, seq_dim, heads_dim,
                                                              split_feature_dim)).swapaxes(1, 2)
        if encoder_x is None:
            key_x = np.einsum('klj,vj', x, self.key).reshape((example_dim, seq_dim, heads_dim,
                                                              split_feature_dim)).swapaxes(1, 2)
            value_x = np.einsum('klj,vj', x, self.value).reshape((example_dim, seq_dim, heads_dim,
                                                                  split_feature_dim)).swapaxes(1, 2)
        else:
            key_x = np.einsum('klj,vj', encoder_x, self.key).reshape((example_dim, seq_dim, heads_dim,
                                                                      split_feature_dim)).swapaxes(1, 2)
            value_x = np.einsum('klj,vj', encoder_x, self.value).reshape((example_dim, seq_dim, heads_dim,
                                                                          split_feature_dim)).swapaxes(1, 2)
        query_x = query_x.reshape((heads_dim * example_dim, seq_dim, split_feature_dim))
        key_x = key_x.reshape((heads_dim * example_dim, seq_dim, split_feature_dim))
        value_x = value_x.reshape((heads_dim * example_dim, seq_dim, split_feature_dim))
        key_x = key_x / (feature_dim ** (1 / 4))
        query_x = query_x / (feature_dim ** (1 / 4))
        w = np.einsum('bnm,bmp->bnp', query_x, key_x.swapaxes(1, 2))
        if mask_amount != 0:
            w = w + self.get_mask(w, mask_amount)
        softmax = np.apply_along_axis(self.set_softmax, 2, w.copy())
        y = np.einsum('bnm,bmp->bnp', softmax, value_x).reshape((example_dim, heads_dim, split_feature_dim, seq_dim))
        y = y.swapaxes(1, 2).reshape(example_dim, seq_dim, feature_dim)
        return y

    @staticmethod
    def get_encoding(x, text):
        """
        Adds together the embedded and positional encoding
        :param x: The embedded encoding
        :param text: The text to be positionally encoded
        :return: Matrix representing the combined positional and embedded encoding
        """
        encoder = TextEncoder(text)
        sentence_dim, word_per_sentence_dim, embed_dim = x.shape
        positional_encoding = encoder.get_positional_encoding(sentence_dim, word_per_sentence_dim, embed_dim)
        encoding = x + positional_encoding
        return encoding

    @staticmethod
    def set_softmax(column):
        """
        Sets the softmax for the given layer
        :param column: The column to softmax
        :return: The softmax column
        """
        column = np.power(np.e, column)
        denominator = column.sum()
        return np.divide(column, denominator)

    @staticmethod
    def normalize(matrix1, matrix2=None):
        """
        Normalizes a given vector with the distribution centered at zero
        :param matrix1: The primary matrix to normalize
        :param matrix2: A secondary matrix to add before normalization; used for normalizing the attention head and
        the residual input
        :return: matrix normalized at zero
        """
        if matrix2 is not None:
            matrix = matrix1 + matrix2
        else:
            matrix = matrix1.copy()
        mean = matrix.mean()
        std = matrix.std()
        return (matrix - mean) / std

    @staticmethod
    def get_mask(x_matrix, number_shown):
        """
        Calculates a mask for the given matrix based on the number of words/sequences not masked
        :param x_matrix: The matrix to mask
        :param number_shown: The number of words/sequences that should be shown/not masked
        :return: A matrix containing a mask for x_matrix
        """
        mask = np.zeros((x_matrix.shape[-2], x_matrix.shape[-1]))
        if number_shown < x_matrix.shape[-1]:
            indices = x_matrix.shape[1] - number_shown
            y_s, x_s = np.diag_indices(indices)
            x_s = x_s + number_shown
            min_row = y_s.min()
            if min_row != 0:
                mask[0:min_row, :] = -np.inf
            for diagonal_x, diagonal_y in zip(x_s, y_s):
                mask[diagonal_y, diagonal_x:] = -np.inf
            mask = np.tile(mask, (x_matrix.shape[0], 1, 1))
        return mask
