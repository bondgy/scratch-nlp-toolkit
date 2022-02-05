import math
import re
import pandas as pd
import numpy as np


class TextEncoder:
    """
    Class for encoding text into vectors using common encoding methods.
    Attributes
    ----------
    delimiter : str
        delimiter used for splitting text into words via the string.split() method
    """

    delimiter = " "

    sentence_delimiter = "\?”|\.”|\”!|\?\"|.\"|\!\"|[.?!]+"

    word_delimiter = "\?”|\.”|\”!|\?\"|.\"|\!\"|[.?! ]+"

    def __init__(self, text_arg, clean_regex=None, lower=True):
        if clean_regex is not None:
            self.text = TextEncoder.clean_text(text_arg, clean_regex)
        else:
            self.text = TextEncoder.clean_text(text_arg)
        if lower:
            self.text = self.text.lower()
        self.sentences = self.get_sentences(self.text, self.sentence_delimiter)
        self.words = self.get_words()
        self.data_corpus = pd.Series(self.words, dtype=pd.Int64Dtype).unique()

    def get_positional_encoding(self, sentences_length, word_per_sentence, embedding_length):
        """
        Creates a positional encoding vector from supplied text for a transformer/self-attention unit input
        :param embedding_length: The length of the embedding
        :param word_per_sentence: The maximum amount of words per sentence
        :param sentences_length: The maximum amount of sentences
        :return: a three dimensional numpy array containing the positional encoding
        """
        position_counter = 1
        encoding = np.zeros((sentences_length, word_per_sentence, embedding_length))
        embedding_vector = np.zeros((1, 1, embedding_length))
        for embed_position in range(0, embedding_length, 1):
            embedding_vector[0, 0, embed_position] = 10000 ** ((2 * (embed_position + 1)) / embedding_length)
        for sentence_index in range(len(self.sentences)):
            sentence = self.sentences[sentence_index]
            sentence_split = sentence.split(TextEncoder.delimiter)
            for word_index in range(len(sentence_split)):
                embed_vector_copy = embedding_vector[0, 0, :].copy()
                embed_vector_copy = position_counter / embed_vector_copy
                if position_counter % 2 == 0:
                    embed_vector_copy = np.sin(embed_vector_copy)
                else:
                    embed_vector_copy = np.cos(embed_vector_copy)
                encoding[sentence_index: sentence_index + 1, word_index:word_index + 1, 0:embedding_length] =\
                    embed_vector_copy
                position_counter += 1
        return encoding

    @staticmethod
    def clean_text(text_arg, regex='[A-Z a-z.?!\'’]+'):
        """
        Cleans the text by removing all characters outside of end-of-sentence punctuation and roman alphabet characters
        :param regex: optional regex expression to clean the text with
        :param text_arg: the text to be cleaned
        :return: a cleaned copy of the string
        """
        matches = re.findall(regex, text_arg)
        cleaned_text = "".join(matches).strip().replace("  ", " ")
        return cleaned_text

    def get_words(self):
        """
        Gets an array of words from the sentence class attribute
        :return: an array of the unique words
        """
        words = re.split(self.word_delimiter, self.text)
        return words

    @staticmethod
    def get_sentences(text, sentence_delimiter):
        """
        Gets the sentences from the text class attribute; sentences split on '?', '.', and '!'.
        :return: A list containing the individual sentences
        """
        sentences = re.split(sentence_delimiter, text)
        for index in range(len(sentences)):
            sentences[index] = sentences[index].strip()
        if sentences[-1] == '':
            sentences.pop(-1)
        return sentences

    def get_sentence_punctuation(self):
        punctuation = re.findall(self.sentence_delimiter, self.text)
        return punctuation

    def get_bag_of_words(self, method='binary'):
        """
        Uses the bag of words methodology to encode the text
        :param method: string indicating which type of bag of words to use; possible values are binary or common.
        :return: a pandas DataFrame containing the vectors
        """
        vectors = pd.DataFrame(index=range(len(self.sentences)), columns=[self.data_corpus], dtype=int)
        for sentence_index in range(len(self.sentences)):
            sentence = self.sentences[sentence_index]
            vector = pd.Series(index=vectors.columns, dtype=int)
            for word in sentence.split(self.delimiter):
                if vector[word] == 0 or method != 'binary':
                    vector[word] += 1
            vectors.loc[sentence_index, :] = vector
        return vectors

    def get_tf_idf(self, sentences=None):
        """
        Uses the Term Frequency - Inverse Document Frequency methodology to encode the text
        :return: a pandas DataFrame containing the encoded text
        """
        if sentences is None:
            sentences = self.sentences
        vectors = pd.DataFrame(index=range(len(sentences)), columns=[self.data_corpus], dtype=float)
        idf = pd.Series(index=vectors.columns, dtype=float)
        total_words = len(self.words)
        for word in self.data_corpus:
            word_counter = 0
            for sentence in sentences:
                if word in sentence:
                    word_counter += 1
            idf[word] = math.log(total_words / word_counter)
        for sentence_index in range(len(sentences)):
            sentence = sentences[sentence_index].strip()
            vector = pd.Series(index=vectors.columns, dtype=float)
            sentence_word_count = len(sentence)
            for word in sentence.split(self.delimiter):
                if vector[word] == 0:
                    word_count = sentence.count(word)
                    tf = word_count / sentence_word_count
                    vector[word] = tf * idf[word]
            vectors.loc[sentence_index] = vector
        return vectors

    def get_index_encoding(self):
        """
        Encodes the text via the index encoding methodology
        :return: a pandas DataFrame containing the encoded text
        """
        vectors = pd.DataFrame(index=range(len(self.sentences)), columns=[self.data_corpus], dtype=int)
        max_length = 0
        corpus_list = list(self.data_corpus)
        for sentence in self.sentences:
            sentence_length = len(sentence.split(self.delimiter))
            if sentence_length > max_length:
                max_length = sentence_length
        for sentence_index in range(len(self.sentences)):
            sentence = self.sentences[sentence_index]
            vector = pd.Series(index=range(max_length), dtype=int)
            sentence_words = sentence.split(self.delimiter)
            for word_index in range(len(sentence_words)):
                vector[word_index] = corpus_list.index(sentence_words[word_index])
            vectors.loc[sentence_index] = vector
        return vectors
