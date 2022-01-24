import math
import re
import pandas as pd


class TextEncoder:

    """
    Class for encoding text into vectors using common encoding methods.
    Attributes
    ----------
    delimiter : str
        delimiter used for splitting text into words via the string.split() method
    """

    delimiter = " "

    def __init__(self, text_arg):
        self.text = TextEncoder.clean_text(text_arg).lower()
        self.sentences = self.get_sentences()
        self.words = self.get_words()
        self.data_corpus = pd.Series(self.words, dtype=pd.Int64Dtype).unique()

    @staticmethod
    def clean_text(text_arg):
        """
        Cleans the text by removing all characters outside of end-of-sentence punctuation and roman alphabet characters
        :param text_arg: the text to be cleaned
        :return: a cleaned copy of the string
        """
        cleaned_text = re.match('([A-Z a-z.?!])+', text_arg).string
        return cleaned_text

    def get_words(self):
        """
        Gets an array of unique words from the sentence class attribute
        :return: an array of the unique words
        """
        words = []
        for sentence in self.sentences:
            for word in sentence.split(self.delimiter):
                if word is not None and word != "":
                    words.append(word)
        return words

    def get_sentences(self):
        """
        Gets the sentences from the text class attribute; sentences split on '?', '.', and '!'.
        :return: A list containing the individual sentences
        """
        sentences = re.split("[.!?]", self.text)
        for index in range(len(sentences)):
            sentences[index] = sentences[index].strip()
        sentences.pop(-1)
        return sentences

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
            vectors.loc[sentence_index] = vector
        return vectors

    def get_tf_idf(self):
        """
        Uses the Term Frequency - Inverse Document Frequency methodology to encode the text
        :return: a pandas DataFrame containing the encoded text
        """
        vectors = pd.DataFrame(index=range(len(self.sentences)), columns=[self.data_corpus], dtype=float)
        idf = pd.Series(index=vectors.columns, dtype=float)
        total_words = len(self.words)
        for word in self.data_corpus:
            word_counter = 0
            for sentence in self.sentences:
                if word in sentence:
                    word_counter += 1
            idf[word] = math.log(total_words / word_counter)
        for sentence_index in range(len(self.sentences)):
            sentence = self.sentences[sentence_index]
            vector = pd.Series(index=vectors.columns, dtype=float)
            sentence_word_count = len(sentence)
            for word in sentence.split(self.delimiter):
                if vector[word] == 0:
                    word_count = sentence.count(word)
                    tf = word_count / sentence_word_count
                    vector[word] = tf * idf[word]
            vectors.loc[sentence_index] = vector

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
