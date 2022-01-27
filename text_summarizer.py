import nltk
from nltk.corpus import stopwords
from text_encoder import TextEncoder
from collections import defaultdict
nltk.download('stopwords')


class ExtractiveSummarizer:
    def __init__(self, text):
        self.encoder = TextEncoder(text)
        self.original_sentences = TextEncoder.get_sentences(text, TextEncoder.sentence_delimiter)
        stop_words = stopwords.words('english')
        self.uncommon_words = list(filter(lambda word: word not in stop_words, self.encoder.words))
        self.uncommon_sentences, self.total_uncommon_words = self.get_uncommon_sentences()
        self.punctuations = self.encoder.get_sentence_punctuation()

    def get_uncommon_sentences(self):
        uncommon_sentences = []
        total_uncommon_words = 0
        for sentence in self.encoder.sentences:
            uncommon_sentence = []
            for word in sentence.split(" "):
                if word not in self.uncommon_words:
                    uncommon_sentence.append(word)
            uncommon_sentences.append(uncommon_sentence)
            total_uncommon_words += len(uncommon_sentence)
        return uncommon_sentences, total_uncommon_words

    def summarize_by_word_frequency(self, sentence_amount):
        frequencies = defaultdict(lambda: 0.0, {})
        for sentence in self.uncommon_sentences:
            for word in sentence:
                frequencies[word] += 1
        for word_key in frequencies.keys():
            frequencies[word_key] = frequencies[word_key] / self.total_uncommon_words
        sentence_ranks = []
        for sentence in self.uncommon_sentences:
            rank = 0
            for word in sentence:
                rank += frequencies[word]
            rank = rank / len(sentence)
            sentence_ranks.append(rank)
        summarized_text = self.get_summarized_sentences(sentence_ranks, len(self.uncommon_sentences), sentence_amount)
        return summarized_text

    def summarize_by_if_idf(self, sentence_amount):
        tf_idf = self.encoder.get_tf_idf()
        sentence_ranks = []
        for row_index in range(tf_idf.shape[0]):
            sentence_ranks.append(tf_idf.loc[row_index].mean())
        summarized_text = self.get_summarized_sentences(sentence_ranks, len(self.encoder.sentences), sentence_amount)
        return summarized_text

    def get_summarized_sentences(self, sentence_ranks, sentences_length, sentence_amount):
        sentence_indices = list(range(sentences_length))
        sentence_ranks, sentence_indices = zip(*sorted(zip(sentence_ranks, sentence_indices)))
        summarized_text = ""
        total_sentences = sentence_amount if sentence_amount <= len(sentence_indices) else len(sentence_indices)
        for index in range(total_sentences):
            sentence_index = sentence_indices[index]
            summarized_text += self.original_sentences[sentence_index] + self.punctuations[sentence_index] + " "
        return summarized_text.strip()


# sample implementation
sample_text = "Wikipedia’s “List of sexually active popes” is both useful and frivolous, impressive and incomplete. " \
              "When I showed it to Caitlin here at the Intersect, she declared it “simultaneously the perfect example " \
              "of Wikipedia’s promise and its failings.” So maybe I’m bad at finding the “best” one, but I feel like " \
              "this particular list is one of its most emblematic. The idea of collecting a list of popes who were " \
              "sexually active at some point speaks for itself – it’s a uniquely Wikipedian way of categorizing " \
              "history in the age of search engines. But I’m really into the entry’s talk page. Since its creation " \
              "more than a decade ago, this particular list has been argued over, corrected, and expanded with vigor. " \
              "One user went after some pretty egregious errors when it comes to Catholic terminology. There are also " \
              "substantial arguments over the neutrality of the entire concept of collecting sexually active popes – " \
              "whether historically confirmed or not. One lengthy talk thread is titled “Encyclopedia or tabloid” " \
              "and several Wikipedians drop in to accuse the entire article of being anti-Catholic, while others " \
              "defend it. It might not surprise you to find out that it’s been nominated for deletion, twice, " \
              "and somehow survived. Despite its critics, the page itself has been immortalized (kind of) in an XKCD " \
              "comic, view-able in the Alt text that pops up when you mouse over one of its images. "
summarized_sentence_amount = 4
summarizer = ExtractiveSummarizer(sample_text)
print(summarizer.summarize_by_word_frequency(summarized_sentence_amount))
print(summarizer.summarize_by_if_idf(summarized_sentence_amount))