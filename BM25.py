import math
import numpy as np

class BM25:
    """
    Best Match 25.

    Parameters
    ----------
    k1 : float, default 1.5

    b : float, default 0.75
    """

    def __init__(self, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):
        """
        Fit the various statistics that are required to calculate BM25 ranking
        score using the corpus given.

        Parameters
        ----------
        corpus : list[list[str]]
            Each element in the list represents a document, and each document
            is a list of the terms.

        Returns
        -------
        self
        """
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            tf.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query):
        scores = [self._score(query, index) for index in range(self.corpus_size_)]
        return scores

    def _score(self, query, index):
        score = 0.0
        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index] # index represent document number
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)
        return score

def makeindex(sentences_train):
    # bm25Invbins = {}
    bm25 = BM25()
    print ("fitting ")
    bm25.fit(sentences_train)
    # allwords = np.unique(flatten(sentences_train))

    Invbins = {}
    InvbinsScore = {}
    for i,sentence in enumerate(sentences_train):
        for word, freq in bm25.tf_[i].items():
            # freq = bm25.tf_[i][word]
            doc_len = bm25.doc_len_[i]
            numerator = bm25.idf_[word] * freq * (bm25.k1 + 1)
            denominator = freq + bm25.k1 * (1 - bm25.b + bm25.b * doc_len/bm25.avg_doc_len_)
            score = (numerator / denominator)
            try:
                Invbins[word].append(i)
                InvbinsScore[word].append(score)
            except(KeyError):
                Invbins[word] = []
                InvbinsScore[word] = []
                Invbins[word].append(i)
                InvbinsScore[word].append(score)
    # sort based on BM25 scores
    for term in Invbins:
        InvbinsScore[term] = np.array(InvbinsScore[term])
        order = np.argsort(InvbinsScore[term])[::-1]
        InvbinsScore[term] = InvbinsScore[term][order]
        Invbins[term] = np.array(Invbins[term])[order]    
    return [Invbins,InvbinsScore]