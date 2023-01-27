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
        self.Invbins = {}
        self.InvbinsScore = {}
        return self

    def search(self, query):
        for term in query:
            try: 
                print (self.Invbins[term], self.InvbinsScore[term])
            except(KeyError):
                print (" ")

    def makeindex(self, sentences_train):
        print ("fitting ")
        self.fit(sentences_train)
        # allwords = np.unique(flatten(sentences_train))
        for i,sentence in enumerate(sentences_train):
            for word, freq in self.tf_[i].items():
                # freq = self.tf_[i][word]
                doc_len = self.doc_len_[i]
                numerator = self.idf_[word] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len/self.avg_doc_len_)
                score = (numerator / denominator)
                try:
                    self.Invbins[word].append(i)
                    self.InvbinsScore[word].append(score)
                except(KeyError):
                    self.Invbins[word] = []
                    self.InvbinsScore[word] = []
                    self.Invbins[word].append(i)
                    self.InvbinsScore[word].append(score)
        # sort based on BM25 scores
        for term in self.Invbins:
            self.InvbinsScore[term] = np.array(self.InvbinsScore[term])
            order = np.argsort(self.InvbinsScore[term])[::-1]
            self.InvbinsScore[term] = self.InvbinsScore[term][order]
            self.Invbins[term] = np.array(self.Invbins[term])[order]    
        return [self.Invbins,self.InvbinsScore]