import argparse
import numpy as np
import sys

class WordEmbedding(object):
    def __init__(self, pretrainedVectorsFile):
        self.initialize(pretrainedVectorsFile, pretrainedVectorsFile)

    def initialize(self,vocab_file,vectors_file):
        #This piece of code of initialization is taken from official Github repository of GloVe
        with open(vocab_file, 'r') as f:
            words = [x.rstrip().split(' ')[0] for x in f.readlines()]

        with open(vectors_file, 'r') as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                vectors[vals[0]] = [float(x) for x in vals[1:]]

        vocab_size = len(words)
        vocab = {w: idx for idx, w in enumerate(words)}
        ivocab = {idx: w for idx, w in enumerate(words)}

        vector_dim = len(vectors[ivocab[0]])
        W = np.zeros((vocab_size, vector_dim))
        for word, v in vectors.items():
            if word == '<unk>':
                continue
            W[vocab[word], :] = v

        # normalize each word vector to unit variance
        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T

        self.W = W_norm
        self.vocab = vocab
        self.ivocab = ivocab

    def find_similar_words(self,text,refs,thresh):
        C = np.zeros((len(refs),self.W.shape[1]))
        for idx, term in enumerate(refs):
            if term in self.vocab:
                C[idx,:] = self.W[self.vocab[term], :]

        tokens = text.split(' ')
        scores = [0.] * len(tokens)
        found=[]

        for idx, term in enumerate(tokens):
            if term in self.vocab:
                vec = self.W[self.vocab[term], :]
                cosines = np.dot(C,vec.T)
                score = np.mean(cosines)
                scores[idx] = score
                if (score > thresh):
                    found.append({"term": term, "score": score})

        return sorted(found, key = lambda entry: entry["score"], reverse=True)

    def find_cosine_distance(self, firstWord, secondWord):
        firstVec = self.W[self.vocab[firstWord], :]
        secondVec = self.W[self.vocab[secondWord], :]
        cosine_similarity = np.dot(firstVec, secondVec)/(np.linalg.norm(firstVec)* np.linalg.norm(secondVec))
        return cosine_similarity

    def find_missing_word(self, firstWord, secondWord, thirdWord):
        distance = self.find_cosine_distance(firstWord, secondWord)
        possible_words = []

        for term in self.vocab:
            distance_temp = self.find_cosine_distance(thirdWord, term)
            if abs(distance_temp - distance) < 0.1:
                possible_words.append({"term": term, "diff": abs(distance_temp - distance)})

        sorted_words = sorted(possible_words, key = lambda entry: entry["diff"])
        terms_alone = [term["term"] for term in sorted_words]
        return terms_alone[:100]

    def find_close_words(self, word):
        possible_words = []
        for term in self.vocab:
            distance_temp = 1.0 - self.find_cosine_distance(word, term)
            if distance_temp >= 0 and distance_temp < 1.0:
                possible_words.append({"term": term, "distance": abs(distance_temp)})

        sorted_words = sorted(possible_words, key = lambda entry: entry["distance"])
        terms_alone = [term["term"] for term in sorted_words]
        return terms_alone[:200]

    def sum_vecs(self,text):

      tokens = text.split(' ')
      vec = np.zeros(self.W.shape[1])

      for idx, term in enumerate(tokens):
          if term in self.vocab:
              vec = vec + self.W[self.vocab[term], :]
      return vec

    def sum_weighted_vecs(self,text, weightVector):

      tokens = text.split(' ')
      vec = np.zeros(self.W.shape[1])

      for idx, term in enumerate(tokens):
          if term in self.vocab:
              wordVector = self.W[self.vocab[term], :];
              vec = vec + [weightVector[idx] * value for value in wordVector]
      return vec

    def get_wordvec_size(self):
        return self.W.shape[1]


