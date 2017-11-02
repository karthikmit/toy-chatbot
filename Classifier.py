import WordEmbedding
import numpy as np

class IntentClassifier:
  def __init__(self, WordEmbedding):
    self.embed = WordEmbedding
    self.intents = {}
    self.tfMap = {}
    self.idfMap = {}

  def get_centroid(self, trainingSet, docName):
    C = np.zeros((len(trainingSet), self.embed.get_wordvec_size()))
    for idx, text in enumerate(trainingSet):
        C[idx,:] = self.embed.sum_weighted_vecs(text.lower(), self.getTextVector(text, docName))

    centroid = np.mean(C,axis=0)
    assert centroid.shape[0] == self.embed.get_wordvec_size()
    return centroid

  def get_intent(self,text):
    keys = self.intents.keys();
    
    tokens = text.split();
    processedTokens = [token for token in tokens if token.isalpha()];
    processedText = " ".join(processedTokens);

    #vec = self.embed.sum_weighted_vecs(processedText.lower(), self.getSingleTextVector(processedText))
    #scores = np.array([ np.linalg.norm(vec - self.intents[label]["centroid"]) for label in keys])
    #print keys
    #print scores

    # go over every document, and try what if the new text part of the existing training set and calculate the score by taking diff of old and new centroid.
    scores = []
    for label in keys:
      old_centroid = self.intents[label]["centroid"];
      
      for token in processedTokens:
        if token in self.tfMap[label]:
          self.tfMap[label][token] += 1
        else:
          self.tfMap[label][token] = 1

        if token in self.idfMap:
          self.idfMap[token] += 1
        else:
          self.idfMap[token] = 1

      training_set = self.data[label]["examples"];
      training_set.append(processedText);
      new_centroid = self.get_centroid(training_set, label);
      scores.append(np.linalg.norm(old_centroid - new_centroid));
      training_set.remove(processedText);

      for token in processedTokens:
        self.tfMap[label][token] -= 1
        self.idfMap[token] -= 1
        if self.tfMap[label][token] == 0:
          self.tfMap[label].pop(token, None);
        if self.idfMap[token] == 0:
          self.idfMap.pop(token, None);

    print "Scores :: "
    print scores
    return keys[np.argmin(scores)]

  def train_examples(self, data):
    self.preprocess_data(data);
    self.data = data
    for key in data:
      trainingSet = data[key]["examples"];
      self.populateTfMap(key, trainingSet)

    for label in data.keys():
      if label not in self.intents:
        self.intents[label] = {}

      self.intents[label]["centroid"] = self.get_centroid(data[label]["examples"], label)
      self.intents[label]["centroid_norm"] = np.linalg.norm(self.intents[label]["centroid"])
 
    print "Centroid Info"
    print self.intents

  def preprocess_data(self, data):
    for key in data:
      trainingSet = data[key]["examples"];
      processedSet = [];
      for trainingSample in trainingSet:
        tokens = trainingSample.split();
        processedTokens = [token for token in tokens if token.isalpha()];
        processedSample = " ".join(processedTokens);
        processedSet.append(processedSample);

      data[key]["examples"] = processedSet;

    print "Processed Data :: "
    print data


  def getTextVector(self, text, docName):
    text = text.lower();
    tokens = text.split();
    tokenFreqMap = {};
    for token in tokens:
      tfValue = self.tfMap[docName][token];
      idfValue = self.idfMap[token];
      tfIdfValue = tfValue / float(idfValue);
      tokenFreqMap[token] = tfIdfValue;

    tfIdfSum = sum([tokenFreqMap[key] for key in tokenFreqMap]);
    return [tokenFreqMap[key] / float(tfIdfSum) for key in tokenFreqMap];

  def getSingleTextVector(self, text):
    text = text.lower();
    tokens = text.split();
    tokenFreqMap = {};
    for token in tokens:
      tfValue = 1.0;
      if token in self.idfMap:
        idfValue = self.idfMap[token];
      else:
        idfValue = 1.0;

      tfIdfValue = tfValue / float(idfValue);
      tokenFreqMap[token] = tfIdfValue;

    tfIdfSum = sum([tokenFreqMap[key] for key in tokenFreqMap]);
    return [tokenFreqMap[key] / float(tfIdfSum) for key in tokenFreqMap];

  def populateTfMap(self, docName, trainingSet):
    if docName not in self.tfMap:
      self.tfMap[docName] = {}

    for trainingDoc in trainingSet:
      tokens = trainingDoc.lower().split();
      for token in tokens:
        if token in self.tfMap[docName]:
          self.tfMap[docName][token] += 1
        else:
          self.tfMap[docName][token] = 1

        if token in self.idfMap:
          self.idfMap[token] += 1
        else:
          self.idfMap[token] = 1


