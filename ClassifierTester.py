#ClassifierTester.py
import Classifier
import WordEmbedding

classifier = Classifier.IntentClassifier(WordEmbedding.WordEmbedding("./model.txt"))

data={
  "greet": {
    "examples" : ["hello","hey there","howdy","hello","hi","hey","hey ho"]
  },

  "pnr_status_query": {
    "examples":[
      "What is my pnr status 2436014775",
      "what is the pnr status of 2436014789",
      "status of pnr 2436014889",
      "Please tell my PNr status 2436014775",
      "Status of my PNR 2436014775",
      "PNR status for 2436014775",
      "what is the status of Pnr 2436014775",
      "Please tell me the status of my PNR 2436014775"
    ]
  },

  "train_status_query": {
    "examples": [
      "Where is the train 16236 now",
      "show me the status of train #16236",
      "when is the train 16236 expected at CRLM",
      "where is the train 16236 now",
      "tell me the train status, 16236",
      "live status of train 16236",
      "what is the current status of train 16236"
    ]
  }
}

classifier.train_examples(data);

for text in ["hi", "tell me the pnr status for 2436014234", "what is the status of train 16236"]:
    print "text : '{0}', predicted_label : '{1}'".format(text, classifier.get_intent(text))
