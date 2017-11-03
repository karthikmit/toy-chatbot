import cherrypy
import Classifier
import WordEmbedding
import simplejson

class ClassifierController(object):
    @cherrypy.expose
    def index(self):
        return "Hello world!";

    @cherrypy.expose
    def health(self):
    	return "I am alright";

    @cherrypy.expose
    def classify(self, text):
    	print "text:"
    	print text
    	return self.classifier.get_intent(text);

    @cherrypy.expose
    def distance(self, first, second):
    	return str(self.wordEmbedding.find_cosine_distance(first, second));

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def closeWords(self, word):
    	return self.wordEmbedding.find_close_words(word);	

    def __init__(self):
		self.wordEmbedding = WordEmbedding.WordEmbedding("./model.txt");
		self.classifier = Classifier.IntentClassifier(self.wordEmbedding);
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
		  },
		  "hotel_status_query": {
		    "examples": [
		      "What is the status of hotel booking NH1234566",
		      "show me the status of hotel booking NH1234566",
		      "Status of my hotel booking NH23467",
		      "Status of hotel booking NH23467",
		      "current Status of hotel booking NH23467",
		    ]
		  }
		}

		self.classifier.train_examples(data);


if __name__ == '__main__':
    cherrypy.quickstart(ClassifierController())