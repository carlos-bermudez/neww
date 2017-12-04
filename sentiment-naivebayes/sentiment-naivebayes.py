import nltk
import csv
from itertools import islice
from nltk.tokenize import word_tokenize
from senpy.plugins import SentimentPlugin, ShelfMixin
from senpy.models import Sentiment


class NaiveBayesPlugin(SentimentPlugin, ShelfMixin):
    def _labelize(self, polarity):
        polarity = float(polarity)
        if polarity >= 0:
            return 'pos'
        else:
            return 'neg'
    
    def activate(self):
        nltk.download('punkt')
        if 'NaiveBayesClassifier' not in self.sh:
            train = []
            with open(self.corpora_path) as f:
                vader = csv.reader(f, delimiter='\t')
                for row in vader:
                    train.append((word_tokenize(row[2]),self._labelize(row[1])))
            self._all_words = set(word.lower()
                            for passage in train for word in passage[0])
            data_Bayes = [({word: (word in x[0])
                            for word in self._all_words}, x[1]) for x in train]
            classifier = nltk.NaiveBayesClassifier.train(data_Bayes)
            self.sh['NaiveBayesClassifier'] = classifier
        self._NaiveBayesClassifier = self.sh['NaiveBayesClassifier']
        self.save()
    
    def analyse_entry(self, entry, params):
        text = entry.get("text", None)
        features = {
            word.lower(): (word in word_tokenize(text.lower()))
            for word in self._all_words
        }
        result = self._NaiveBayesClassifier.classify(features)
        polarity = "marl:Neutral"
        polarity_value = 0

        if result == 'pos':
            polarity = "marl:Positive"
            polarity_value = self.maxPolarityValue
        elif result == 'neg':
            polarity = "marl:Negative"
            polarity_value = self.minPolarityValue
        sentiment = Sentiment({
            "marl:hasPolarity": polarity,
            "marl:polarityValue": polarity_value
        })
        entry.sentiments.append(sentiment)
        yield entry
