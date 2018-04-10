from abc import ABCMeta, abstractmethod
from collections import defaultdict

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        # TODO
        self.label = label

    def __str__(self):
        # TODO
        return str(self.label)

    def __eq__(self, other):
        return self.label == other

class FeatureVector:
    def __init__(self):
        # TODO
        self.vector = defaultdict(lambda: 0.0)

    def add(self, index, value):
        # TODO
        self.vector[index] = value

    def get(self, index):
        # TODO
        return self.vector[index]
        

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

       
