import numpy as np
from scipy.special import expit
from utils import get_best_features


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class SumOfFeatures(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        pass

    def fit(self, X, y):
        # NOTE: Not needed for SumOfFeatures classifier. However, do not modify.
        pass

    def predict(self, X):
        # TODO: Write code to make predictions.
        h = X.shape[1] // 2
        return np.array([1 if X[i, :h].sum() - X[i, -h:].sum() >= 0 else 0 for i in range(X.shape[0])])


class Perceptron(Model):

    def __init__(self, rate, iter):
        super().__init__()
        # TODO: Initializations etc. go here.
        self._w = None
        self.learning_rate = rate
        self.iterations = iter

    def fit(self, X, y):
        # TODO: Write code to fit the model.
        self.num_input_features = X.shape[1]
        self._w = np.zeros(self.num_input_features)
        for k in range(self.iterations):
            for i in range(X.shape[0]):
                yp = 1 if X[i].dot(self._w)[0] >= 0 else -1
                if (yp == 1 and y[i] == 0) or (yp == -1 and y[i] == 1):
                    self._w += self.learning_rate * (-1 if y[i] == 0 else 1) * X[i].toarray().ravel()

    def predict(self, X):
        # TODO: Write code to make predictions.
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        n = min(X.shape[1], self._w.shape[0])
        return np.array([1 if X[i, :n].dot(self._w[:n])[0] >= 0 else 0 for i in range(X.shape[0])])


# TODO: Add other Models as necessary.

class LogisticRegression(Model):

    def __init__(self, rate, iterations, num_of_features):
        super().__init__()
        self._w = None
        self.learning_rate = rate
        self.iterations = iterations
        self.num_of_features_to_select = num_of_features
        self.best_feature_idx_list = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        delta_y = None
        X_selected = X
        if self.num_of_features_to_select > 0 and self.num_input_features > self.num_of_features_to_select:
            self._w = np.zeros(self.num_of_features_to_select)
            X_selected, self.best_feature_idx_list = get_best_features(self.num_of_features_to_select, X, y)
        else:
            self._w = np.zeros(self.num_input_features)
        for k in range(self.iterations):
            delta_y = y - expit(X_selected.dot(self._w))
            self._w += self.learning_rate * X_selected.T.dot(delta_y)

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        if self.num_of_features_to_select > 0 and X.shape[1] > self.num_of_features_to_select:
            X_selected = np.zeros([X.shape[0], self.num_of_features_to_select])
            for i in range(X.shape[0]):
                for j in range(len(self.best_feature_idx_list)):
                    X_selected[i][j] = (0 if self.best_feature_idx_list[j] >= X.shape[1] else X[i, self.best_feature_idx_list[j]])
            n = min(self.num_of_features_to_select, self._w.shape[0])
            return np.array([1 if expit(np.dot(X_selected[i, :n], self._w[:n])) >= 0.5 else 0 for i in range(X.shape[0])])
        else:
            n = min(X.shape[1], self._w.shape[0])
            return np.array([1 if expit(X[i, :n].dot(self._w[:n])) >= 0.5 else 0 for i in range(X.shape[0])])