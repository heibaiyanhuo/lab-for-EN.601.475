import numpy as np


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y, **kwargs):
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

    def fit(self, X, y, **kwargs):
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


class LambdaMeans(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.R_mat = None
        self.mu_mat = None
        self.K = 1

    def fit(self, X, _, **kwargs):
        """  Fit the lambda means model  """
        assert 'lambda0' in kwargs, 'Need a value for lambda'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        # TODO: Write code to fit the model.  NOTE: labels should not be used here.

        # init
        X = X.toarray()
        n, m = X.shape
        lambda0 = kwargs['lambda0']
        iterations = kwargs['iterations']

        self.mu_mat = np.mean(X, axis=0).reshape(1, -1)
        if lambda0 == 0.0:
            lambda0 = np.mean(np.linalg.norm(X - self.mu_mat, axis=1))

        for _ in range(iterations):
            self.R_mat = np.zeros([self.K, n])
            # E-step
            for i in range(n):
                distance = np.linalg.norm(self.mu_mat - X[i, :], axis=1)
                min_mu_idx = np.argmin(distance)
                if distance[min_mu_idx] <= lambda0:
                    self.R_mat[min_mu_idx, i] = 1
                else:
                    self.K += 1
                    self.mu_mat = np.row_stack((self.mu_mat, X[i, :]))
                    r_new = np.zeros(n)
                    r_new[i] = 1
                    self.R_mat = np.row_stack((self.R_mat, r_new))

            # M-step
            r_weight_mat = 1 / np.sum(self.R_mat, axis=1).reshape(self.K, -1)
            self.mu_mat = np.dot(self.R_mat, X) * r_weight_mat


    def predict(self, X):
        # TODO: Write code to make predictions.
        X = X.toarray()
        n, m = X.shape
        cutoff = min(m, self.mu_mat.shape[1])
        y_predictions = np.zeros(n)
        for i in range(n):
            y_predictions[i] = np.argmin(np.linalg.norm(self.mu_mat[:, :cutoff] - X[i, :cutoff], axis=1))
        return y_predictions


class StochasticKMeans(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        pass

    def fit(self, X, _, **kwargs):
        assert 'num_clusters' in kwargs, 'Need the number of clusters (K)'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        num_clusters = kwargs['num_clusters']
        iterations = kwargs['iterations']
        # TODO: Write code to fit the model.  NOTE: labels should not be used here.

    def predict(self, X):
        # TODO: Write code to make predictions.
        pass
