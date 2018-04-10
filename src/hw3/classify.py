import os
import argparse
import sys
import pickle

import numpy as np

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

from data import load_matrix_data
from utils import AdaBoostHelper

class AdaBoost:
    def __init__(self, iterations):
        super()
        self.h_args = []
        self.alpha_list = []
        self.T = iterations

    def train(self, X, y):
        n = X.shape[0]
        m = X.shape[1]
        helper = AdaBoostHelper(X, y, n, m)
        for i in range(self.T):

            h_arg = helper.calc_ht()
            self.h_args.append(h_arg)


            if h_arg[-1] < 0.000001:
                self.T = i
                break

            alpha = helper.calc_alphat(h_arg[-1])
            self.alpha_list.append(alpha)

            # print('{}, {}, {}'.format(h_arg[:2], h_arg[3:], alpha))
            helper.set_next_distribution(alpha, h_arg[2])

    def predict(self, X):
        X = X.toarray()
        num_of_examples = X.shape[0]
        y_hat = np.zeros(num_of_examples)
        for i in range(self.T):
            h_arg = self.h_args[i]
            y_hat += self.alpha_list[i] * AdaBoostHelper.ht(X, h_arg)
        y = np.zeros(num_of_examples, dtype=np.int)
        y[y_hat >= 0] = 1
        y[y_hat < 0] = 0
        return y

def load_data(filename):
    instances = []
    # max dimension
    m = -1
    with open(filename) as reader:
        for line in reader:
            if len(line.strip()) == 0:
                continue
            
            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")

            label = ClassificationLabel(int_label)
            feature_vector = FeatureVector()

            m = max(int(split_line[-1].split(':')[0]), m)

            for item in split_line[1:]:
                try:
                    index = int(item.split(":")[0])
                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                
                if value != 0.0:
                    feature_vector.add(index, value)

            instance = Instance(feature_vector, label)
            instances.append(instance)

    return instances, m


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")

    # TODO This is where you will add new command line options

    parser.add_argument("--num-boosting-iterations", type=int, help="The number of boosting iterations to run.",
                        default=10)
    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--algorithm should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def train(X, y, algorithm, iterations):
    # TODO Train the model using "algorithm" on "data"
    # TODO This is where you will add new algorithms that will subclass Predictor

    if algorithm == 'adaboost':
        model = AdaBoost(iterations)
        model.train(X, y)
        return model

    return None


def write_predictions(predictor, X, predictions_file):
    labels = predictor.predict(X)
    try:
        with open(predictions_file, 'w') as writer:

            for label in labels:
                # label = predictor.predict(instance)
                writer.write(str(label))
                writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    args = get_args()

    if args.mode.lower() == "train":
        # Load the training data.
        # instances, m = load_data(args.data)

        X, y = load_matrix_data(args.data)

        # Train the model.
        predictor = train(X, y, args.algorithm, args.num_boosting_iterations)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        # instances, m = load_data(args.data)
        X, y = load_matrix_data(args.data)
        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        write_predictions(predictor, X, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()
    # file = 'datasets/easy.train'
    # X, y = load_matrix_data(file)
    #
    # model = AdaBoost(10)
    # model.train(X, y)
    # instances, m = load_data(file)
    # print(m)
    # model = AdaBoost(m, 10)

    # print('start to train')
    # model.train(instances)
    # helper = Helper()
    # print(helper.calc_yhat(instances, 36, 0.4844385))

    # test_file = 'datasets/easy.dev'
    # test_X, test_y = load_matrix_data(test_file)
    # print(model.predict(test_X))
    # print('start to test')

