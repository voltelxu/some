from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

iris_train = "iris_training.csv"
iris_test = "iris_test.csv"

def main():
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
         filename=iris_train,
         target_dtype=np.int,
         features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=iris_test,
        target_dtype=np.int,
        features_dtype=np.float32)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3,
                                                model_dir="")

    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)

        return x, y

    classifier.fit(input_fn=get_train_inputs, steps=2000)

    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)
        return x, y

    accuracy = classifier.evaluate(input_fn=get_test_inputs,
                                   steps=1)["accuracy"]
    print("\nTest Accuracy:{0:f}\n".format(accuracy))

    def new_samples():
        return np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    predictions = list(classifier.predict(input_fn=new_samples))

    print("New Samples, Class Predictions:{}\n".format(predictions))

if __name__ == "__main__":
    main()
