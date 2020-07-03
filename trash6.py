import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import  src.psga.train.evaluation.functional as FF
from time import time
import tensorflow as tf

input = [2,2,2,3,4,5,5,5,5,5]
target = [2,2,2,3,2,1,1,1,1,3]
sklearn_score = cohen_kappa_score(input, target, weights="quadratic")


@tf.function
def cohen_kappa_loss(y_true, y_pred, row_label_vec, col_label_vec, weight_mat, eps=1e-6, dtype=tf.float64):
    labels = tf.matmul(y_true, col_label_vec)

    weight = tf.pow(tf.tile(labels, [1, tf.shape(y_true)[1]]) - tf.tile(row_label_vec, [tf.shape(y_true)[0], 1]), 2)
    weight /= tf.cast(tf.pow(tf.shape(y_true)[1] - 1, 2), dtype=dtype)
    numerator = tf.reduce_sum(weight * y_pred)
    denominator = tf.reduce_sum(
        tf.matmul(
            tf.reduce_sum(y_true, axis=0, keepdims=True),
            tf.matmul(weight_mat, tf.transpose(tf.reduce_sum(y_pred, axis=0, keepdims=True)))
        )
    )
    denominator /= tf.cast(tf.shape(y_true)[0], dtype=dtype)

    return tf.math.log(numerator / denominator + eps)


class CohenKappaLoss(tf.keras.losses.Loss):
    def __init__(self,
                 num_classes,
                 name='cohen_kappa_loss',
                 eps=1e-6,
                 dtype=tf.float64):
        super(CohenKappaLoss, self).__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)

        self.num_classes = num_classes
        self.eps = eps
        self.dtype = dtype
        label_vec = tf.range(num_classes, dtype=dtype)
        self.row_label_vec = tf.reshape(label_vec, [1, num_classes])
        self.col_label_vec = tf.reshape(label_vec, [num_classes, 1])
        self.weight_mat = tf.pow(
            tf.tile(self.col_label_vec, [1, num_classes]) - tf.tile(self.row_label_vec, [num_classes, 1]),
            2) / tf.cast(tf.pow(num_classes - 1, 2), dtype=dtype)

    def call(self, y_true, y_pred, sample_weight=None):
      return cohen_kappa_loss(
            y_true, y_pred, self.row_label_vec, self.col_label_vec, self.weight_mat, self.eps, self.dtype
        )

kappa = CohenKappaLoss(4)
y_true = tf.constant([[0, 0, 1, 0], [0, 1, 0, 0],
                      [1, 0, 0, 0], [0, 0, 0, 1]])
y_pred = tf.constant([[0.1, 0.2, 0.6, 0.1], [0.1, 0.5, 0.3, 0.1],
                      [0.8, 0.05, 0.05, 0.1], [0.01, 0.09, 0.1, 0.8]])
y_true = tf.cast(y_true, dtype=tf.float64)
y_pred = tf.cast(y_pred, dtype=tf.float64)
loss = kappa(y_true, y_pred)


print(loss)


# input = torch.tensor(input)
# target = torch.tensor(target)
# torch_score = FF.cohen_kappa_score(input, target, weights="quadratic").item()

