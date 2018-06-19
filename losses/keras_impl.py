import keras.backend as K
import tensorflow as tf

def focal_loss(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(tf_focal_loss(y_true, y_pred))

def tf_focal_loss(y_true, y_pred, alpha=.5, gamma=2.):
    """
    Straightforward implementation of focal loss.
    See https://arxiv.org/abs/1708.02002
    # Arguments:
        y_true: actual labels {0., 1.}
        y_pred: predicted labels {0., 1.}
        alpha: see paper
        gamma: see paper
    """
    assert alpha >= 0.
    assert alpha <= 1.
    assert gamma >= 0.
    # Compute the complementaries
    y_true_ = tf.subtract(1., y_true)
    y_pred_ = tf.subtract(1., y_pred)
    alpha_ = tf.subtract(1., alpha)
    # When y is 1
    loss1 = tf.multiply(y_true, tf.log(y_pred))
    loss1 = tf.multiply(tf.pow(y_pred_, gamma), loss1)
    loss1 = tf.multiply(alpha, loss1)
    loss1 = -loss1
    # When y is 0
    loss2 = tf.multiply(y_true_, tf.log(y_pred_))
    loss2 = tf.multiply(tf.pow(y_pred, gamma), loss2)
    loss2 = tf.multiply(alpha_, loss2)
    loss2 = -loss2
    return tf.add(loss1, loss2)
