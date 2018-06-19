import keras.backend as K
import numpy as np

from losses.keras_impl import focal_loss

def test_focal_loss():
    y_pred = K.variable(np.array([[0.3, 0.2, 0.1],
                                  [0.1, 0.2, 0.7]]))
    y_true = K.variable(np.array([[0, 1, 0],
                                  [1, 0, 0]]))
    print(K.eval(focal_loss(y_true, y_pred)))
