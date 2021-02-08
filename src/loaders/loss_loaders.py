from src.helper import url_solver as US
import tensorflow as tf


class LossLoaderJSON:
  def __init__(self, json):
    self.loss_object = tf.keras.losses.deserialize(json["tester"]["losses"])
    

  def loss(self, model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=False)
    return self.loss_object(y_true = y, y_pred = y_)

  def grad(self, model, inputs, targets):
    with tf.GradientTape() as tape:
      loss_value = self.loss(model, inputs, targets, True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


