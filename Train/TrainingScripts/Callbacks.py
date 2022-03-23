import tensorflow as tf
import numpy as np
class OwnReduceLROnPlateau(tf.keras.callbacks.Callback):
  """Reduce learning rate when a metric has stopped improving.
  Models often benefit from reducing the learning rate by a factor
  of 2-10 once learning stagnates. This callback monitors a
  quantity and if no improvement is seen for a 'patience' number
  of epochs, the learning rate is reduced.
  Example:
  ```python
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
  model.fit(X_train, Y_train, callbacks=[reduce_lr])
  ```
  Args:
      monitor: quantity to be monitored.
      factor: factor by which the learning rate will be reduced.
        `new_lr = lr * factor`.
      patience: number of epochs with no improvement after which learning rate
        will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
        the learning rate will be reduced when the
        quantity monitored has stopped decreasing; in `'max'` mode it will be
        reduced when the quantity monitored has stopped increasing; in `'auto'`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      min_delta: threshold for measuring the new optimum, to only focus on
        significant changes.
      cooldown: number of epochs to wait before resuming normal operation after
        lr has been reduced.
      min_lr: lower bound on the learning rate.
  """

  def __init__(self,
               monitor='val_loss',
               factor=0.1,
               patience=10,
               verbose=0,
               mode='auto',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0,
               **kwargs):
    super(OwnReduceLROnPlateau, self).__init__()

    self.monitor = monitor
    self.factor = factor
    self.min_lr = min_lr
    self.min_delta = min_delta
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.best = 0
    self.mode = mode
    self.monitor_op = None
    self._reset()

  def _reset(self):
    """Resets wait counter and cooldown counter.
    """
    if (self.mode == 'min' or
        (self.mode == 'auto' and 'acc' not in self.monitor)):
      self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
      self.best = np.Inf
    else:
      self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
      self.best = -np.Inf
    self.cooldown_counter = 0
    self.wait = 0

  def on_train_begin(self, logs=None):
    self._reset()

  def on_epoch_end(self, epoch, logs=None,lr=0):
    logs = logs or {}
    logs['lr'] =lr
    current = logs.get(self.monitor)
    if self.in_cooldown():
      self.cooldown_counter -= 1
      self.wait = 0
      return lr

    if self.monitor_op(current, self.best):
      self.best = current
      self.wait = 0
      return lr
    elif not self.in_cooldown():
      self.wait += 1
      if self.wait >= self.patience:
        old_lr = lr
        if old_lr > np.float32(self.min_lr):
          new_lr = old_lr * self.factor
          new_lr = max(new_lr, self.min_lr)

          if self.verbose > 0:
            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                    'rate to %s.' % (epoch + 1, new_lr))
          self.cooldown_counter = self.cooldown
          self.wait = 0
          return new_lr
        else:
          return lr
      else:
        return lr
    

  def in_cooldown(self):
    return self.cooldown_counter > 0

from tensorflow.keras import backend

def ModifiedHuberDelta(threshold):
  
  def ModifiedHuber(y_true, y_pred):
    """Computes Huber loss value.
    For each value x in `error = y_true - y_pred`:
    ```
    loss = 0.5 * x^2                  if |x| <= d
    loss = d * |x| - 0.5 * d^2        if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      delta: A float, the point where the Huber loss function changes from a
        quadratic to linear.
    Returns:
      Tensor with one scalar loss entry per sample.
    """
    y_pred = tf.cast(y_pred, dtype=backend.floatx())
    y_true = tf.cast(y_true, dtype=backend.floatx())
    delta = tf.cast(threshold, dtype=backend.floatx())
    error = tf.subtract(y_pred, y_true)
    abs_error = tf.abs(error)
    half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)

    mean = abs(backend.mean((y_pred - y_true)))
    huber = backend.mean(
        tf.where(abs_error <= delta, half * tf.square(error),
                          delta * abs_error - half * tf.square(delta)))
    return huber + mean

  return ModifiedHuber

