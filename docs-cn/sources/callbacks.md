## 回调（callbacks）的用法

回调函数是一组用于训练过程中某些阶段的函数。在训练的过程中使用回调函数，可以检查模型的内部状态和统计信息。在 `Sequential` （序列）模型的 `.fit()` （训练）方法中，使用关键字 `callbacks` 参数，可以传递多个回调函数。训练时，就会自动调用与这些回调函数相关的方法。

---

## 基类

```python
keras.callbacks.Callback()
```
- __属性（Properties）__:
    - __参数（params）__: dict. Training parameters (eg. verbosity, batch size, number of epochs...).
    - __模型（model）__: `keras.models.Model`. Reference of the model being trained.
- __方法（Methods）__:
    - __on_train_begin__(logs={}): 训练开始时调用该方法。
    - __on_train_end__(logs={}): 训练结束时调用该方法。
    - __on_epoch_begin__(epoch, logs={}): Method called at the beginning of epoch `epoch`.
    - __on_epoch_end__(epoch, logs={}): Method called at the end of epoch `epoch`.
    - __on_batch_begin__(batch, logs={}): Method called at the beginning of batch `batch`.
    - __on_batch_end__(batch, logs={}): Method called at the end of batch `batch`.

The `logs` dictionary will contain keys for quantities relevant to the current batch or epoch. Currently, the `.fit()` method of the `Sequential` model class will include the following quantities in the `logs` that it passes to its callbacks:
- __on_epoch_end__: logs optionally include `val_loss` (if validation is enabled in `fit`), and `val_accuracy` (if validation and accuracy monitoring are enabled).
- __on_batch_begin__: logs include `size`, the number of samples in the current batch.
- __on_batch_end__: logs include `loss`, and optionally `accuracy` (if accuracy monitoring is enabled).

---

## Available callbacks

```python
keras.callbacks.ModelCheckpoint(filepath, verbose=0, save_best_only=False)
```

Save the model after every epoch. If `save_best_only=True`, the latest best model according to the validation loss will not be overwritten.
`filepath` can contain named formatting options, which will be filled the value of `epoch` and keys in `logs` (passed in `on_epoch_end`).

For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then multiple files will be save with the epoch number and the validation loss.


```python
keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0)
```

Stop training after no improvement of the metric `monitor` is seen for `patience` epochs.

---


## Create a callback

You can create a custom callback by extending the base class `keras.callbacks.Callback`. A callback has access to its associated model through the class property `self.model`.

Here's a simple example saving a list of losses over each batch during training:
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

---

### Example: recording loss history

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, callbacks=[history])

print history.losses
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
```

---

### Example: model checkpoints

```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])

```

