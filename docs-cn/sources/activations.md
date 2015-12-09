
## 激活函数（activations）的用法

激活函数通过 `Activation` （激活）层来使用，也可以用 `activation` 参数在所有的前向传播层使用。举例如下：

```python
from keras.layers.core import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```
等价于：
```python
model.add(Dense(64, activation='tanh'))
```

元素对元素的 Theano 函数也可以用作激活函数：

```python
def tanh(x):
    return theano.tensor.tanh(x)

model.add(Dense(64, activation=tanh))
model.add(Activation(tanh))
```

## 激活函数列表

- __softmax__: Softmax 应用于所有输入的最末一维。形如 `(nb_samples, nb_timesteps, nb_dims)` 或者 `(nb_samples, nb_dims)`
- __softplus__
- __relu__
- __tanh__
- __sigmoid__
- __hard_sigmoid__
- __linear__

## 高级激活函数

比 Theano 函数更复杂的函数（例如，基于学习的激活函数、基于配置的激活函数等）也可以作为激活函数，参见 [Advanced Activation layers （高级激活层）](layers/advanced_activations.md)。参见如下模块 `keras.layers.advanced_activations`。高级激活层包括 PReLU 和 LeakyReLU。
