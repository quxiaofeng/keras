
## 目标函数（objectives）的使用方法

目标函数（又称损失函数（loss function）或优化目标函数（optimization score function））是模型（model）所必需的两个参数之一（目标函数和优化方法）。

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

目标函数可以是一个内置目标函数的名称（字符串），也可以是一个输出为一个标量的 Theano 符号函数（Theano symbolic function）。该符号函数的输入为如下两个参数：

- __y_true__: 数据类别标签。Theano 张量（tensor）。
- __y_pred__: 模型输出的数据标签。 Theano 张量，与 y_true 相同大小。

计算之后的优化目标函数值是所有输出数据点的输出数组的均值。

优化函数样例, 详见 [目标函数 源代码](https://github.com/fchollet/keras/blob/master/keras/objectives.py).

## 内置目标函数

- __mean_squared_error__ / __mse__： 均方差
- __root_mean_squared_error__ / __rmse__： 均方根误差
- __mean_absolute_error__ / __mae__： 平均绝对误差
- __mean_absolute_percentage_error__ / __mape__： 平均绝对百分比误差
- __mean_squared_logarithmic_error__ / __msle__： 平均对数平方误差
- __squared_hinge__： L2 平方转折点损失(平方合页损失)
- __hinge__： 转折点损失（合页损失）
- __binary_crossentropy__： 二值叉熵（也称对数损失） 
- __categorical_crossentropy__： 多类叉熵（也称多类对数损失）。 _注意_：该目标函数要求类别标签应为形如 `(样本数，类别数)` （`(nb_samples, nb_classes)`） 的二进制数组。
