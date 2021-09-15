import numpy as np


# 感知机原始形式
class Perceptron:
    def __init__(self, length):
        self.w = np.ones(length - 1, dtype=np.float32)
        self.b = 0
        self.learning_rate = 0.1

    # 模型
    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # 算法 策略
    def fit(self, x_train, y_train):
        is_wrong = False
        while not is_wrong:
            # 误分类的个数
            wrong_count = 0
            for d in range(len(x_train)):
                x = x_train[d]
                y = y_train[d]
                if y * self.sign(x, self.w, self.b) <= 0:
                    # 更新 w b
                    self.w = self.w + self.learning_rate * np.dot(y, x)
                    self.b = self.b + self.learning_rate * y
                    wrong_count += 1
                print(self.w)
                print(self.b)
            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!'
