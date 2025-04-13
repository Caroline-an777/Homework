import numpy as np

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        # 参数初始化
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros(output_size)
        self.activation = activation

    def forward(self, X):
        # 前向传播
        self.Z1 = X.dot(self.W1) + self.b1
        if self.activation == 'relu':
            self.A1 = np.maximum(0, self.Z1)
        elif self.activation == 'sigmoid':
            self.A1 = 1 / (1 + np.exp(-self.Z1))
        self.Z2 = self.A1.dot(self.W2) + self.b2
        
        # Softmax
        exp_scores = np.exp(self.Z2 - np.max(self.Z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, probs, reg_lambda):
        num_samples = X.shape[0]
        
        # 计算梯度
        y_one_hot = np.eye(probs.shape[1])[y]
        dZ2 = (probs - y_one_hot) / num_samples
        
        dW2 = self.A1.T.dot(dZ2) + reg_lambda * self.W2
        db2 = np.sum(dZ2, axis=0)
        
        dA1 = dZ2.dot(self.W2.T)
        if self.activation == 'relu':
            dZ1 = dA1 * (self.Z1 > 0)
        elif self.activation == 'sigmoid':
            dZ1 = dA1 * self.A1 * (1 - self.A1)
        
        dW1 = X.T.dot(dZ1) + reg_lambda * self.W1
        db1 = np.sum(dZ1, axis=0)
        
        return dW1, db1, dW2, db2

    def get_params(self):
        return self.W1, self.b1, self.W2, self.b2

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2 = params