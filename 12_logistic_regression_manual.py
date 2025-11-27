#cell 1 通用数据预处理部分

import jieba
import jieba.analyse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# 加载停用词
stopwords = set()
with open("stopwords.txt", "r", encoding="utf-8") as f:
    for line in f:
        stopwords.add(line.strip())

# 加载数据函数
def load_data(filepath):
    data, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                data.append(parts[2])
                labels.append(int(parts[1]))
    return data, labels

# 文本预处理：分词 + 去停用词 + TextRank关键词
def preprocess_data(data, use_textrank=True):
    seg_data = []
    textrank_data = []
    for text in data:
        seg_list = [word for word in jieba.cut(text) if word not in stopwords]
        seg_data.append(" ".join(seg_list))

        if use_textrank:
            keywords = " ".join(jieba.analyse.textrank(text, topK=5, withWeight=False))
            textrank_data.append(keywords)
        else:
            textrank_data.append("")
    return seg_data, textrank_data

# 读取训练与测试数据
train_data, train_labels = load_data("train.txt")
test_data, test_labels = load_data("test.txt")

# 预处理
train_seg, train_textrank = preprocess_data(train_data, use_textrank=True)
test_seg, test_textrank = preprocess_data(test_data, use_textrank=True)

# 合并 TF-IDF 与 TextRank 特征文本
combined_train_features = [seg + " " + textrank for seg, textrank in zip(train_seg, train_textrank)]
combined_test_features = [seg + " " + textrank for seg, textrank in zip(test_seg, test_textrank)]

# 向量化
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(combined_train_features)
test_features = vectorizer.transform(combined_test_features)

# 特征选择（卡方）
selector = SelectKBest(chi2, k=20)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 打印选中特征词
feature_words = [vectorizer.get_feature_names_out()[idx] for idx in selector.get_support(indices=True)]
print("选中特征词：", feature_words)

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. 数据准备

# 将稀疏矩阵转换为稠密数组以进行矩阵运算
X_train = train_features_selected.toarray()
X_test = test_features_selected.toarray()
y_train = np.array(train_labels, dtype=np.float64)
y_test = np.array(test_labels, dtype=np.float64)

# 2. 模型定义

class ManualLogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=1000, reg_lambda=1.0, verbose=False):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        # 数值稳定性处理：防止 exp 溢出
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        # 初始化参数（零初始化）
        self.w = np.zeros(n, dtype=np.float64)
        self.b = 0.0

        for i in range(self.max_iter):
            # 前向传播
            z = X.dot(self.w) + self.b
            h = self._sigmoid(z)

            # 梯度计算 (含 L2 正则化导数)
            error = h - y
            dw = (1.0 / m) * (X.T.dot(error)) + (self.reg_lambda / m) * self.w
            db = (1.0 / m) * np.sum(error)

            # 参数更新
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # 打印日志
            if self.verbose and i % 200 == 0:
                loss = (-1.0/m) * np.sum(y * np.log(h + 1e-9) + (1 - y) * np.log(1 - h + 1e-9))
                print(f"Iter {i}: Loss = {loss:.6f}")

    def predict_proba(self, X):
        z = X.dot(self.w) + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# 3. 模型训练与推理

manual_lr = ManualLogisticRegression(
    learning_rate=0.1, 
    max_iter=1000, 
    reg_lambda=1.0, 
    verbose=False
)
manual_lr.fit(X_train, y_train)

manual_pred = manual_lr.predict(X_test)

# 4. 结果输出

precision = precision_score(y_test, manual_pred)
recall = recall_score(y_test, manual_pred)
f1 = f1_score(y_test, manual_pred)

print("=" * 40)
print(f"【模型 2】自编写逻辑回归 (Basic GD)")
print("-" * 40)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("=" * 40)
