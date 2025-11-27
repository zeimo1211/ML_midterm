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


# 将稀疏矩阵转换为稠密数组以进行矩阵运算
X_train = train_features_selected.toarray()
X_test = test_features_selected.toarray()
y_train = np.array(train_labels, dtype=np.float64)
y_test = np.array(test_labels, dtype=np.float64)

# 1. 模型定义

class OptimizedLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=32, 
                 alpha=0.0001, l1_ratio=0.5, verbose=False):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.alpha = alpha        # 正则化强度
        self.l1_ratio = l1_ratio  # L1/L2 混合比例
        self.verbose = verbose
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        
        # 参数初始化：使用随机小值打破对称性
        np.random.seed(42)
        self.w = np.random.randn(n) * 0.01
        self.b = 0.0
        
        # Adam 优化器参数初始化
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        mw, vw = np.zeros(n), np.zeros(n)  # 权重的一阶/二阶矩
        mb, vb = 0.0, 0.0                  # 偏置的一阶/二阶矩
        t = 0 

        for epoch in range(self.max_iter):
            # 数据混洗 (Shuffle)
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y[indices]

            # Mini-Batch 迭代
            for start_idx in range(0, m, self.batch_size):
                t += 1
                end_idx = min(start_idx + self.batch_size, m)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                curr_m = len(y_batch)

                # 前向传播
                z = X_batch.dot(self.w) + self.b
                h = self._sigmoid(z)

                # 基础梯度计算
                error = h - y_batch
                grad_w = (1.0 / curr_m) * X_batch.T.dot(error)
                grad_b = (1.0 / curr_m) * np.sum(error)

                # 添加 ElasticNet 正则化梯度项
                l2_grad = self.alpha * (1 - self.l1_ratio) * self.w
                l1_grad = self.alpha * self.l1_ratio * np.sign(self.w)
                grad_w += (l2_grad + l1_grad)

                # Adam 参数更新：权重 w
                mw = beta1 * mw + (1 - beta1) * grad_w
                vw = beta2 * vw + (1 - beta2) * (grad_w ** 2)
                mw_hat = mw / (1 - beta1 ** t)
                vw_hat = vw / (1 - beta2 ** t)
                self.w -= self.lr * mw_hat / (np.sqrt(vw_hat) + epsilon)

                # Adam 参数更新：偏置 b
                mb = beta1 * mb + (1 - beta1) * grad_b
                vb = beta2 * vb + (1 - beta2) * (grad_b ** 2)
                mb_hat = mb / (1 - beta1 ** t)
                vb_hat = vb / (1 - beta2 ** t)
                self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + epsilon)

            # 日志记录
            if self.verbose and (epoch % 50 == 0):
                z_full = X.dot(self.w) + self.b
                h_full = self._sigmoid(z_full)
                loss = -np.mean(y * np.log(h_full + 1e-9) + (1 - y) * np.log(1 - h_full + 1e-9))
                self.loss_history.append(loss)
                #print(f"Epoch {epoch}: Loss = {loss:.6f}")

    def predict_proba(self, X):
        z = X.dot(self.w) + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# 2. 模型训练与推理


# 配置优化后的超参数
opt_model = OptimizedLogisticRegression(
    learning_rate=0.01,
    max_iter=500,
    batch_size=32,
    alpha=0.0001,
    l1_ratio=0.6,
    verbose=True
)
opt_model.fit(X_train, y_train)

# 3. 预测与评估 (包含阈值调整策略)

# 提升阈值至 0.5 以优化 Precision
target_threshold = 0.5 
opt_pred = opt_model.predict(X_test, threshold=target_threshold)

precision = precision_score(y_test, opt_pred)
recall = recall_score(y_test, opt_pred)
f1 = f1_score(y_test, opt_pred)

# 4. 结果输出

print("=" * 40)
print(f"【模型 3】优化逻辑回归 (Adam + ElasticNet)")
print("-" * 40)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("=" * 40)
