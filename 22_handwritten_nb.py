import numpy as np
import common

# 区别说明：
# - 自定义 HandwrittenMultinomialNB，实现手动朴素贝叶斯（拉普拉斯平滑）。
# - 相比 midtern.ipynb 的 sklearn NB：使用手动实现，并与 sklearn 比较。
# - 相比 "自编写贝叶斯.py"：添加平滑版本的 vectorizer 参数 (ngram_range 等)。
# - 不包含决策树或曲线绘制。

class HandwrittenMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 平滑参数
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None
        self.feature_count_ = None
        self.class_count_ = None

    def fit(self, X, y):
        # 转换为稠密数组以便处理
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        # 计算每个类别的先验概率
        m, n = X_dense.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.feature_count_ = np.zeros((n_classes, n))
        self.class_count_ = np.zeros(n_classes)

        for i, cls in enumerate(self.classes_):
            # 获取属于当前类别的样本
            mask = (y == cls)
            X_cls = X_dense[mask]
            # 计算特征计数和类别计数
            self.feature_count_[i, :] = X_cls.sum(axis=0)
            self.class_count_[i] = X_cls.shape[0]

        # 计算类别的先验概率（对数形式）
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())

        # 计算特征的条件概率（对数形式）
        # 应用拉普拉斯平滑
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

    def predict(self, X):
        # 转换为稠密数组以便处理
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        # 计算每个样本在每个类别下的对数概率
        jll = X_dense.dot(self.feature_log_prob_.T) + self.class_log_prior_
        # 返回概率最大的类别
        return self.classes_[np.argmax(jll, axis=1)]


# 加载数据等使用 common
stopwords = common.load_stopwords()
train_data, train_labels = common.load_data("train.txt")
test_data, test_labels = common.load_data("test.txt")

train_seg, train_textrank = common.preprocess_data(train_data, stopwords)
test_seg, test_textrank = common.preprocess_data(test_data, stopwords)

# 使用平滑版本的参数
vectorizer_params = {"ngram_range": (1, 2), "max_df": 0.95, "min_df": 5}
train_features, test_features = common.extract_features(
    train_seg, train_textrank, test_seg, test_textrank, train_labels,
    vectorizer_params=vectorizer_params, k=20
)

nb = HandwrittenMultinomialNB(alpha=1.0)
nb.fit(train_features, train_labels)
predictions = nb.predict(test_features)

common.evaluate_model(test_labels, predictions, "Handwritten NB")

# 可选：与 sklearn 比较
# from sklearn.naive_bayes import MultinomialNB
# sklearn_nb = MultinomialNB(alpha=1.0)
# sklearn_nb.fit(train_features, train_labels)
# sklearn_predictions = sklearn_nb.predict(test_features)
# common.evaluate_model(test_labels, sklearn_predictions, "sklearn NB (比较)")