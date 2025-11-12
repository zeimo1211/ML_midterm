import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB  # 导入朴素贝叶斯模型
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import numpy as np
warnings.filterwarnings("ignore")

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

# 加载训练和测试数据
train_data, train_labels = load_data("train.txt")
test_data, test_labels = load_data("test.txt")

# 分词、去除停用词及提取TextRank关键词
def preprocess_data(data, use_textrank=True):
    seg_data = []
    textrank_data = []
    for text in data:
        seg_list = [word for word in jieba.cut(text) if word not in stopwords]
        seg_data.append(" ".join(seg_list))

        if use_textrank:
            keywords = " ".join(jieba.analyse.textrank(text, topK=5, withWeight=False))
            textrank_data.append(keywords)
    return seg_data, textrank_data

train_seg, train_textrank = preprocess_data(train_data)
test_seg, test_textrank = preprocess_data(test_data)

# 结合TF-IDF特征和TextRank特征
combined_train_features = [seg + " " + textrank for seg, textrank in zip(train_seg, train_textrank)]
combined_test_features = [seg + " " + textrank for seg, textrank in zip(test_seg, test_textrank)]

# 特征选择
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(combined_train_features)
test_features = vectorizer.transform(combined_test_features)

selector = SelectKBest(chi2, k=20)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 打印选中的特征词
feature_words = [vectorizer.get_feature_names_out()[idx] for idx in selector.get_support(indices=True)]
print("特征词：", feature_words)


class HandwrittenMultinomialNB:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None
        self.feature_count_ = None
        self.class_count_ = None

    def fit(self, X, y):
        # 计算每个类别的先验概率
        m, n = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.feature_count_ = np.zeros((n_classes, n))
        self.class_count_ = np.zeros(n_classes)

        for i in range(n_classes):
            Xi = X[y == self.classes_[i]]
            self.feature_count_[i, :] = Xi.sum(axis=0)
            self.class_count_[i] = Xi.shape[0]

        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())
        self.feature_log_prob_ = np.log(
            (self.feature_count_ + 1) / (self.feature_count_ + 1).sum(axis=1, keepdims=True))

    def predict(self, X):
        jll = []
        is_sparse = hasattr(X, 'getrow')  # 检查X是否为稀疏矩阵

        for i in range(X.shape[0]):
            row = X.getrow(i) if is_sparse else X[i]
            log_prob = row.dot(self.feature_log_prob_.T)
            log_prob = log_prob.ravel()  # 将log_prob展平为一维数组
            log_prob += self.class_log_prior_
            jll.append(log_prob)
        return self.classes_[np.argmax(jll, axis=1)]


# 使用手写模型替MultinomialNB
nb_classifier = HandwrittenMultinomialNB()
nb_classifier.fit(train_features_selected, train_labels)
nb_predictions = nb_classifier.predict(test_features_selected)

# 性能指标
precision = precision_score(test_labels, nb_predictions)
recall = recall_score(test_labels, nb_predictions)
f1 = f1_score(test_labels, nb_predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)