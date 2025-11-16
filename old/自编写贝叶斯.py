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
with open("../stopwords.txt", "r", encoding="utf-8") as f:
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
train_data, train_labels = load_data("../train.txt")
test_data, test_labels = load_data("../test.txt")


# 分词、去除停用词及提取TextRank关键词
def preprocess_data(data, use_textrank=True):
    seg_data = []
    textrank_data = []
    for text in data:
        seg_list = [word for word in jieba.cut(text) if word not in stopwords and word.strip()]
        seg_data.append(" ".join(seg_list))

        if use_textrank:
            try:
                # 添加异常处理，防止空文本导致的错误
                if text.strip():  # 确保文本不为空
                    # 使用更宽松的参数
                    keywords = jieba.analyse.textrank(text, topK=5, withWeight=False, allowPOS=('n', 'v', 'a'))
                    if keywords:  # 确保有关键词被提取
                        textrank_data.append(" ".join(keywords))
                    else:
                        textrank_data.append("")
                else:
                    textrank_data.append("")
            except Exception as e:
                # 如果出现任何异常，使用空字符串代替
                # print(f"TextRank处理文本时出错: {e}，文本内容: '{text}'")
                textrank_data.append("")
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

# 调整k值，确保不超过特征数量
k = min(20, train_features.shape[1])
selector = SelectKBest(chi2, k=k)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 打印选中的特征词
feature_names = vectorizer.get_feature_names_out()
selected_indices = selector.get_support(indices=True)
feature_words = [feature_names[idx] for idx in selected_indices]
print("特征词：", feature_words)


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


# 使用手写模型替代MultinomialNB
print("训练手写朴素贝叶斯模型...")
nb_classifier = HandwrittenMultinomialNB(alpha=1.0)
nb_classifier.fit(train_features_selected, train_labels)
nb_predictions = nb_classifier.predict(test_features_selected)

# 性能指标
precision = precision_score(test_labels, nb_predictions)
recall = recall_score(test_labels, nb_predictions)
f1 = f1_score(test_labels, nb_predictions)

print("手写朴素贝叶斯模型性能:")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# 与sklearn的MultinomialNB比较
print("\n训练sklearn朴素贝叶斯模型...")
sklearn_nb = MultinomialNB(alpha=1.0)
sklearn_nb.fit(train_features_selected, train_labels)
sklearn_predictions = sklearn_nb.predict(test_features_selected)

# 性能指标
sklearn_precision = precision_score(test_labels, sklearn_predictions)
sklearn_recall = recall_score(test_labels, sklearn_predictions)
sklearn_f1 = f1_score(test_labels, sklearn_predictions)

print("Sklearn朴素贝叶斯模型性能:")
print("Precision:", sklearn_precision)
print("Recall:", sklearn_recall)
print("F1-score:", sklearn_f1)