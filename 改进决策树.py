import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
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

# 分词、去除停用词
def preprocess_data(data):
    seg_data = []
    for text in data:
        seg_list = [word for word in jieba.cut(text) if word not in stopwords]
        seg_data.append(" ".join(seg_list))
    return seg_data

train_seg = preprocess_data(train_data)
test_seg = preprocess_data(test_data)

# 构建TF-IDF特征
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_seg)
test_features = vectorizer.transform(test_seg)

# 特征选择
selector = SelectKBest(chi2, k=20)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 优化决策树算法
class ImprovedDecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini_impurity(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        impurity = 1.0
        total_samples = len(y)
        for count in class_counts:
            p_cls = count / total_samples
            impurity -= p_cls ** 2
        return impurity

    def information_gain(self, X, y, feature_index, threshold):
        # 计算信息增益
        left_indices = X[:, feature_index] < threshold
        right_indices = X[:, feature_index] >= threshold

        left_y = y[left_indices]
        right_y = y[right_indices]

        total_impurity = self.gini_impurity(y)
        left_impurity = self.gini_impurity(left_y)
        right_impurity = self.gini_impurity(right_y)

        p_left = len(left_y) / len(y)
        p_right = len(right_y) / len(y)

        information_gain = total_impurity - (p_left * left_impurity + p_right * right_impurity)
        return information_gain

    def find_best_split(self, X, y, n_features):
        best_feature_index, best_threshold, best_information_gain = None, None, -1
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                threshold = float(threshold)  # 将阈值转换为浮点数
                information_gain = self.information_gain(X, y, feature_index, threshold)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold

    def split_dataset(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] < threshold
        right_indices = X[:, feature_index] >= threshold

        left_X = X[left_indices]
        right_X = X[right_indices]

        left_y = y[left_indices]
        right_y = y[right_indices]

        return left_X, left_y, right_X, right_y

    def build_tree(self, X, y, depth=0):
        y = np.array(y)  # 确保y是NumPy数组

        if len(X) == 0 or len(X) < self.min_samples_split:
            return self.most_common_class(y)

        n_features = X.shape[1]
        best_feature_index, best_threshold = self.find_best_split(X, y, n_features)

        if best_feature_index is not None:
            left_X, left_y, right_X, right_y = self.split_dataset(X, y, best_feature_index, best_threshold)

            if len(left_X) == 0 or len(right_X) == 0:
                return self.most_common_class(y)

            left_subtree = self.build_tree(left_X, left_y, depth + 1)
            right_subtree = self.build_tree(right_X, right_y, depth + 1)

            return (best_feature_index, best_threshold, left_subtree, right_subtree)

        return self.most_common_class(y)

    def most_common_class(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        return unique_classes[np.argmax(class_counts)]


    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        predictions = []
        for instance in X:
            node = self.tree
            while isinstance(node, tuple):  # 检查节点是否为元组
                feature_index, threshold, left_subtree, right_subtree = node
                if instance[feature_index] < threshold:
                    node = left_subtree
                else:
                    node = right_subtree
            predictions.append(node)
        return predictions


# 使用改进的决策树类训练和预测
improved_tree = ImprovedDecisionTreeClassifier(max_depth=10, min_samples_split=50)
improved_tree.fit(train_features_selected.toarray(), train_labels)
predictions = improved_tree.predict(test_features_selected.toarray())

# 计算和打印性能指标
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)