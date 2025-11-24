import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# 加载停用词
stopwords = set()
try:
    with open("stopwords.txt", "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
except FileNotFoundError:
    print("警告：停用词文件未找到，将不使用停用词")


# 加载数据函数
def load_data(filepath):
    data, labels = [], []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    data.append(parts[2])
                    labels.append(int(parts[1]))
    except Exception as e:
        print(f"加载数据时出错: {e}")
    return data, labels


# 加载训练和测试数据
train_data, train_labels = load_data("train.txt")
test_data, test_labels = load_data("test.txt")

# 检查数据是否为空
if not train_data or not test_data:
    print("错误：训练数据或测试数据为空")
    exit()


# 分词、去除停用词
def preprocess_data(data):
    seg_data = []
    for text in data:
        try:
            seg_list = [word for word in jieba.cut(text) if word not in stopwords and word.strip()]
            seg_data.append(" ".join(seg_list))
        except Exception as e:
            print(f"分词时出错: {e}")
            seg_data.append("")
    return seg_data


train_seg = preprocess_data(train_data)
test_seg = preprocess_data(test_data)

# 构建TF-IDF特征
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_seg)
test_features = vectorizer.transform(test_seg)

# 特征选择 - 安全地设置k值
k = min(20, train_features.shape[1])
selector = SelectKBest(chi2, k=k)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

print(f"特征选择后的特征数量: {train_features_selected.shape[1]}")


# 完整的决策树节点类
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # 用于分割的特征索引
        self.threshold = threshold  # 分割阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 如果是叶子节点，存储预测值


# 改进的决策树分类器
class ImprovedDecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
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

        if len(left_y) == 0 or len(right_y) == 0:
            return 0

        total_impurity = self.gini_impurity(y)
        left_impurity = self.gini_impurity(left_y)
        right_impurity = self.gini_impurity(right_y)

        p_left = len(left_y) / len(y)
        p_right = len(right_y) / len(y)

        information_gain = total_impurity - (p_left * left_impurity + p_right * right_impurity)
        return information_gain

    def find_best_split(self, X, y):
        best_feature_index, best_threshold, best_gain = None, None, -1
        n_features = X.shape[1]

        for feature_index in range(n_features):
            # 随机选择一部分阈值以提高效率
            unique_values = np.unique(X[:, feature_index])
            if len(unique_values) > 10:
                # 如果唯一值太多，随机抽样
                thresholds = np.random.choice(unique_values, size=min(10, len(unique_values)), replace=False)
            else:
                thresholds = unique_values

            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold, best_gain

    def build_tree(self, X, y, depth=0):
        # 停止条件
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            # 返回叶子节点，值为最常见的类别
            unique_classes, class_counts = np.unique(y, return_counts=True)
            value = unique_classes[np.argmax(class_counts)]
            return TreeNode(value=value)

        # 寻找最佳分割
        feature_index, threshold, gain = self.find_best_split(X, y)

        # 如果信息增益为0或负，停止分割
        if gain <= 0:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            value = unique_classes[np.argmax(class_counts)]
            return TreeNode(value=value)

        # 分割数据
        left_indices = X[:, feature_index] < threshold
        right_indices = X[:, feature_index] >= threshold

        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]

        # 如果分割后某个子集为空，停止分割
        if len(left_y) == 0 or len(right_y) == 0:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            value = unique_classes[np.argmax(class_counts)]
            return TreeNode(value=value)

        # 递归构建子树
        left_subtree = self.build_tree(left_X, left_y, depth + 1)
        right_subtree = self.build_tree(right_X, right_y, depth + 1)

        return TreeNode(feature_index, threshold, left_subtree, right_subtree)

    def fit(self, X, y):
        X_array = X.toarray() if hasattr(X, 'toarray') else np.array(X)
        y_array = np.array(y)
        self.root = self.build_tree(X_array, y_array)

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] < node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X):
        X_array = X.toarray() if hasattr(X, 'toarray') else np.array(X)
        predictions = []
        for instance in X_array:
            pred = self._predict_single(instance, self.root)
            predictions.append(pred)
        return np.array(predictions)


# 使用改进的决策树类训练和预测
try:
    improved_tree = ImprovedDecisionTreeClassifier(max_depth=10, min_samples_split=50)
    improved_tree.fit(train_features_selected, train_labels)
    predictions = improved_tree.predict(test_features_selected)

    # 计算和打印性能指标
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
except Exception as e:
    print(f"训练或预测过程中出错: {e}")