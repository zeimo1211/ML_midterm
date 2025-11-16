import numpy as np
import common

# 区别说明：
# - 自定义 ImprovedDecisionTreeClassifier，使用信息增益 (Gini 在 32_handwritten_dt.py 中)。
# - 手动实现树构建，添加 max_depth 和 min_samples_split。
# - 相比 "hd_dt.py"：使用信息增益而非 Gini。

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

# 使用 common 处理数据和特征
stopwords = common.load_stopwords()
train_data, train_labels = common.load_data("train.txt")
test_data, test_labels = common.load_data("test.txt")

train_seg, _ = common.preprocess_data(train_data, stopwords, use_textrank=False)  # 无 TextRank
test_seg, _ = common.preprocess_data(test_data, stopwords, use_textrank=False)

train_features, test_features = common.extract_features(
    train_seg, [""] * len(train_seg), test_seg, [""] * len(test_seg), train_labels, k=20
)

dt = ImprovedDecisionTreeClassifier(max_depth=10, min_samples_split=50)
dt.fit(train_features, train_labels)
predictions = dt.predict(test_features)

common.evaluate_model(test_labels, predictions, "Improved DT")