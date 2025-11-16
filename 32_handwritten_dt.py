import numpy as np
import common

# 区别说明：
# - 自定义决策树，使用 Gini 不纯度。
# - 手动实现树节点和构建。
# - 相比 "改进决策树.py"：使用 Gini 而非信息增益；更简单的停止条件。
def split_gini(X, y, feature_index, threshold):
    left_y, right_y = [], []
    for i, sample in enumerate(X):
        if sample[feature_index] < threshold:
            left_y.append(y[i])
        else:
            right_y.append(y[i])

    gini_left = gini_impurity(left_y) if left_y else 0
    gini_right = gini_impurity(right_y) if right_y else 0
    n_left, n_right = len(left_y), len(right_y)
    n_total = n_left + n_right

    if n_total == 0:  # 防止除以零
        return float("inf")

    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    return weighted_gini

class DecisionTreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None


def gini_impurity(y):
    if len(y) == 0:
        return 0
    unique_classes, class_counts = np.unique(y, return_counts=True)
    impurity = 1.0
    total_samples = len(y)
    for count in class_counts:
        p_cls = count / total_samples
        impurity -= p_cls ** 2
    return impurity


def find_best_split(X, y, n_features):
    best_feature_index, best_threshold, best_gini = None, None, float("inf")

    # 如果数据为空，直接返回
    if len(X) == 0:
        return best_feature_index, best_threshold

    for feature_index in range(n_features):
        # 获取该特征的所有唯一值作为候选阈值
        thresholds = np.unique([row[feature_index] for row in X])
        for threshold in thresholds:
            gini = split_gini(X, y, feature_index, threshold)
            if gini < best_gini:
                best_gini, best_feature_index, best_threshold = gini, feature_index, threshold
    return best_feature_index, best_threshold


def split_dataset(X, y, feature_index, threshold):
    left_X, right_X, left_y, right_y = [], [], [], []
    for sample, label in zip(X, y):
        if sample[feature_index] < threshold:
            left_X.append(sample)
            left_y.append(label)
        else:
            right_X.append(sample)
            right_y.append(label)
    return left_X, left_y, right_X, right_y


def build_tree(X, y, depth=0, max_depth=10, min_samples_split=5):
    if len(X) == 0:
        return None

    # 计算每个类别的样本数
    unique_classes, class_counts = np.unique(y, return_counts=True)
    num_samples_per_class = [0] * (max(unique_classes) + 1) if len(unique_classes) > 0 else [0]
    for cls, count in zip(unique_classes, class_counts):
        if cls < len(num_samples_per_class):
            num_samples_per_class[cls] = count

    predicted_class = np.argmax(num_samples_per_class) if len(num_samples_per_class) > 0 else 0

    node = DecisionTreeNode(
        gini=gini_impurity(y),
        num_samples=len(y),
        num_samples_per_class=num_samples_per_class,
        predicted_class=predicted_class
    )

    # 停止条件：达到最大深度、样本数小于最小分割样本数、或者已经是纯节点
    if (depth >= max_depth or
            len(X) < min_samples_split or
            len(np.unique(y)) == 1):
        return node

    n_features = len(X[0]) if len(X) > 0 else 0
    feature_index, threshold = find_best_split(X, y, n_features)

    if feature_index is not None:
        left_X, left_y, right_X, right_y = split_dataset(X, y, feature_index, threshold)

        # 确保分割后两个子集都不为空
        if len(left_X) > 0 and len(right_X) > 0:
            node.feature_index = feature_index
            node.threshold = threshold
            node.left = build_tree(left_X, left_y, depth + 1, max_depth, min_samples_split)
            node.right = build_tree(right_X, right_y, depth + 1, max_depth, min_samples_split)

    return node


def predict(sample, node):
    if node is None:
        return 0  # 返回默认类别

    # 如果是叶子节点，返回预测类别
    if node.left is None and node.right is None:
        return node.predicted_class

    # 如果有左右子树，根据特征值决定走向
    if node.feature_index is not None and node.threshold is not None:
        if sample[node.feature_index] < node.threshold:
            return predict(sample, node.left)
        else:
            return predict(sample, node.right)

    # 如果无法判断，返回当前节点的预测类别
    return node.predicted_class


# 使用 common
stopwords = common.load_stopwords()
train_data, train_labels = common.load_data("train.txt")
test_data, test_labels = common.load_data("test.txt")

train_seg, train_textrank = common.preprocess_data(train_data, stopwords)
test_seg, test_textrank = common.preprocess_data(test_data, stopwords)

train_features, test_features = common.extract_features(
    train_seg, train_textrank, test_seg, test_textrank, train_labels, k=20
)

tree = build_tree(train_features.toarray(), train_labels, max_depth=10, min_samples_split=50)
predictions = [predict(instance, tree) for instance in test_features.toarray()]

common.evaluate_model(test_labels, predictions, "Handwritten DT")