import jieba
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

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

# 构建决策树
print("开始构建决策树...")
tree = build_tree(train_features_selected.toarray(), train_labels, max_depth=10, min_samples_split=50)

# 进行预测
print("开始预测...")
predictions = []
for instance in test_features_selected.toarray():
    pred = predict(instance, tree)
    predictions.append(pred)

# 性能指标
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)