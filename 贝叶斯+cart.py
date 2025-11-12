import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 加载停用词
stopwords = set()
with open("stopwords.txt", "r", encoding="utf-8") as f:
    for line in f:
        stopwords.add(line.strip())

# 加载训练数据
train_data = []
train_labels = []
with open("train.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 3:
            train_data.append(parts[2])
            train_labels.append(int(parts[1]))

# 加载测试数据
test_data = []
test_labels = []
with open("test.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 3:
            test_data.append(parts[2])
            test_labels.append(int(parts[1]))

# 分词和去除停用词
train_seg = []
for text in train_data:
    seg_list = [word for word in jieba.cut(text) if word not in stopwords]
    train_seg.append(" ".join(seg_list))

test_seg = []
for text in test_data:
    seg_list = [word for word in jieba.cut(text) if word not in stopwords]
    test_seg.append(" ".join(seg_list))

# 特征选择和打印特征词
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_seg)
test_features = vectorizer.transform(test_seg)

# 创建 SelectKBest 对象，使用卡方检验作为特征选择方法，选择前 15000 个最相关的特征
selector = SelectKBest(chi2, k=15000)

# 在训练集上拟合并应用特征选择
train_features_selected = selector.fit_transform(train_features, train_labels)

# 在测试集上应用相同的特征选择，以确保训练和测试数据集使用相同的特征
test_features_selected = selector.transform(test_features)


feature_words = [vectorizer.get_feature_names_out()[idx] for idx in selector.get_support(indices=True)]
print("特征词：", feature_words)

# 贝叶斯模型训练和预测
clf = MultinomialNB()
clf.fit(train_features_selected, train_labels)

predictions = clf.predict(test_features_selected)

# 计算PRF值
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

# # 计算AUC分数
# auc = roc_auc_score(test_labels, clf.predict_proba(test_features)[:, 1])

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
# print("AUC:", auc)