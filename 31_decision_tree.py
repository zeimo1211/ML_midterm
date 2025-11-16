from sklearn.tree import DecisionTreeClassifier
import common

# 区别说明：
# - 使用 sklearn DecisionTreeClassifier，与 midtern.ipynb 的决策树部分类似。
# - 相比自定义版本：无手动实现树结构。
# - 使用默认参数。

stopwords = common.load_stopwords()
train_data, train_labels = common.load_data("train.txt")
test_data, test_labels = common.load_data("test.txt")

train_seg, train_textrank = common.preprocess_data(train_data, stopwords)
test_seg, test_textrank = common.preprocess_data(test_data, stopwords)

train_features, test_features = common.extract_features(
    train_seg, train_textrank, test_seg, test_textrank, train_labels, k=20
)

clf = DecisionTreeClassifier()
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)

common.evaluate_model(test_labels, predictions, "sklearn DecisionTree")