from sklearn.naive_bayes import MultinomialNB
import common  # 导入共享模块

# 区别说明：
# - 调用 sklearn MultinomialNB。
# - 添加了 cart.py 的 vectorizer 参数 (min_df=2 等) 和更大 k 值。

stopwords = common.load_stopwords()
train_data, train_labels = common.load_data("train.txt")
test_data, test_labels = common.load_data("test.txt")

train_seg, train_textrank = common.preprocess_data(train_data, stopwords)
test_seg, test_textrank = common.preprocess_data(test_data, stopwords)

# 使用 cart.py 的参数
vectorizer_params = {"min_df": 2, "max_df": 0.95}
train_features, test_features = common.extract_features(
    train_seg, train_textrank, test_seg, test_textrank, train_labels,
    vectorizer_params=vectorizer_params, k=15000
)

clf = MultinomialNB(alpha=1.0)
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)

common.evaluate_model(test_labels, predictions, "sklearn MultinomialNB")