import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

## nb+cart
# 加载停用词
stopwords = set()
try:
    with open("stopwords.txt", "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:  # 确保不是空行
                stopwords.add(word)
except FileNotFoundError:
    print("警告：停用词文件未找到，将不使用停用词")
except Exception as e:
    print(f"加载停用词时出错: {e}")

# 加载数据函数
def load_data(filepath):
    data, labels = [], []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    text = parts[2].strip()
                    if text:  # 确保文本不为空
                        data.append(text)
                        labels.append(int(parts[1]))
    except FileNotFoundError:
        print(f"文件 {filepath} 未找到")
    except Exception as e:
        print(f"加载数据时出错: {e}")
    return data, labels

# 加载训练和测试数据
train_data, train_labels = load_data("train.txt")
test_data, test_labels = load_data("test.txt")

# 检查数据是否加载成功
if not train_data or not test_data:
    print("错误：未能加载训练或测试数据")
    exit()

print(f"训练数据数量: {len(train_data)}")
print(f"测试数据数量: {len(test_data)}")

# 分词和去除停用词
def segment_texts(texts):
    seg_texts = []
    for text in texts:
        try:
            seg_list = [word for word in jieba.cut(text) if word not in stopwords and word.strip()]
            seg_texts.append(" ".join(seg_list))
        except Exception as e:
            print(f"分词时出错: {e}，文本: {text}")
            seg_texts.append("")  # 添加空字符串作为占位符
    return seg_texts

train_seg = segment_texts(train_data)
test_seg = segment_texts(test_data)

# 特征提取
vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)  # 添加参数过滤罕见和常见词
train_features = vectorizer.fit_transform(train_seg)
test_features = vectorizer.transform(test_seg)

print(f"原始特征数量: {train_features.shape[1]}")

# 特征选择 - 安全地设置k值
k = min(15000, train_features.shape[1])
if k < 15000:
    print(f"警告：实际特征数量只有 {train_features.shape[1]}，将k调整为 {k}")

selector = SelectKBest(chi2, k=k)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 打印选中的特征词
feature_names = vectorizer.get_feature_names_out()
selected_indices = selector.get_support(indices=True)
if len(selected_indices) > 0:
    feature_words = [feature_names[idx] for idx in selected_indices[:10]]  # 只显示前10个
    print("前10个特征词：", feature_words)
else:
    print("警告：没有选择到任何特征词")
    # 如果不选择特征，使用原始特征
    train_features_selected = train_features
    test_features_selected = test_features

# 贝叶斯模型训练和预测
clf = MultinomialNB(alpha=1.0)  # 添加平滑参数
clf.fit(train_features_selected, train_labels)

# 预测
predictions = clf.predict(test_features_selected)

# 计算PRF值
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

print("\n模型性能:")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# 如果需要计算AUC，取消注释以下代码
# try:
#     # 对于二元分类，获取正类的概率
#     if hasattr(clf, "predict_proba"):
#         probas = clf.predict_proba(test_features_selected)
#         # 确保是二元分类
#         if probas.shape[1] == 2:
#             auc = roc_auc_score(test_labels, probas[:, 1])
#             print("AUC:", auc)
#         else:
#             print("AUC计算需要二元分类")
#     else:
#         print("模型不支持概率预测，无法计算AUC")
# except Exception as e:
#     print(f"计算AUC时出错: {e}")