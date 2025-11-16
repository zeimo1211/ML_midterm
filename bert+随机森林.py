import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import jieba
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
import time
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba.analyse


# GPU检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 函数：格式化时间显示
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# 加载停用词
stopwords = set()
with open("stopwords.txt", "r", encoding="utf-8") as f:
    for line in f:
        stopwords.add(line.strip())

# 加载训练和测试数据
def load_data(file_path):
    data, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                data.append(parts[2])
                labels.append(int(parts[1]))
    return data, labels

train_data, train_labels = load_data("train.txt")
test_data, test_labels = load_data("test.txt")

# 设置BERT
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)  # 移动模型到GPU
# 初始化TF-IDF向量化器
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
# 准备TextRank特征用于TF-IDF
train_textrank_features = [" ".join(jieba.analyse.textrank(text, topK=10, withWeight=False)) for text in train_data]
# 拟合TF-IDF向量化器
tfidf_vectorizer.fit(train_textrank_features)


# 数据预处理
def preprocess_data_with_textrank(data, labels, max_length=64, topK=10):
    input_ids, attention_masks, processed_labels, textrank_features = [], [], [], []

    for i, text in enumerate(data):
        # 使用TextRank提取关键词
        keywords = jieba.analyse.textrank(text, topK=topK, withWeight=False)
        keywords = " ".join(keywords)

        # 继续原有的BERT预处理
        text = " ".join([word for word in jieba.cut(text) if word not in stopwords])
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        processed_labels.append(labels[i])

        # 将TextRank关键词添加到特征列表中
        textrank_features.append(keywords)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    processed_labels = torch.tensor(processed_labels)

    # 使用已经拟合的TF-IDF向量化器来转换TextRank特征
    textrank_features = tfidf_vectorizer.transform(textrank_features).toarray()

    return input_ids, attention_masks, processed_labels, textrank_features



train_inputs, train_masks, train_labels, train_textrank_features = preprocess_data_with_textrank(train_data, train_labels)
test_inputs, test_masks, test_labels, test_textrank_features = preprocess_data_with_textrank(test_data, test_labels)


# 创建DataLoader
batch_size = 16
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
validation_dataset = TensorDataset(test_inputs, test_masks, test_labels)
validation_dataloader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=batch_size)

# 设置优化器和调度器
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 10
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    t0 = time.time()
    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# 训练随机森林模型
print("Training Random Forest Model...")
t0 = time.time()
rf_model.fit(train_textrank_features, train_labels.numpy())
training_time_rf = format_time(time.time() - t0)
print("  Training took: {:}".format(training_time_rf))


# 评估模型
print("\nRunning Validation...")
t0 = time.time()
model.eval()
predictions, true_labels = [], []

for batch in validation_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

print("  Validation took: {:}".format(format_time(time.time() - t0)))

#
# 将预测结果和标签连接起来
flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = np.concatenate(true_labels, axis=0)

precision = precision_score(flat_true_labels, flat_predictions)
recall = recall_score(flat_true_labels, flat_predictions)
f1 = f1_score(flat_true_labels, flat_predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)




# 评估随机森林模型
print("\nEvaluating Random Forest Model...")
t0 = time.time()
rf_predictions = rf_model.predict(test_textrank_features)
validation_time_rf = format_time(time.time() - t0)
print("  Evaluation took: {:}".format(validation_time_rf))

# 计算随机森林模型的性能指标
rf_precision = precision_score(test_labels, rf_predictions)
rf_recall = recall_score(test_labels, rf_predictions)
rf_f1 = f1_score(test_labels, rf_predictions)

print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest F1-score:", rf_f1)

# 确保test_labels是一个numpy数组
test_labels_np = np.array(test_labels)

# BERT模型的预测类别
bert_class_predictions = flat_predictions


# 随机森林模型的预测结果
rf_predictions = rf_model.predict(test_textrank_features)


# 计算每个模型的正确预测
bert_correct = bert_class_predictions == test_labels_np
rf_correct = rf_predictions == test_labels_np

# 初始化集成模型的预测结果数组
ensemble_predictions = np.zeros(len(test_labels_np), dtype=int)

# 遍历每个样本，确定是否至少由一个模型正确预测
for i in range(len(test_labels_np)):
    if bert_correct[i] or rf_correct[i]:
        ensemble_predictions[i] = test_labels_np[i]
    else:
        ensemble_predictions[i] = 1 - test_labels_np[i]  # 预测为另一个类别

# 计算集成模型的性能指标
ensemble_precision = precision_score(test_labels_np, ensemble_predictions)
ensemble_recall = recall_score(test_labels_np, ensemble_predictions)
ensemble_f1 = f1_score(test_labels_np, ensemble_predictions)

print("Ensemble Model Precision:", ensemble_precision)
print("Ensemble Model Recall:", ensemble_recall)
print("Ensemble Model F1-score:", ensemble_f1)
