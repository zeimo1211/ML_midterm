import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
import random
import time
import datetime
import pickle

from common import load_data

# GPU检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 函数：格式化时间显示
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# 数据预处理
def preprocess_data(data, labels, tokenizer, max_length=128):
    input_ids, attention_masks, processed_labels = [], [], []

    for i, text in enumerate(data):
        # 使用 BERT 分词器处理文本
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        processed_labels.append(labels[i])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    processed_labels = torch.tensor(processed_labels)

    return input_ids, attention_masks, processed_labels


def train_bert():
    # 加载数据
    train_data, train_labels = load_data("train.txt")
    test_data, test_labels = load_data("test.txt")

    print(f"训练数据数量: {len(train_data)}")
    print(f"测试数据数量: {len(test_data)}")

    # 直接使用 transformers 加载 BERT 模型和分词器
    model_name = 'bert-base-chinese'

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 加载模型，设置分类数量（根据你的任务调整）
    num_labels = len(set(train_labels))  # 自动检测标签数量
    print(f"检测到 {num_labels} 个分类标签")

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )

    model.to(device)

    print("预处理训练数据...")
    train_inputs, train_masks, train_labels_tensor = preprocess_data(train_data, train_labels, tokenizer)
    print("预处理测试数据...")
    test_inputs, test_masks, test_labels_tensor = preprocess_data(test_data, test_labels, tokenizer)

    # 创建DataLoader
    batch_size = 16
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels_tensor)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

    # 设置优化器和调度器
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 4
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 设置随机种子
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    # 训练模型
    print("\n开始训练...")
    for epoch_i in range(epochs):
        print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
        t0 = time.time()
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                print(f'  Batch {step:>5,} of {len(train_dataloader):>5,}. Elapsed: {elapsed}')

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print(f"  平均训练损失: {avg_train_loss:.2f}")
        print(f"  训练用时: {training_time}")

    # 保存模型
    model.save_pretrained('./saved_bert_model')
    tokenizer.save_pretrained('./saved_bert_model')

    # 保存测试数据用于后续推理
    test_data_dict = {
        'test_inputs': test_inputs,
        'test_masks': test_masks,
        'test_labels': test_labels,
        'test_labels_tensor': test_labels_tensor
    }

    with open('./test_data.pkl', 'wb') as f:
        pickle.dump(test_data_dict, f)

    print("BERT模型训练完成并已保存")


if __name__ == "__main__":
    train_bert()