import os
os.environ["HF_ENDPOINT"] = "https://alpha.hf-mirror.com"
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel
from transformers import EarlyStoppingCallback
import torch
from torch.utils.data import Dataset, DataLoader

os.environ["HF_ENDPOINT"] = "https://alpha.hf-mirror.com"

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_hoc_data():
    train_df = pd.read_parquet("./data/HoC/train.parquet")
    test_df = pd.read_parquet("./data/HoC/test.parquet")
    return train_df, test_df

def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    return accuracy, macro_f1, micro_f1

def train_log_linear(train_texts, train_labels, test_texts, test_labels):
    print("Training log-linear model...")
    
    # 特征提取
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=100000)
    # vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=100000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # 模型训练
    # model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model = LogisticRegression(penalty='l2',max_iter=1000, n_jobs=-1)
    model.fit(X_train, train_labels)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    print("\nLog-linear Model Training Results:")
    train_accuracy, train_macro_f1, train_micro_f1 = evaluate_metrics(train_labels, train_preds)
    print(f"Accuracy: {train_accuracy:.4f}, Macro-F1: {train_macro_f1:.4f}, Micro-F1: {train_micro_f1:.4f}")
    
    print("\nLog-linear Model Test Results:")
    test_accuracy, test_macro_f1, test_micro_f1 = evaluate_metrics(test_labels, test_preds)
    print(f"Accuracy: {test_accuracy:.4f}, Macro-F1: {test_macro_f1:.4f}, Micro-F1: {test_micro_f1:.4f}")
    
    return test_accuracy, test_macro_f1, test_micro_f1

def train_bert(train_texts, train_labels, test_texts, test_labels, num_labels):
    print("\nTraining BERT model...")
    
    # 加载tokenizer和模型
    # model_name = 'bert-base-uncased'
    model_name = 'google-bert/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        accuracy = accuracy_score(p.label_ids, preds)
        macro_f1 = f1_score(p.label_ids, preds, average="macro")
        micro_f1 = f1_score(p.label_ids, preds, average="micro")
        return {"accuracy": accuracy, "macro_f1": macro_f1, "micro_f1": micro_f1}
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    
    print("\nBERT Model Evaluation Results:")
    eval_results = trainer.evaluate()
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}, Macro-F1: {eval_results['eval_macro_f1']:.4f}, Micro-F1: {eval_results['eval_micro_f1']:.4f}")
    
    return eval_results['eval_accuracy'], eval_results['eval_macro_f1'], eval_results['eval_micro_f1']

def process_20_newsgroups():
    print("Loading 20 Newsgroups dataset...")
    dataset = load_dataset('SetFit/20_newsgroups')
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    num_classes = len(set(train_labels))
    
    print("\n=== 20 Newsgroups Dataset ===")
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Number of classes: {num_classes}")
    
    print("\n=== Log-linear Model ===")
    ll_acc, ll_macro, ll_micro = train_log_linear(train_texts, train_labels, test_texts, test_labels)
    
    print("\n=== BERT Model ===")
    bert_acc, bert_macro, bert_micro = train_bert(train_texts, train_labels, test_texts, test_labels, num_classes)
    
    return {
        "20_newsgroups": {
            "log_linear": {"accuracy": ll_acc, "macro_f1": ll_macro, "micro_f1": ll_micro},
            "bert": {"accuracy": bert_acc, "macro_f1": bert_macro, "micro_f1": bert_micro}
        }
    }

def process_hoc():
    print("\nLoading HoC dataset...")
    train_df, test_df = load_hoc_data()
    
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    num_classes = len(np.unique(train_labels))
    
    print("\n=== HoC Dataset ===")
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Number of classes: {num_classes}")
    
    print("\n=== Log-linear Model ===")
    ll_acc, ll_macro, ll_micro = train_log_linear(train_texts, train_labels, test_texts, test_labels)
    
    print("\n=== BERT Model ===")
    bert_acc, bert_macro, bert_micro = train_bert(train_texts, train_labels, test_texts, test_labels, num_classes)
    
    return {
        "hoc": {
            "log_linear": {"accuracy": ll_acc, "macro_f1": ll_macro, "micro_f1": ll_micro},
            "bert": {"accuracy": bert_acc, "macro_f1": bert_macro, "micro_f1": bert_micro}
        }
    }

def main():
    """主函数"""
    results = {}
    
    # 处理20 Newsgroups数据集
    news_results = process_20_newsgroups()
    results.update(news_results)
    
    # 处理HoC数据集
    hoc_results = process_hoc()
    results.update(hoc_results)
    
    # 最终结果
    print("\n=== Final Results ===")
    for dataset_name, metrics in results.items():
        print(f"\nDataset: {dataset_name}")
        for model_name, scores in metrics.items():
            print(f"{model_name}: Accuracy={scores['accuracy']:.4f}, Macro-F1={scores['macro_f1']:.4f}, Micro-F1={scores['micro_f1']:.4f}")

if __name__ == "__main__":
    main()