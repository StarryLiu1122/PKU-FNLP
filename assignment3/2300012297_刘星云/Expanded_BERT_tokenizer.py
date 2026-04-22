import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_ENDPOINT"] = "https://alpha.hf-mirror.com"
import json
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datasets import load_dataset
import random
from tqdm import tqdm
os.environ["HF_ENDPOINT"] = "https://alpha.hf-mirror.com"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Step 1: Train tokenizer ======
def train_tokenizer(corpus_path, vocab_size=100000):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    def batch_iterator():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
            # for line in f:
                sample = json.loads(line)
                if i % 1000 == 0:
                  print(f"Processing sample #{i}")
                yield sample["text"]
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save("./biomedical_wordpiece_tokenizer.json")
    print("✅ Step 1: Trained tokenizer saved as biomedical_wordpiece_tokenizer.json")

# ====== Step 2: Expand vocab ======
def is_valid_token(token):
    # 去掉过短或仅符号的 token
    if len(token) <= 2:
        return False
    return True

def filter_tokens(tokens):
    # 筛选有意义的 token
    filtered = [t for t in tokens if is_valid_token(t)]
    return filtered

def expand_bert_vocab(base_model="google-bert/bert-base-uncased", new_tokenizer_path="./biomedical_wordpiece_tokenizer.json", num_new_tokens=5000):
    original_tokenizer = BertTokenizer.from_pretrained(base_model)
    new_tokenizer = Tokenizer.from_file(new_tokenizer_path)
    # 获取新 tokenizer 的词表：token -> id
    new_vocab_dict = new_tokenizer.get_vocab()
    # 排除原始 BERT 中已有的 token
    original_vocab_set = set(original_tokenizer.vocab)
    new_tokens_only = [token for token in new_vocab_dict if token not in original_vocab_set]
    # print(len(new_tokens_only))
    # 排序
    # new_tokens_sorted = sorted(new_tokens_only, key=lambda x: new_vocab_dict[x])
    new_tokens_sorted = sorted(new_tokens_only, key=lambda x: len(x), reverse=True)
    # 排除冗余 token
    filtered_tokens = filter_tokens(new_tokens_sorted)[:num_new_tokens]
    new_tokens = filtered_tokens
    
    # 随机抽取 50 个
    # sampled_tokens = random.sample(new_tokens, 50)
    # for token in sampled_tokens:
    #     print(token)

    original_tokenizer.add_tokens(new_tokens)
    original_tokenizer.save_pretrained("./expanded_bert_tokenizer")
    print(f"✅ Step 2: Added {len(new_tokens)} new tokens. Saved tokenizer to expanded_bert_tokenizer")
    return len(original_tokenizer)

# ====== Step 3: Resize model embedding ======
def resize_bert_embeddings(model_name="google-bert/bert-base-uncased", tokenizer_path="./expanded_bert_tokenizer", num_labels=11):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained("./bert_with_expanded_vocab")
    tokenizer.save_pretrained("./bert_with_expanded_vocab")
    print("✅ Step 3: Resized model saved to bert_with_expanded_vocab")

def compare_parameters():
  # 加载原始 tokenizer 和模型
  orig_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
  orig_model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

  # 读取原始和新 tokenizer 的词表大小
  old_vocab_size = orig_model.get_input_embeddings().num_embeddings
  new_tokenizer = BertTokenizer.from_pretrained("./expanded_bert_tokenizer")
  new_vocab_size = len(new_tokenizer)

  # 嵌入维度
  embedding_dim = orig_model.get_input_embeddings().embedding_dim

  # 计算新增参数数量
  new_params = (new_vocab_size - old_vocab_size) * embedding_dim

  print(f"原始词表大小: {old_vocab_size}")
  print(f"新词表大小: {new_vocab_size}")
  print(f"嵌入维度: {embedding_dim}")
  print(f"新增参数总数: {new_params}")

# ====== Step 4: Train model on HoC ======

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

def evaluate_metrics(true, pred):
    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro")
    micro_f1 = f1_score(true, pred, average="micro")
    return acc, macro_f1, micro_f1

def train_bert(train_texts, train_labels, test_texts, test_labels, num_labels):
    tokenizer = BertTokenizer.from_pretrained("./bert_with_expanded_vocab")
    model = BertForSequenceClassification.from_pretrained("./bert_with_expanded_vocab", num_labels=num_labels)
    model.to(device)

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)

    training_args = TrainingArguments(
      output_dir="./results",
      num_train_epochs=15,
      learning_rate=3e-5,
    #   warmup_ratio=0.1,
      per_device_train_batch_size=8,
      per_device_eval_batch_size=64,
      warmup_steps=500,
      weight_decay=0.01,
      logging_dir='./logs',
      logging_steps=100,
      evaluation_strategy="epoch",  
      save_strategy="epoch",        
      load_best_model_at_end=True, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    preds = trainer.predict(test_dataset)
    preds_labels = preds.predictions.argmax(axis=1)
    acc, macro_f1, micro_f1 = evaluate_metrics(test_labels, preds_labels)
    print(f"✅ Step 4: Final Test - Acc: {acc:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
    return acc, macro_f1, micro_f1


def load_hoc_data():
    train_df = pd.read_parquet("./data/HoC/train.parquet")
    test_df = pd.read_parquet("./data/HoC/test.parquet")
    return train_df, test_df

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
    
    print("\n=== BERT Model ===")
    bert_acc, bert_macro, bert_micro = train_bert(train_texts, train_labels, test_texts, test_labels, num_classes)
    
    return {
        "hoc": {
            "bert": {"accuracy": bert_acc, "macro_f1": bert_macro, "micro_f1": bert_micro}
        }
    }

def load_tokenizers():
    bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert_tokenizer_expanded = BertTokenizer.from_pretrained("./expanded_bert_tokenizer") 
    return bert_tokenizer, bert_tokenizer_expanded

def sample_hoc_sentences(n=3):
    train_df, _ = load_hoc_data()
    sampled_rows = train_df.sample(n=n, random_state=24)

    print("\n=== Sampled Sentences from HoC Training Set ===")
    for i, row in sampled_rows.iterrows():
        print(f"Sample {i+1}: {row['text']}")
    return sampled_rows["text"].tolist()

def compare_tokenizers_on_samples():
    sentences = sample_hoc_sentences()
    bert_tokenizer, expanded_tokenizer = load_tokenizers()

    for i, sent in enumerate(sentences):
        print(f"\n=== Sample {i+1} ===")
        print(f"Original sentence:\n{sent}\n")
        
        orig_tokens = bert_tokenizer.tokenize(sent)
        expanded_tokens = expanded_tokenizer.tokenize(sent)
        
        print(f"Original BERT tokenizer:\n{orig_tokens}\n")
        print(f"Expanded BERT tokenizer:\n{expanded_tokens}\n")
    
def compare_tokenizers_length():
    bert_tokenizer, expanded_tokenizer = load_tokenizers()
    train_df, _ = load_hoc_data()
    # 加载训练数据
    texts = train_df["text"].tolist()

    # 统计平均长度
    orig_lengths = [len(bert_tokenizer.tokenize(text)) for text in tqdm(texts)]
    exp_lengths = [len(expanded_tokenizer.tokenize(text)) for text in tqdm(texts)]

    print("Original BERT 平均长度：", sum(orig_lengths) / len(orig_lengths))
    print("Expanded BERT 平均长度：", sum(exp_lengths) / len(exp_lengths))


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # Step 1
    train_tokenizer("./pubmed_sampled_corpus.jsonline")

    # Step 2
    expand_bert_vocab()

    # Step 3
    resize_bert_embeddings()

    # # report
    # compare_tokenizers_on_samples()
    # compare_tokenizers_length()
    # compare_parameters()

    # Step 4 
    results = {}
    # 处理HoC数据集
    hoc_results = process_hoc()
    results.update(hoc_results)
    
    # 最终结果
    print("\n=== Final Results ===")
    for dataset_name, metrics in results.items():
        print(f"\nDataset: {dataset_name}")
        for model_name, scores in metrics.items():
            print(f"{model_name}: Accuracy={scores['accuracy']:.4f}, Macro-F1={scores['macro_f1']:.4f}, Micro-F1={scores['micro_f1']:.4f}")

