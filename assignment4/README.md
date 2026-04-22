# Project 1: Machine Translation for Low-Resource Languages

本项目为低资源语言壮语构建了一个机器翻译系统

## 📦 Code Structures

```bash
附录/
├── code/                            
    ├── Sub-Task1.py                    # Sub-Task 1 主文件
    ├── Sub-Task2.py                    # Sub-Task 2 主文件
    ├── model.py                        # qwen-max API
    ├── grammar.py                      
    ├── dictionary.py                   
    ├── corpus.py                       
    ├── prompt.py                       # construct prompt
    ├── tokenizer.py                   
    └── save2csv.py                     # used for evaluation
├── output/                           
    ├── submission1.jsonl               # prompt for Sub-Task 1
    └── submission2.jsonl               # prompt for Sub-Task 2
├── Data_project_1                    
├── Data_project_2
├── requirements.txt                    # dependencies
├── report.pdf                    
└── README.md                   

```

## ⚙️ Dependencies

要设置此项目的环境，请执行以下步骤：

请使用 Python 3.8+ 环境，推荐使用虚拟环境或 conda 环境管理。

```bash
# Install core dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

1.Sub-Task 1: Simple and Controlled

```bash
python Sub_Task1.py
```

2.Sub-Task 2: Difficult and Realistic

```bash
python Sub_Task2.py
```

## 📌 Notes

- 运行前请修改api-key或设置环境变量
