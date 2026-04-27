# PKU-FNLP: 自然语言处理基础

![PKU](https://img.shields.io/badge/University-PKU-red)
![Course](https://img.shields.io/badge/Course-FNLP-blue)
![Field](https://img.shields.io/badge/Field-NLP-green)

> **北京大学《自然语言处理基础》课程 (FNLP) 学习笔记与实验代码仓库**

---

## 📌 课程简介

本仓库记录了作者于25春季在北京大学修读《自然语言处理基础》课程期间的学习成果，包含课程笔记、实验作业及项目实现。课程内容涵盖了从传统语言学方法到现代深度学习架构的 NLP 核心原理。

**核心技术栈：**
* **经典 NLP：** 文本分类、序列标注、语言模型、句法分析。
* **深度学习：** 神经网络语言模型、RNN/LSTM、Transformer、预训练模型。
* **前沿应用：** 大语言模型 (LLM) 基础、提示学习、文本生成。

---

## 📂 仓库结构

```text
PKU-FNLP/
├── 📂 Lecture slides/                # 课程PPT
├── 📂 assignment1/                   # 报告一：设计挑战性问题
├── 📂 assignment2/                   # 作业二：文本分类模型训练
├── 📂 assignment3/                   # 作业三：生物医学领域的文本分类任务
├── 📂 assignment4/                   # 项目四：壮语翻译
├── 📂 quiz/                          # 课程quiz
└── 📂 参考教材/                  
```

## 🧪 实验内容详述

<table>
  <thead>
    <tr>
      <th width="15%">项目</th>
      <th width="20%">主题</th>
      <th width="50%">核心算法与任务</th>
      <th width="15%">交付物</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>assignment 1</b></td>
      <td><b>报告</b></td>
      <td>设计 LLMs 无法正确回答的挑战性问题</td>
      <td>💻 <a href="./assignment1/">代码</a></td>
    </tr>
    <tr>
      <td><b>assignment 2</b></td>
      <td><b>文本分类模型训练</b></td>
      <td>实现并训练两个文本分类模型：log-linear 模型和 BERT 模型</td>
      <td>💻 <a href="./assignment2/">代码</a></td>
    </tr>
    <tr>
      <td><b>assignment 3</b></td>
      <td><b>生物医学领域的文本分类任务</b></td>
      <td>扩展BERT分词器的词汇量并用于生物医学领域的文本分类任务</td>
      <td>💻 <a href="./assignment3/">代码</a></td>
    </tr>
    <tr>
      <td><b>Project</b></td>
      <td><b>期末项目：壮语翻译</b></td>
      <td>综合运用课程所学，完成一个完整的 NLP 应用系统（机器翻译</td>
      <td>💻 <a href="./assignment4/">代码</a></td>
    </tr>
  </tbody>
</table>

## 🚀 运行实验

### 1. 环境准备

本项目建议在 `Python 3.8+` 环境下运行。推荐使用 [Conda](https://www.anaconda.com/) 管理虚拟环境以避免依赖冲突：

```bash
# 创建并激活环境
conda create -n fnlp python=3.8
conda activate fnlp

# 安装核心依赖
pip install numpy scipy matplotlib torch torchvision transformers scikit-learn jieba
```

### 2. 实验执行

每个作业文件夹（assignment 1-4）中均包含对应的启动脚本。请进入相应目录后运行，例如启动 assignment 2 的文本分类实验：

```Bash
cd assignment2
python text_classification.py
```

## 💻 核心技术点

本项目的代码实现深度涉及以下自然语言处理核心领域：

- 语言学基础 (Linguistics): 中文分词、词性标注、命名实体识别、句法分析。

- 统计学习 (Statistical Learning): 隐马尔可夫模型 (HMM)、最大熵模型、条件随机场 (CRF)。

- 深度学习 (Deep Learning): 循环神经网络 (RNN/LSTM)、注意力机制、Transformer 架构。

- 预训练技术 (Pre-training): 词嵌入 (Word2Vec/GloVe)、BERT、GPT 等大语言模型基础。

## 📚 参考文献与致谢

在学习与实验过程中，参考了以下优秀的资源与教材：

* **补充学习资料：**
    * **Stanford CS224n：** 深度学习自然语言处理课程。
    * **Hugging Face Transformers：** 预训练模型库与文档。
* **学术教材：**
    * **Jurafsky & Martin：** *Speech and Language Processing (3rd ed. draft)*。
    * **Yoav Goldberg：** Neural Network Methods for Natural Language Processing。

## ⚖️ 许可与规范

1.  **学术诚信：** 本仓库的所有代码及报告仅供个人学习记录与学术交流使用。请后修同学切勿直接搬运代码提交，共同维护北京大学良好的学术氛围。
2.  **版权声明：**
    * **官方素材：** 课程讲义、官方实验框架及相关多媒体素材的版权归 **北京大学 FNLP 课程组**所有。
    * **原创实现：** 本仓库中由作者独立编写的算法实现、代码修改及实验报告文字版权归作者本人所有。
3. **权利维护 (Take-down Policy)**：本仓库致力维护尊重原创的学术环境。若其中包含的某些素材（如第三方笔记、课件等）无意中侵犯了您的版权，请发送邮件至 **[i2793521817@outlook.com]**，我会在核实后第一时间进行删除或更正标注。

## 👤 联系作者

邮箱： i2793521817@outlook.com

研究方向：机器人具身智能

最后更新日期: 2026年4月22日

## 

<div align="center">

**⭐ Star us on GitHub if the repository helps your research!**

Made with ❤️ by Xingyun Liu

</div>