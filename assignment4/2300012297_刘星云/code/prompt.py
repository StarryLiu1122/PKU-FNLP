from corpus import lang2tokenizer
import random
import json

model_to_chat_template = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}

# Sub_task 2
def zero_shot_task2(src_sent, dictionary, parallel_corpus, grammar, args):
    prompt = f"""你是一个专业的语言学家，擅长将壮语翻译成汉语。
    请将以下壮语句子准确翻译成符合日常表达习惯的汉语句子：
    壮语句子：{src_sent}
    请只输出最终翻译结果，不要附加解释、语法分析或其他说明；标点符号和原句保持一致，不要附加特殊符号
    """
    return prompt

def construct_prompt_za2zh(src_sent, dictionary, parallel_corpus, grammar, args):
    # retrieve parallel sentences
    if args.num_parallel_sent > 0:
        top_k_sentences_with_scores = parallel_corpus.search_by_bm25(src_sent, query_lang=args.src_lang, top_k=args.num_parallel_sent)
    else:
        top_k_sentences_with_scores = []


    def get_word_explanation_prompt(text):
        prompt = "## 在上面的句子中，"
        tokenized_text = lang2tokenizer[args.src_lang].tokenize(text, remove_punc=True)
        for word in tokenized_text:
            # 先看是否有精确匹配
            exact_match_meanings = dictionary.get_meanings_by_exact_match(word, max_num_meanings=4)
            if exact_match_meanings is not None:
                concated_meaning = "”或“".join(exact_match_meanings)
                concated_meaning = "“" + concated_meaning + "”"
                prompt += f"壮语词语“{word}”在汉语中可能的翻译是{concated_meaning}；\n"
            else:
                # 如果没有精确匹配，则看是否有模糊匹配
                fuzzy_match_meanings = dictionary.get_meanings_by_fuzzy_match(word, top_k=5, max_num_meanings_per_word=3)
                for item in fuzzy_match_meanings[:5]:
                    concated_meaning = "”或“".join(item["meanings"])
                    concated_meaning = "“" + concated_meaning + "”"
                    prompt += f"壮语词语“{item['word']}”在汉语中可能的翻译是{concated_meaning}；\n"
        return prompt
    
    prompt = "你是一个专业的语言学家，擅长将壮语翻译成汉语。"

    if args.num_parallel_sent > 0:
        prompt += "# 请仿照样例，参考给出的词汇和语法，将壮语句子翻译成汉语。\n\n"
        for i in range(len(top_k_sentences_with_scores)):
            item = top_k_sentences_with_scores[i]["pair"]
            prompt += f"## 样例{i+1} 请将下面的壮语句子翻译成汉语：{item[args.src_lang]}\n"

            # 附上参考词汇
            prompt += get_word_explanation_prompt(item[args.src_lang])
            prompt += f"## 所以，该壮语句子完整的汉语翻译是：{item['zh']}\n\n"
    
    # prompt最后是需要翻译的句子
    prompt += f"## 现在，请将下面的壮语句子准确翻译成符合日常表达习惯的汉语句子：{src_sent}\n"
    prompt += get_word_explanation_prompt(src_sent)
    if grammar is not None:  # 加上相关的语法
        relevant_grammars = grammar.search_relevant_grammars(src_sent, top_k=10)
        if relevant_grammars:
            prompt += "# 以下是相关的语法：\n"
            for i, grammar in enumerate(relevant_grammars):
                # prompt += f"## 语法{i+1}：{grammar['grammar_description']}例子：{grammar['examples']}\n"
                prompt += f"## 语法{i+1}：{grammar['grammar_description']}\n"
                for j, example in enumerate(grammar.get('examples', [])):
                    za = example.get('za', '')
                    zh = example.get('zh', '')
                    rw = json.dumps(example.get('related_words', {}), ensure_ascii=False)
                    prompt += f"- 例子{j+1}：壮语：“{za}”，汉语：“{zh}”，相关词汇：{rw}\n"
        # if relevant_grammars:
        #     prompt += "# 以下是与该句子相关的语法说明：\n"
        #     for i, grammar in enumerate(relevant_grammars):
        #         prompt += f"## 语法{i+1}：{grammar['grammar_description']}\n"
    prompt += f"## 所以，该壮语句子{src_sent}的完整的汉语翻译是（请直接给出最终的汉语翻译，不用给出分析）："

    return prompt


def construct_prompt_za2zh_with_all_grammar(src_sent, dictionary, parallel_corpus, grammar, args):
    # retrieve parallel sentences
    if args.num_parallel_sent > 0:
        top_k_sentences_with_scores = parallel_corpus.search_by_bm25(src_sent, query_lang=args.src_lang, top_k=args.num_parallel_sent)
    else:
        top_k_sentences_with_scores = []


    def get_word_explanation_prompt(text):
        prompt = "## 在上面的句子中，"
        tokenized_text = lang2tokenizer[args.src_lang].tokenize(text, remove_punc=True)
        for word in tokenized_text:
            # 先看是否有精确匹配
            exact_match_meanings = dictionary.get_meanings_by_exact_match(word, max_num_meanings=4)
            if exact_match_meanings is not None:
                concated_meaning = "”或“".join(exact_match_meanings)
                concated_meaning = "“" + concated_meaning + "”"
                prompt += f"壮语词语“{word}”在汉语中可能的翻译是{concated_meaning}；\n"
            else:
                # 如果没有精确匹配，则看是否有模糊匹配
                fuzzy_match_meanings = dictionary.get_meanings_by_fuzzy_match(word, top_k=10, max_num_meanings_per_word=3)
                for item in fuzzy_match_meanings[:5]:
                    concated_meaning = "”或“".join(item["meanings"])
                    concated_meaning = "“" + concated_meaning + "”"
                    prompt += f"壮语词语“{item['word']}”在汉语中可能的翻译是{concated_meaning}；\n"
        return prompt
    
    all_grammar_text = grammar.get_all_grammar_descriptions()
    prompt = f"""你是一个专业的语言学家，擅长将壮语翻译成汉语。
    下面是一本壮语语法书中整理出的全部语法规则，每条规则有助于理解壮语句子的结构和含义：
    {all_grammar_text}
    """

    if args.num_parallel_sent > 0:
        prompt += "# 请仿照样例，参考给出的词汇和语法，将壮语句子翻译成汉语。\n\n"
        for i in range(len(top_k_sentences_with_scores)):
            item = top_k_sentences_with_scores[i]["pair"]
            prompt += f"## 样例{i+1} 请将下面的壮语句子翻译成汉语：{item[args.src_lang]}\n"

            # 附上参考词汇
            prompt += get_word_explanation_prompt(item[args.src_lang])
            prompt += f"## 所以，该壮语句子完整的汉语翻译是：{item['zh']}\n\n"
    
    # prompt最后是需要翻译的句子
    prompt += f"## 现在，请将下面的壮语句子准确翻译成符合日常表达习惯的汉语句子：{src_sent}\n"
    prompt += get_word_explanation_prompt(src_sent)
    if grammar is not None:  # 加上相关的语法
        relevant_grammars = grammar.search_relevant_grammars(src_sent, top_k=10)
        if relevant_grammars:
            prompt += "# 以下是从语法书中检索到的与该句子最相关的语法规则及示例（供参考）：\n"
            for i, grammar in enumerate(relevant_grammars):
                # prompt += f"## 语法{i+1}：{grammar['grammar_description']}例子：{grammar['examples']}\n"
                prompt += f"## 语法{i+1}：{grammar['grammar_description']}\n"
                for j, example in enumerate(grammar.get('examples', [])):
                    za = example.get('za', '')
                    zh = example.get('zh', '')
                    rw = json.dumps(example.get('related_words', {}), ensure_ascii=False)
                    prompt += f"- 例子{j+1}：壮语：“{za}”，汉语：“{zh}”，相关词汇：{rw}\n"
        # if relevant_grammars:
        #     prompt += "# 以下是与该句子相关的语法说明：\n"
        #     for i, grammar in enumerate(relevant_grammars):
        #         prompt += f"## 语法{i+1}：{grammar['grammar_description']}\n"
    prompt += f"## 所以，该壮语句子{src_sent}的完整的汉语翻译是（请直接给出最终的汉语翻译，不用给出分析）："

    return prompt

# def construct_prompt_za2zh_with_related_words(src_sent, grammar, related_words, args):
#     prompt = ""

#     prompt += f"## 请将下面的壮语翻译成汉语：{src_sent}\n"
#     # 构造词汇提示文本
#     for word, meaning in related_words.items():
#         prompt += f"已知壮语词汇“{word}”在汉语中的翻译是“{meaning}”；\n"
        
#     if grammar is not None:  # 加上相关的语法
#         # relevant_grammars = grammar.search_relevant_grammars(src_sent, top_k=5)
#         relevant_grammars = grammar.search_relevant_grammars_forward(src_sent, related_words, top_k=5)
#         if relevant_grammars:
#             prompt += "# 以下是与该句子相关的语法说明：\n"
#             for i, grammar in enumerate(relevant_grammars):
#                 prompt += f"## 语法{i+1}：{grammar['grammar_description']}，例子：{grammar['examples']}\n"
#     prompt += f"## 所以，该壮语句子的完整的汉语翻译是（请直接给出最终的汉语翻译句子；要注意标点符号，如果原句没有标点，不用加标点）："

#     return prompt



# def construct_prompt_za2zh_with_related_words(src_sent, grammar, related_words, args):
#     prompt = f"## 你是一个资深的汉语言学家，请将下面的壮语翻译成汉语：{src_sent}\n"
#     prompt += "下面是翻译过程中你可能用到的相关语法：\n"
#     if grammar is not None:  # 相关的语法
#         # relevant_grammars = grammar.search_relevant_grammars(src_sent, top_k=5)
#         relevant_grammars = grammar.search_relevant_grammars_forward(src_sent, related_words, top_k=20)
#         if relevant_grammars:
#             # prompt += "# 以下是和你即将需要翻译的句子相关的语法：\n"
#             for i, grammar in enumerate(relevant_grammars):
#                 # prompt += f"## 语法{i+1}：{grammar['grammar_description']}例子：{grammar['examples']}\n"
#                 prompt += f"## 语法{i+1}：{grammar['grammar_description']}\n"
#                 for j, example in enumerate(grammar.get('examples', [])):
#                     za = example.get('za', '')
#                     zh = example.get('zh', '')
#                     rw = json.dumps(example.get('related_words', {}), ensure_ascii=False)
#                     prompt += f"- 例子{j+1}：壮语：“{za}”，汉语翻译：“{zh}”，相关词语：{rw}\n"
#     # 构造词汇提示文本
#     for word, meaning in related_words.items():
#         prompt += f"已知壮语词汇“{word}”在汉语中的翻译是“{meaning}”；\n"
#     prompt += f"## 请根据语法和已知的词汇，直接给出{src_sent}的汉语翻译（直接给出最终的答案，不要外加特殊符号；要注意标点符号及有无要和原句保持一致，且翻译要符合日常用语的规范）："

#     return prompt


# Sub_task 1
def zero_shot(src_sent, grammar, related_words, args):
    prompt = f"""你是一个专业的语言学家，擅长将壮语翻译成汉语。
    请将以下壮语句子准确翻译成符合日常表达习惯的汉语句子：
    壮语句子：{src_sent}
    请只输出最终翻译结果，不要附加解释、语法分析或其他说明；标点符号和原句保持一致，不要附加特殊符号
    """
    return prompt

def construct_prompt_za2zh_with_related_words(src_sent, grammar, related_words, args):
    prompt = f"""你是一个专业的语言学家，擅长将壮语翻译成汉语。
    请将以下壮语句子准确翻译成符合日常表达习惯的汉语句子：
    壮语句子：{src_sent}
    为了帮助翻译，下面是从语法书中检索到的相关的语法规则，同时每条语法后面提供了相应的例句和相关词汇（供参考）：
    """

    if grammar is not None:
        relevant_grammars = grammar.search_relevant_grammars_forward(src_sent, related_words, top_k=20)
        for i, g in enumerate(relevant_grammars):
            prompt += f"\n【语法{i+1}】{g['grammar_description']}\n"
            for j, ex in enumerate(g.get('examples', [])):
                za = ex.get('za', '')
                zh = ex.get('zh', '')
                rw = json.dumps(ex.get('related_words', {}), ensure_ascii=False)
                prompt += f"- 示例{j+1}：壮语：“{za}”，汉语：“{zh}”，相关词汇：{rw}\n"

    if related_words:
        prompt += "\n此外，以下是句子{src_sent}中各壮语词汇的翻译：\n"
        for word, meaning in related_words.items():
            prompt += f"- “{word}” → “{meaning}”\n"

    prompt += f"""\n请根据上面的语法和词汇，直接给出上述壮语句子的标准汉语翻译。
⚠️ 只输出最终翻译结果，不要附加解释、语法分析或其他说明；标点符号和原句保持一致，不要附加特殊符号。"""

    return prompt



def construct_prompt_za2zh_with_related_words_new(src_sent, grammar, related_words, args):
    all_grammar_text = grammar.get_all_grammar_descriptions()
    prompt = f"""你是一个专业的语言学家，擅长将壮语翻译成汉语。
    下面是一本壮语语法书中整理出的全部语法规则，每条规则有助于理解壮语句子的结构和含义：
    {all_grammar_text}
    ---
    现在请将以下壮语句子准确翻译成符合日常表达习惯的汉语句子：
    壮语句子：{src_sent}
    为了帮助翻译，以下是从语法书中检索到的与该句子最相关的语法规则及示例（供参考）：
    """

    if grammar is not None:
        relevant_grammars = grammar.search_relevant_grammars_forward(src_sent, related_words, top_k=20)
        for i, g in enumerate(relevant_grammars):
            prompt += f"\n【语法{i+1}】{g['grammar_description']}\n"
            for j, ex in enumerate(g.get('examples', [])):
                za = ex.get('za', '')
                zh = ex.get('zh', '')
                rw = json.dumps(ex.get('related_words', {}), ensure_ascii=False)
                prompt += f"- 示例{j+1}：壮语：“{za}”，汉语：“{zh}”，相关词汇：{rw}\n"

    if related_words:
        prompt += "\n此外，以下是句子{src_sent}中各壮语词汇的翻译：\n"
        for word, meaning in related_words.items():
            prompt += f"- “{word}” → “{meaning}”\n"

    prompt += f"""\n请根据上面的语法和词汇，直接给出上述壮语句子的标准汉语翻译。
⚠️ 只输出最终的汉语翻译结果，不要附加解释、语法分析或其他说明；标点符号和原句保持一致，不要附加特殊符号。"""

    return prompt


if __name__ == '__main__':
    pass