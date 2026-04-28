import json
import re

class GrammarBook:
    def __init__(self, grammar_path):
        self.grammar_path = grammar_path
        self.grammar_entries = []
        self.load_grammar()

    def load_grammar(self):
        with open(self.grammar_path, 'r', encoding='utf-8') as f:
            self.grammar_entries = json.load(f)

    def split_za_by_related_words(self, za_sentence, related_words):
        za_sentence = za_sentence.replace(" ", "")
        idx = 0
        length = len(za_sentence)
        sorted_keys = sorted(related_words.keys(), key=lambda x: -len(x))
        
        tokens = []
        buffer = ""  # 收集未匹配字符
        
        while idx < length:
            matched = False
            for key in sorted_keys:
                if za_sentence.startswith(key, idx):
                    if buffer:
                        tokens.append(buffer)
                        buffer = ""
                    tokens.append(key)
                    idx += len(key)
                    matched = True
                    break
            if not matched:
                buffer += za_sentence[idx]
                idx += 1

        if buffer:
            tokens.append(buffer)
        
        return tokens


    def search_relevant_grammars(self, sentence, top_k=5):
        """
        返回最相关的 top_k 条语法规则
        匹配原则：根据句子中出现的关键词（如例句词）在 grammar entries 中出现的频率排序
        """
        # 提取关键词（简单分词器，也可换为 tokenizer）
        words = re.findall(r"\b\w+\b", sentence.lower())

        scores = []
        for entry in self.grammar_entries:
            score = 0
            for ex in entry.get('examples', []):
                for za_word in re.findall(r"\b\w+\b", ex['za'].lower()):
                    if za_word in words:
                        score += 1
            scores.append((score, entry))

        # 按相关度排序并取前 top_k 个
        scores.sort(key=lambda x: x[0], reverse=True)
        top_grammar_entries = [entry for score, entry in scores[:top_k] if score > 0]
        return top_grammar_entries
    
    
    def search_relevant_grammars_forward(self, sentence, related_words=None, top_k=20):
        def is_reduplication(word):
            """判断是否是完全重复的双音节（如 'mbanjmbanj'）"""
            half = len(word) // 2
            return word[:half] == word[half:] and len(word) % 2 == 0

        def split_reduplication(word):
            """将 reduplication 拆解为基本单词"""
            if is_reduplication(word):
                return word[:len(word) // 2]
            return None
        
        za = sentence.replace(" ", "")  # 去掉空格以模拟连写形式
        tokens = self.split_za_by_related_words(za, related_words or {})  # 拆词
        token_set = set(tokens)

        scored_grammars = []

        for entry in self.grammar_entries:
            score = 0
            # 1. 在 grammar_description 中查找 token
            description = entry.get("grammar_description", "")
            for token in token_set:
                if token in description:
                    score += 5

            # 2. 在 related_words 中查找 token
            for ex in entry.get("examples", []):
                ex_related_words = ex.get("related_words", {})
                for word in ex_related_words.keys():
                    if word in token_set:
                        score += 1
                        
            # 检查是否有 token 被重叠使用，例如 mbanjmbanj -> mbanj 重叠
            for token in tokens:
                if token + token in za:
                    # 检查 grammar 中是否描述了重叠现象
                    if "重叠" in description or "每一" in description or "重复" in description:
                        score += 5 

            if score > 0:
                scored_grammars.append((score, entry))

        # 按得分排序
        scored_grammars.sort(key=lambda x: x[0], reverse=True)

        # 只返回 top_k 个 entry
        return [entry for score, entry in scored_grammars[:top_k]]
    
    def get_all_grammar_descriptions(self):
        """
        将所有 grammar_description 组织成可插入 prompt 的长字符串。
        每条语法用编号开头，适合直接拼入提示词。
        """
        lines = []
        for i, entry in enumerate(self.grammar_entries):
            desc = entry.get("grammar_description", "").strip()
            if desc:
                lines.append(f"【语法{i+1}】{desc}")
        return "\n".join(lines)
            


if __name__ == "__main__":
    pass
