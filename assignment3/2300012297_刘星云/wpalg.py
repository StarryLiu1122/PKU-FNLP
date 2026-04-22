from transformers import AutoTokenizer
from collections import defaultdict


def wordpiece(training_corpus, vocab_size):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    word_freqs = defaultdict(int)
    for text in training_corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    alphabet = []
    for word in word_freqs.keys():
        if word[0] not in alphabet:
            alphabet.append(word[0])
        for letter in word[1:]:
            if f"##{letter}" not in alphabet:
                alphabet.append(f"##{letter}")

    alphabet.sort()

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

    # Do NOT add your above this line.
    #======
    
    # Add your code here.
    splits = {
        word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
        for word in word_freqs.keys()
    }
    
    def compute_pair_scores(splits):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores

    def merge_pair(a, b, splits):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits
    
    while len(vocab) < vocab_size:
        scores = compute_pair_scores(splits)
        best_pair, max_score = "", None
        for pair, score in scores.items():
            if max_score is None or max_score < score:
                best_pair = pair
                max_score = score
        splits = merge_pair(*best_pair, splits)
        new_token = (
            best_pair[0] + best_pair[1][2:]
            if best_pair[1].startswith("##")
            else best_pair[0] + best_pair[1]
        )
        vocab.append(new_token)
    #======
    # Do NOT add your below this line.

    return vocab

if __name__ == "__main__":
    default_training_corpus = [
        "peking university is located in haidian district",
        "computer science is the flagship major of peking university",
        "the school of electronic engineering and computer science enrolls approximately five hundred new students each year"  
    ]

    default_vocab_size = 120

    my_vocab = wordpiece(default_training_corpus, default_vocab_size)

    print('The vocab:', my_vocab)

    def encode_word(custom_vocab, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in custom_vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize(custom_vocab, text):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        encoded_words = [encode_word(custom_vocab, word) for word in pre_tokenized_text]
        return sum(encoded_words, [])

    print('Tokenization result:', tokenize(my_vocab, 'nous etudions a l universite de pekin'))
