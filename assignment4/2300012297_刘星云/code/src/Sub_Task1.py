import argparse
import os
import json

import numpy as np

from corpus import *
from grammar import *
from model import *
from tokenizer import *
from prompt import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # linguistic resources
    parser.add_argument('--src_lang', type=str, default='za')
    parser.add_argument('--tgt_lang', type=str, default='zh')
    parser.add_argument('--grammar_path', type=str, default='../Data_project_1/grammar_book.json')
    parser.add_argument('--test_data_path', type=str, default='../Data_project_1/test_data.json')

    # model
    parser.add_argument('--model_name', type=str, default='qwen-max')
    parser.add_argument('--chat_mode', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=1)

    # cofig for generation
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.05)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=100)

    # config for prompt
    parser.add_argument('--prompt_type', type=str, default='za2zh')

    # output path
    parser.add_argument('--output_path', type=str, default="../output/submission1.jsonl")

    args = parser.parse_args()

    
    # load grammar
    grammar = GrammarBook(args.grammar_path)

    # load test data
    test_data = json.load(open(args.test_data_path, "r", encoding='utf-8'))
    
    # sampling params
    if args.do_sample:
        # set seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # construct prompt 
    prompt_func = construct_prompt_za2zh_with_related_words
    # prompt_func = construct_prompt_za2zh_with_related_words_new
    # prompt_func = zero_shot

    # chat mode
    if args.chat_mode:
        chat_template = model_to_chat_template['qwen']

    # output path
    print("output_path:", args.output_path)
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    fout = open(args.output_path, "w", encoding="utf-8")

    # API 请修改api或设置环境变量
    api_key = "sk-697feb11d7ae42da8c28e047977eb422"

    # do test
    for item in tqdm(test_data):
        src_sentence = item[args.src_lang]
        related_words = item['related_words']
        
        # construct prompt
        prompt = prompt_func(src_sentence, grammar, related_words, args)
        # print(prompt)

        # special treatment for chat mode
        if args.chat_mode:
            prompt = chat_template.format(prompt=prompt)

        # generate
        pred = get_pred_qwen(prompt, api_key)

        print("input:", src_sentence)
        print("pred:", pred)
        # print("prompt:", prompt)

        fout.write(json.dumps({"query": src_sentence, "pred": pred, "prompt": prompt, "id": item['id']}, ensure_ascii=False) + "\n")

