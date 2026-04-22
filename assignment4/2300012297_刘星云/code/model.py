import json
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from openai import OpenAI

def load_model(model_name, model_path, n_gpu=1):
    print("loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    print("loaded!")
    return llm, tokenizer
    
    
def get_pred(llm, tokenizer, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = len(inputs['input_ids'][0])
    # print("input_len", input_len)
    inputs = inputs.to('cpu')
    # inputs = inputs.to('cuda')
    preds = llm.generate(
        **inputs,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )
    pred = tokenizer.decode(preds[0][input_len:], skip_special_tokens=True)
    output = pred
    result = output.strip().split('\n')[0]
    return result


def get_pred_qwen(prompt, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    try:
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        result = completion.choices[0].message.content.strip()
        return result
    except Exception as e:
        print("出错了：", e)
        return None
    

if __name__ == '__main__':
    pass