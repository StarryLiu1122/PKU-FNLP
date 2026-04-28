import os
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def load_model(model_name, model_path, n_gpu=1, use_vllm=True):
    print("loading model...")
    if use_vllm:
        llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=n_gpu, max_model_len=8192)
        print("loaded!")
        return llm
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
        print("loaded!")
        return llm, tokenizer
    
    

def get_pred(llm, sampling_params, prompt):
    # print("prompt:", prompt)
    outputs = llm.generate(prompt, sampling_params)
    result = outputs[0].outputs[0].text.strip().split('\n')[0]
    result = result.split("<|endoftext|>")[0].strip()
    result = result.split("<|im_end|>")[0].strip()
    
    # print("result:", result)
    return result

try:
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_key="sk-697feb11d7ae42da8c28e047977eb422",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ],
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
    
    
