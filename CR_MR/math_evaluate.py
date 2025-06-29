import copy
import json
import os
import re# 导入 re 模块，用于正则表达式操作，用于从字符串中提取数字或字母
import sys
import argparse# 导入 argparse 模块，用于解析命令行参数，方便用户通过命令行指定参数。

import fire

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,# - share_gradio: 是否通过 Gradio 共享模型（默认 False，未在代码中使用）。
):
    args = parse_args()# 调用 parse_args 函数解析命令行参数，存储在 args 变量中

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
    ):
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True,  # 启用采样
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # 添加attention_mask
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,  # 启用缓存以提高性能
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """
    save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}.json'
    create_dir('experiment/')

    dataset = load_data(args)# 调用 load_data 函数加载指定数据集，返回数据集列表。
    tokenizer, model = load_model(args)# 调用 load_model 函数加载分词器和模型，返回元组 (tokenizer, model)。

    if args.adapter == "LoRA" or args.adapter == "DoRA":# 如果适配器类型是 LoRA 或 DoRA，执行权重合并操作。
        print("Merge LoRA/DoRA weights into the original weights")# 打印信息，表示正在将 LoRA/DoRA 权重合并到原始模型权重中。
        key_list = [(key,module) for key, module in model.model.named_modules()]# 获取模型中所有命名模块的键和模块对象，存储为列表。
        for key,module in key_list:# 遍历所有模块。
            if isinstance(model.peft_config.target_modules, str):# 检查 PEFT 配置中的 target_modules 是否为字符串。
                target_module_found = re.fullmatch(model.peft_config.target_modules, key)# 使用正则表达式检查模块名是否完全匹配 target_modules。
            else:
                target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.target_modules)

            if args.adapter == "DoRA":
                if model.peft_config.Wdecompose_target_modules != None:
                    if isinstance(model.peft_config.Wdecompose_target_modules, str):
                        wdecompose_target_module_found = re.fullmatch(model.peft_config.Wdecompose_target_modules, key)
                    else:
                        wdecompose_target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.Wdecompose_target_modules)
                else: 
                    wdecompose_target_module_found = False
            else:
                wdecompose_target_module_found = False

            if target_module_found:# 如果找到目标模块。
                print(f"found {key}")#打印
                # print(f"module.merged {module.merged}")
                # print(f"module.merge_weights {module.merge_weights}")
                module.merge_weights = True#标记为True已经完成合并
                module.train(mode=False)#不能进行训练

            elif wdecompose_target_module_found:
                print(f"found {key}")
                # print(f"module.merged {module.merged}")
                # print(f"module.merge_weights {module.merge_weights}")
                module.merge_weights = True
                module.train(mode=False)
                
    total = len(dataset)# 获取数据集的总长度。
    correct = 0# 初始化正确预测的计数器
    miss = 0.001# 设置数值预测的误差容忍度（用于判断预测是否正确）。
    output_data = []# 初始化输出数据列表，用于存储预测结果。
    pbar = tqdm(total=total)# 创建 tqdm 进度条，用于显示测试进度。
    for idx, data in enumerate(dataset):# 遍历数据集中的每个数据项。
        instruction = data.get('instruction')

        outputs = evaluate(instruction)
        label = data.get('answer')
        flag = False# 初始化标志变量，表示预测是否正确。
        if args.dataset.lower() in ['aqua']:# 如果数据集是 AQuA（选择题数据集）。
            predict = extract_answer_letter(args, outputs)# 提取模型输出中的字母答案（A, B, C, D, E）。
            if label == predict:
                correct += 1
                flag = True
        else:# 对于其他数据集（数值型答案）。
            if isinstance(label, str):# 如果答案是字符串，转换为浮点数。
                label = float(label)
            predict = extract_answer_number(args, outputs)# 提取模型输出中的数值答案。
            if abs(label - predict) <= miss:
                correct += 1
                flag = True# 如果预测值与正确答案的差值在误差范围内，正确计数加 1，设置 flag 为 True。
        new_data = copy.deepcopy(data)# 创建数据项的深拷贝，避免修改原始数据。
        new_data['output_pred'] = outputs
        new_data['pred'] = predict
        new_data['flag'] = flag
        output_data.append(new_data)# 将新数据项添加到输出数据列表。
        print(' ')
        print('---------------')
        print(outputs)
        print('prediction:', predict)
        print('label:', label)
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')
        with open(save_file, 'w+') as f:# 将输出数据写入 JSON 文件，带缩进格式化。
            json.dump(output_data, f, indent=4)
        pbar.update(1)# 更新进度条。
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'# 构造数据集文件路径，格式为 dataset/{数据集名}/test.json。
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))# 读取 JSON 文件并解析为 Python 对象。
    return json_data


def parse_args():# 定义 parse_args 函数，解析命令行参数。
    parser = argparse.ArgumentParser()# 创建 ArgumentParser 对象。
    parser.add_argument('--dataset', choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP'],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', 'BLOOM-7B', 'GPT-j-6B','Qwen2.5-7B','Qwen2.5-7B-Instruct'], required=True)#修改
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix','Dislora'],
                        required=True)
    parser.add_argument('--base_model', required=True)# 添加 base_model 参数，指定基础模型路径或名称，必须提供。
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    # 定义 load_model 函数，加载分词器和模型。
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model# 获取基础模型路径或名称。
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights# 获取 LoRA 权重路径或名称
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    if args.model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq# 加载基础模型，支持 8 位精度，使用 float16 类型，自动分配设备。
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
        )# 加载 PEFT 模型，应用 LoRA 权重,就是将训练好的适配器权重加上来
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        # 设置模型和分词器的填充、开始、结束 token ID，修复某些模型的配置问题。
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()# 将模型设置为评估模式。从而防止修改参数
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:# 定义 load_instruction 函数，加载指令（未使用）。
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction

# 定义 extract_answer_number 函数，从输出中提取数值答案。
def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()# 获取数据集名称并转换为小写。
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:# 如果数据集是数值型数据集。
        sentence = sentence.replace(',', '')# 移除输出字符串中的逗号。
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]# 使用正则表达式提取所有数字（包括负数和小数）。
        if not pred:
            return float('inf')# 如果没有提取到数字，返回无穷大。
        # 取最后一个数字并转换为浮点数，作为答案
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))# 如果数据集不支持，所选的数据集不在字典中，抛出未实现错误。
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

# 定义 extract_answer_letter 函数，从输出中提取字母答案（用于选择题）
def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()# 去除输出字符串的首尾空格。
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)# 使用正则表达式提取 A, B, C, D, E 中的任意一个。
    if pred_answers:
        return pred_answers[0]# 如果提取到答案，返回第一个答案。
    else:
        return ''


if __name__ == "__main__":
    fire.Fire(main)
