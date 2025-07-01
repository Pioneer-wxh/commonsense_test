#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import argparse

def extract_subset(input_file, output_file, num_samples=1000):
    """
    从原始数据集中提取指定数量的样本，创建一个子集用于测试
    
    参数:
        input_file (str): 输入JSON文件的路径
        output_file (str): 输出JSON文件的路径
        num_samples (int): 要提取的样本数量
    """
    print(f"从 {input_file} 读取数据...")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 根据数据结构提取样本
    if isinstance(data, list):
        # 如果数据是列表格式
        subset = data[:num_samples]
        print(f"从列表中提取了 {len(subset)} 条数据")
    elif isinstance(data, dict):
        if 'train' in data:
            # 如果数据是包含'train'键的字典
            subset = data['train'][:num_samples]
            print(f"从'train'键中提取了 {len(subset)} 条数据")
        else:
            # 如果数据是普通字典，取第一个键值对中的值
            first_key = list(data.keys())[0]
            subset = data[first_key][:num_samples]
            print(f"从键'{first_key}'中提取了 {len(subset)} 条数据")
    else:
        raise ValueError("不支持的数据格式")
    
    # 保存到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    
    print(f"已成功将 {len(subset)} 条数据保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='从数据集中提取子集用于测试')
    parser.add_argument('--input', '-i', type=str, 
                        default='/root/autodl-tmp/commonsense_test/CR_MR/math_10k.json',
                        help='输入JSON文件路径')
    parser.add_argument('--output', '-o', type=str, 
                        default='/root/autodl-tmp/commonsense_test/CR_MR/math_1k.json',
                        help='输出JSON文件路径')
    parser.add_argument('--num', '-n', type=int, default=1000,
                        help='要提取的样本数量')
    
    args = parser.parse_args()
    extract_subset(args.input, args.output, args.num)

if __name__ == "__main__":
    main() 
