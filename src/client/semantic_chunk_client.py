# -*- coding: utf-8 -*-
# 声明文件使用 UTF-8 编码，确保能正确处理包含非 ASCII 字符的文本

# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------

# 导入 random 模块，用于生成随机数
import random
# 导入 json 模块，用于处理 JSON 数据的编码和解码
import json
# 导入 requests 模块，用于发送 HTTP 请求
import requests
# 导入 os 模块，用于与操作系统进行交互，如文件路径操作
import os
# 导入 pickle 模块，用于对象的序列化和反序列化
import pickle
# 从 typing 模块导入 List 类型注解，用于指定变量为列表类型
from typing import List
# 导入 sys 模块，用于访问与 Python 解释器紧密相关的变量和函数
import sys
# 再次导入 os 模块，虽然重复导入，但 Python 会忽略重复导入的影响
import os
# 将当前文件所在目录的上一级目录添加到 Python 模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从 constant 模块中导入 clean_docs_path 变量
from constant import clean_docs_path

# 定义语义切分服务的请求 URL
URL = "http://0.0.0.0:6000/v1/semantic-chunks"

# 定义请求语义切分的函数，接收句子列表和分组大小作为参数
def request_semantic_chunk(sentences, group_size):
    # 定义请求头，指定请求内容类型为 JSON
    headers = {
        "Content-Type":"application/json"
    }
    # 将请求数据转换为 JSON 字符串
    payload = json.dumps({
        "sentences": sentences,
        "group_size": group_size
    })
    try:
        # 发送 POST 请求到指定 URL，携带请求头和请求数据
        response = requests.post(
            URL,
            headers=headers,
            data=payload
        )
        # 将响应内容解析为 JSON 格式
        res = response.json()
        # 从响应结果中提取切分后的文本块
        text = res["chunks"]
    except Exception as e:
        # 若请求过程中出现异常，打印错误信息
        print(f"call reject failed:{e}")
        # 异常发生时，返回原始句子列表
        text = sentences
    # 返回切分后的文本块
    return text

# 主程序入口
if __name__ == '__main__':
    # 从指定文件中反序列化加载处理后的文档数据
    data = pickle.load(open("data/processed_docs/clean_docs.pkl", "rb"))
    # 从文档数据的索引范围中随机抽取 10 个索引
    index = random.sample(range(len(data)), 10)
    # 遍历随机抽取的索引
    for idx in index:
        # 获取对应索引文档的文本内容
        doc = data[idx].page_content
        # 调用请求语义切分的函数，传入文档内容和分组大小 10
        res = request_semantic_chunk(doc, 10)
        # 打印分隔线
        print("="*100)
        # 遍历切分后的文本块
        for r in res:
            # 打印每个文本块
            print(r)
            # 打印分隔线
            print("="*100)
