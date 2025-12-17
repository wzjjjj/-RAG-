import gc
# 导入正则表达式模块，用于字符串的匹配和分割操作
import re
# 导入系统模块，提供了一些与 Python 解释器和操作系统交互的功能
import sys
# 导入数学模块，提供了一些常用的数学函数
import math
# 导入 requests 库，用于发送 HTTP 请求
import requests
# 导入 uvicorn 库，一个 ASGI 服务器，用于运行 FastAPI 应用
import uvicorn
# 导入 PyTorch 库，用于深度学习任务
import torch
# 导入 Pandas 库，用于数据处理和分析
import pandas as pd
# 从 contextlib 模块导入异步上下文管理器，用于管理异步资源
from contextlib import asynccontextmanager
# 从 typing 模块导入一些类型注解，方便代码的类型检查
from typing import List, Literal, Optional, Tuple, Union
# 从 fastapi 库导入 FastAPI 类和 HTTPException 异常类
from fastapi import FastAPI, HTTPException
# 从 fastapi.middleware.cors 模块导入跨域资源共享中间件
from fastapi.middleware.cors import CORSMiddleware
# 从 pydantic 库导入 BaseModel 类和 Field 类，用于数据验证和序列化
from pydantic import BaseModel, Field
# 从 sse_starlette.sse 模块导入 EventSourceResponse 类，用于服务器发送事件
from sse_starlette.sse import EventSourceResponse
# 从 sentence_transformers 库导入 SentenceTransformer 类，用于生成句子嵌入向量
from sentence_transformers import SentenceTransformer
# 从 sklearn.cluster 模块导入 AgglomerativeClustering 类，用于层次聚类
from sklearn.cluster import AgglomerativeClustering
# 再次导入系统模块（重复导入，可删除）
import sys
# 导入操作系统模块，提供了与操作系统交互的功能
import os
# 将当前文件所在目录的上一级目录添加到 Python 模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 注释掉的导入语句，从 src 包导入 constant 模块
# from src import constant
# 导入 constant 模块，可能包含一些常量配置
import constant
import pickle
from constant import raw_docs_path, clean_docs_path, split_docs_path,clean_docs_path

def create_chat_completion(sentences, group_size=5) -> List:
    """将句子按语义相似性分组

    Args:
        sentences: 待分组的句子列表
        group_size: 每组的目标最大句子数（实际可能略多）

    Returns:
        合并后的分组文本列表

    Raises:
        HTTPException: 当输入参数不合法时
    """

    # 声明 embedding_model 为全局变量
    global embedding_model

    # 参数校验：检查 group_size 是否小于 1，如果是则抛出 400 错误
    # if request.group_size < 1:
    #     raise HTTPException(status_code=400, detail="Invalid request")

    # 当输入句子的长度小于等于最小文档大小时，直接返回原句子  sentences就是page_content
    if len(sentences) <= 256:
        return sentences

    # 考虑文档标题，使用正则表达式按 '###' 分割句子
    split_docs = re.split(f'(###)', sentences) 
    # 过滤掉分割结果中的空字符串
    split_docs = [k for k in split_docs if k.strip()]
    # 如果分割结果的第一个元素是 '###'，则将相邻的两个元素合并
    if split_docs[0] == "###":
        split_docs = [''.join(split_docs[i:i+2]) for i in range(0, len(split_docs), 2)]
    else:
        # 否则，第一个元素单独保留，后面相邻的两个元素合并
        split_docs = [split_docs[0]] + [''.join(split_docs[i:i+2]) for i in range(1, len(split_docs), 2)]

    # 如果分割后的文档数量大于 1，直接返回分割后的文档列表
    if len(split_docs) > 1:
        return split_docs

    # 按两个换行符分割句子
    split_docs = sentences.split("\n\n")

    # 如果分割后的文档数量小于等于 group_size，直接返回分割后的文档列表
    if len(split_docs) <= group_size:
        return split_docs

    # 计算合理的聚类数量，向上取整
    n_clusters = max(1, math.ceil(len(split_docs) / group_size))

    # 生成句子的嵌入向量，使用预训练模型进行编码，已自动使用 GPU 加速  (512,)
    embeddings = embedding_model.encode(split_docs)

    # 使用余弦相似度的层次聚类算法
    clustering = AgglomerativeClustering(
        # 指定聚类的数量
        n_clusters=n_clusters,
        # 使用余弦距离作为度量标准
        metric="cosine",  
        # 使用平均链接算法
        linkage="average",  
        # 自动计算完整的聚类树
        compute_full_tree="auto"
    )

    try:
        # 对嵌入向量进行聚类，得到每个句子的聚类标签
        labels = clustering.fit_predict(embeddings)
    except Exception as e:
        # 如果聚类过程中出现异常，抛出 400 错误
        raise HTTPException(status_code=400, detail=f"Clustering failed: {str(e)}")

    # 将分割后的句子和对应的聚类标签存储到 Pandas 的 DataFrame 中
    df = pd.DataFrame({"sentence": split_docs, "label": labels})

    # 按聚类标签分组，将同一组的句子合并成一个字符串
    result = (df.groupby("label", sort=True)['sentence']
              .agg(lambda x: " ".join(x))
              .to_dict())

    # 合并误切分的小文档块
    docs = list(result.values())
    merged_docs = []
    index = 0
    while index < len(docs):
        # 获取当前文档
        cur_doc = docs[index]
        plus = 1
        # 遍历后续文档
        for sub_idx in range(index+1, len(docs)):
             # 如果后续文档的长度小于最小块大小，则合并到当前文档中
             if len(docs[sub_idx]) < 10:
                 cur_doc += docs[sub_idx]
                 plus += 1
             else:
                 break
        index += plus
        # 将合并后的文档添加到结果列表中
        merged_docs.append(cur_doc)

    # 返回合并后的文档列表
    return merged_docs

if __name__ == '__main__':
    from tqdm import tqdm
    clean_docs = pickle.load(open(clean_docs_path, "rb"))
    embedding_model = SentenceTransformer(constant.m3e_small_model_path)
    for doc in tqdm(clean_docs):
        # 调用 create_chat_completion 函数进行语义切分
        result = create_chat_completion(doc.page_content, group_size=5)


