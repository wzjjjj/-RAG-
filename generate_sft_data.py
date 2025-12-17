# -*- coding: utf-8 -*-
# 声明文件使用 UTF-8 编码，确保能正确处理中文等非 ASCII 字符

# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  @丁师兄大模型
# --------------------------------------------

# 导入 os 模块，用于与操作系统进行交互，比如文件和目录操作
import os
# 导入 pickle 模块，用于 Python 对象的序列化和反序列化
import pickle
# 导入 time 模块，用于处理时间相关的操作
import time
# 导入 json 模块，用于处理 JSON 数据的编码和解码
import json
# 导入 re 模块，用于正则表达式操作，方便进行字符串匹配和替换
import re
# 导入 random 模块，用于生成随机数
import random
# 从 tqdm 模块导入 tqdm 类，用于在循环中显示进度条
from tqdm import tqdm
# 从 src.retriever.bm25_retriever 模块导入 BM25 类，用于基于 BM25 算法的文档检索
from src.retriever.bm25_retriever import BM25
# 从 src.retriever.milvus_retriever 模块导入 MilvusRetriever 类，用于基于 Milvus 的文档检索
from src.retriever.milvus_retriever import MilvusRetriever 
# 从 src.client.llm_chat_client 模块导入 request_chat 函数，用于与大语言模型进行对话请求
from src.client.llm_chat_client import request_chat
# 注释掉的代码，从 src.reranker.bge_m3_reranker 模块导入 BGEM3ReRanker 类，用于文档重排序
# from src.reranker.bge_m3_reranker import BGEM3ReRanker 
# 从 src.reranker.qwen3_reranker_vllm 模块导入 Qwen3ReRankervLLM 类，用于基于 Qwen3 的文档重排序
from src.reranker.qwen3_reranker_vllm import Qwen3ReRankervLLM 
# 从 src.constant 模块导入 bge_reranker_model_path 变量，可能是 BGE 重排序模型的路径
from src.constant import bge_reranker_model_path
# 从 src.constant 模块导入 qwen3_4b_reranker_model_path 变量，可能是 Qwen3 4B 重排序模型的路径
from src.constant import qwen3_4b_reranker_model_path
# 从 src.utils 模块导入 merge_docs 和 post_processing 函数，分别用于合并文档和后处理响应
from src.utils import merge_docs, post_processing

# 设置随机数种子为 42，保证每次运行代码时随机结果一致
random.seed(42)

# 定义大语言模型的对话提示模板，包含上下文信息和任务描述，以及输出格式要求
LLM_CHAT_PROMPT = """
### 信息
{context}

### 任务
你是特斯拉电动汽车Model 3车型的用户手册问答系统，你具备{{信息}}中的知识。
请回答问题"{query}"，答案需要精准，语句通顺，并严格按照以下格式输出

{{答案}}【{{引用编号1}},{{引用编号2}},...】
如果无法从中得到答案，请说 "无答案" ，不允许在答案中添加编造成分。
"""



# warmstart
# 初始化 BM25 检索器，传入文档为 None，设置为检索模式
bm25_retriever = BM25(docs=None, retrieve=True)
# 初始化 Milvus 检索器，传入文档为 None，设置为检索模式
milvus_retriever = MilvusRetriever(docs=None, retrieve=True) 
# 注释掉的代码，初始化 BGE M3 重排序器，传入模型路径
# bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_model_path)
# 初始化 Qwen3 重排序器，传入模型路径
qwen3_reranker = Qwen3ReRankervLLM(model_path=qwen3_4b_reranker_model_path)
# 使用 Milvus 检索器进行一次测试检索，查询语句为 "这是一条测试数据"，返回前 3 个文档
milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)


# 以只读模式打开训练 QA 对的 JSON 文件
fd = open("data/qa_pairs/train_qa_pair.json")
# 加载 JSON 文件内容到 test_qa_pairs 变量
test_qa_pairs = json.load(fd)
# 以写入模式打开训练数据的 JSON 文件，用于后续写入处理结果
output_handler = open("data/qa_pairs/train_data.json", "w")
# 遍历测试 QA 对列表，使用 tqdm 显示进度条
for item in tqdm(test_qa_pairs):
    #try:
        # 提取问题并去除首尾空格
        query = item["question"].strip()
        # 使用 BM25 检索器检索与问题相关的前 5 个文档
        bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
        # 使用 Milvus 检索器检索与问题相关的前 10 个文档
        milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
        # 合并 BM25 和 Milvus 检索到的文档
        merged_docs = merge_docs(bm25_docs, milvus_docs)
        # 使用 Qwen3 重排序器对合并后的文档进行重排序，返回前 5 个文档
        ranked_docs = qwen3_reranker.rank(query, merged_docs, topk=5)
        # 将重排序后的文档内容按编号拼接成上下文信息
        context = "\n".join([str(idx+1) + "." + doc.page_content for idx, doc in enumerate(ranked_docs)])
        '''
        【1】### 离车后自动上锁
    带着手机钥匙或配对的遥控钥匙离开时，车门和行李箱可以自动锁定（如果订购日期是在大约 2019 年 10 月 1 日之后）。要打开或关闭此功能，可点击控制 > 车锁 > 离车后自动上锁。
    **注**：如果已将 Apple 手表认证为钥匙，也可以将该手表用于离车后自动上锁功能。
    【2】车门锁闭时，外部车灯闪烁一次，后视镜折叠（如果折叠后视镜开启）。要在 Model 3 锁定时听到提示音，可点击控制 > 车锁 > 锁定提示音。
    【3】### 大灯延时照明
    停止驾驶并将 Model 3 停在照明较差的环境中时，外部车灯会短暂亮起。它们会在一分钟后或您锁闭 Model 3 时（以较早者为准）自动关闭。当您使用 Tesla 手机应用程序锁定 Model 3 时，大灯将立即熄灭。但是，如果车辆因启用了“离车后自动上锁”功能而锁定（请参阅离车后自动上锁 页码 7），则大灯将在一分钟后自动熄灭。要打开或关闭此功能，请点击控制 > 车灯 > 大灯延时照明。关闭大灯延时照明后，当换入驻车挡并打开车门时，大灯会立即熄灭。
        '''
        # 调用 request_chat 函数，传入问题和上下文信息，获取大语言模型的响应
        response = request_chat(query, context)
        # 对大语言模型的响应进行后处理
        answer = post_processing(response, ranked_docs)
        # 提取重排序后的文档内容
        context = [q.page_content for q in ranked_docs]
        # 提取合并后的所有文档内容
        all_docs = [q.page_content for q in merged_docs]
        # 构建包含问题、上下文、响应和合并文档的信息字典
        info = {"query": query, "context": context, "response": response, "merged_docs": all_docs}
        # 将信息字典转换为 JSON 字符串，不转义 ASCII 字符
        info = json.dumps(info, ensure_ascii=False)
        # 将 JSON 字符串写入文件并换行
        output_handler.write(info+'\n')
        # 刷新文件缓冲区，确保数据及时写入文件
        output_handler.flush()
    #except:
    #    pass


# 定义输入的最大长度为 4096 个字符
MAX_INPUT_SIZE = 4096
# 定义重排序开发集的大小为 1000 条数据
RERANK_DEV_SIZE = 1000
# 定义测试集的比例为 0.08，即 8%
TEST_RATE = 0.08

# 以写入模式打开摘要训练数据的 JSON 文件
summary_train_handler = open("./data/summary_data/train.json", "w")
# 以写入模式打开摘要测试数据的 JSON 文件
summary_test_handler = open("./data/summary_data/test.json", "w")
# 以写入模式打开重排序训练数据的 JSON 文件
rerank_train_handler = open("./data/rerank_data/train.json", "w")
# 以写入模式打开重排序开发数据的 JSON 文件
rerank_dev_handler = open("./data/rerank_data/dev.json", "w")
# 以写入模式打开重排序测试数据的 JSON 文件
rerank_test_handler = open("./data/rerank_data/test.json", "w")

# 初始化空列表，用于存储摘要训练数据
summary_train = []
# 初始化空列表，用于存储摘要测试数据
summary_test = []
# 初始化空列表，用于存储重排序训练数据
rerank_train = []
# 初始化空列表，用于存储重排序测试数据
rerank_test = []
# 以只读模式打开训练数据的 JSON 文件
fd = open("data/qa_pairs/train_data.json")
# 逐行读取文件内容
for line in fd:
    # 将每行 JSON 字符串解析为 Python 字典
    info = json.loads(line)
    # 提取响应信息
    response = info["response"]
    # 使用正则表达式提取响应中的引用编号
    all_cites = re.findall("[【](.*?)[】]", response)
    # 初始化空列表，用于存储处理后的引用编号
    cites = []
    # 遍历所有引用编号
    for cite in all_cites:
        # 去除引用编号中的特殊字符
        cite = re.sub("[{} 【】]", "", cite)
        # 将逗号替换为中文逗号
        cite = cite.replace(",", "，")
        # 将引用编号按中文逗号分割并转换为整数列表
        cite = [int(k) for k in cite.split("，") if k.isdigit()]
        # 将处理后的引用编号添加到 cites 列表中
        cites.extend(cite)
    # 对引用编号进行去重和排序
    cites = sorted(list(set(cites)))
    # 将引用编号用逗号连接成字符串
    cites = ",".join([str(c) for c in cites])
    # 去除响应中的引用编号部分
    answer = re.sub("[【](.*?)[】]", "", response)
    # 去除答案中的特殊字符
    answer = re.sub("[{}【】]", "", answer)
    # 如果有引用编号，将答案和引用编号按格式拼接
    if cites:
        format_answer = answer + f"【{cites}】"
    else:
        # 没有引用编号，答案设置为 "无答案"
        format_answer = "无答案"
    # 将上下文信息按编号拼接
    context = "\n".join([str(idx+1) + "." + doc for idx, doc in enumerate(info["context"])])
    # 如果上下文长度超过最大输入长度，进行截断
    if len(context) > MAX_INPUT_SIZE:
        context = context[:MAX_INPUT_SIZE]
    # 提取问题并去除首尾空格
    query = info["query"].strip()
    # 根据提示模板和问题、上下文信息生成指令
    instruction = LLM_CHAT_PROMPT.format(query=query, context=context)
    # 构建包含问题、上下文、指令、输入和输出的字典
    item = {
        "query": query,
        "context": context,
        "instruction": instruction,
        "input": "",
        "output": format_answer
    }
    # 提取不在上下文信息中的文档作为负样本
    neg_docs = [doc for doc in info["merged_docs"] if doc not in info["context"]]
    # 生成一个 0 到 1 之间的随机数，如果小于测试集比例
    if random.random() < TEST_RATE:
        # 将当前数据项添加到摘要测试数据列表中
        summary_test.append(item)
        # 如果答案不是 "无答案"
        if format_answer != "无答案":
            # 选取上下文信息中的第一个和最后两个中的一个文档
            content_list = [info["context"][0], random.choice(info["context"][-2:])]
            # 如果有负样本，随机选取一个添加到列表中
            if neg_docs:
                content_list.append(random.choice(neg_docs))
            # 将包含问题和文档列表的字典添加到重排序测试数据列表中
            rerank_test.append({"query": query, "content": content_list})

    else:
        # 将当前数据项添加到摘要训练数据列表中
        summary_train.append(item)

        # 重排序数据处理
        if format_answer != "无答案":
            # 选取上下文信息中的第一个文档作为正样本
            positive = info["context"][0]
            # 从上下文信息的最后两个文档中随机选取一个
            middle = random.choice(info["context"][-2:])
            # 将正样本添加到重排序训练数据列表中，标签为 2
            rerank_train.append({"query": query, "content": positive, "label": 2})
            # 将中间样本添加到重排序训练数据列表中，标签为 1
            rerank_train.append({"query": query, "content": middle, "label": 1})
            # 如果有负样本
            if neg_docs:
                # 随机选取一个负样本
                negative = random.choice(neg_docs)
                # 将负样本添加到重排序训练数据列表中，标签为 0
                rerank_train.append({"query": query, "content": negative, "label": 0})
        else:
            # 答案为 "无答案" 时，随机选取一个合并文档作为负样本
            negative = random.choice(info["merged_docs"])
            # 将负样本添加到重排序训练数据列表中，标签为 0
            rerank_train.append({"query": query, "content": negative, "label": 0})

# 过滤掉重排序训练数据中问题或文档内容为空的数据
rerank_train = [item for item in rerank_train if len(item["query"]) > 0 and len(item["content"]) > 0]
# 从重排序训练数据中截取最后 RERANK_DEV_SIZE 条数据作为重排序开发集
rerank_dev = rerank_train[-RERANK_DEV_SIZE:]
# 打乱重排序训练数据的顺序
random.shuffle(rerank_train)
# 打印重排序训练集和测试集的大小
print("Rerank Train size:", len(rerank_train), "Rerank Test size:", len(rerank_test))

# 遍历重排序训练数据列表，将每个数据项转换为 JSON 字符串并写入文件
for item in rerank_train:
    rerank_train_handler.write(json.dumps(item, ensure_ascii=False) + "\n")
# 遍历重排序开发数据列表，将每个数据项转换为 JSON 字符串并写入文件
for item in rerank_dev:
    rerank_dev_handler.write(json.dumps(item, ensure_ascii=False) + "\n")
# 遍历重排序测试数据列表，将每个数据项转换为 JSON 字符串并写入文件
for item in rerank_test:
    rerank_test_handler.write(json.dumps(item, ensure_ascii=False) + "\n")

# 打印摘要训练集和测试集的大小
print("Summary Train size:", len(summary_train), "Summary Test size:", len(summary_test))
# 将摘要训练数据列表转换为 JSON 字符串，不转义 ASCII 字符，缩进 4 个空格，写入文件
summary_train_handler.write(json.dumps(summary_train, ensure_ascii=False, indent=4))
# 将摘要测试数据列表转换为 JSON 字符串，不转义 ASCII 字符，缩进 4 个空格，写入文件
summary_test_handler.write(json.dumps(summary_test, ensure_ascii=False, indent=4))
