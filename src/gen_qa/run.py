# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------

# 导入操作系统相关模块，用于环境变量操作、文件路径处理等
import os
# 导入正则表达式模块，用于字符串匹配和替换
import re
# 导入pickle模块，用于对象的序列化和反序列化
import pickle
# 导入time模块，用于处理时间相关操作，如休眠
import time
# 导入random模块，用于生成随机数
import random
# 导入threading模块，用于多线程编程
import threading
# 导入json模块，用于处理JSON数据的编码和解码
import json
# 导入hashlib模块，用于生成哈希值
import hashlib

# 导入并发执行器模块，用于线程池和进程池管理
import concurrent.futures
# 从tqdm库中导入自动进度条，用于显示任务进度
from tqdm.auto import tqdm
# 从openai库中导入OpenAI客户端类
from openai import OpenAI
# 从langchain_core.documents模块中导入Document类，用于表示文档
from langchain_core.documents import Document

# 设置随机数种子，确保随机结果可复现
random.seed(42)

# 定义最小文本块大小，用于过滤较短的文本块
MINMAL_CHUNK_SIZE = 100
# 定义线程池的最大工作线程数
MAX_WORKERS = 20
# 定义输入文件路径，包含处理后的文档
INPUT_PATH = './data/processed_docs/clean_docs.pkl'
# 定义QA对文件路径，用于存储生成的QA对
QA_PATH = "./data/qa_pairs/qa_pair.json"
# 定义通用聊天记录文件路径，用于生成负样本
CHATS_PATH = "./data/ut/raw_general_chats.txt"
# 定义扩展QA对文件路径，用于存储泛化后的问题
OUTPUT_PATH = "./data/qa_pairs/expand_qa_pair.json"
# 定义训练集QA对文件路径
TRAIN_PATH = "./data/qa_pairs/train_qa_pair.json"
# 定义测试集QA对文件路径
TEST_PATH = "./data/qa_pairs/test_qa_pair.json"
# 定义测试集关键词文件路径，用于存储抽取的关键词
TEST_KEYWORDS_PATH = "./data/qa_pairs/test_keywords_pair.json"

# 抽取QA的模板，用于生成问题和答案
CONTEXT_PROMPT_TPL = """
我会给你一段文本（<document></document>之间的部分），你需要阅读这段文本，分别针对这段文本生成5个问题，和基于这段文本对问题的回答，回答请保持完整，无须重复问题。

对问题、答案的要求：
1.问题：问题要与这段文本相关，不要询问类似“这个问题的答案在哪一章”这样的问题;
2.答案：回答请保持完整且简洁，无须重复问题。答案要能够独立回答问题，而不是引用其他章节和页码，例如答案内容不能出现请参阅xx页码;
3.5个问题里面至少要包含一个需要综合*大段*文本才能回答的问题，但不要问类似“这一段主要讲了什么内容”这样的问题;

对输出的要求：
1.返回结果以JSON形式组织，格式为[{"question": "...", "answer": "..."}, ...]。
2.如果当前文本主要是目录，或者是一些人名、地址、电子邮箱等没有办法生成有意义的问题时，可以返回[]。

下方是文本：
<document>
{{document}}
</document>

请生成结果：
"""
# 泛化问题的模板，用于根据已有问题生成同义问题
GENERALIZE_PROMPT_TPL = """
你是一个造句大师，请根据我输入的问题，生成具有意思相近的5个问题。

要求：
1.生成的问题要表达相近的意思，请探索采用不同的问法。
2.生成问题尽量口语化一点，可以不遵循原来的句式，例如：怎么打开车窗=》这个车的窗子要怎么才能开启
3.每一个问题用回车符连接，前面用序号开头，例如：1., 2., 3.

注意：不是回答问题，任务是输出5个同义句。

下方是输入的问题：
<question>
{{document}}
</question>

请生成结果：
"""
# 抽取关键词的模板，用于从汽车行业文本中提取核心关键词
KEYWORDS_PROMPT_TPL = """
你是一名专业的汽车领域NLP工程师，任务是从给定的汽车行业文本中提取核心关键词。请按以下要求操作：

抽取原则：
1.优先提取汽车专业术语（如"涡轮增压"、"ADAS系统"）
2.保留产品型号/规格（如"MQB平台"、"2023款Model Y"）
3.包含技术特征（如"L2级自动驾驶"、"48V轻混"）
4.提取关键动作（如"召回"、"OTA升级"）

输出要求：
1.重点关注：动力总成/车身结构/汽车零部件/智能网联/辅助驾驶/新能源技术/充电设施/售后服务
2.过滤通用词汇（如"使用"、"包括"）
3.请输出最重要的关键词，关键词数量不得超过5个
4.如果没有关键词，请直接输出“无”

输出格式：
关键词列表，用逗号分隔
例如：行车记录仪,探测功能,辅助驾驶,车辆功率

下方是输入的问题：
<question>
{{document}}
</question>

请生成结果：
"""
# 质量评估提示词模板，用于对QA对进行打分
QA_QUALITY_PROMPT_TPL = """
你是一个汽车领域的专家，现在有人根据一份汽车用车手册，构造了一些问题，并对问题进行了回答。
你的任务是对这些问题（<question></question>之间的部分）和回答（<answer></answer>）进行打分。

结果请以JSON形式组织，格式如下（<result></result>之间的部分）：
<result>
{"score": ..., "reason": ...}
</result>
其中score是对问题-回答的打分，分值是一个int类型的值，取值范围为1-5。reason是打分的理由。

好的问题，应该是询问事实、观点等，不好的问题，通常要求做一些文本摘要等初级文字处理工作，类似于“这一段描述了什么”，“文本描述了什么”；或者询问的内容是图相关的，例如“图4展示了什么数据？”。
好的答案，应该能够回应问题，而不是回答无关的内容，不好的回答，会给出在原文中的引用，例如“第3章”等。

问题：
<question>
{{question}}
</question>

答案：
<answer>
{{answer}}
</answer>

请进返回JSON格式的数据即可，不要添加其他任何描述性信息。
"""

# 定义OpenAI API的密钥
api_key = "sk-zk2a9f16937452e24e032eab82729dd155668247d5b7b511"
# 定义OpenAI API的基础URL
base_url = "https://api.zhizengzeng.com/v1"
# 初始化OpenAI客户端
llm_client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# 构建QA提示词的函数，将模板中的占位符替换为实际文本
def build_qa_prompt(prompt_tmpl, text):
    prompt = prompt_tmpl.replace('{{document}}', text).strip()
    return prompt

# 与LLM进行对话的函数，包含重试机制
def chat(prompt, max_retry=3, debug=False, temperature=0.85, top_p=0.95):
    # 内部函数，用于实际调用LLM进行对话
    def do_chat(prompt):
        completion = llm_client.chat.completions.create(
            model='qwen2.5-14b-instruct',
            messages=[
                {"role": "system", "content": "你是一个有用的人工智能助手."},
                {"role": "user", "content": prompt}
            ],
            top_p=top_p,
            temperature=temperature
        )
        return completion.choices[0].message.content

    x = do_chat(prompt)  # 调用do_chat函数获取对话结果，但该结果未被使用
    while max_retry > 0:
        try:
            return do_chat(prompt)  # 尝试调用LLM进行对话并返回结果
        except Exception as e:
            max_retry -= 1  # 重试次数减1
            sleep_seconds = random.randint(1, 4)  # 随机生成休眠时间
            if debug:
                print(f"{str(e)}, remain retry: {max_retry}, sleeping {sleep_seconds}s {prompt}")  # 打印调试信息
            time.sleep(sleep_seconds)  # 休眠指定时间
    return None  # 重试次数用完仍失败，返回None

# 生成QA对的函数
def gen_qa(splitted_docs, prompt_tmpl, qa_ckpt_filename, expand=False):
    qa_ckpt = {}  # 初始化QA检查点字典
    file_lock = threading.Lock()  # 初始化文件锁，用于线程安全
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        if expand:
            # 扩展模式下，为每个文档提交任务
            futures = {doc.page_content: executor.submit(chat, build_qa_prompt(
                prompt_tmpl, doc.page_content), 3, True) for doc in splitted_docs if
                       doc.metadata['unique_id'] not in qa_ckpt}
        else:
            # 非扩展模式下，过滤较短或者重复的文本块并提交任务
            futures = {doc.metadata['unique_id']: executor.submit(chat, build_qa_prompt(
                prompt_tmpl, doc.page_content), 3, True) for doc in splitted_docs
                       if len(doc.page_content.replace('\n', '')) >= MINMAL_CHUNK_SIZE and
                           doc.metadata['unique_id'] not in qa_ckpt}
        for unique_id in tqdm(futures):  # 遍历所有任务
            future = futures[unique_id]
            result = future.result()  # 获取任务结果
            if result is None:
                continue

            item = {'unique_id': unique_id, 'raw_resp': result}  # 构建QA项
            qa_ckpt[unique_id] = item  # 将QA项添加到检查点字典

            file_lock.acquire()  # 获取文件锁

            try:
                with open(qa_ckpt_filename, 'a') as f:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')  # 将QA项写入文件
            except Exception as e:
                print(e)  # 打印写入文件时的异常信息
            finally:
                file_lock.release()  # 释放文件锁
    return qa_ckpt  # 返回QA检查点字典

if __name__ == "__main__":
    splitted_docs = pickle.load(open(INPUT_PATH, "rb"))  # 加载处理后的文档
    print("待处理文件数：", len(splitted_docs))  # 打印待处理文件数量
    print(splitted_docs[0])  # 打印第一个文档

    qa_dict = gen_qa(splitted_docs, CONTEXT_PROMPT_TPL, QA_PATH)  # 生成QA对
    question_docs = []
    fd = open(QA_PATH, "r")
    idx = 0
    for line in fd:
        info = json.loads(line)
        resp = json.loads(info["raw_resp"])
        for qa in resp:
            question_docs.append(Document(page_content=qa["question"], metadata={"unique_id": str(idx)}))  # 构建问题文档
            idx += 1
    print("待泛化问题数：", len(question_docs))  # 打印待泛化问题数量
    expand_qa_dict = gen_qa(question_docs, GENERALIZE_PROMPT_TPL, OUTPUT_PATH, expand=True)  # 泛化问题

        # 初始化一个空字典，用于存储从文件中加载的QA对
    qa_dict = {}
    # 以只读模式打开存储QA对的文件
    with open(QA_PATH) as fd:
        # 逐行读取文件内容
        for line in fd:
            # 将每行JSON格式的字符串解析为Python字典
            info = json.loads(line)
            # 以unique_id为键，将解析后的信息存入qa_dict
            qa_dict[info["unique_id"]] = info  # 加载QA对

    # 初始化一个空字典，用于存储从文件中加载的泛化后的问题 '```json\n[\n    {\n        "question": "Model 3的手机钥匙可以支持多少部手机或手表？",\n        "answer": "Model 3总共支持19把钥匙，其中包括手机钥匙。这意味着您可以将多部手机或手表设置为手机钥匙。"\n    },\n    {\n        "question": "如何在没有互联网连接的情况下使用手机钥匙？",\n        "answer": "手机认证成功后，无需互联网连接即可用作Model 3的手机钥匙，但是要免提使用手机、访问联系人、播放媒体等功能，需将手机配对并连接为蓝牙设备。"\n    },\n    {\n        "question": "如果在Model 3内将已配对的手机留在车内，有什么风险？",\n        "answer": "如果在蓝牙启用时将已配对手机留在车内，基本相当于将钥匙留在解锁的车内，即使在手机应用程序上按锁定图标，拉动车门外把手也会解锁车门。因此，这可能会导致车门被他人打开。"\n    },\n    {\n        "question": "卡片钥匙如何使用来解锁和锁定Model 3？",\n        "answer": "将卡片钥匙放置在驾驶侧车门柱上方约三分之一处的读卡器上可解锁或锁定Model 3，检测到卡片钥匙后，外部车灯闪烁，后视镜展开或折叠（若启用“折叠后视镜”），喇叭鸣响（若启用“锁定提示音”）。\\n可能需要实际接触无线手机充电器或驾驶侧门柱，且可能需靠住收发器1到2秒。"\n    },\n    {\n        "question": "手机钥匙和卡片钥匙有何区别？",\n        "answer": "手机钥匙支持自动锁定和解锁，并且可以使用蓝牙技术与车辆进行通信。而卡片钥匙则通过短距离射频识别 (RFID) 信号与Model 3进行通信，不支持自动锁定和解锁，主要用于替代手机钥匙在手机电量耗尽、丢失或被盗等情况下的操作。"\n    }\n]\n```'
    expand_qa_dict = {}
    # 以只读模式打开存储泛化后问题的文件
    with open(OUTPUT_PATH) as fd:
        # 逐行读取文件内容
        for line in fd:
            # 将每行JSON格式的字符串解析为Python字典
            info = json.loads(line)
            # 以unique_id为键，将解析后的信息存入expand_qa_dict
            expand_qa_dict[info["unique_id"]] = info  # 加载泛化后的问题

    # 初始化一个空字典，用于存储处理后的泛化问题
    expand_qa_pairs = {}
    # 遍历expand_qa_dict中的每个键值对
    for unique_id, info in expand_qa_dict.items():
        # 提取问题
        question = info["unique_id"]
        # 提取泛化后的问题字符串
        expand_questions = info["raw_resp"]
        # 按换行符分割泛化后的问题字符串，得到问题列表
        expand_questions = expand_questions.split("\n")
        # 去除每个问题前的序号（如1., 2.等）并去除首尾空格
        expand_questions = [re.sub(r'^\d[.. ]', '', item).strip() for item in expand_questions]  # 处理泛化后的问题
        # 以原始问题为键，将处理后的泛化问题列表存入expand_qa_pairs
        expand_qa_pairs[question] = expand_questions

    # 初始化一个空列表，用于存储训练集的QA对
    train_qa_pairs = []
    # 初始化一个空列表，用于存储测试集的QA对
    test_qa_pairs = []
    # 遍历qa_dict中的每个键值对
    for unique_id, info in qa_dict.items():
        # 将raw_resp字段的JSON字符串解析为Python列表
        resp = json.loads(info["raw_resp"])
        # 遍历解析后的QA对列表
        for qa in resp:
            # 提取问题并去除首尾空格
            question = qa["question"].strip()
            # 提取答案并去除首尾空格
            answer = qa["answer"].strip()
            # 如果答案中包含“无法准确”或“未提及”，则跳过当前QA对
            if "无法准确" in answer or "未提及" in answer:
                continue
            # 将原始问题和泛化后的问题合并成一个列表
            expand_questions = [question] + expand_qa_pairs[question]  # 合并原始问题和泛化问题
            # 遍历合并后的问题列表
            for query in expand_questions:
                # 为每个问题生成一个唯一的哈希ID
                unique_id = hashlib.md5(query.encode('utf-8')).hexdigest()  # 生成唯一ID
                # 构建一个包含唯一ID、问题和答案的字典
                item = {
                    "unique_id": unique_id,
                    "question": query,
                    "answer": answer 
                }
                # 生成一个0到1之间的随机数，如果小于0.9
                if random.random() < 0.9:
                    # 将当前QA对添加到训练集列表中
                    train_qa_pairs.append(item)  # 添加到训练集
                else:
                    # 将当前QA对添加到测试集列表中
                    test_qa_pairs.append(item)  # 添加到测试集

    # 打印训练集和测试集的QA对数量
    print("训练集QA数：", len(train_qa_pairs), "测试集QA数：", len(test_qa_pairs))  # 打印训练集和测试集的QA数量

    # 初始化一个空列表，用于存储测试集答案文档
    test_answer_docs = []
    # 提取测试集QA对中的所有唯一答案，存储在集合中
    unique_test_answers = set([item["answer"] for item in test_qa_pairs])
    # 遍历唯一答案集合
    for idx, answer in enumerate(unique_test_answers):
        # 为每个答案创建一个Document对象，并添加到测试集答案文档列表中
        test_answer_docs.append(Document(page_content=answer, metadata={"unique_id": str(idx)}))  # 构建测试集答案文档

    # 打印待抽取关键词的文档数量
    print("待抽取关键词docs数：", len(test_answer_docs))  # 打印待抽取关键词的文档数量
    # 调用gen_qa函数，从测试集答案文档中抽取关键词
    keywords_dict = gen_qa(test_answer_docs, KEYWORDS_PROMPT_TPL, TEST_KEYWORDS_PATH, expand=True)  # 抽取关键词

    # 初始化一个空字典，用于存储关键词映射
    keywords_mapping = {}
    # 遍历关键词字典中的每个键值对
    for unique_id, info in keywords_dict.items():
        # 按逗号分割raw_resp字段中的关键词字符串，得到关键词列表
        keywords = info["raw_resp"].split(",")
        # 过滤掉无效关键词（如“无”和“Model 3”）
        kewyords = [item for item in keywords if item not in ["无", "Model 3"]]  # 过滤无效关键词
        # 以unique_id为键，将过滤后的关键词列表存入keywords_mapping
        keywords_mapping[info["unique_id"]] = kewyords 

    # 遍历测试集QA对列表
    for info in test_qa_pairs:
        # 根据答案从关键词映射中获取对应的关键词列表
        keywords = keywords_mapping[info["answer"]]
        # 为测试集QA对添加关键词字段
        info["keywords"] = keywords  # 为测试集QA对添加关键词

    # 读取通用聊天记录文件内容，按行分割成列表
    chats_data = open(CHATS_PATH).readlines()
    # 去除每行聊天记录的首尾空格
    chats_data = [item.strip() for item in chats_data]

    # 设置随机数种子，确保随机结果可复现
    random.seed(42)
    # 遍历聊天记录列表
    for line in chats_data:
        # 生成一个0到1之间的随机数，如果小于0.95
        if random.random() < 0.95:
            # 构建一个负样本字典，并添加到训练集列表中
            train_qa_pairs.append({
                "unique_id": hashlib.md5(line.encode('utf-8')).hexdigest(),
                "question": line, 
                "answer": "无答案"
            })  # 添加负样本到训练集
        else:
            # 构建一个负样本字典，并添加到测试集列表中
            test_qa_pairs.append({
                "unique_id": hashlib.md5(line.encode('utf-8')).hexdigest(),
                "question": line, 
                "answer": "无答案",
                "keywords": []
            })  # 添加负样本到测试集

    # 设置随机数种子，确保随机结果可复现
    random.seed(42)
    # 以写入模式打开训练集文件
    with open(TRAIN_PATH, "w") as fd:
        # 打乱训练集QA对列表的顺序
        random.shuffle(train_qa_pairs)
        # 将训练集QA对列表以JSON格式写入文件，设置不转义ASCII字符并缩进2个空格
        fd.write(json.dumps(train_qa_pairs, ensure_ascii=False, indent=2))
        # 打印训练集文件路径和写入的QA对数量
        print("训练集已写入:", TRAIN_PATH, len(train_qa_pairs))  # 将训练集写入文件

    # 设置随机数种子，确保随机结果可复现
    random.seed(42)
    # 以写入模式打开测试集文件
    with open(TEST_PATH, "w") as fd:
        # 打乱测试集QA对列表的顺序
        random.shuffle(test_qa_pairs)
        # 将测试集QA对列表以JSON格式写入文件，设置不转义ASCII字符并缩进2个空格
        fd.write(json.dumps(test_qa_pairs, ensure_ascii=False, indent=2))
        # 打印测试集文件路径和写入的QA对数量
        print("测试集已写入:", TEST_PATH, len(test_qa_pairs))  # 将测试集写入文件

