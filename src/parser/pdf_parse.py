# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------

# 导入正则表达式模块，用于文本的模式匹配和分割
import re
# 导入 PyMuPDF 库，用于处理 PDF 文件
import fitz
# 导入 JSON 模块，用于处理 JSON 数据的编码和解码
import json
# 导入复制模块，用于对象的深拷贝
import copy
# 导入哈希模块，用于生成唯一的哈希值
import hashlib
# 导入 tiktoken 库，用于计算文本的 token 数量
import tiktoken
# 导入 tqdm 库，用于显示进度条
from tqdm import tqdm
# 从 langchain_core 模块导入 Document 类，用于表示文档对象
from langchain_core.documents import Document
# 从 langchain_text_splitters 模块导入递归字符文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 从 pymongo 库的 collection 模块导入 Collection 类，用于操作 MongoDB 集合
from pymongo.collection import Collection
# 从 typing_extensions 模块导入 List 类型注解
from typing_extensions import List
# 导入系统模块，用于与 Python 解释器进行交互
import sys
# 导入操作系统模块，用于与操作系统进行交互
import os
# 将当前文件所在目录的上一级目录添加到 Python 模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入自定义的常量模块
import constant
# 从 fields 包的 manual_images 模块导入 ManualImages 类
from fields.manual_images import ManualImages
# 从 fields 包的 manual_info_mongo 模块导入 ManualInfo 类
from fields.manual_info_mongo import ManualInfo
# 从 client 包的 mongodb_config 模块导入 MongoConfig 类
from client.mongodb_config import MongoConfig
# 导入自定义的图片处理模块
import parser.image_handler as image_handler
# 从 client 包的 semantic_chunk_client 模块导入请求语义切分的函数
from client.semantic_chunk_client import request_semantic_chunk

# 全局配置
# 定义文本分割的块大小
_chunk_size = 256
# 定义文本分割的块重叠大小
_chunk_overlap = 50
# 定义过滤 PDF 页面的最小页码
_min_filter_pages = 4
# 定义过滤 PDF 页面的最大页码
_max_filter_pages = 247
# 定义语义分组的大小
_semantic_group_size = 10
# 定义父文档的最大长度
_max_parent_size = 512
# 定义裁剪页面底部的像素数
_page_clip = 50
# 获取 tiktoken 的 cl100k_base 编码
encoding = tiktoken.get_encoding("cl100k_base")
# 从 MongoDB 配置类中获取 manual_text 集合
manual_text_collection: Collection = MongoConfig.get_collection("manual_text")
# 获取 PDF 文件的路径
file_path = constant.pdf_path

# ===== TextSplitter 设置 =====
# 创建递归字符文本分割器实例
text_splitter = RecursiveCharacterTextSplitter(
    # 设置块大小
    chunk_size=_chunk_size,
    # 设置块重叠大小
    chunk_overlap=_chunk_overlap,
    # 按这个优先级递归切分文本
    separators=["\n\n", "\n"],
    # 定义计算文本长度的函数，使用 tiktoken 编码后的 token 数量
    length_function=lambda text: len(encoding.encode(text))
)

# ===== 文本预处理部分 =====
def sentence_split(text: str) -> list[str]:
    """
    按中文/英文标点切句

    Args:
        text (str): 输入的文本

    Returns:
        list[str]: 切分后的句子列表
    """
    # 使用正则表达式按中文句号、换行符和制表符进行分割
    sentences = re.split(r'(?<=[。\n\t])+', text.strip())
    # 过滤掉空字符串，并去除每个句子的前后空格
    return [s.strip() for s in sentences if s.strip()]

def load_pdf() -> list[Document]:
    """
    加载 PDF 文件，提取文本和图片信息，并转换为 Document 对象列表

    Returns:
        list[Document]: 包含 PDF 文本和元数据的 Document 对象列表
    """
    # 打开 PDF 文件
    pdf = fitz.open(file_path)
    # 初始化原始文档列表
    raw_docs = []

    # 遍历 PDF 的每一页，并显示进度条
    for idx, page_num in enumerate(tqdm(range(len(pdf)))):
        # 过滤封面和目录
        if idx < _min_filter_pages or idx > _max_filter_pages:
            continue
        # 加载当前页
        page = pdf.load_page(page_num)
        # 定义裁剪区域，裁剪页面底部 _page_clip 像素
        crop = fitz.Rect(0, 0, page.rect.width, page.rect.height-_page_clip)
        # 提取裁剪区域内的文本
        text = page.get_text(clip=crop)
        # 获取当前页的所有图片信息
        images = page.get_images(full=True)

        # 初始化手动图片信息列表
        manual_images_list: List[ManualImages] = []
        # 遍历当前页的所有图片
        for img_index, img in enumerate(images):
            # 调用图片处理函数处理图片
            manual_image: ManualImages = image_handler.handle_image(img, img_index, page)
            if manual_image:
                # 将图片信息转换为 JSON 格式并添加到列表中
                manual_images_list.append(json.loads(manual_image.json()))

        if text.strip():
            # 生成文本的唯一哈希值
            unique_id = hashlib.md5(text.encode('utf-8')).hexdigest()
            # 定义元数据字典
            metadata = {
                "unique_id": unique_id,
                "source": file_path,
                "page": page_num + 1,
                "images_info": manual_images_list
            }

            # 创建 Document 对象并添加到原始文档列表中
            raw_docs.append(Document(page_content=text, metadata=metadata))

    return raw_docs

def texts_split(raw_docs: list[Document]) -> list[Document]:
    """
    句子级 + 语义感知切分

    Args:
        raw_docs (list[Document]): 原始的 Document 对象列表

    Returns:
        list[Document]: 切分后的 Document 对象列表
    """
    # 初始化所有切分后的文档列表
    all_split_docs = []

    # 遍历原始文档列表，并显示进度条
    for doc in tqdm(raw_docs):

        # 语义切分
        grouped_chunks = request_semantic_chunk(doc.page_content, group_size=_semantic_group_size)

        # 父doc
        # 初始化父文档列表
        parent_docs = []
        for group in grouped_chunks:
            # 生成父文档的唯一哈希值
            parent_id = hashlib.md5(group.encode('utf-8')).hexdigest()
            # 深拷贝原始文档的元数据
            parent_metadata = copy.deepcopy(doc.metadata)
            # 更新元数据中的唯一 ID
            parent_metadata["unique_id"] = parent_id 
            # 创建父文档对象
            parent_doc = Document(page_content=group, metadata=parent_metadata)
            # 将父文档添加到父文档列表中
            parent_docs.append(parent_doc)
            if len(group) < _max_parent_size:
                # 如果父文档长度小于最大长度，添加到所有切分后的文档列表中
                all_split_docs.append(parent_doc)
        # 将父文档保存到 MongoDB 中
        save_2_mongo(parent_docs)

        # 子doc  chunk是携带了metadata的Document对象 group只是文本而已
        for chunk in parent_docs:
            # 带overlap继续句子级切分
            # 使用文本分割器对父文档进行切分 langchain的RecursiveCharacterTextSplitter
            split_docs = text_splitter.create_documents([chunk.page_content], metadatas=[chunk.metadata])
            # 初始化重新编号的子文档列表
            reid_split_docs = []
            for child_doc in split_docs:
                # 生成子文档的唯一哈希值
                child_id = hashlib.md5(child_doc.page_content.encode('utf-8')).hexdigest()
                if child_doc.page_content == chunk.page_content:
                    continue
                # 深拷贝父文档的元数据
                child_metadata = copy.deepcopy(chunk.metadata)
                # 更新元数据中的唯一 ID
                child_metadata["unique_id"] = child_id
                # 添加父文档的唯一 ID 到元数据中
                child_metadata["parent_id"] = chunk.metadata["unique_id"]
                # 创建重新编号的子文档对象
                reid_child_doc = Document(page_content=child_doc.page_content, metadata=child_metadata)
                # 将重新编号的子文档添加到列表中
                reid_split_docs.append(reid_child_doc)

            # 将重新编号的子文档保存到 MongoDB 中
            save_2_mongo(reid_split_docs)
            # 将重新编号的子文档添加到所有切分后的文档列表中
            all_split_docs.extend(reid_split_docs)

    return all_split_docs

def save_2_mongo(split_docs):
    """
    将切分后的文档保存到 MongoDB 中

    Args:
        split_docs: 切分后的文档列表
    """
    for doc in split_docs:
        # 从 metadata 中提取关键参数
        metadata = doc.metadata

        # 构造唯一性 unique_id
        unique_id = metadata.get("unique_id")
        if not unique_id:
            continue

        # 创建文档记录对象
        doc_record = ManualInfo(
            unique_id=unique_id,
            page_content=doc.page_content,
            metadata=metadata
        )

        # 更新数据库操作
        manual_text_collection.update_one(
            {"unique_id": doc_record.unique_id},
            {"$set": doc_record.model_dump()},
            upsert=True
        )


