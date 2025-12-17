# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------

# 导入操作系统相关模块
import os
# 导入 PyMuPDF 库，用于处理 PDF 文件
import fitz
# 导入元组类型注解
from typing import Tuple
# 导入 MongoDB 集合类型
from pymongo.collection import Collection
# 导入列表类型注解
from typing_extensions import List
# 导入系统模块
import sys
import os
# 将项目路径添加到系统路径中，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入自定义常量模块
import constant
# 导入手动图片信息模型
from fields.manual_images import ManualImages
# 导入 MongoDB 配置模块
from client.mongodb_config import MongoConfig

# 全局配置
# 获取 MongoDB 中 manual_images 集合
manual_images_collection: Collection = MongoConfig.get_collection("manual_images")
# 图片保存目录
image_save_dir = constant.image_save_dir
# PDF 文件路径
pdf_path = constant.pdf_path

# 标题判断配置
TITLE_PROPERTIES = {
    "min_size": 10,  # 标题最小字体大小
    "max_lines": 3,  # 标题最大行数
    "max_length": 30,  # 标题最大长度
    "bold_weight": 0.7,  # 粗体权重
    "page_clip": 50,  # 页面底部裁剪高度
    "bottom_size": -200  # 图片区域底部扩展大小
}

def handle_image(img: Tuple, img_index: int, page: fitz.Page) -> ManualImages | None:
    """处理单个图片"""
    # 获取图片的 XREF 编号
    xref = img[0]
    # 从 PDF 页面中提取图片信息
    base_image = page.parent.extract_image(xref)

    # 跳过小图标
    if base_image["ext"] == "png" or base_image["width"] <= 34:
        return None

    # 保存图片并获取路径 
    image_path = save_image(base_image, img_index, page.number)

    # 获取扩展后的图片区域
    img_rect = page.get_image_bbox(img)
    expanded_rect = get_expanded_rect(img_rect, page.rect)

    # 获取关联文本块
    related_blocks = get_related_text_blocks(page, expanded_rect, img_rect.y0)
    title_blocks = [text for is_title, text in related_blocks if is_title]

    return ManualImages(
        image_path=image_path,
        page=page.number + 1,
        title="\n".join(title_blocks)
    )

def save_image(base_image: dict, img_index: int, page_number: int) -> str:
    """保存图片到本地并返回路径"""
    # 生成图片文件名
    image_name = f"page{page_number + 1}_img{img_index + 1}.{base_image['ext']}"
    # 拼接图片保存路径
    image_path = os.path.join(image_save_dir, image_name)
    # 以二进制写入模式打开文件并写入图片数据
    with open(image_path, "wb") as f:
        f.write(base_image["image"])
    return image_path

def get_expanded_rect(img_rect: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
    """获取扩展后的搜索区域"""
    # 原始图片区域进行扩展
    expanded = img_rect + (0, TITLE_PROPERTIES["bottom_size"], 0, img_rect.height * 3)
    # 限制扩展区域的底部边界，避免超出页面底部减去裁剪高度后的范围
    expanded[3] = min(expanded[3], page_rect[3] - TITLE_PROPERTIES["page_clip"])
    # 返回扩展区域与页面区域的交集，确保扩展区域不超出页面范围
    return expanded.intersect(page_rect)

def get_related_text_blocks(page: fitz.Page, rect: fitz.Rect, img_y: float) -> List[Tuple[bool, str]]:
    """获取与图片相关的文本块"""
    related_blocks = []
    # 遍历页面中的所有文本块
    for block in page.get_text("blocks"):
        # 获取文本块的矩形区域
        block_rect = fitz.Rect(block[:4])
        # 如果文本块与扩展区域不相交，则跳过
        if not block_rect.intersects(rect):
            continue

        # 获取文本块的文本内容并去除首尾空格
        block_text = block[4].strip()
        # 判断文本块是否在图片上方
        above = block_rect.y1 < img_y
        # 判断文本块是否为标题候选块
        is_title_block = is_title_block_candidate(page, block, above)
        # 将判断结果和文本块内容添加到列表中
        related_blocks.append((is_title_block, block_text))

    return related_blocks

def is_title_block_candidate(page: fitz.Page, block: tuple, above: bool) -> bool:
    """判断是否为标题候选块"""
    # 如果文本块类型不为 0 或者文本内容为空，则不是标题候选块
    if block[6] != 0 or not block[4].strip():
        return False

    try:
        # 获取文本块中第一行第一个跨度的信息
        span = page.get_text("dict")["blocks"][block[5]]["lines"][0]["spans"][0]
    except (IndexError, KeyError):
        # 出现异常则不是标题候选块
        return False

    # 获取文本块的文本内容并去除首尾空格
    text = block[4].strip()
    # 获取文本块的字体大小
    font_size = span["size"]
    # 判断文本块是否为粗体
    is_bold = "bold" in span["font"].lower()

    # 排除带句尾标点的文本
    if text.endswith(('.', '。', '!', '！')):
        return False

    # 计分规则
    score = 0
    # 字体大小满足要求则加分
    score += 2 if font_size >= TITLE_PROPERTIES["min_size"] else 0
    # 粗体则加分
    score += 1 if is_bold else 0
    # 行数满足要求则加分
    score += 0.5 if (text.count('\n') + 1) <= TITLE_PROPERTIES["max_lines"] else 0
    # 长度满足要求则加分
    score += 0.5 if len(text) <= TITLE_PROPERTIES["max_length"] else 0
    # 在图片上方则加分，否则减分
    score += 2 if above else -1

    # 总分大于等于 3 则为标题候选块
    return score >= 3
