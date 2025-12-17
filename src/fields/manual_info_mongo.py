# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------

from typing import Optional

from pydantic import BaseModel, Field
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.manual_images import ManualImages


class ManualInfo(BaseModel):
    unique_id: str = Field(description="唯一标识符")
    metadata: dict = Field(description="存储文档的meta信息")
    page_content: Optional[str] = Field(description="文档分片的内容")
