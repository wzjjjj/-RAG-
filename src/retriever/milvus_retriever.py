# -*- coding: utf-8 -*-
# 这行代码指定了文件的编码格式为 UTF-8，确保文件中可以正确处理中文等特殊字符。
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------
# 以上是项目的基本信息，包括项目名称和版权声明。

# 导入 time 模块，该模块提供了与时间相关的函数，比如获取当前时间、暂停程序运行等。
import time
# 导入 hashlib 模块，用于实现各种哈希算法，比如 MD5、SHA 等，常用来生成唯一的标识符。
import hashlib
# 导入 pandas 模块，通常用别名 pd 表示。pandas 是一个强大的数据处理和分析库，不过在当前代码里未实际使用。
#import pandas as pd
# 从 pymilvus 库中导入多个类和函数，pymilvus 是 Milvus 向量数据库的 Python SDK。
from pymilvus import (
    # connections 用于建立与 Milvus 数据库的连接。
    connections,
    # utility 提供了一些实用功能，比如检查集合是否存在等。
    utility,
    # FieldSchema 用于定义 Milvus 集合中字段的结构。
    FieldSchema,
    # CollectionSchema 用于定义 Milvus 集合的整体结构，由多个 FieldSchema 组成。
    CollectionSchema,
    # DataType 定义了 Milvus 中支持的数据类型。
    DataType,
    # Collection 表示 Milvus 中的一个集合，可对集合进行各种操作，如插入数据、搜索等。
    Collection,
    # AnnSearchRequest 用于创建一个向量搜索请求。
    AnnSearchRequest,
    # RRFRanker 是一个重排器，用于对搜索结果进行重新排序。
    RRFRanker,
    # WeightedRanker 也是一个重排器，可根据权重对不同类型的搜索结果进行加权排序，但在代码中未实际使用。
    WeightedRanker
)
# 从 langchain_core.documents 模块导入 Document 类，Document 用于表示文档对象，包含文本内容和元数据。
from langchain_core.documents import Document
# 从 pymilvus.model.hybrid 模块导入 BGEM3EmbeddingFunction 类，该类用于将文本转换为向量（嵌入）。
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
# 从 pymilvus.model.reranker 模块导入 BGERerankFunction 类，用于对搜索结果进行重新排序，但在代码中未实际使用。
from pymilvus.model.reranker import BGERerankFunction
# 导入 sys 模块，该模块提供了一些与 Python 解释器和系统环境交互的功能。
import sys
# 导入操作系统模块，用于与操作系统进行交互，比如获取文件路径、创建目录等。
import os
# 将当前文件所在目录的上一级目录添加到 Python 模块搜索路径中，这样 Python 就能找到上一级目录里的模块。
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从 fields 包下的 manual_images 模块导入 ManualImages 类，但在代码中未实际使用。
from fields.manual_images import ManualImages
# 从 constant 模块导入测试文档路径、BGE M3 模型路径和 Milvus 数据库路径。
from constant import test_doc_path, bge_m3_model_path, milvus_db_path
# 从 client 包下的 mongodb_config 模块导入 MongoConfig 类，用于配置和连接 MongoDB 数据库。
from client.mongodb_config import MongoConfig

# 定义批量插入向量时每次插入的文档数量为 50。
EMB_BATCH = 50
# 定义存储在 Milvus 中文本字段的最大长度为 512。
MAX_TEXT_LENGTH = 512 
# 定义存储在 Milvus 中唯一 ID 字段的最大长度为 100。
ID_MAX_LENGTH = 100
# 定义 Milvus 集合的名称为 "hybrid_bge_m3"。
COL_NAME = "hybrid_bge_m3" 

# 调用 MongoConfig 类的 get_collection 方法，获取名为 "manual_text" 的 MongoDB 集合。
mongo_collection = MongoConfig.get_collection("manual_text")
# 连接到 Milvus 数据库，uri 是 Milvus 数据库的连接地址。
connections.connect(uri=milvus_db_path)
# 初始化 BGEM3EmbeddingFunction 类的实例，使用指定的模型路径和设备（这里是 GPU），用于将文本转换为向量。
embedding_handler = BGEM3EmbeddingFunction(model_name= bge_m3_model_path, device="cuda")

# 定义一个名为 MilvusRetriever 的类，用于实现基于 Milvus 数据库的文本检索功能。
class MilvusRetriever:
    # 类的构造函数，在创建 MilvusRetriever 类的实例时会自动调用。
    # docs 是待处理的文档列表，retrieve 是一个布尔值，指示是否处于检索阶段，默认为 False。
    def __init__ (self, docs, retrieve=False):
        # 定义 Milvus 集合的字段，每个字段用 FieldSchema 表示。
        fields = [
            # 构建唯一 ID 字段，作为主键，数据类型为字符串，最大长度为 ID_MAX_LENGTH。
            FieldSchema(name="unique_id", dtype=DataType.VARCHAR, is_primary=True, max_length=ID_MAX_LENGTH),
            # 存储原文的字段，数据类型为字符串，最大长度为 MAX_TEXT_LENGTH。
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH),
            # 存储稀疏向量的字段，数据类型为稀疏浮点向量。
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            # 存储密集向量的字段，数据类型为浮点向量，维度由 embedding_handler 的 dense 维度决定。
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_handler.dim["dense"]),
        ]
        # 根据定义好的字段创建集合的模式。
        schema = CollectionSchema(fields)

        # 如果不是检索阶段，并且 Milvus 中已经存在名为 COL_NAME 的集合，则删除该集合。
        if not retrieve and utility.has_collection(COL_NAME):
            Collection(COL_NAME).drop()
        # 创建或获取名为 COL_NAME 的集合，并设置一致性级别为强一致性。
        self.col = Collection(COL_NAME, schema, consistency_level="Strong")

        # 定义稀疏向量的索引类型和度量类型，这里使用稀疏倒排索引和内积度量。
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        # 定义密集向量的索引类型和度量类型，这里使用自动索引和内积度量。
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        # 在稀疏向量字段上创建索引。
        self.col.create_index("sparse_vector", sparse_index)
        # 在密集向量字段上创建索引。
        self.col.create_index("dense_vector", dense_index)
        # 将集合加载到内存中，以便进行搜索操作。
        self.col.load()

        # 如果是非检索阶段，调用 save_vectorstore 方法构建向量库。
        if not retrieve:
            self.save_vectorstore(docs)

    # 该方法用于将文档的向量信息保存到 Milvus 集合中。
    # docs 是待处理的文档列表，类型为字符串列表。
    def save_vectorstore(self, docs: list[str]): 
        # 从文档列表中提取每个文档的文本内容。
        raw_texts = [doc.page_content for doc in docs]
        # 从文档列表中提取每个文档的唯一 ID。
        unique_ids = [doc.metadata["unique_id"] for doc in docs]

        # 调用 embedding_handler 计算文本的嵌入向量，得到稀疏向量和密集向量。
        texts_embeddings = embedding_handler(raw_texts)

        # 按批量插入向量信息到 Milvus 集合中。
        for i in range(0, len(docs), EMB_BATCH):
            # 取出当前批次的唯一 ID、文本内容、稀疏向量和密集向量。
            batched_entities = [
                unique_ids[i : i + EMB_BATCH],
                raw_texts[i : i + EMB_BATCH],
                texts_embeddings["sparse"][i : i + EMB_BATCH],
                texts_embeddings["dense"][i : i + EMB_BATCH],
            ]
            # 将当前批次的数据插入到 Milvus 集合中。
            self.col.insert(batched_entities)
        # 打印索引构建完成的信息，显示插入的数据条数。
        print("索引构建完成，插入了{}条数据:".format(self.col.num_entities))

    # 该方法用于基于密集向量进行搜索。
    # query_dense_embedding 是查询的密集向量，limit 是返回结果的最大数量。
    def dense_search(self, query_dense_embedding, limit):
        # 定义搜索参数，使用内积度量。
        search_params = {"metric_type": "IP", "params": {}}
        # 在密集向量字段上进行搜索，返回符合条件的结果。
        res = self.col.search(
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=limit,
            output_fields=["unique_id", "text"],
            param=search_params,
        )
        return res

    # 该方法用于基于稀疏向量进行搜索。
    # query_sparse_embedding 是查询的稀疏向量，limit 是返回结果的最大数量。
    def sparse_search(self, query_sparse_embedding, limit):
        # 定义搜索参数，使用内积度量。
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        # 在稀疏向量字段上进行搜索，返回符合条件的结果。
        res = self.col.search(
            [query_sparse_embedding],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=["unique_id", "text"],
            param=search_params,
        )
        return res

    # 该方法用于进行混合搜索，结合稀疏向量和密集向量的搜索结果。
    # query_dense_embedding 是查询的密集向量，query_sparse_embedding 是查询的稀疏向量。
    # sparse_weight 是稀疏向量的权重，dense_weight 是密集向量的权重，limit 是返回结果的最大数量。
    def hybrid_search(
        self,
        query_dense_embedding,
        query_sparse_embedding,
        sparse_weight=1.0,
        dense_weight=1.0,
        limit=10,
    ):
        # 定义密集向量搜索的参数，使用内积度量。
        dense_search_params = {"metric_type": "IP", "params": {}}
        # 创建密集向量的搜索请求。
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
        )
        # 定义稀疏向量搜索的参数，使用内积度量。
        sparse_search_params = {"metric_type": "IP", "params": {}}
        # 创建稀疏向量的搜索请求。
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
        )
        # 原本计划使用 WeightedRanker 进行重排，但被注释掉了。
        # rerank = WeightedRanker(sparse_weight, dense_weight)
        # 使用 RRFRanker 对搜索结果进行重排。
        rerank = RRFRanker()
        # 进行混合搜索，结合稀疏向量和密集向量的搜索请求，并使用重排器。
        res = self.col.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["unique_id", "text"]
        )
        return res

    # 该方法用于根据查询文本检索前 topk 个相关文档。
    # query 是查询文本，topk 是返回结果的最大数量，默认为 10。
    def retrieve_topk(self, query, topk=10):
        # 记录当前时间，用于计算检索耗时。
        t1 = time.time()
        # 调用 embedding_handler 的 encode_queries 方法，计算查询文本的嵌入向量。
        query_embeddings = embedding_handler.encode_queries([query])

        # 调用 hybrid_search 方法进行混合搜索，获取前 topk 个结果。
        hybrid_results = self.hybrid_search(
            query_embeddings["dense"][0],
            query_embeddings["sparse"][[0]],
            sparse_weight=0.7,
            dense_weight=1.0,
            limit=topk
        )[0]

        # 存储从 MongoDB 中获取的相关文档。
        related_docs = []
        # 遍历混合搜索的结果。
        for result in hybrid_results:
            # 根据结果中的唯一 ID 从 MongoDB 中查找对应的文档。
            search_res = mongo_collection.find_one({"unique_id": result["id"]})
            # 原本计划处理图片信息，但被注释掉了。
            #images_list = []
            #for image in search_res["metadata"]["images_info"]:
            #    images_list.append(ManualImages(**image))
            #search_res["metadata"]["images_info"] =  images_list 
            # 创建 Document 对象，包含文本内容和元数据。
            doc = Document(page_content=search_res["page_content"], metadata=search_res["metadata"])
            # 将 Document 对象添加到相关文档列表中。
            related_docs.append(doc)

        return related_docs 

# 如果该脚本作为主程序运行，则执行以下代码。
if __name__ == "__main__":
    # 读取测试文档文件，将每一行作为一个文本。
    texts = [k for k in open(test_doc_path).readlines()]
    # 存储文档对象的列表。
    docs = []
    # 遍历每个文本。
    for text in texts:
        # 使用 MD5 哈希算法生成文本的唯一 ID。
        unique_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        # 定义文档的元数据，包含唯一 ID。
        metadata = {"unique_id": unique_id}
        # 创建 Document 对象，添加到文档列表中。
        docs.append(Document(page_content=text, metadata=metadata))
    # 创建 MilvusRetriever 类的实例，传入文档列表。
    retriever = MilvusRetriever(docs)
    # 定义查询文本。
    query = "Model3支持的钥匙类型"
    # 调用 retrieve_topk 方法进行检索，获取前 2 个相关文档。
    results = retriever.retrieve_topk(query, 2)
    # 遍历检索结果，打印每个文档和分隔线。
    for res in results:
        print(res)
        print("="*100)
