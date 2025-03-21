# Simple RAG

构建一个简单的RAG模型，包含了RAG的核心功能，即Retrieval和Generation。

## RAG 原理
检索增强生成技术（Retrieval-Augmented Generation，RAG）是为了解决LLM会产生误导性的“幻觉”

RAG通过在大模型生成回答之前，先从广泛的文档数据库中检索相关信息，然后利用这些信息来引导生成过程，极大地提升了内容的准确性和相关性。RAG技术能够有效缓解模型幻觉问题，提高了知识更新的速度，并增强了内容生成的可追溯性，使得大模型在实际应用中变得更加实用和可信。

### RAG的基本结构
- 向量化模块，用来将文档片段向量化
- 文档加载和切分模块
- 数据库，用于存放文档片段和对应的向量表示
- 检索模块，用来根据 Query 问题 检索相关的文档片段
- 大模型模块，用来根据检索出来的文档回答用户问题

### RAG流程
- 索引：将文档库分割成较短的 Chunk，并通过编码器构建向量索引
- 检索：根据问题和 chunks 的相似度检索相关文档片段
- 生成：以检索到的上下文为条件，生成问题的回答

## 向量化

## 文档加载和切分