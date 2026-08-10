# 公开客服语料实验

## 目的

该实验用于验证“知识补充是否改善特定客服意图的检索支持”，不是为了宣称真实企业业务指标。数据与产品演示工单隔离，禁止将公开语料误标为真实客户数据。

来源：`bitext/Bitext-customer-support-llm-chatbot-training-dataset`；许可证：`cdla-sharing-1.0`。运行前仍应在数据集页面复核其最新 Dataset Card 与许可条款。

## 构造方式

1. 从公开数据中筛选售后相关记录。
2. 固定将 `track_refund` 作为受控知识缺口：V1 不含该意图。
3. 用未见过的 `track_refund` 工单建立 30 条盲测集。
4. V2 仅增加另一批 `track_refund` 知识条目；盲测工单不能进入任何知识版本。
5. 脚本以稳定哈希划分数据，并输出 `manifest.json`。若测试数据泄漏到知识数据，构建直接失败。

## 运行

```bash
python -m pip install -r supportflow/requirements-data.txt
python -m supportflow.build_public_corpus
```

输出目录 `supportflow/evals/public_corpus/`：

- `knowledge_v1.jsonl`：基线知识；
- `knowledge_v2_additions.jsonl`：知识更新条目；
- `blind_test_track_refund.jsonl`：从未进入知识库的盲测工单；
- `manifest.json`：来源、许可证、数量与泄漏校验结果。
- `retrieval_comparison.json`：V1/V2 在盲测 `track_refund` 工单上的 Top-1 意图检索命中率与差值。

## 汇报口径

可以说：“基于公开 Bitext 客服语料构建了一个固定意图的 V1/V2 知识更新实验，并通过盲测隔离和稳定分割防止数据泄漏。”

不可以说：“该实验验证了真实企业客户满意度提升。”真实满意度、人工节省和线上转人工率必须来自授权的业务数据。
