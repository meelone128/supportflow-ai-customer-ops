# Kaggle 公开工单路由审计

该模块使用 `suraj520/customer-support-ticket-dataset` 的公开工单字段，验证 SupportFlow 在一批独立工单上的分流与人工复核路由。源数据页标注为 CC0；运行时仍应复核最新许可和数据集条款。

## 隐私边界

原始数据中可能有 Customer Name、Customer Email、Age、Gender 等字段。本模块只读取并保留：工单 ID、主题、描述、类型、优先级、状态、渠道、解决方案和满意度。个人字段不会写入任何评测产物。

## 运行

```bash
python -m pip install -r supportflow/requirements-data.txt
python -m supportflow.run_kaggle_ticket_audit --limit 100
```

首次下载可能需要在 Kaggle 登录并同意相关条款。结果写入 `supportflow/evals/reports/kaggle-ticket-audit.json`，原始下载缓存位于被 Git 忽略的 `supportflow/evals/kaggle_cache/`。

## 指标解释

`high_priority_review_capture` 表示公开数据中标为 High/Critical 的工单，有多少被 SupportFlow 送入人工审批或人工升级。它用于审计“系统是否保守处理高优先级工单”，不表示 Kaggle 优先级就是模型安全标签，也不能推导真实企业的客户满意度提升。
