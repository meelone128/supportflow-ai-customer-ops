# SupportFlow

SupportFlow 是一个面向企业客服工单的多 Agent + RAG 运营平台。它不让模型直接“代替客服”，而是把工单分流、只读业务上下文、证据检索、回复草稿、人工审核、知识库运营和离线评测放进可审计的工作流。

## 主要能力

- LangGraph 编排：分流、客户上下文、订单上下文、知识检索、回复生成、质检六个环节；高风险工单跳过生成并升级人工。
- 证据约束 RAG：每份回复可追溯至知识来源和版本，检索不可用时安全降级。
- 权限与审计：JWT/RBAC 控制审核、知识发布与指标查看；模型没有退款、改订单或改客户资料的写权限。
- 运营闭环：审核反馈会形成知识缺口，版本化发布后可基于独立标注与前后队列对比验证效果。
- 工程化：SQLite 本地开箱即用；可通过环境变量切换 PostgreSQL；Docker Compose 提供 Redis/Celery 异步拓扑。

## 本地运行

```powershell
python -m unittest discover -s supportflow\tests -v
python -m uvicorn supportflow.api:app --host 127.0.0.1 --port 8765
```

- 运营工作台：`http://127.0.0.1:8765/workbench`
- 客户入口：`http://127.0.0.1:8765/customer`
- API 文档：`http://127.0.0.1:8765/docs`

完整架构、评测、部署与面试说明见 [supportflow/README.md](supportflow/README.md)。

## 安全与公开仓库说明

- 本仓库不包含 API Key、数据库连接串、CRM/客服渠道凭据、真实客户数据或本地 `.dev.env`。
- 公开的评测报告只使用脱敏样例与公开数据处理后的汇总结果。
- 线上部署应在平台环境变量中配置密钥，并关闭本地演示角色头：`SUPPORTFLOW_ALLOW_DEMO_ROLE_HEADER=false`。
- 如发现安全问题，请按 [SECURITY.md](SECURITY.md) 中的方式私下报告。
