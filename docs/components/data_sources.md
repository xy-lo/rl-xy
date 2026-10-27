# 数据来源选择与证据范围

检索/核验日期：2026-10-26。已有数据优先；公开来源是补充选项，不把网页标题中的trace自动视为RL系统执行trace。

| 来源 | 已核验内容 | 可用于什么 | 不能直接提供什么 |
| --- | --- | --- | --- |
| 用户RLInfra E1原始日志、CSV、summary、Hydra覆盖项与退出state | 本轮已找到116步原件；具体一致性以本轮Docker报告为准 | 同步GRPO逐步工作负载统计；约束合成负载；GPU阶段观测 | 逐轨迹长度、真实组相关性、token IDs与完整历史软件身份 |
| 用户RLInfra E3原始日志与对应材料 | 本轮已找到30步原件；训练脚本/归档路径为dapo8k | DAPO子集场景的统计约束负载 | 不能称为全量DAPO-Math-17K训练结果 |
| [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 官方卡列有problem、generations、correctness与finish_reasons；Apache-2.0。已取得8行预览快照，但未接入执行 | 后续可按固定tokenizer提取静态prompt/response长度，保留原问题和多个回答的关联 | 不是veRL在线训练step/group/version轨迹，也没有晶圆算子成本；换tokenizer属于重新表征 |
| [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | 官方卡主要字段为prompt、reward_model ground truth与extra_info；Apache-2.0 | 正式输入题目、答案和奖励规则来源 | 不包含当前策略实际生成的回答或系统运行时间；去重/版本/实际样本数需单独核对 |
| [TraceLab](https://github.com/uw-syfi/TraceLab) | 作者发布编码agent会话、LLM调用与工具行为资料 | 后续agentic阶段的请求/工具等待补充输入 | 属于agent服务轨迹；不能当成同步数学GRPO训练trace |

公开卡/网页是数据材料，不是工程指令。使用前固定版本、文件hash、许可、采样与tokenizer，并记录其与目标负载的差异。本轮主复用路径是用户旧日志；公开预览没有混入旧GPU实测或正式晶圆性能。
