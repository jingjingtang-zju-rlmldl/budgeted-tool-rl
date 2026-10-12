# 逐类结果与错误位置分解

这是固定实验完成后的描述性统计，没有用于挑选 checkpoint、修改任务或采样配置。重复种子仍属于同一个事实任务。

| 同事实配对类别 | v2 措辞A | v2 措辞B | v3 措辞A | v3 措辞B |
|---|---:|---:|---:|---:|
| chain_four | 8/8 | 0/8 | 8/8 | 8/8 |
| chain_three | 6/8 | 0/8 | 8/8 | 8/8 |
| prefix_redirect | 8/8 | 8/8 | 8/8 | 8/8 |
| prefix_retry | 8/8 | 8/8 | 8/8 | 8/8 |
| redirect_four | 8/8 | 0/8 | 8/8 | 8/8 |
| redirect_three | 8/8 | 0/8 | 8/8 | 8/8 |
| retry_four | 8/8 | 0/8 | 8/8 | 8/8 |
| retry_three | 8/8 | 0/8 | 8/8 | 8/8 |

| 自由起始任务 | v2 简洁 | v2 强提示 | v3 简洁 |
|---|---:|---:|---:|
| anchor_two | 8/16 | 14/16 | 16/16 |
| chain_four | 8/16 | 16/16 | 16/16 |
| chain_three | 8/16 | 16/16 | 14/16 |
| given_answer | 16/16 | 16/16 | 16/16 |
| one_lookup | 16/16 | 16/16 | 16/16 |
| redirect_two | 8/16 | 16/16 | 16/16 |
| retry_two | 8/16 | 14/16 | 16/16 |
| skip_owner | 16/16 | 16/16 | 16/16 |
| skip_prefix | 16/16 | 16/16 | 16/16 |

新模型对原模型的同任务／同种子配对结果：{"old/protocol": {"gained": 38, "lost": 0, "both_correct": 104, "both_failed": 2}, "old/examples": {"gained": 4, "lost": 2, "both_correct": 138, "both_failed": 0}}。

自由测试首个工具调用的非法字段频次：{"old/protocol": {"accountable_for": 35, "accountable": 2, "staff_member": 1}, "old/examples": {"accountable_staff": 2}, "new/protocol": {}}。

新 signal 首个工具调用的非法字段频次：{'department': 3}。

| 注入错误／位置 | 条件 | 首动作匹配最少调用纠错目标 | 最终可溯源成功 | 后续非法查询 | 正确轨迹额外调用（扣除固定注入错误） |
|---|---|---:|---:|---:|---:|
| unknown_entity/stage0 | old/protocol | 4/8 | 4/8 | 16 | 0 |
| unknown_entity/stage0 | old/examples | 6/8 | 6/8 | 10 | 0 |
| unknown_entity/stage0 | new/protocol | 8/8 | 8/8 | 0 | 0 |
| unknown_entity/stage1 | old/protocol | 5/8 | 5/8 | 12 | 0 |
| unknown_entity/stage1 | old/examples | 8/8 | 8/8 | 0 | 0 |
| unknown_entity/stage1 | new/protocol | 8/8 | 8/8 | 0 | 0 |
| unknown_entity/stage2 | old/protocol | 8/8 | 6/8 | 0 | 0 |
| unknown_entity/stage2 | old/examples | 8/8 | 8/8 | 0 | 0 |
| unknown_entity/stage2 | new/protocol | 8/8 | 8/8 | 0 | 0 |
| unknown_entity/stage3 | old/protocol | 2/8 | 2/8 | 6 | 0 |
| unknown_entity/stage3 | old/examples | 4/8 | 4/8 | 8 | 0 |
| unknown_entity/stage3 | new/protocol | 8/8 | 8/8 | 0 | 0 |
| unknown_field/stage0 | old/protocol | 3/8 | 3/8 | 18 | 0 |
| unknown_field/stage0 | old/examples | 6/8 | 6/8 | 8 | 0 |
| unknown_field/stage0 | new/protocol | 8/8 | 8/8 | 0 | 0 |
| unknown_field/stage1 | old/protocol | 8/8 | 8/8 | 0 | 0 |
| unknown_field/stage1 | old/examples | 8/8 | 8/8 | 0 | 0 |
| unknown_field/stage1 | new/protocol | 8/8 | 8/8 | 0 | 0 |
| unknown_field/stage2 | old/protocol | 8/8 | 8/8 | 0 | 0 |
| unknown_field/stage2 | old/examples | 8/8 | 8/8 | 0 | 0 |
| unknown_field/stage2 | new/protocol | 8/8 | 8/8 | 0 | 0 |
| unknown_field/stage3 | old/protocol | 6/8 | 6/8 | 0 | 0 |
| unknown_field/stage3 | old/examples | 8/8 | 8/8 | 0 | 0 |
| unknown_field/stage3 | new/protocol | 8/8 | 8/8 | 0 | 0 |

位置 0/1/2/3 分别是 project、employee、department、cost_center。注入错误由脚本执行，不是模型自行产生；首动作匹配与最终任务成功分别统计。

恢复集有一个设计限制：unknown_field 使用完整措辞模板 0，unknown_entity 使用模板 1，错误类型与措辞耦合，且两类使用不同事实。模型条件间比较仍保持相同问题和状态；不能把两类错误的分数差直接解释为错误类型自身的难度差。后续若专门比较错误类型，需要同事实交叉两类错误和两种措辞。

| 全新奖励诊断类别 | 答对 | 可溯源答对 | 额外查询 | 非法查询 |
|---|---:|---:|---:|---:|
| chain_four | 16/16 | 16/16 | 0 | 0 |
| chain_three | 16/16 | 16/16 | 0 | 0 |
| prefix_redirect | 16/16 | 16/16 | 0 | 0 |
| prefix_retry | 16/16 | 16/16 | 10 | 6 |
| redirect_four | 16/16 | 16/16 | 0 | 0 |
| redirect_three | 16/16 | 16/16 | 0 | 0 |
| retry_four | 16/16 | 16/16 | 0 | 0 |
| retry_three | 16/16 | 16/16 | 0 | 0 |

新自由测试失败停止原因：{'tool_budget_exhausted': 2}。新 signal 失败停止原因：{}。

## 新模型剩余失败

- sft3_test_chain_three_0004，种子 11217：应复制 `project_226323d6da84`，首动作实际为 `{"action": "lookup", "entity": "project_22632d6da84", "field": "owner"}`，字符遗漏=True，停止=tool_budget_exhausted。
- sft3_test_chain_three_0004，种子 1217：应复制 `project_226323d6da84`，首动作实际为 `{"action": "lookup", "entity": "project_22632d6da84", "field": "owner"}`，字符遗漏=True，停止=tool_budget_exhausted。

这类真实复制失败与本轮注入的类型词／_typo 后缀错误不同。注入错误测试高分不能证明所有 unknown_entity 都能恢复。后续若补数据，应另生成事实，在新训练样本中采集学生失败前缀或加入字符遗漏／替换／插入等不同扰动；当前测试事实保留为历史证据，不回填训练。


## 奖励诊断决策

当前只有 1/32 个任务在四次采样中产生非恒定奖励（3.1%）。类别为 {"all_correct_same_cost": 31, "all_correct_different_cost": 1}。

本轮没有成功／失败混合组。若有有效组，其差异来自答对所需的调用成本，不能解释为模型已展现可供成功率学习的组内差异。全对等成本组即使成本偏高，也没有当前组内相对优势。

不建议现在直接对这批诊断题启动正式 GRPO：有效组比例较低，而且这是留出的诊断集。下一步优先在全新训练事实中增加需要根据观察选择分支、利用已知信息跳过查询、权衡不同合法查询成本的任务，再独立检查奖励正确性和有效组覆盖率。自然 ID 复制错误应先用不同事实的学生错误前缀处理，不能把本轮失败测试题回填训练。温度扫描如要做，应提前固定并使用另一个开发集，不能追着当前 32 题调参。

[逐项机器可读统计与原始证据哈希](descriptive-breakdown.json)。统计口径与主报告一致，新增分层属于事后分析。
