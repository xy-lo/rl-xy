# veRL同步GRPO → 晶圆级：完整实验设计与放行门禁

日期：2026-10-28。任务T42；本文件由设计子任务编写，最终数值与运行预算由父任务锁定。配套机器合同为[parameter_contract.json](../experiments/2026-10-28-v0.9.5/design_contract.json)，来源、revision与原始文件SHA为[reference_manifest.json](../experiments/2026-10-28-v0.9.5/design_sources.json)。**本文没有产生新的性能结果，也不宣布正式3B基线已放行。**

**实验顺序确定为：先验证3B GPU受限同步基线，再将相同策略、相同48个逻辑rank迁到晶圆，最后才引入wafer-aware策略。最终模型是3B、7B、14B、32B。** 当前第一3B控制点为P512/R128；P1024/R1024固定主控制与8K输出长度分布是后续必需覆盖，均有预算与成本门禁。工程探针、第一控制点、论文主表分开命名，不互相替代。

选用的是固定veRL的**合法受限配置**，不是宣称整个veRL默认配置或全空间最优：Megatron distributed optimizer开启，sequence parallel关闭，计算/通信overlap关闭。SP/overlap会显著影响强GPU上界，当前未覆盖的优化必须在报告中披露；完成一个受限3B对比不自动意味着“论文强GPU基线已完成”。

## 1. 研究问题、三层对照与允许结论

| 层次 | 实验对象 | 固定什么/改变什么 | 为什么做 | 来源与状态 |
|---|---|---|---|---|
| G：GPU受限基线 | 48×A100成本/网络模型上的固定veRL协议 | 固定算法和W，在已支持、容量合法的GPU候选中选低成本策略 | 得到可审计的GPU出发点，避免人为固定低效microbatch制造差距 | [R03]§6、[R10]；候选搜索待执行，不称全局最优 |
| WD：晶圆直接迁移 | 同一policy、同48rank与group，映到48active/54allocated H0 | 只换硬件成本、物理拓扑与容量准入；不重选TP/微批/角色 | 单独观察硬件迁移影响。直接迁移可以更快也可以更慢，不要求先输 | [R03]的配置/硬件分层；本项目控制设计，待验收 |
| WA：晶圆感知方法 | 同一晶圆上的新映射/放置/调度 | 一次改变一个机制；先仍用48active | 新方法收益应相对WD衡量，不能把硬件差异都算成方法贡献 | [R13][R14]协同探索方法；本阶段不实施 |
| 54active扩展 | 使用H0全部54die | 单列活动资源扩展，并重新处理batch/DP/padding | 避免把额外6die算力混入“同策略”或“调度”收益 | 项目预算控制；延后 |

定义硬件迁移比为`T_G/T_WD`，方法比为`T_WD/T_WA`。二者分开报告，保留小于1的结果。当前只准备G与WD，不以“晶圆略快”作为通过条件。时间来自声明算子成本和native网络下的仿真时钟，不是实体设备测量；质量与能耗没有因此被验证。

## 2. 模型为什么选3B/7B/14B/32B

| 规模 | 工程/研究用途 | 结构来源与理由 | 本阶段状态 |
|---|---|---|---|
| 3B | 验证完整协议、映射/成本域和第一条件性对照 | Qwen2.5-3B-Instruct固定config；36层、H2048、query/KV=16/2；ReSpec有3B系统验证先例 | 现在做；模型revision与SHA见[R19] |
| 7B | 主流dense RL负载；检验并行选择变化 | RollPacker/Weave等覆盖；28层、H3584、28/4 | 后续同W接入，不复活旧7B补跑队列 |
| 14B | 层深和KV/token明显增加，测试状态/通信压力 | 同一家族48层、H5120、40/8；有直接RL系统先例 | 需TP8/成本域与容量验收 |
| 32B | 检查DO/PP/局部内存边界与更大模型推广 | 同家族64层、H5120、40/8；DAPO、RollPacker等已有此规模 | 先按真实DO账本准入；不得沿旧复制20P口径宣布必然可行或必然需PP |

来源：[R11][R12][R18][R19]。**模型名的B只是规模标签**，正式计数从固定config精确枚举；不同规模的层数、GQA结构和绑权配置不同，不用单一“参数倍数”替代算子图。0.5B仅保留历史回归，不占最终四模型主表。当前不扩大72B/MoE/agentic范围。

旧PP1复制状态的32B在TP8时仅持久状态约81.9GB/rank，这只是旧合同下的容量障碍。当前改为DO，master/moments/shard/bucket驻留不同，须重新核算，不能免费把所有对象都除DP。若DO仍不能准入，才实现并验证PP或其它明示策略；Qwen14/32B的40query heads也不能直接用常规均匀TP16。[R07][R18][R20]

## 3. 算法合同：原文、官方工程值、本项目选择分列

| 项目 | 一手文献/工程事实 | 本项目选择及原因 | 当前放行状态 |
|---|---|---|---|
| 算法 | DeepSeekMath提出GRPO；DAPO另外包含clip-higher、动态过滤与超长处理 | 固定veRL-v0.9.1同步outcome-GRPO、token-mean变体。维持原目标，减少算法变化干扰硬件对照 | 算法名已选；IR接入待验收。[R01][R02][R04] |
| Critic/reward | GRPO无需critic；数学任务可用规则验证，原DeepSeekMath亦研究reward model | 无critic/无神经reward模型；性能W的reward为给定回放值/未知值，不冒充实际判题 | 不得从给定reward推导训练质量。[R01] |
| Advantage | 原GRPO按组减均值并除std；veRL有对应实现 | `norm_adv_by_std=True`、epsilon1e-6，匹配固定源码sample-std/zero-variance语义 | T40小型oracle已过，性能IR必须真的消费相应对象。[R06][R17] |
| Policy loss | 原GRPO含ratio clip；veRL vanilla支持相应配置 | 对称clip low/high=.2/.2、dual_clip_c=3、vanilla；明确保存ratio和old logprob输入 | loss/梯度/通信形状一致性待验收。[R04][R06] |
| Loss reduction | 原GRPO为sequence长度归一化；veRL默认token-mean | 采用**veRL token-mean变体**，用全mini-batch有效response token数归一化 | 不声称原论文逐公式完全复制；padding不计分母。[R01][R04] |
| KL/reference | 原GRPO实验KL=.04；veRL例子KL loss=.001 | `use_kl_loss=True,k3(low_var_kl),coef=.001`；reward内KL关闭，Reference冻结 | 选择官方工程值而非混用原论文.04；禁止免费删ref。[R01][R04][R09] |
| Entropy | veRL例子系数0 | 系数0；不额外请求全vocab entropy | 具体loss成本与不计算entropy的图需一致。[R04][R09] |
| 内部epoch | veRL`ppo_epochs=1`；不等同dataset epoch | 每个rollout batch重用1epoch | 本阶段完整mini-batch只更新一次。[R04][R05] |
| 迭代/质量 | 原论文跑完整学习过程；系统性能窗口不是收敛证据 | 初始Actor版本v，单步发布v+1，Reference版本固定；数值质量训练预算不在本轮 | 不模拟收敛、time-to-quality或准确率。[R01][R03] |
| 采样配方 | 固定veRL温度1/top_p1/top_k−1/seed42；EOS通常生效 | 若以后接真trace，保存这些配置与实际finish_reason。固定W只指定形状，不宣称采样发生 | seed不产生真实回答；控制R与真实EOS长度分开。[R16] |
| 数据/随机性 | DAPO数据集是prompt题库，不是完整3B执行trace | 当前不切成DAPO；不通过换数据/关KL缩短step | 未来算法扩展单列ID/生成过滤统计。[R02] |

同步step的逻辑顺序固定为：权重准备/训推切换→生成→样本重分布/释放推理状态→reward→old-policy logprob→Reference logprob→advantage→actor forward/backward/gradient accumulation→DO归约/clip/optimizer/参数发布→版本屏障。该顺序应和选定worker逐项对照；允许后端内部重叠的阶段必须明示，不能由汇总表隐藏。旧policy logprob是否可复用rollout输出属于另一优化，不能先删阶段。[R03][R05]

## 4. 数据、trace与三类W：来源决定能说什么

| 输入类型 | 来源及理由 | 数量如何记录 | 可用范围/状态 |
|---|---|---|---|
| 固定形状控制W | 参考HybridFlow的固定P/R系统对照；不需要编造真实回答 | 全部prompt/group/trajectory长度明确生成，逐行manifest/hash | 现在优先做；必须标`synthetic_control`。[R03] |
| 公开统计约束W | 官方链接7B FSDP2/Megatron聚合日志：本地实际解析105/210步 | 保存所选step窗口、原日志SHA、seed、合成样本数、约束误差和组内相关性假设 | 只可称“7B统计约束的合成/跨模型反事实W”，非真实3Btrace。[R16] |
| 真实语义/质量锚点 | 已锁`openai/gsm8k/main`、snapshot、veRL preprocess及Qwen3B tokenizer/chat template，见下表 | 原始train7473/test1319；项目train/tuning身份拆分单列，去重/分词/过滤后重新计数 | 题库不是rollout trace；本轮未新采集或数值训练。[R22]–[R26] |
| 真实执行trace | 本次定向官方检索未取得dense3B48A100完整文件；已有0.5B锚点独立保留 | token/group/version/rank/事件窗口与原始归档 | 不把0.5B来源改成3B，不把日志均值反解成唯一真轨迹。[R16] |

### 4.1 首个真实语义/质量锚点已锁：GSM8K

选择GSM8K是为了先建立来源明确、可独立判分的小规模语义锚点；这不表示它足以覆盖长链reasoning，也不表示所有RL论文有统一最低轨迹条数。[R23] 原始题库、实际生成轨迹、固定长度W分别保存。

| 项目 | 官方源事实 | 本项目已锁选择及原因 | 状态/门禁 |
|---|---|---|---|
| 数据集 | `openai/gsm8k`、subset=`main`；train7473/test1319，MIT | 固定snapshot `740312add88f781978c0658806c59bc2815b9866`，避免main漂移 | 两个parquet已下载且SHA与Hub LFS一致；不是生成trace。[R22] |
| Train/tuning/test | 官方只有train与test；不提供本项目500行tuning split | seed42，对7473个原始train索引按约定SHA排序，前500为tuning，其余6973为train；1319test只作最终eval | 身份清单与SHA已生成；500是项目工程选择，不是论文标准。重复题审查/分词尚未执行 |
| 防泄漏 | 原始split身份不保证语义零重复 | 原始行ID保持；NFKC+空白归一化question指纹，隔离训练/调参/测试交叉重复；异常删除/隔离后重新报告行数 | train/tuning/test索引互斥；语义去重完成前不声称无污染。test不调算法、切分、成本常量或W分布 |
| Preprocess | 固定veRL官方gsm8k.py将question与指令构成单user prompt，答案另存ground_truth | 直接锁其commit/SHA与指令；不把teacher答案接入prompt或充作模型回答 | source已锁；完整预处理未执行。[R24] |
| Tokenizer/chat | Qwen3B固定模型revision提供tokenizer.json、tokenizer_config与chat template | `aa8e72537993ba99e69dfaafa59ed015b17504d1`；fast tokenizer；单user、无tools/自定义system；`apply_chat_template(tokenize=True,add_generation_prompt=True)` | tokenizer两文件及模板SHA已保存；保留模板自带default system。执行前保存库版本与token fixture。[R25] |
| Prompt抽样 | 源7473行不等于本轮使用7473个prompt | seed42、独立SHA域排序derived train IDs，无放回；所有模型复用相同源ID；真实prompt cap1024，超长拒绝并计数 | 清单见机器合同；尾批不静默丢弃，padding/mask须计代价。W0/W1/W2仍是独立synthetic_control |
| 真实rollout | 题库没有给定3B on-policy回答长度 | 后续真实采样配方温度1、top_p1、top_k−1、seed42、G8、Rcap8192，EOS生效；实际R从生成结果或有来源的逐条公开回答取得 | 本轮未执行；不能用teacher answer长度、均值或Rcap冒充真实R；cap8K不保证8K长尾 |
| Reward/eval | 固定veRL strict rule提取最终答案并做源码定义的字符串匹配，format0/correct1 | 如做未来质量eval，test1319题全部用greedy单回答、Rcap8192、EOS生效，并报格式失败/空回答/截断 | 配方先锁；不是本轮执行授权。本轮无数值训练，不给accuracy或reward学习曲线。[R26] |
| 后续长reasoning | DAPO论文提供独立题库，算法与数据是两个维度 | `BytedTsinghua-SIA/DAPO-Math-17k` snapshot `65877096c24ffa7abc4e4fa5edb95cf3413a5674`作为后续四模型补充；保持本项目GRPO算法合同 | 只核metadata，299MB数据未下载；总行/唯一题/重复与decontamination先审计，不能按名字当17k独立题或执行trace。[R02][R27] |

`raw/gsm8k-split-indices.json`固定源行身份和独立抽样顺序，机器合同保存SHA与确定性算法。已下载来源文件不等于已经完成分词、去重、rollout或训练。W0/W1/W2的P/R只是控制形状，**不能标成GSM8K真实轨迹**；GSM8K锚点也不能替代W3的8K分布代表性门禁。

父任务已确定“不以旧64条R32结果作为当前主负载”。最终reasoning主表需要8K级输出及长度分布；Azure serving trace较短且不是on-policy RL，不可单独替代。[R11][R12][R18]

## 5. 工作负载阶梯：每个点的角色不同

P为每条回答的prompt token数，R为生成response token数，B为独立prompt数，G为每prompt回答数。`N=B×G`为trajectory数。固定长度场景只模拟这些形状，不暗示模型真的生成了指定回答。

| ID | B×G=N | P/R | 单步有效response tokens | 用途、理由与状态 |
|---|---:|---:|---:|---|
| W0工程探针 | 48×2=96 | 512/128 | 12,288 | 有足够group满足48/24DP且易验收；不作为论文主W、不替代3B代表性结果 |
| W1第一3B控制 | 384×8=3072 | 512/128 | 393,216 | 检验较完整批量/生成的中等控制点；P/R为项目执行预算选择，不称原论文默认值；**待预算准入** |
| W2固定主控制 | 384×8=3072 | 1024/1024 | 3,145,728 | P/R继承HybridFlow控制方法；B/G是48active下项目合法选择，非原论文逐参数复制；**待成本/预算门禁** |
| W3长输出分布 | 384×8=3072 | P固定1K或2K；R为分布、cap8K | 从实际生成manifest精确统计 | 满足reasoning长度覆盖；cap=8K不等于确有长尾，必须报告尾部质量/比例；**来源与预算待锁定** |

W1的prompt+response总token为1,966,080，W2为6,291,456；这是工作量计数，不是仿真吞吐，也不是训练token上限。teacher-forcing输入是否采用P+R−1必须和算子合同一致。

W3优先寻找许可明确的公开逐trajectory长度分布；若仍不可得，可以单独采用**假设性长度压力测试**：R buckets={128,256,512,1024,2048,4096,6144,8192}，项目拟定质量={10%,15%,20%,20%,15%,10%,5%,5%}。这些比例**没有论文/真实3B数据背书**，只是可检验的压力场景；固定seed、舍入方法和最终3072行manifest，再向所有方案提交同一W。与公开统计W分开报告，不称实测reasoning分布。[R11][R12]只支持研究8K长度的必要性，不支持上述比例。

最终四模型主表必须在同一W家族上比较；不要同时把3B设R1K、7B设8K、14B设16K后再宣称“纯模型规模趋势”。另保留按各模型现实任务长度的应用场景，解释为另一实验。

## 6. Batch、mini-batch与microbatch联合合法性

固定veRL版本会把`ppo_mini_batch_size`乘以`rollout.n`，因此这里mini-batch同时记prompt-equivalent与trajectory两个单位。[R05] 在当前48active、PP1/CP1的dense候选中：

`DP=48/(TP×PP×CP)`；要求`N % DP=0`、`Nmini % DP=0`、`N % Nmini=0`，静态无尾批时还要求`(Nmini/DP) % microbatch=0`。生成与logprob各自也检查其DP/批量，不只检查训练。

| W | TP/DP | 本地trajectory mini-batch | 双平台可用训练microbatch | 原因 |
|---|---|---:|---|---|
| W0 N=96 | TP1/DP48；TP2/DP24 | 2；4 | 共同1/2 | micro4对DP48不合法 |
| W1/W2 N=3072 | TP1/DP48；TP2/DP24 | 64；128 | 共同1/2/4/8/16 | 每个候选均能整除；容量/速度仍须独立准入 |

主候选fullmini=384prompt-equivalent=3072trajectory、ppo_epochs1，即一次optimizer更新。多mini-batch候选192/96/48prompt可分别形成2/4/8次更新，但当前主比较不混入这些变化；特别mini48时DP48本地仅8trajectory，不能仍选micro16。

原先256×8=2048对DP48/24分别余32/8；附件128×8=1024分别余16/16。它们可以在别的活动数或有显式padding时合法，不能按常见batch值默认放行。128×8可保留为**32active设备的文献对照点**，须另列资源预算；不冒充48active主点。

384的合法性不延伸到所有54die切分。直接迁移只用48active；以后54active方案重新求共同合法batch或显式记录padding/mask及代价，不能悄悄多生成样本。

## 7. GPU候选切分与“强基线”的范围

| 维度 | 当前选定/候选 | 项目原因 | 来源、未覆盖范围与状态 |
|---|---|---|---|
| 训练/生成TP | (1→1)、(2→1)、(2→2)；(1→2)为需通用reshard支持的候选 | 3B query/KV维度允许此无KV复制子集，分开训练/生成并行 | [R03][R19][R20]；不支持的候选必须报unsupported，不给它虚构慢时间 |
| PP/CP | 初始PP1/CP1 | 降低第一3B验收范围，避免未实现pipeline bubble | 项目合法子集；最终32B如需PP/CP须新增图与通信验收 |
| distributed optimizer | True | 对齐固定Megatron默认，保留真实状态分片/通信成本 | [R07]；必须RS/AG与账本都实现，不是将内存除DP |
| sequence parallel | **False** | 父任务选择的合法受限配置；TP1本来无SP切分，TP2先不冒称SP已实现 | [R07]默认True，本文明确override。TP2 SPtrue未覆盖，因此不能称完整strong GPU最优 |
| gradient/parameter overlap | **False/False** | 保守串行variant便于核验初始时间/依赖 | 可合法配置，但不是更快或默认；须做有overlap的强GPU补充或显式收窄论文claim |
| 训练/logprob microbatch | 分别从1/2/4/8/16合法候选中选；不强制相同 | 形状与内存不同，不能固定单trajectory压低GPU | [R10]；必须真正合并算子/累积而非改metadata |
| generation batching | 固定并记录max_num_seqs/token budget、prefill/chunk/cache选项 | rollout并发决定KV峰值与decode GEMM形状 | [R10]；未实现FlashAttention/continuous batching/prefix cache时逐项披露，不用别的场景标签替代 |
| activation checkpoint | 初3B显式none；容量不足时作为新候选 | 保持原图可核对；若启用须计重算与内存释放 | [R10]；不无代价扣激活 |
| GPU选优 | 只在支持且准入的候选中，以同W完整model step最小者为选定policy | GPU先选，之后冻结再迁移，防止按晶圆结果反向挑策略 | [R03]；保存所有候选/容量失败，不称全空间最优 |

**论文门禁判断：SPfalse/no-overlap配置足以定义一个合法、可复现的受限基线；它本身不足以支撑“比优化veRL/GPU最好方案更快”。** 第一3B对照必须如此命名。若要使用“强GPU baseline”，至少补TP2 SPtrue/可用overlap/合理生成融合缓存的覆盖或可信上界，且不通过隐藏这些选项构造晶圆优势。

### 7.1 实际数据分发与官方运行时的边界（R29）

当前受限IR按训练DP连续分配完整group；每个训练TP块内的生成replica则使用`local_trajectories[j::micro]`静态交错分发，回收时按原owner显式搬运。W1每训练DP128条，拆成两个生成replica各64条；实际每个生成replica收到16个group各4条，而非8个完整group。GRPO按完整源group计算，生成时拆group不改变既定目标；它是两平台共同的**项目静态分发选择**，原因是等量形状控制与明确owner，不是官方默认或最优请求调度。[R29]

固定veRL源码先按`n`相邻repeat样本；`AgentLoopManager`按`DataProto.chunk`给agent workers连续分块。实际server选择还经过`LLMServerClient`和全局负载均衡，不能从worker chunk推断每个GPU replica固定拥有连续完整group。当前IR没有复制该异步请求路由/continuous batching/cache实现；`async`请求API也不自动等于异步RL更新。[R29] 两平台必须保持本轮同一静态映射，但这只证明**受限IR的严格迁移**，不证明官方default rollout执行器已被复制。G8强GPU须补实际server路由/批处理策略或有依据的更强上界。

## 8. Optimizer、梯度与状态移动必须实际出现在图里

| 组件 | 必需行为 | 为什么/来源 | 放行检查 |
|---|---|---|---|
| 混合精度Adam | 参数BF16；main grad/master/m/v FP32；lr1e-6、betas(.9,.999)、wd.01、gradclip1 | [R08][R09]；保存具体backend的decoupled decay定义，不能混Adam/AdamW数值语义 | 小型oracle＋dtype/对象字节/读写检查；非3B数值训练 |
| DO归约 | 沿真实DP组做梯度reduce-scatter，每rank只更新自己的master/m/v shard | [R07]，降低状态驻留同时增加明确集体通信 | native组/payload/dtype、bucket与owner、RS完成前不更新 |
| norm/clip | 对应全局参数集合的norm，各复制/分片贡献只计一次 | 同一优化问题要求，不能每DP局部clip后再错误求和 | TP owner/DP shard计数与独立小张量值对照 |
| 参数发布 | 更新后BF16参数shard all-gather恢复所选训练驻留布局 | DO状态与参数状态不同 | AG之前不能发布新版本，不能保留旧全梯度all-reduce再重复收费 |
| 真实microbatch | 线性GEMM按独立样本合并，注意力/KV不跨请求；dweight累积、optimizer频次与fullmini一致 | [R03][R10]；批量是GPU效率的重要变量 | FLOP/样本/梯度/token分母守恒及独立oracle，不只比较操作数变少 |
| 训推转换 | 保留选定naive full-tensor export、distinct rollout权重、转换scratch及endpoint读写 | 固定worker的合法variant，不称论文零冗余3D-HybridEngine | 相同policy两平台相同逻辑字节与驻留时序 |
| 版本/释放 | Actor v→v+1；Reference冻结；轨迹旧行为版本保留；KV按已完成生命周期释放 | 同步GRPO数据语义 | 多轮无泄漏/无stale-read；不把已完成KV假设成必须跨轮保留 |

### 8.1 普通采样prefill的LM head行数纠偏（R28）

vLLM v0.10.1固定commit `aab549870df50edf0512f0a59b574f692f546465` 的`LogitsProcessor.forward`在选定路径中先调用`_prune_hidden_states`，再调用`_get_logits`；辅助`sampling_metadata`源码显示：普通prefill采样且`prompt_logprobs=None`时，每个请求只追加最后一个sample index。普通单token decode的query_len默认1。[R28] 这不是“整个prefill只算一个token”：Transformer/KV仍处理原P行，减少的是生成端LM head所用的hidden行。

| 算子/阶段 | 固定来源事实 | 本项目共同GPU/wafer合同 | 状态与理由 |
|---|---|---|---|
| rollout prefill Transformer/KV | 完整prompt参与prefill；logits前剪行不取消prompt hidden/KV计算 | 保持每请求P行Transformer/KV及原生命周期 | 不借LMhead优化删掉注意力或KV |
| logits hidden row选择 | `selected_token_indices`先对hidden做index_select | 每请求选最后1行；BF16显式读2H字节、写2H字节，形成selected hidden buffer后接LMhead | 本地读写是明示成本合同，不假设零成本alias或声称实测融合kernel |
| rollout prefill LM head | 无prompt-logprob请求时只需每请求最后1行的下一token logits | 单请求M=1；N个同形独立请求合批后M=N，vocab/TP分片与精度合同不变 | 旧M=P计费不符合此普通采样合同；两平台一起改 |
| rollout decode LM head | 普通单token decode每请求本就1行 | 仍为单请求M=1／合批M=N | 不从本修订宣称decode新增收益 |
| old-policy/ref/train teacher forcing | 本次推理sampling源码不证明teacher只计算R行logits | 保留当前声明的全S图和loss mask/分母；未验证前不改成R-only | 后续若优化，另立源码/IR/成本合同；不能从生成路径免费外推 |

本纠偏只针对`logits_as_input=False`、`prune_hidden_states=True`、存在selected indices、`prompt_logprobs=None`的普通采样控制；不覆盖prompt logprobs、多token speculative decode、beam、chunked prefill的其它路径，也不声称部署了完整vLLM运行时。新增operator合同ID为`vllm_0.10.1_selected_sampling_logits_rows_v1`；即使policy FP未变，也必须用operator标记与source SHA区分新旧图。

**旧W0全prefill行LMhead结果仅保留为工程历史，不升级论文baseline。** 不修改其原始数据或“回填”新合同；旧GPU时间不能与新wafer时间比较。新结果须验收selected hidden的BF16读写/依赖/释放、prefill与decode的M、完整P/KV不变、teacher全S不变、样本与FLOP守恒，重新生成exact GEMM census与成本域覆盖，并执行两平台同合同native对照。更少LMhead行数不是已测端到端加速。[R28]

DO完整内存以实际bucket/alias/存活区间核算，不能把actor、Reference、梯度、KV等全部除DP；18P/20P只用于特定复制状态反事实，不是所有后端的峰值公式。

## 9. 硬件与资源预算：allocated和active必须同时列

GPU是48张A10080GB SXM，6节点×8；晶圆总54die、每die256核，直接迁移仅48die活动、6die闲置。GPU逻辑rank r映到node=floor(r/8)、local=r%8；晶圆固定映到die r、x=r%6/y=floor(r/6)，形成6×8活动子阵列；最后一行6die不参加逻辑TP/DP组。该固定映射在看结果前锁定，不按赢家重排。

| 资源 | 48GPU活动/分配 | 晶圆48active | 晶圆54allocated | 来源与口径 |
|---|---:|---:|---:|---|
| dense BF16/参考FP16峰值 | 14.976 PFLOPS | 12.53376 PFLOPS | 14.10048 PFLOPS | GPU[R15]；wafer1.02TF/core[R13]，BF16等同FP16为假设 |
| DRAM/HBM容量 | 3.840 TB | 3.072 TB | 3.456 TB | 十进制GB；不是全部可免费跨die统一访问 |
| 聚合DRAM/HBM带宽 | 97.872 TB/s | 96 TB/s | 108 TB/s | GPU2039GB/s[R15]；wafer2TB/s/die[R13] |
| 私有SRAM/shared合计 | 约0.87058GB shared | 15.36GB SRAM | 17.28GB SRAM | GPU这里只列SM shared，不是完整L1/L2；两者不是同类总缓存清单 |

48active口径下GPU名义算力比wafer高约19.49%、容量高25%；54allocated口径下则是约6.209%、11.111%。两套口径不混用，不称所有资源严格等预算，更不声称等功耗/面积。

| 网络/执行项 | 主场景 | 原因/来源与状态 |
|---|---|---|
| GPU节点内NVLink | 300GB/s/方向 | 将[R15]600GB/s聚合解释为双向；具体口径是假设，非测得集群有效带宽 |
| GPU节点间shared uplink | **400GB/s/方向/节点** | 保留WSC-LLM paper-caliber参考场景[R13]，与原有模型一致；不能叫实测商用cluster |
| GPU网络敏感性 | 25GB/s/节点作为另一场景 | [R03]200Gbps=25GB/s；这是另一论文硬件场景，不是把主场景偷偷“改单位” |
| Wafer D2D | 2TB/s/edge/direction | 数值来自H0/WSC参考；per-edge/per-direction是明确建模口径，原论文未完整校准本H0 |
| Wafer本地执行 | BF16/FP32/vector、SRAM/NoC带宽和latency均逐字段保留假设标签 | H0资源模板不是实体芯片测量；不把WATOS的DNN或误差继承到本项目 |
| Host/控制/launch | host传输与resume/sleep、GPU launch逐项记录，并做零/小/大值敏感性 | 当前固定1µs等不是统一硬件事实；CUDA graph热GEMM标签不包括所有host开销 |
| 物理几何 | 只称资源/拓扑模板 | 不声称已经证明WSC Case3完整模块可按某一固定外形摆入封装 |

## 10. 成本可信度：子模型误差不能当全GPU校准

现有4,811条A100热cuBLAS GEMM标签及T40审计是有用证据，但cache regime、融合和目标shape不完全一致。审计差异不等于实际veRL端到端误差，不能用一个统一倍数修正整个step。[R17]

| 成本层 | 首个3B对照所需检查 | 论文主表增加的要求 |
|---|---|---|
| GEMM | 精确M/N/K/batch、dtype/output/accumulator、方向/layout签名域；缺失拒绝或另标解析场景 | 目标形状的热/streaming/cold选择依据、频次加权误差、held-out形状，不只unique-shape平均 |
| attention/向量 | 因果mask、GQA、ragged/padding、norm/activation/softmax/optimizer的真实字节与操作合同 | fused attention/cache/实际输出头等与GPU强实现的失配界；不能拿GEMM校准覆盖所有算子 |
| memory/local mapping | 容量合法、共享带宽不被私有核倍增、通信buffer/alias不漏 | 缓存/复用/重算情形敏感性；wafer成本需可信来源及可辩护的范围 |
| 通信 | 独立P2P/RS/AG/native组与同payload fixture；带宽/latency/fence等待分开解释 | 节点拓扑/共享uplink、算法选择与GPU通信实测或可信参考误差；phase跨度不能直接叫纯链路时间 |
| 整step | 同协议同W的native事件/账本一致性，明确未校准部分 | 端到端校验锚点、差异归因和结论稳健性；不能由单测PASS推出硬件精度 |

主表若仍是未校准解析成本，只能写“声明模型下的条件性架构对比”。热GEMM lookup不得二次收同一内存服务，也不得用其耗时替代含fusion/host的全部kernel时间。不同成本场景分列，不隐藏不利结果。

### 10.1 实现中的向量计数与host占位参数

以下值来自本项目显式IR合同，不是论文或实体硬件测量值；源码入口为`workflow.py`、`aligned_graph.py`、`operator_ir.py`和固定case的`costs`。它们保留同策略对照的运算/依赖口径，不能充作生产kernel精度证据。

| 字段／当前值 | 使用与设计原因 | 待验证范围 |
|---|---|---|
| `reward_cpu_ns=12` | 每个stage-local replica batch的rule-reward占位固定时长；两平台相同，保留host依赖 | 不是12ns/条的实测判分；真实规则解析、调度与共享CPU吞吐未校准 |
| `advantage_cpu_ns=8` | 每个stage-local batch的host优势计算占位；aligned另保留`5n+4g`的sample-std语义工作量计数 | 当前host_controller路径仍使用固定时长，不把语义计数假称为已收费的CPU/vector时延 |
| `sample_vector_ops_per_token=16` | 实际在每个输出token的vocabulary元素上乘16作为采样proxy | 不是已校准categorical sampler；无工具/多模态/投机解码 |
| softmax／RMSNorm／SwiGLU 每元素 `5／6／5` | 可审计标量表达式的统一向量op-count proxy，两平台同公式 | exp/sqrt/带宽/融合与真实kernel需要各自校验，不能只按FP32峰值证明精度 |
| `optimizer_vector_ops_per_local_parameter=12`、`gradient_clip_vector_ops_per_local_parameter=3` | legacy标量计数；新DO路径另按owned shard和显式读写建立KernelSpec，必须以实际提交IR为准 | 不由这两个case字段推断新DO总cost；状态所有权与真实融合Adam另核 |
| `control_marker_ns=1`、resume／sleep各`1000ns` | 保留正时长和阶段/版本fence，后两项为显式训推控制假设 | 非真实API/resume测量；GPU launch/Host/转换延迟敏感性与整步校准仍是G8前置项 |

因此本轮第一组结果称为**包含host占位阶段的受限IR控制实验**。若论文要主张实际veRL端到端cycle time，必须补齐这些host/向量/控制成本或明确收窄设备侧指标，不能因两边都用同一常数就宣称全部成本准确。

## 11. 验证计划与论文放行层级

| 门禁 | 实际要验证什么 | 通过后能说什么 | 失败时 |
|---|---|---|---|
| G0设计锁定 | model/revision、算法、W、policy、48active映射、成本/预算manifest | 明确这次究竟跑什么 | 参数仍未定就不启动大图 |
| G1算法/数据流 | std/zero variance、clip/KL/reduction小型oracle；实际IR读写/版本/有效token分母 | 指定算法合同与数据流一致 | 不用仅有数学函数PASS放行图 |
| G2DO/批处理 | 真实RS/AG、状态shard与bucket、normowner、真正微批和累积 | 所选并行/更新语义实际落地 | unsupported候选剔除并披露，不虚构候选时间 |
| G3严格迁移 | 同policy/W hash、48rank/group/dispatch/微批；6idle不偷算 | 硬件迁移对照成立 | 同active或同W不符就NOT_COMPARABLE |
| G4成本域 | 目标kernel域/精度/cache/launch/网络口径及未校准标记 | 条件性预测有明确成本解释 | 域外不静默fallback |
| G5预算 | 全量展开之前计数/估内存/ET/I/O并检查容器限制 | 有受限资源下的执行方案 | 不直接重启无上限任务 |
| G6回归 | 小例TP1/TP2/idle/通信/两轮版本与资源；加速前后完整事件/时钟/峰值对照 | 工程改动未改模拟语义 | 不以更少记录或少执行阶段换通过 |
| G7第一3B对比 | 两侧完整退出0、原始事件/资源/版本通过、独立重放与manifest归档 | 可交付“3B受限配置条件性模型结果” | OOM/timeout/磁盘失败是失败，不填性能数字 |
| G8论文主表 | 强GPU子空间与未覆盖优化披露、目标成本验证/敏感性、四模型同W、8K分布及容量失败 | 足以组成带范围限制的体系结构仿真实验章节 | 不称完整veRL部署、实体加速/质量保持 |

**独立审查意见：只有150测试、小型公式oracle、native小例、active-rank fingerprint、DO代码存在，还不足以放行论文主表。** 它们是G1/G2/G3/G6的必要证据。尤其SPtrue/overlap未覆盖、目标3B成本域/融合缓存未校准、8K分布未覆盖时，应先交付G7受限结果，不能越级宣布G8完成。晶圆没有实体H0标签时，可以做新架构的条件性仿真研究，但必须对参考组件验证、假设范围和敏感性负责。

## 12. 仿真墙钟、内存与预算护栏

模型时间与仿真程序实际耗时分开记录。后者至少拆成建图/成本预检、ET和输入写入、native、校验/报告、归档。旧3B数小时不是芯片模拟时间本身，也不能凭不同W的0.5B/3B两点外推参数量的幂律。

W1为3072trajectory，旧3B控制仅64trajectory/R32；按逐trajectory/decode直接展开有48倍trajectory、4倍decode的**计数放大风险**，不是“必然192倍耗时”的预测。R1K/R8K更应先做预算，不直接提交。[R17]

| 护栏 | 执行前必须保存 | 触发后的处理 |
|---|---|---|
| 节点/边/对象计数 | 每阶段唯一模板数、展开次数、真实物化/流式节点数、ET/JSON/日志字节估计 | 超批准预算先优化表示；不能少模拟样本后线性乘回完整step |
| RAM/CPU/PID | 无GPU的Docker；CPU4、RAM8GiB、memory-swap=8GiB（无额外swap）、PID512；planner/native均受约束 | 超限归budget_failed，保留失败现场；禁止自动加资源或后台swap续命 |
| 时间 | planner与encode各180s；native300s；post180s；整条命令900s | 任一阶段或总时限超出归budget_failed，不回填估计完成时间 |
| 磁盘 | 单候选输出≤2GiB；T42新运行合计≤12GiB；启动空闲≥24GiB（由30GiB显式修订，见下），运行中至少保留20GiB | 触阈值归budget_failed；共享盘空闲不是独占保证，不删他人或旧冻结证据 |
| 顺序 | 同时最多1个run；W0完整3B先做，W1/W2先过count/RSS/磁盘dry-run | 不重复启动同ID，不自动恢复旧7B/发布链 |
| 加速 | 模板/签名缓存、共享不可变ID、增量账、流式展开/解析 | 必须与展开参考在小/中规模同事件或证明等价；不悄悄去掉资源竞争 |

**上述上限已由父任务锁定**，机器值保存在`parameter_contract.json`，来源为项目工程约束[R21]，不是论文参数或已测运行时保证。它们用于控制共享服务器的有限成本，不意味着“8GiB肯定够”。先做W0完整3B；W1/W2必须先通过count/RSS/磁盘dry-run。若W0不满足预算，先优化IR与执行表示，不扩W、不自动增加资源，也不截短原W。所有超限候选归`budget_failed`，不作为数值/性能样本；W0成功仍不等于W1/W2完成。

**R21工程预算修订记录（v2）**：父任务现场日志报告共享NVMe空闲由约51GiB降至26GiB，而本轮T42仅占约1.7GiB；原30GiB启动门槛因此阻止了tiny验收。为使新任务可准入，启动空闲门槛显式从30改为24GiB，按`2GiB单候选监测阈值＋20GiB始终保留的动态reserve＋2GiB额外margin`计算。**动态空闲低于20GiB就停止自己的run，不删除他人文件或旧证据**。CPU、RAM、swap、PID、各阶段/整条timeout、单候选监测阈值/总输出预算均未增加；这只是共享环境的准入预算修订，不是硬件或算法参数变化，也不是新运行的时长保证。旧30GiB被阻止的命令/快照保留原样；51→26GiB的原始快照由父任务随后归档，本设计子任务未重新测量远端磁盘。[R21]

## 13. 统计、指标与失败公开

| 项目 | 定义/项目理由 | 参考及边界 |
|---|---|---|
| 完整step latency | 从指定同步step开始到所有active rank发布完成的native callback区间 | [R03]系统step概念；初次加载/编译是否计入须说明，不能混冷首步与稳态 |
| 生成吞吐 | 实际生成有效response tokens/Tstep；另报(P+R)tokens/Tstep | 两个分母口径并列，参考[R03]但不混同 |
| 训练吞吐 | 实际参与训练的非padding tokens/Tupdate，另报全step | 不把生成token当训练token或goodput |
| 阶段构成 | 全rank最早issue/最晚callback、关键路径、空闲等待、通信logical/network字节 | 阶段跨度含等待/端点/归约，不直接叫纯带宽时间 |
| 内存 | 每die/device的DRAM、逐core SRAM、KV、激活、DO/通信scratch峰值与生命周期 | 全片总容量不是局部可放置证明 |
| 确定性 | 同W重复只验证确定性，不能把5次相同仿真当n=5独立样本 | 项目统计约束；不伪造置信区间 |
| 分布性 | W3/公开统计W至少预先锁定5个独立输入窗口/seed，成对比较比值，报告所有点与范围 | 描述输入波动而非硬件误差；少数窗口不保证真实任务总体代表性 |
| Warm-up | 真GPU参考如有按来源划分warm-up/测量；仿真不机械套10+5重复制造样本 | [R03]10warm+5测量是该文实机方法，不继承为本模型已执行证据 |
| 失败/unsupported | 原命令、退出/信号、日志、资源峰值、源与配置SHA、失败类型均保存 | 不从commands删失败冒充单次成功；拼装需显式assembly和failed_attempts |
| 发布 | 先不可变source/input/image/result manifest，再MD与代码提交；commit不回填历史 | 不自动上传许可未明作者日志；报告负结果和未跑项 |

性能仿真不报告pass@1/time-to-quality提升；若未来做真实数值质量，独立固定题库、eval prompt、采样次数、训练预算与多seed，再给相应统计。不得用预设reward或仿真吞吐证明学习质量。[R01][R02][R16]

## 14. 执行顺序、停止条件与交付清单

1. 父任务审查本设计与机器合同，锁定已选算法、W0/W1、GPU候选范围、网络主/敏感性；按[R21]已经锁定的硬资源预算执行。源事实与项目override均保留，设计审查通过不代替运行门禁。
2. 先完成G1–G6：真实DO、微批、48active映射、成本域、预算计数与小例原生回归。未支持的SP/overlap候选明确留缺口。
3. 受限W0只验证全3B模型结构与所有新路径；成功后才按预算执行W1的GPU候选，冻结policy并跑WD。不中途根据wafer胜负重选GPU策略。
4. 保存失败/完整日志与独立重放后，产出第一3B受限比较；若W2/W3或strongGPU门禁未过，报告仍限G7，不把第一结果写成正式reasoning主表。
5. 继续W2/W3和7/14/32B的源/容量/成本/语义覆盖，再做WA、单变量消融与完整论文图表。纯training/serving及checkpoint扩展另立阶段，不抢当前主线。

**立即停止并保持未完成状态的条件**：参数/policy/W/源hash不一致；漏阶段/缺callback；内存越界、swap、timeout或磁盘阈值；域外成本静默外推；偷偷少生成/丢group/漏padding；将DO内存免费除DP；根据赢家改常量；完整证据还没保存就宣称成功。对应失败不填补性能数字，也不要求结果朝某一平台倾斜。

每个成功/失败run至少交付：model/data/W/algorithm/policy/hardware/成本来源manifest、source/image/上游版本、完整argv与退出、预算与墙钟分段、原始native事件/资源日志、所有检查、峰值与阶段指标、比较门禁、报告限制。设计→实现→执行三者的差异记录到同一验收入口，不以“文档里有字段”代替实际执行。

## 15. 来源入口

本轮设计修订已明示两项替代：R28纠正普通采样prefill LM head行数，旧全prefill工程结果不进入新主表；R21将新准入startup free disk由30修订为24GiB，保留20GiB动态reserve及其它资源上限。前者是执行算子合同纠偏，后者是环境预算调整，均不伪装成新性能结果。GSM8K锁定后的本地prompt审计另见[当前数据审计结论](RESULTS.md#数据与成本验证)，它仍不是rollout或学习质量证据。

引用键与逐项SHA/版本的权威清单为[reference_manifest.json](../experiments/2026-10-28-v0.9.5/design_sources.json)。文中每项“项目选择/假设/待验证”不被伪装成论文原文值。

[R01]: https://arxiv.org/html/2402.03300v3
[R02]: https://arxiv.org/html/2503.14476v2
[R03]: https://arxiv.org/html/2409.19256v2
[R04]: https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/config/actor/actor.yaml
[R05]: https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/ray_trainer.py
[R06]: https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/core_algos.py
[R07]: https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/config/engine/megatron.yaml
[R08]: https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/config/optim/megatron.yaml
[R09]: https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/examples/grpo_trainer/run_qwen2-7b_math_megatron_fsdp.sh
[R10]: https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/docs/perf/perf_tuning.rst
[R11]: https://arxiv.org/abs/2509.21009
[R12]: https://www.usenix.org/conference/osdi26/presentation/wu-tianyuan
[R13]: https://doi.org/10.1145/3695053.3731101
[R14]: https://arxiv.org/abs/2512.12279
[R15]: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
[R16]: ../experiments/2026-10-28-v0.9.5/design_sources.json
[R17]: ../experiments/2026-10-28-v0.9.5/design_sources.json
[R18]: ../experiments/2026-10-28-v0.9.5/design_sources.json
[R19]: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/aa8e72537993ba99e69dfaafa59ed015b17504d1/config.json
[R20]: https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/model_executor/layers/linear.py#L713-L765

[R21]: ../experiments/2026-10-28-v0.9.5/design_contract.json

[R22]: https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866
[R23]: https://arxiv.org/abs/2110.14168v2
[R24]: https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/examples/data_preprocess/gsm8k.py
[R25]: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/tokenizer_config.json
[R26]: https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/reward_score/gsm8k.py
[R27]: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/tree/65877096c24ffa7abc4e4fa5edb95cf3413a5674

[R28]: https://github.com/vllm-project/vllm/blob/aab549870df50edf0512f0a59b574f692f546465/vllm/model_executor/layers/logits_processor.py

**R21执行证据补充（v3）**：a7在2GiB监测阈值触发停止，原5秒终止宽限导致实际保留约2.416GB；这是失败，未获得性能结果，不隐去过冲。后续使用立即终止自己的容器缩短过冲，阈值仍2GiB、总预算仍12GiB、动态reserve仍20GiB。该用户态监测不是文件系统quota，实际结束字节必须记录。新专属`ROOT/runs-home-T42`位于有空闲的rootFS，沿用相同准入条件；旧NVMe `runs` symlink不变。各自已完成证据只在完整归档核验后回收，不能删除未知用户文件。

[R29]: https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/experimental/agent_loop/agent_loop.py

本文件是数值合同不变的仓库阅读版。R16–R18为项目既有审计材料的来源索引，原SHA保留，正文材料留在离线证据包；论文PDF不随仓库分发。设计中的“待执行”是锁定时状态，最新3B实际验收见[RESULTS.md](RESULTS.md)。
