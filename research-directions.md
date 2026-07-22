# DDPM 项目选题调研（2026-07-22）

> **✅ 已定方向（2026-07-22）：CIFAR-10 单卡像素空间 Speedrun** —— 从零复现 DDPM（FID 3.17 锚点），
> 逐步现代化至 EDM 级配方，发布锁死评测协议的开源 time-to-FID 基准。
> 下文保留完整调研记录备查；§四"复现纪律"直接适用于本方向。

> 方法：6 个角度并行文献调研（每个 ≥5 轮检索 + 原文核对），产出 18 个候选选题，
> 再由两个独立"怀疑者"对抗校验：一个专门推翻 GPU 时数估计（含 FID 评测成本），
> 一个专门在 arXiv 2022–2026 上找"已经有人做过"的证据。
> 最终计分：0 个新颖性被推翻，10 个 sound，8 个 questionable。
> 目标画像：单卡 RTX 5090（32GB），从零复现 + 受控消融，全程 ≤150 GPU 时，
> 论文形状对齐上一篇《Capacity Returns Are Conditioned on Data Quality》（WMT 复现 + 2×2 消融）。

## 一、2026 年中的领域地图（六个角度的结论）

1. **基线**：小规模扩散的"正确形式"已收敛到 EDM（Karras 2022, arXiv 2206.00364）：
   预条件化 + log-normal 噪声采样 + Heun 35-NFE，CIFAR-10 FID 1.97（无条件）/1.79（条件）。
   EDM2（arXiv 2312.02696, CVPR 2024）加了 magnitude-preserving 层和 **post-hoc EMA**。
   原版 DDPM（FID 3.17，arXiv 2006.11239）在单卡 5090 上约 15–25h 可复现
   （锚点：tqch/ddpm-torch 复现到 FID 3.188；FutureXiang/Diffusion 用 4×3080Ti 14h 完成 EDM，FID 2.06/2.22）。
2. **开放问题**：Kingma-Gao 的 ELBO 观点说时间步采样分布只应通过其诱导的损失加权起作用，
   但 2024–26 的论文（Improved Noise Schedule 2407.03297 等）不断报告"改采样密度优于等价重加权"——
   至今没有方差受控的干净因子分解实验。REPA 一线在像素空间还有未解释的符号翻转。
3. **数据质量 × 容量**：该方向被方法型论文主导（Ambient Diffusion → Ambient-o 2506.10038），
   实证研究都是单轴的：2411.02780 变数据量×噪声但不变容量；2405.20494（NeurIPS 2024 spotlight，
   "轻度腐蚀反而有益"）变腐蚀类型/强度但容量不是受控轴。**容量×质量交互在图像扩散上是真空白**。
4. **少步采样**：三大 from-scratch 家族（consistency/iCT/sCM、shortcut、MeanFlow 及其 2025-26 后继）
   数字漂亮但对比文献一团糟——没有匹配算力的公平对比，且所有少步方法只在干净数据上训过。
   注意：**iCT from-scratch 官方配方约 8 卡一周（1300+ GPU 时）**，单卡小预算复现必须承认 under-training。
5. **FM vs Diffusion**：理论共识（DeepMind "Diffusion Meets Flow Matching" 博客 + ICLR 2025 blog track）
   是两者同一模型类，仅差参数化/加权/采样器。但"匹配加权后差距归零"这个可证伪预测**没有人做过实验验证**。
   SiT（2401.08740）是顺序式而非因子式，从不交叉匹配加权。
6. **评测方法论**："The FID Lottery"（arXiv 2606.20536, 2026-06）刚用 10 万 H100 时量化了 FID 的种子随机性
   （重训练比重采样的方差大 3.2×），但只测了 SiT/ImageNet-256 一个设定，明确把其它规模列为未知。
   clean-fid（2104.11222）的插值坑、EDM 的 min-of-3 协议、EMA 长度敏感性（EDM2）是所有消融实验的噪声地板。

## 二、幸存选题（可行性 + 新颖性双 SOUND）

### ★ A.（推荐）Does Capacity Return Change Sign with Data Quality? — 图像扩散版 2×2
- **问题**：匹配训练预算下，额外容量在干净数据上提升 FID、在污染数据上收益缩小甚至反号吗？
  （上一篇 MT 论文核心发现的跨模态检验；分类任务已有 2208.08003 证明反号，生成任务空白）
- **设计**：CIFAR-10 无条件，DDPM UNet + EDM 预条件化提效。
  容量 {Small ~12–15M, Big ~50–56M} × 质量 {干净 50k, 50% 按 Ambient-o 腐蚀集污染（模糊/JPEG/噪声）}，
  预算匹配 ~20M images seen，主 2×2 × 3 seeds；副实验：污染比例扫描 {0,25,50,75%} 定位过零点。
  指标：FID-50k 主指标 + precision/recall（保真 vs 覆盖分解，对应 MT 论文的 correctness×alignment 分解）。
- **校验后预算**：~75–110h（含校验者建议增加的 clean-25k 数据量控制组，堵住"污染=减少干净数据量"的混淆）。
- **开工前必读**：2405.20494 全文附录（确认没有埋藏的容量交叉分析）、2411.02780、2208.08003。
- **风险**：CIFAR 规模下两个尺寸可能被污染同等伤害（无交互）→ 剂量扫描把问题转为"过零点在哪"的测量；
  诚实负结果也可发。

### B. Same Coin, Different Change? — Diffusion–FM 等价性的实验压力测试
- **问题**：DDPM(VP, ε-pred) 与 FM(线性插值, v-pred) 在匹配架构/算力/采样器、
  **且在 log-SNR 空间交叉匹配有效损失加权**后，FM 优势是否归零？残差多少归于输出参数化？
- **设计**：2×2 {formulation} × {native vs cross-matched weighting}，~25–35M UNet，CIFAR-10 主 + FFHQ/AFHQ-64 一对确认组，
  2 seeds，同一 DDIM=Euler-flow 确定性采样器，NFE ∈ {10,35,100}，交付"gap 分解条形图"。
- **校验后预算**：~100–140h（校验者修正了 FID 评测 8–15h 与 FFHQ 单组 15–25h）。
- **新颖性**：理论预测明确、无人实验验证；须引用并区分 FasterDiT（2410.10356，最重要的未引近邻，做过
  四种 schedule×formulation 但从不等化有效加权）。
- **风险**：匹配组可能统计不可分（种子噪声 ~0.1–0.3 FID）→ "何时参数化开始起作用"（NFE/Heun 扫描）保底。

### C. Few-Step Fragility — 数据质量 × 采样步压缩
- **问题**：consistency/MeanFlow 这类自举目标的少步方法，对训练数据腐蚀是否比多步扩散**更**敏感？
  （结构上与上一篇论文同构："X 的收益取决于数据质量"，X 从容量换成步数压缩）
- **设计**：{干净, 腐蚀} × {EDM 35-NFE, consistency training, MeanFlow}，同 UNet 同预算，2 seeds，
  终点是各方法内部的 ΔFID（对弱绝对 FID 稳健）；附加探针：从腐蚀数据教师蒸馏 vs 直接在腐蚀数据上 CT
  （呼应上一篇的 QE 过滤负结果——教师是不是数据质量过滤器）。
- **校验后预算**：100–140h（含值得保护的教师-过滤探针）。
- **必须写进论文的 caveat**：单卡预算下 CT/MeanFlow 相对官方配方 under-train 50–100×，
  方法敏感性与欠训练部分混淆（组内对比缓解但不消除）；MeanFlow 的 JVP+大 batch 需先跑显存试点。
- **参考**：2502.01441 已证明 CT 对数据统计异常敏感（用 Cauchy loss 修）——假设有据，惊喜度略降；
  开工前查 2604.23552（CD 的记忆化）。

### D. The Price of a Number — 单卡评测算力核算（方法论笔记，最便宜）
- **问题**：每 GPU 分钟评测算力，哪个指标（clean-FID / FD-DINOv2 / CMMD / PRDC）在哪个样本量下
  给出最小可靠可检差异？诚实评测占小规模论文总算力的多大比例？
- **设计**：4–6 个已知质量梯度的 checkpoint（自训 + 公开 EDM），200k+ 样本池 bootstrap 出
  "评测成本 vs 最小可检效应"的 Pareto 前沿；检验 CMMD 的省样本声明在 32×32（CLIP 分布外）是否反转。
- **校验后预算**：~40–70h。**若做任何主选题，这个可以先跑——它就是主选题的评测基础设施。**
- **风险**：形状偏 TMLR/workshop（"finding 在哪"）；适合做配套而非唯一主线。

### 另一个值得记住的：E. Slight Corruption Helps — But Whom?（新颖性 questionable 但便宜）
- 2405.20494 的"轻度条件腐蚀有益"效应做容量交叉的剂量-响应（{0,2.5,10,25%} 标签翻转 × {12M,50M}），
  ~60–75h（校验者建议把 CEP 嵌入扰动组当核心）。是 A 的天然姊妹实验/扩展章节，单独成文偏薄。

## 三、被砍掉/降级的（附原因）

| # | 题目 | 判定 |
|---|------|------|
| 0 | EDM 组件阶梯 × 预算 | 可行性**被推翻**（3 seeds 全网格 = 375h）；砍种子可救，新颖性 sound |
| 1 | CIFAR-10 像素空间 speedrun 基准 | 新颖性 sound，但 1000 步采样验证成本被低估；工程味重 |
| 2 | EMA×评测协议审计 | 可行，但 clean-fid/EDM2/FID Lottery 已各占一轴 |
| 4 | REPA vs Dispersive at small scale | PixelREPA(2603.14366) + iREPA(2512.10794) 已把机制做掉了 |
| 5/8 | 重复数据 × 容量记忆化 | "Bigger Isn't Always Memorizing"(2505.16959) 等三面夹击，缝隙在变窄 |
| 9/11 | 少步方法匹配算力对比 / 蒸馏总账 | 新颖性 sound 但预算算不平（~185h/165h），且 from-scratch CT 欠训练问题最重 |
| 13 | 时间步分布 × formulation 因子分解 | FasterDiT(2410.10356) 覆盖大半 |
| 14 | FM 优势 × 容量 | SiT 已报告"FM 全尺寸一致更好"，预期无交互 |
| 15 | 小规模 FID Lottery | FID Lottery(2606.20536) 框架已立，只剩"换个规模重跑" |
| 16 | 指标排序一致性 | 新颖性 sound 但评测成本漏乘（12 runs × 4 NFE × 2 EMA × …） |

## 四、无论选哪个都适用的复现纪律（从调研中提炼）

- FID 协议锁死：clean-fid 实现、50k 样本 vs 训练集统计、EDM 式 min-of-3；否则数字不可比。
- EMA 是一等公民混淆源：存快照、用 EDM2 post-hoc EMA 扫描，每格报自己的 best-EMA FID。
- 增广要用 EDM 的 non-leaky（增广条件化）版本，否则水平翻转会漏进生成样本。
- bf16 + torch.compile 是 5090 上的默认；EDM 官方 fp16 需要显式 loss scaling (--ls=100)。
- 吞吐锚点：~5M images/h（35M UNet，CIFAR-10，单卡 5090 bf16），由 STF(2302.00670) 与
  FutureXiang/Diffusion 两个独立来源交叉验证。
