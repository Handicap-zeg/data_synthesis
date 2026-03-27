# Data-Centric Math Question Synthesis

---

## 3. 离线资产构建

### 3.1 知识点与种子题

- 节点：细粒度知识点（携带领域标签）。
- 种子：MATH / AMC / AIME 等数据中的题目与解答。
- 每题抽取 `K(q)`（知识点集合），建立 `Seeds(v)` 索引。

### 3.2 轻量图结构

构建两类边：

- `E_pre`：前置依赖边（DAG），来源于规则 + LLM 多投票。
- `E_sem`：语义近邻边（无向或弱有向），来源于 embedding 相似度。

保留原则：
- 仅保留高置信边。
- 对 `E_pre` 强制无环。
- 对 `E_sem` 限制每节点 top-k（控制噪声）。

### 3.3 先验缓存

 **CAS（Concept Affinity Score）**：

\[
CAS(t_i,t_j)=\lambda_1 A_{sem}+\lambda_2 A_{pre}+\lambda_3 A_{link}
\]

其中默认 `\lambda=(0.50, 0.30, 0.20)`，并定义：

- `A_sem = (1 + cos(μ_ti, μ_tj))/2`，`μ_t` 为 topic 质心 embedding。
- `A_pre = exp(-d_pre(t_i,t_j)/τ)`，`d_pre` 为 `E_pre` 上 topic 间最短依赖距离，不可达记为 0，默认 `τ=2`。
- `A_link = |E_sem(t_i,t_j)| / min(k|t_i|, k|t_j|)`，表示跨 topic 的语义边密度（裁剪到 `[0,1]`）。

额外缓存：
- `cov(v)`：节点覆盖计数。
- `d_v`：节点难度。
- `d_hat(C)`：链难度。

### 3.4 `d_hat` 可复现计算标准

#### 3.4.1 题目难度归一化 `y(q)`

统一映射到 `[0,1]`：

- MATH level `l in {1..5}`：`y(q)=(l-1)/4`。
- AMC10：`y(q)=0.35`。
- AMC12：`y(q)=0.45`。
- AIME：`y(q)=0.70`。

#### 3.4.2 节点难度 `d_v`

- 若 `|Seeds(v)| >= 15`：
  `d_v = median({y(q) | q in Seeds(v)})`。
- 若样本不足：用图平滑补全（迭代 10 轮）：

\[
d_v \leftarrow \frac{\eta\,d_{dom(v)}+\sum_{u\in N(v)} w_{uv} d_u}{\eta+\sum_{u\in N(v)} w_{uv}},\ \eta=2.0
\]

其中 `d_dom(v)` 为该领域节点难度中位数，`w_uv` 使用边置信度。

#### 3.4.3 链难度 `d_hat(C)`

设 `C=(v_1,...,v_m)`，定义四个特征：

- `f_node = mean(d_v)`
- `f_len = (m-2)/2`（当 `m in [2,4]` 时落在 `[0,1]`）
- `f_jump = mean(1-conf(e_i))`（语义边默认 `conf=0.5`）
- `f_cross = (#domain_switches)/(m-1)`

最终：

\[
d_{hat}(C)=clip(0.55 f_{node}+0.15 f_{len}+0.15 f_{jump}+0.15 f_{cross}, 0, 1)
\]

难度桶默认阈值：
- Easy: `[0, 0.38)`
- Medium: `[0.38, 0.68)`
- Hard: `[0.68, 1.00]`

每月用最近一批通过 Lean4 的样本做一次单调校准（isotonic regression），仅更新阈值，不改主公式。

---

## 4. 算法

### 4.1 算法名

**RNBS：Robust Novelty Beam Search**

目标：在同一算法下同时支持“巩固题”和“跨域桥接题”，并保持低复杂度。

### 4.2 输入与输出

输入：
- 图 `G=(V, E_pre, E_sem)`
- 目标难度 `d_target`
- 最大链长 `L<=4`
- 束宽 `B`

输出：
- 候选知识链集合 `Cands = {C_1, ..., C_m}`

### 4.3 统一扩展策略

每一步对当前链 `C` 扩展一个新节点 `u`：

- 邻居集合：
  - 巩固模式：`N = Out_pre(last(C)) ∪ TopSem(last(C))`
  - 桥接模式：若需跨域，强制 `dom(u) != dom(last(C))`（至少一次）

- 候选打分：

\\[
S(C \oplus u)=w_1 S_{novel}+w_2 S_{comp}+w_3 S_{diff}+w_4 S_{edge}
\]

其中：
- `S_novel = 1 / (1 + avg(cov(v), v in C⊕u))`
- `S_comp = mean(CAS(topic(v_i), topic(v_{i+1})))`
- `S_diff = -| d_hat(C⊕u) - d_target |`
- `S_edge = min(edge_confidence on path)`

默认权重：`w = (0.35, 0.25, 0.20, 0.20)`。

### 4.4 鲁棒性过滤（关键）

对 Top-M 候选链做三重鲁棒过滤：

1. 结构扰动一致性：
- 随机丢边（例如 10%）重复 `R=20` 次。
- 候选链可重构率 `>= 0.8` 才保留。

2. 语义可教性一致性：
- 3 次独立 LLM 判定“能否自然融入同一道题的解题过程”。
- 通过阈值：`>=2/3`。

3. 去重与反模板化：
- 与历史链做 MinHash/Jaccard 去重。
- 与种子题 embedding 相似度过高则丢弃（避免改写型伪新题）。

### 4.5 伪代码

```text
RNBS(G, d_target, mode, L=4, B=16):
  beam <- [[sample_start_node()]]
  for step in 1..L-1:
    pool <- []
    for C in beam:
      for u in candidate_neighbors(C, mode):
        if violate_topology(C, u): continue
        C2 <- C + [u]
        pool.add((C2, score(C2, d_target)))
    beam <- topB(pool)

  cands <- topM(beam)
  cands <- robust_filter(cands)   # 扰动一致性 + 语义一致性 + 去重
  return cands
```

复杂度近似：`O(L * B * deg)`，工程上可稳定控制。

---

## 5. 链路检验与题目生成

### 5.1 链路检验 Gate（生成前）

对每条候选链按代价递增执行：

1. `Gate-T` 拓扑合法：`E_pre` 不逆序，不含环冲突。
2. `Gate-N` 新颖性：不与历史链/题高重合。
3. `Gate-D` 难度区间：`d_hat(C)` 落入目标桶。
4. `Gate-C` 兼容性下界：链上相邻 topic 的 `CAS` 均值需 `>= θ_c`（默认 `0.45`）。
5. `Gate-R` 可解性草检：LLM 给出“最短解题骨架”（3-6 步），失败则拒绝。

### 5.2 题目合成

- 输入：通过 Gate 的链 + 对应 seeds 示例。
- 输出：`problem, answer, solution_outline, used_concepts`。
- 要求：每个链节点在 `solution_outline` 中必须出现对应作用点。

---

## 6. Lean4 二次校验（链路后主路径）

### 6.1 校验目标

对每道生成题，验证三件事：

1. 一致性（无自相矛盾条件）。
2. 可满足性（至少存在一个解）。
3. 答案正确性（给定答案满足约束；若应唯一则证明唯一）。

### 6.2 形式化流程（HERALD 风格）

1. `NL -> FL`：将题目翻译为 Lean4 声明（含依赖检索）。
2. 构造校验定理：
- `theorem wellposed : ∃ x, constraints x`
- `theorem answer_ok : constraints proposed_answer`
- 可选：`theorem unique_ans : ∃! x, constraints x`
3. Lean4 编译与 proof check（超时控制）。
4. 对通过样本执行回译 + NLI 一致性检查（软约束）。

### 6.3 判定分级

- `HIGH`：Lean4 通过（`wellposed` + `answer_ok`，必要时 `unique_ans`）。
- `MEDIUM`：Lean失败但多模型一致性通过（兜底）。
- `REJECT`：两路都失败。

### 6.4 失败回退策略

- 若 Lean 报语义缺失：补充显式定义后重试 1 次。
- 若 Lean 报不可满足：回传链路层，标记该链为“不可出题链”。
- 若 Lean 报答案不成立：触发重解与再验证。

---

## 7. 难度分层与课程式出数

按 `d_hat(C)` 与验证信号分桶：

- Easy：`d_hat < 0.38`，短链、单域优先。
- Medium：`0.38 <= d_hat < 0.68`，中链、弱跨域。
- Hard：`d_hat >= 0.68`，跨域桥接、形式化复杂。

训练建议：`Easy -> Medium -> Hard` 逐阶段混入。

---

## 8. 评测与消融（必须项）

### 8.1 质量指标

- Correctness：答案正确率（Lean / 人审）。
- Novelty：与种子题去重后新颖率。
- Robustness：图扰动下通过率下降幅度。
- Usefulness：下游 SFT/RL 性能增益。

### 8.2 消融实验

至少做四组：

1. 去掉结构扰动一致性。
2. 去掉 `CAS` 兼容性项。
3. 去掉 Lean4（仅投票验证）。
4. RNBS vs 旧版 PCRW+Bridge。



