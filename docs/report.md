# data_synthesis 工作

## 

## 1. 图生成流程

### 1.1 Step 0：MSC 目录准备
- 脚本：`scripts/00_build_msc_catalog.py`
- 作用：构建 MSC full/l1/l2 目录与描述映射，给后续节点标准化用。

### 1.2 Step 1：题库种子预处理
- 脚本：`scripts/01_prepare_math_seed.py`
- 作用：清洗原题库，得到 `seed.parquet`（qid, subject, level, y 等）。

### 1.3 Step 2：概念抽取
- 脚本：`scripts/02_extract_concepts.py`
- 输出：
  - `data/interim/concept_mentions.parquet`
  - `data/interim/concept_sets.parquet`
- 作用：从题目文本抽 topic/concept/msc 等概念信息。

### 1.4 Step 3：主节点与子节点构图（主节点目前使用msc code用来生成边，子节点再用concept细化一遍用来生成题目，保证题目的知识点更加精细更加贴近seed）
- 脚本：`scripts/03_build_nodes_and_seeds.py`
- 主节点：按 `node_mode`（当前为 `msc_code`）构建 `nodes.parquet`。
- 子节点：若 `use_subnodes=true`，在每个 MSC 内对 concept 聚类，生成：
  - `subnodes.parquet`
  - `seeds_sub_index.parquet`
  - `q2subnodes.parquet`
  - `subnode_map.parquet`

### 1.5 Step 4：先修边E_pre和语义边 E_sem 

脚本：scripts/04_build_edges_pre.py和 `scripts/05_build_edges_sem.py`

- 当前规则要点：

     #  先修关系先由共现程度和msc_code相似度 top_k筛选后 LLM投票（3vote ,LLM每次从a->b/b->a/none选）

  - 语义分数由词/字向量相似度 + 共现（这里参考KPDDS） + 层级相似融合得到，规则制定，不投票。
  - 候选 sem 边需满足：`跨 domain` 或者 `在 pre 图上的距离 >= 2`。（提供一些跳跃性和新题）

### 1.6 Step 5：先验与难度估计
- 脚本：`scripts/06_build_priors_and_difficulty.py`
- 输出：
  - `topic_cas.parquet`
  - `node_difficulty.parquet`
- 作用：融合 pre/sem 结构与种子难度，得到节点难度平滑估计。（对节点下属的qid题目难度用LLM投票后取平均，如果题目太少会在图上取临近节点作平滑）



## 2. 题目生成图算法
- 脚本：`scripts/10_generate_questions.py`

### 2.1 链路搜索
在主节点图上做 beam search（目标链长 `chain-len`（1-4均可，取决于难度和其它需求）），候选扩展策略与打分如下。

```text
Algorithm BeamSearchChain(start, L, beam, k_pre, k_sem, max_sem_edges):
    B <- { Chain(nodes=[start], sem_used=0, score=0) }
    for step in 1..L-1:
        P <- empty list
        for ch in B:
            u <- last(ch.nodes)
            sem_cross <- top-k cross-domain sem neighbors of u
            sem_same  <- top-k same-domain sem neighbors of u
            pre_next  <- top-k pre neighbors of u

            if ch.sem_used < max_sem_edges and sem_cross not empty:
                # 优先走跨域 sem
                for v in sem_cross:
                    P.add( extend(ch, v, edge_type='sem', sem_used+1) )
            else:
                # 常规 pre 扩展
                for v in pre_next:
                    P.add( extend(ch, v, edge_type='pre', sem_used same) )
                # 同域 sem 视作 pre-like 扩展
                for v in sem_same:
                    P.add( extend(ch, v, edge_type='pre', sem_used same) )

        score each chain in P by:
            s = -|d_hat(chain)-target_d| + 0.2*min(edge_conf) + 0.05*domain_switches
        B <- top `beam` chains in P by s

    return best chain in B
```

- 评分综合：链难度接近目标、边置信度、跨域切换等。

  

### 2.2 子节点映射与样例检索
- 若开启 `use_subnodes`：每个主节点选一个代表子节点（优先频次和覆盖）。
- 从 `seeds_sub_index` 按子节点检索题目样例，挑选一道例题并注入生成提示。

### 2.3 结构约束与重试（提高题目质量，强制要求生成solution_outline)
- 强制 `solution_outline` 每步带 `[NODE:...]` 标签，并覆盖整条链节点。
- 若缺标签、结构不合规或链覆盖不足，触发重试（含 tag 重试）。

### 2.4 难度判定
- 由 judge 模块根据题干+解纲估计 `d_pred`。
- 需满足 `|d_pred - target_difficulty| <= difficulty_tol`。

## 3. 目前图的统计结果

- 主节点（nodes）：`113`
- 子节点（subnodes）：`339`



## 4. 示意图（局部）

![local_graph_N0000035_both_clean](C:\Users\34775\Downloads\local_graph_N0000035_both_clean.png)

## 5. 示意题生成结果（双曲线题）

### 5.1 链路信息
- `chain_nodes`：`[N0000035, N0000087, N0000081, N0000024]`
- `chain_subnodes`：`[S0000297, S0000314, S0000296, S0000310]`
- 边类型：`pre -> pre -> pre`
- 目标难度：`0.80`

### 5.2 题目（原始）
- 题意：在双曲线
  \[
  -x^2 + 2y^2 - 10x - 16y + 1 = 0
  \]
  上找点 \(P\)，使过 \(P\) 与焦点中点的直线斜率等于焦点向量夹角的一半角正切 \(\tan(\theta/2)\)，并求全部精确坐标。

### 5.3 生成器判定结果
- `difficulty_eval.pass = true`

- `d_pred = 0.81`

### 5.4 示意题答案（人工复核）
原方程配方后为
$$
2(y-4)^2-(x+5)^2=6.
$$
记
$$
X=x+5,\quad Y=y-4,
$$
则双曲线为
$$
2Y^2-X^2=6.
$$
焦点为 $(0,\pm 3)$，焦点中点为原点。设 $u=\overrightarrow{PF_1},v=\overrightarrow{PF_2}$，则
$$
\tan\frac{\theta}{2}
=\frac{|u\times v|}{|u||v|+u\cdot v}.
$$
代入 $u=(-X,3-Y),v=(-X,-3-Y)$，并联立斜率条件
$$
\frac{Y}{X}=\tan\frac{\theta}{2},
$$
与双曲线方程可得
$$
X^2=2,\quad Y=\pm 2,\quad \text{且 } \frac{Y}{X}>0.
$$
因此只有两组解：
$$
(X,Y)=(\sqrt2,2),\;(-\sqrt2,-2).
$$
换回 $(x,y)$：
$$
\boxed{(x,y)=(-5+\sqrt2,\;6),\;(-5-\sqrt2,\;2)}.
$$



## 6. TODO
  ## (1)目前的节点里还是有一些含高等知识的节点（比如椭圆曲线，LLM可能直接理解成椭圆了），接下来可能还需要手工再筛，MSC以大学知识点为主，目前即使在代码里强制只保留05（组合）11（数论）这些MATH数据集真正涉及的，还是会有一些误判。

## (2)pre边的质量我还需要再考察一下，感觉有一些先修关系不太对，在想把pre边和同domain的sem边放在一个地位，然后提高pre边的判定标准，（这个地方LLM投票的准确率貌似也很不高），这里还需要1-2天再调整一下。

##  (3)目前问题的solution_outline低级错误较多，还需要prompt校正或者换题目生成的模型（目前用的deepseek api?)

