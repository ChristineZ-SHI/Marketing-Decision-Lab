# 公式与流程核对：课堂对齐修正版

## 核对依据
用户上传的 Coursework2(1).R、Coursework 2 Brief（Q1–Q14）、W6.R、week2_case clv_stu.R、app.R。仅使用方法与公式，未打包任何课堂原始数据或文档。

## 已修正的偏离

| 项目 | 首版问题 | 修正版 |
|---|---|---|
| 订阅利润 | 简化为手动输入 £20 | goods_revenue × (1−COGS) + subscription_fee − shipping_cost；默认 £19.99 |
| RFM | 直接生成合成RFM列 | recency=last；frequency=home+sports+clothes+health+books+digital+toys；monetary_value=electronics+nonelectronics |
| 拆分 | 60/20/20 分层拆分、额外验证选择 | 原 R sample()，75/25，不额外分层，set.seed(12345) |
| 决策树 | sklearn 自定义深度和叶节点 | 实际调用 rpart(..., method="class")，保持包默认值 |
| 随机森林 | sklearn 250棵、种子42、自定义深度 | 实际调用 ranger(..., num.trees=5000, seed=888, probability=TRUE)，训练前 set.seed(888) |
| ROI阈值政策 | 默认按预算截断，再选择模型 | 严格 predicted_probability > cost_per_offer / profit_per_customer，全选符合阈值客户；预算截断仅保留为明确标注的可选扩展 |
| Q1 blanket | 只展示预算内测试集随机触达 | 在整个模拟pilot上计算原Q1指标；另用同一测试集全触达策略公平比较Q4/Q5 |
| 回归口径 | 品牌周度总销量，额外季节项 | 恢复产品周度 sales ~ final_price + marketing_expense + brand，Philips参照组 |
| 数据合并 | 销量为主表后聚合 | products LEFT JOIN sales on product_id，再 marketing RIGHT JOIN 合并结果 on brand+week_id；final_price=RRP×(1−discount) |
| Q9 | 未展示原固定效应方程 | 屏幕尺寸分类项、价格、投放，控制brand/technology/resolution/energy_class/support_HDR/refresh_rate分类效应 |
| 产品利润 | 自创unit_cost与等销量假设 | 删除。原作业未提供成本，不虚构产品净利润 |
| A/B | 二元转化率、Bonferroni、自动赢家 | 恢复W6的活动量结果、年龄/性别/账号年龄/前期活动量平衡检查、Welch t检验、control基准处理组OLS |
| CLV | 手动输入CAC和年度贡献 | 恢复M&S点击→试用→会员漏斗，计算试用利润、配送成本、CAC，再算M−c和折现CLV |
| K-means | 新增log和标准化 | 恢复app.R原始特征与迭代机制：随机点初始化、平方欧氏距离、均值更新、空簇随机补点、1e−6收敛 |

## 原公式
- profit_per_customer = goods_revenue × (1−COGS) + subscription_fee − shipping_cost。
- break_even = cost_per_offer / profit_per_customer。
- total_profit = 实际响应人数 × profit_per_customer。
- total_cost = 触达人数 × cost_per_offer。
- ROI = (total_profit−total_cost)/total_cost；无触达时ROI无定义。
- CLV = Σ[t=1..N] (M−c) × r^(t−1)/(1+k)^t − CAC。
- M = membership + revenue_each_visit × profit_margin × n_visit。
- c = deliverycost_each_visit × n_visit。
- CAC = cost_per_click/(clicker_to_trier_rate×trier_to_member_rate) + 2×deliverycost_each_visit/trier_to_member_rate − [(revenue−promo)−revenue×(1−margin)+revenue×margin]/trier_to_member_rate。

## 数值核验
默认订阅利润 £19.99；break-even 约10.0050%。课堂M&S默认参数给出CAC £50、M £369、c £200、g £169，按上述留存及折现序列计算CLV。

## 实现边界
1. rpart/ranger及R随机采样采用原调用，不再用sklearn近似。R包版本与用户最初版本可能不同；合成数据不同，因此不会复现原作业的数值ROI。
2. Q8/Q9的线性方程以Python最小二乘求解，与完整秩、无权重情况下相同方程的系数一致。并未调用fixest；Q9用分类虚拟变量控制相同效应。传统OLS标准误不保证等于所有fixest版本/选项的推断输出。页面明确说明。
3. W6的Welch检验以SciPy实现对应R默认t.test(equal variance=FALSE)的数学公式；处理组回归是相同线性方程。
4. 模拟客户预测、敏感性曲线、额外AUC/Brier诊断、可选预算截断属于界面扩展，均标注。它们不改变原ROI流程。
5. 本版没有声称实现Q10–Q12的工具变量识别、Q14断点回归或Week1 NPV。未具备识别数据时不编造因果结论。
6. 所有CSV为独立生成的模拟记录。模拟生成机制是演示设计，不是课堂估计结果。品牌名只沿用题目基准与类别标签，不表示真实品牌销量或效果。
7. R计算使用临时CSV和模型文件，完成后自动移除；模型和结果由应用缓存保留在运行进程内。无外部AI调用。上传数据仍由托管服务器处理，公开展示应使用模拟数据。

## 交叉实现验证
本次独立调用R lm验证Q8系数和标准误，并调用R t.test验证A组活动量差异、p值和置信区间，与Python数学实现一致。原R rpart/ranger完成实际训练，8个页面通过运行检查。
