# Marketing Decision Lab · 课堂公式修正版

Streamlit网页，核心机器学习使用原来的R rpart/ranger。公式核对与首版修正见 FORMULA_AUDIT.md。

## 部署
1. 解压后把所有内容上传到GitHub仓库根目录，不上传ZIP或外层文件夹。
2. 必须包括 app.py、analytics.py、course_backend.R、packages.txt、requirements.txt、.streamlit/config.toml、demo_data和说明文件。
3. Streamlit Community Cloud入口填写 app.py。packages.txt安装R与rpart/ranger系统包，requirements.txt安装Python依赖。
4. 首次需要安装R依赖并训练5000棵树，等待加载完成。缺少R时会明确报错，不自动退回近似模型。
5. 若更新已有部署，上传全部新版文件并重启应用。

本地Linux可先安装 r-base-core r-cran-rpart r-cran-ranger，然后 pip install -r requirements.txt，streamlit run app.py。

## 内容
- RFM按课堂原始类别列计算，R sample()按75/25和种子12345拆分。
- 原rpart默认分类树与ranger 5000棵概率森林，种子888。
- 原订阅利润、盈亏平衡响应率、实际响应ROI；预算扩展单独标注。
- 产品周度Q8回归、Philips基准、Q9类别固定效应；Python OLS求解相同方程，非fixest接口。
- W6活动量Welch t检验、随机化平衡检查和处理组回归。
- M&S完整CAC漏斗与CLV折现公式。
- 原K-means迭代流程，不额外log/标准化。

## 数据与隐私
默认自动加载独立模拟数据；CSV列名遵循课堂。没有原始项目/课堂数据、作业文档或个人资料。上传时需符合模板列名，关联的产品/销量/营销三表应同时替换。R分析期间使用临时文件并清理。无外部AI/API请求；托管服务器仍会处理上传数据。

所有预测与模拟只演示方法。不能将历史响应ROI当成因果uplift，也不能将观察性价格回归当成已识别因果效果。
