# 本地复现约定

- 本目录是独立 Git 仓库。实验流程以父项目 `../../ai_read/实验通用原则.md` 为准。
- 新实验先在父项目 `experiment/` 创建日期和编号目录及 `time.md`，再在本仓库创建同名分支；持续中文提交，小样本检查后按规范 squash 到主分支，提交消息使用实验分支名。
- 实验总结、笔记、日志和图像放在父项目对应实验目录；本仓库保留代码、依赖声明及必要操作约定。入口必须显式指定父项目 `experiment/` 下的 `--output` / `-Output`。
- 初始化提交 `9e6de81` 保留导入的官方源码；历史来源信息见 `../../experiment/exp_1107_@01_lampilot_environment/logs/source/source.json`。
- 历史实验提交映射和流程缺口见 `../../experiment/commit_history.md`；缺失时间、实验分支及 squash 不倒填、不伪造。
- 父项目的 Git 暂存和提交由用户决定，不自动执行。
- 当前目标：Windows 11，Conda 环境 `lampilot_py311`，DeepSeek API，固定场景种子的小规模连通验证。
- 不要求对齐原论文模型和数值，不进行多随机种子实验，不下载模型权重。
- 优先保留官方提示词、解析器、控制器和评分器；适配入口放入 `reproduction/`。
- 禁止提交 API Key、`.env`、日志和结果缓存。GitHub 访问使用 `C:\Program Files\GitHub CLI\gh.exe`。
- 将 Demo 子集结果明确称为小规模连通验证，不称为原论文独立测试集成绩。
