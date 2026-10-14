# 本地 Ultralytics 框架副本

本目录保留本项目使用的 Ultralytics 源码与 AKConv 注册改动。
上游项目：https://github.com/ultralytics/ultralytics

`pyproject.toml` 将此文件作为构建说明。安装时在项目根目录执行
`python -m pip install -r requirements-pc.txt`，其中核心依赖使用
`-e ./ultralytics-main` 安装本地副本。

框架源码及其派生改动沿用上游 AGPL-3.0 许可，详见本目录 `LICENSE`。
当前部署权重是否包含 AKConv 必须读取模型结构核实，不能由目录名判断。
