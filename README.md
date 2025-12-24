# ops-x

基于`PyTorch`原生扩展机制开发高性能昇腾NPU算子，基于Bisheng提供的CPU&NPU异构编程范式，构建`PyTorch`开源生态亲和的算子，为互联网开发者提供具备生产级质量的算子实现参考，助力模型快速迁移与性能优化。


## Features 特性
  - **与PyTorch无缝集成:** 基于PyTorch扩展能力构建，支持将**自定义算子**无缝接入PyTorch框架。
  - **Host & Device 异构编程:** 支持使用CUDA-like `<<<>>>` 语法调用NPU算子，实现高效编程。
  - **专注单文件开发模式:** 通过**单一C++文件**即可完成完整算子功能开发，大幅提升开发效率。


## Installation 安装

### Binaries 二进制安装
请从发布页面下载预编译的whl安装包，执行安装即可。
```sh
$ python3 -m pip install <PACKAGE_NAME>.whl
```

### From Source 从源码编译

#### Prerequisites 环境要求
  - Python 3.8 或更高版本
  - GCC 9.4.0 or 或更新版本
  - PyTorch 2.6.0 或更高版本
  - 对应版本的PyTorch Adapter
  - CANN软件包

对开发者的附加要求:
  - 需要掌握 `C++` 与 `AscendC` 编程能力


环境配置示例如下所示:
```sh
$ conda create -y -n <CONDA_NAME> python=3.10
$ conda activate <CONDA_NAME>
```
一个conda环境不是强制使用的，若系统已安装所有无法通过`pip`获取的必要依赖项，您也可以在标准虚拟环境(例如使用`uv`等工具创建)中执行`build`操作。

#### CANN 支持

为了使用CANN支持进行编译，请先选择受支持的`CANN`版本，并安装对应架构的 `Ascend-cann-toolkit_<VERSION>-{aarch64|x86_64}.run` 安装包。

随后配置CANN环境变量：
```sh
$ source /usr/local/Ascend/ascend-toolkit/set_env.sh
$ # or specified install path: source <INSTALL_PATH>/ascend-toolkit/set_env.sh
```

#### Get the Source Code 获取源码
```sh
git clone https://gitcode.com/cann/ops-ott.git
cd ops-x
```

#### Install Dependencies 安装依赖项
```sh
# Run this command from the ops-x directory after cloning the source code
python3 -m pip install -r requirements.txt
```

#### Install ops-x
```sh
python3 -m pip install --no-build-isolation -v -e .
```

#### Build wheel packages 构建whl包
```sh
python3 -m build --wheel -n
```

## Getting Started 快速开始

- 架构说明: 详解源代码结构与构建系统设计
- 开发示例: 提供完整自定义算子的简易开发范例
- 模块解析: 阐释算子实现中各核心模块的功能
- 测试指南: 指导如何编写新算子的测试用例并执行验证
