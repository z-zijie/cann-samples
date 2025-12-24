# ops-x

基于`PyTorch`原生扩展机制开发高性能昇腾NPU算子，基于Bisheng提供的CPU&NPU异构编程范式，构建`PyTorch`开源生态亲和的算子，为互联网开发者提供具备生产级质量的算子实现参考，助力模型快速迁移与性能优化。


## Features
  - **Seamless Integration with PyTorch:** Built upon PyTorch Extension capabilities, enabling **custom NPU operators** to be effortlessly integrated into the PyTorch framework.
  - **Host & Device Heterogeneous Programming:** Supports calling NPU operators using the familiar `<<<>>>` syntax (similar to CUDA), facilitating heterogeneous computation.
  - **Focused Single-File Development:** Entire operator functionality is developed through a **single C++ file**, streamlining development.


## Installation

### Binaries
Download the pre-compiled whl packages from the release page, then install it.
```sh
$ python3 -m pip install <PACKAGE_NAME>.whl
```

### From Source

#### Prerequisites
  - Python 3.8 or later
  - GCC 9.4.0 or newer is required
  - PyTorch 2.6.0 or later
  - Corresponding version of PyTorch Adapter
  - Supported version of CANN toolkits

Extra requests for developer:
  - understanding of `C++` and `AscendC` programming


An example of environment setup is shown below:
```sh
$ conda create -y -n <CONDA_NAME> python=3.10
$ conda activate <CONDA_NAME>
```
A conda environment is not required. You can also do a `build` in a standard virtual environment, e.g. created with tools like `uv`, provided your system has installed all the necessary dependencies unavailable as pip packages.

#### CANN Support

To compile with CANN support, select a supported version of `CANN`, then install the `Ascend-cann-toolkit_<VERSION>-{aarch64|x86_64}.run`.
Then set CANN environments:
```sh
$ source /usr/local/Ascend/ascend-toolkit/set_env.sh
$ # or specified install path: source <INSTALL_PATH>/ascend-toolkit/set_env.sh
```

#### Get the Source Code
```sh
git clone https://gitcode.com/cann/ops-ott.git
cd ops-x
```

#### Install Dependencies
```sh
# Run this command from the ops-x directory after cloning the source code
python3 -m pip install -r requirements.txt
```

#### Install ops-x
```sh
python3 -m pip install --no-build-isolation -v -e .
```

#### Build wheel packages
```sh
python3 -m build --wheel -n
```

## Getting Started

- Introducing: The source code and the build system.
- Examples: Easy to understand how to develop a custom op across all domains.
- Explaining: The function of each module in an operator implementation.
- Testing: How to write a test for a new custom op and run the tests.
