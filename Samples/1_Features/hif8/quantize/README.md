# Quantize算子直调样例
## 概述
本样例介绍Quantize算子的核函数直调方法，算子支持单核运行。
## 支持的产品
- Atlas A5 系列产品
## 目录结构介绍
```
├── quantize
│   ├── scripts
│   │   ├── gen_data.py          // 输入数据和真值数据生成脚本
│   │   └── verify_result.py     // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt           // 编译工程文件
│   ├── data_utils.h             // 数据读入写出函数
│   └── quantize_custom.asc    // AscendC算子实现 & 调用样例
```
## 算子描述
- 算子功能：  
  Quantize算子实现将数据量化为Hifloat8类型的功能。

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Quantize</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">1 * 2048</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">scale</td><td align="center">1 * 2048</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">offset</td><td align="center">1 * 2048</td><td align="center">float32</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1 * 2048</td><td align="center">hifloat8</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">quantize_custom</td></tr>
  </table>
- 算子实现：  
  Quantize算子的数学表达式为：
  ```
  y = (x / scale + offset).to(hifloat8)
  ```
  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。  

  Quantize算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm，scaleGm和offsetGm搬运到Local Memory，分别存储在xLocal、scaleLocal，offsetLocal，Compute任务负责对xLocal、scaleLocal，offsetLocal进行计算，计算结果存储在yLocal中，CopyOut任务负责将输出数据从yLocal搬运至Global Memory上的输出Tensor yGm中。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。
## 编译运行
- 配置环境变量  
  以命令行方式下载样例代码，master分支为例。
  ```bash
  cd ${git_clone_path}/Samples/1_Features/hif8/quantize
  ```
  请根据当前环境上CANN开发套件包的[安装方式](../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=/usr/local/Ascend/cann
    ```
  - 默认路径，非root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=$HOME/Ascend/cann
    ```
  - 指定路径install_path，安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=${install_path}/cann
    ```
  配置安装路径后，执行以下命令统一配置环境变量。
  ```bash
  # 配置CANN环境变量
  source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
  ```
- 样例执行
  ```bash
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..;make -j;             # 编译工程
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output_y.bin output/golden_y.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass
  ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2026/01/20 | 新增Quantize自定义算子用例 |