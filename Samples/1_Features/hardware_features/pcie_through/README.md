# PCIe Through 特性介绍

## 简要描述

PCIe Through 是一种数据搬运优化特性：其本质是一种零拷贝技术， 允许Device直接访问Host端的内存（或Host直接访问Device内存），避免显式的Memcpy调用。当算子的输入/输出 Tensor 地址落在 Host 内存映射到的 Device地址范围内时，算子内部自动选择 PCIe 安全的 Tiling 路径，避免不支持的 DMA 操作，从而在 Host 内存直接参与计算的场景下保证功能正确性。

### 工作原理

1. **Host 内存注册映射**：通过 `aclrtMallocHost` 在 Host 侧分配内存，再使用 `aclrtHostRegister` 配合 `ACL_HOST_REGISTER_MAPPED` 标志将该内存映射到 Device 地址空间，获取对应的 Device 指针。
2. **PCIe through 判定**：如果要开启PCIe through特性能力，需先设置环境变量 `OP_PCIE_THROUGH_ACCESS_HOST_MEM_CHECK_ENABLE=1` 开启特性；再调用 `aclrtGetDeviceInfo` 查询 Host-Device 连接类型是否为 PCIe 连接（`ACL_HOST_DEVICE_CONNECT_TYPE_PCIE`）；最后通过aclrtHostRegister(ptr, size, ACL_HOST_REGISTER_MAPPED, &devPtr)将host内存注册并映射到device地址空间，判断Host地址到Device地址映射是否成功。如果均满足，则判定为 PCIe through 场景。
3. **算子 Tiling 路径选择**：在 PCIe through 场景下，GatherV2 算子内部强制走 `TILING_SIMD` 模式（纯 SIMD 计算，PCIe 安全），避免不支持的 DMA 操作。
4. **结果直接可见**：计算完成后，输出数据直接写入 Host 内存，无需额外 `aclrtMemcpy` 拷贝。

### 与普通调用的区别

| 对比项 | 普通调用 | PCIe Through 调用 |
|--------|----------|-------------------|
| 输入内存位置 | Device HBM（`aclrtMalloc`） | Host 内存映射（`aclrtMallocHost` + `aclrtHostRegister`） |
| 数据搬运 | 需要 `aclrtMemcpy` H2D/D2H | 算子直接通过 PCIe 访问 Host 内存，无需显式拷贝 |
| Tiling 路径 | 自动选择最优路径（SIMD/SIMT 等） | 强制 SIMD 模式（PCIe 安全） |
| 适用场景 | 常规推理/训练 | Host 内存直接参与计算（减少拷贝开销） |

## 支持架构

NPU ARCH 3510

## 算子实践

本样例以 GatherV2 算子为例，演示 PCIe Through 特性的完整调用流程。

### GatherV2 算子规格

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">GatherV2</td></tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">4 * 8</td><td align="center">float32</td><td align="center">ND</td></tr>
<tr><td align="center">index</td><td align="center">3</td><td align="center">int64</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">attr属性</td><td align="center">dim</td><td align="center">\</td><td align="center">int64</td><td align="center">\</td><td align="center">0</td></tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">3 * 8</td><td align="center">float32</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">aclnnGatherV2</td></tr>
</table>

- 计算公式：
  ```
  y[i][j] = x[index[i]][j]
  ```

### 关键 API 说明

| API | 作用 |
|-----|------|
| `aclrtMallocHost` | 在 Host 侧分配内存（后续可映射到 Device） |
| `aclrtHostRegister(ptr, size, ACL_HOST_REGISTER_MAPPED, &devPtr)` | 将 Host 内存注册并映射到 Device 地址空间，返回 Device 指针 |
| `aclrtGetDeviceInfo(devId, ACL_DEV_ATTR_HD_CONNECT_TYPE, &val)` | 查询 Host-Device 连接类型，判断是否为 PCIe 连接 |
| `aclCreateTensor(..., devPtr)` | 使用 Device 指针创建 aclTensor（该指针实际指向 Host 内存） |
| `aclnnGatherV2GetWorkspaceSize` | GatherV2 第一段接口（Tiling + 计算 workspace） |
| `aclnnGatherV2` | GatherV2 第二段接口（执行计算） |
| `aclrtHostUnregister` | 取消 Host 内存注册 |
| `aclrtFreeHost` | 释放 Host 内存 |

本样例提供两种调用方式，均以 GatherV2 算子为例演示 PCIe Through 特性：

| 样例 | 调用方式 | 说明 |
|------|----------|------|
| `pcie_through_gather_v2_aclnn` | aclnn 直调 | 通过 `aclnnGatherV2GetWorkspaceSize` + `aclnnGatherV2` 直接执行算子 |
| `pcie_through_gather_v2_kernel_launch` | <<<>>> kernel 直调 | 通过 `gather_v2_kernel<<<numBlock, nullptr, stream>>>()` 直接启动 kernel |

## 编译与运行

### 环境准备

在编译运行前，请先完成 CANN Toolkit 的安装与环境变量配置：

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

### 编译

本样例遵循 cann-samples 仓的统一 CMake 构建流程。在仓库根目录执行：

```bash
# 配置构建（以 Ascend950 为例）
cmake -S . -B build -DNPU_ARCH=dav-3510
```

#### aclnn 直调样例 & kernel 直调样例

这两个样例使用默认的 bisheng 编译器即可编译：

```bash
# 编译 aclnn 直调样例
cmake --build build --target pcie_through_gather_v2_aclnn

# 编译 kernel 直调样例
cmake --build build --target pcie_through_gather_v2_kernel_launch
```

编译产物位于：

```
build/Samples/1_Features/hardware_features/pcie_through/
```

### 运行

所有样例均通过环境变量 `OP_PCIE_THROUGH_ACCESS_HOST_MEM_CHECK_ENABLE=1` 触发 PCIe through 判定逻辑：

```bash
cd build/Samples/1_Features/hardware_features/pcie_through

# 运行 aclnn 直调样例
OP_PCIE_THROUGH_ACCESS_HOST_MEM_CHECK_ENABLE=1 ./pcie_through_gather_v2_aclnn

# 运行 kernel 直调样例
OP_PCIE_THROUGH_ACCESS_HOST_MEM_CHECK_ENABLE=1 ./pcie_through_gather_v2_kernel_launch
```

预期输出如下（两个样例输出格式类似，以 aclnn 直调为例）：

 ```
[Init] ACL initialized, device=0
[MallocHost] xHost=0x... idxHost=0x... outHost=0x...
[HostRegister] xDev=0x... idxDev=0x... outDev=0x...
[PCIeThrough] Detected PCIe through scenario, operator will use SIMD tiling path
[Tiling] workspaceSize=0
[Execute] GatherV2 completed
GatherV2 result (PCIe through):
  out[0] = 24.0
  out[1] = 25.0
  ...
[Verify] PASSED: output matches expected values
```

若输出 `[Verify] PASSED: output matches expected values`，说明结果正确。

### 核心代码片段

```c++
// 1. 在 Host 侧分配内存
aclrtMallocHost(&xHost, xBytes);

// 2. 将 Host 内存注册并映射到 Device 地址空间，获取 Device 指针
aclrtHostRegister(xHost, xBytes, ACL_HOST_REGISTER_MAPPED, &xDev);

// 3. 判定是否为 PCIe through 场景
//    通过环境变量 OP_PCIE_THROUGH_ACCESS_HOST_MEM_CHECK_ENABLE=1 开启，再查询 H2D 连接类型
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_HD_CONNECT_TYPE, &hdConnectType);
if (hdConnectType == ACL_HOST_DEVICE_CONNECT_TYPE_PCIE) {
    // PCIe through 场景，算子将走 SIMD tiling 路径
}

// 4. 使用 Device 指针创建 aclTensor
aclTensor *xTensor = aclCreateTensor(xShape.data(), xShape.size(), ACL_FLOAT,
                                     nullptr, 0, ACL_FORMAT_ND, nullptr, 0, xDev);

// 5. 调用 GatherV2
aclnnGatherV2GetWorkspaceSize(xTensor, dim, idxTensor, outTensor, &workspaceSize, &executor);
aclnnGatherV2(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);

// 6. 结果直接在 Host 内存可见，无需 aclrtMemcpy D2H 拷贝
```
