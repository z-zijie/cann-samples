import torch
import torch_npu
import x
import pytest


def test_add_interface_exist():
    """
    Test that the 'x.add' operator is present in torch.ops.
    This existence test asserts that the custom operator registered under the
    'x' namespace is discoverable from Python via torch.ops.x.add.
    It does not exercise operator functionality — only that the Python binding
    and registration are available.
    Rationale:
    The presence of this test guards against a common failure mode where an
    operator is implemented and registered in C++/ATen but is not exposed to
    the Python torch.ops namespace due to mismatches between the PyTorch
    operator schema and the C++ registration signature (argument names, types,
    or overloads). Such schema/signature inconsistencies can cause the
    operator to be hidden or not exported to Python, breaking consumers that
    expect to call torch.ops.x.add. This test will fail loudly if the
    binding is missing, prompting investigation into schema/registration issues.
    """
    # This test specifically protects against discrepancies between the
    # PyTorch operator schema and the C++ signature/registration that can
    # prevent the operator from being visible in torch.ops.x.
    print(torch.ops.x.add)
    assert hasattr(torch.ops.x, "add"), "The 'add' operator is not registered in the 'torch.ops.x' namespace."


# 定义测试参数
SHAPES = [
    (1,),
    (3,),
    (10,),
    (100,),
    (1024,),
    (10000,),
    (10, 10),
    (32, 32),
    (100, 100),
    (10, 100),
    (100, 10),
    (256, 512),
    (5, 10, 15),
    (16, 32, 64),
    (32, 64, 128),
    (1, 3, 32, 32),
    (4, 3, 64, 64),
    (8, 3, 128, 128),
    (1000, 1000),
]

DTYPES = [
    torch.float32,
    torch.float16,
    torch.int32,
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_add_operator(shape, dtype):
    """
    测试 add 操作符的功能，使用精简但覆盖广泛的形状和数据类型组合。

    参数:
        shape: 张量形状
        dtype: 数据类型
    """
    # 创建输入张量
    if dtype in [torch.int32]:
        # 整数类型使用整数随机值
        a = torch.randint(-100, 100, shape, dtype=dtype)
        b = torch.randint(-100, 100, shape, dtype=dtype)
    else:
        # 浮点类型使用正态分布随机值
        a = torch.randn(*shape, dtype=dtype)
        b = torch.randn(*shape, dtype=dtype)

    # 计算期望结果
    expected = a + b

    # 将数据移动到NPU
    a_npu = a.npu()
    b_npu = b.npu()

    # 同步确保数据已传输
    torch.npu.synchronize()

    # 调用自定义操作符
    result_npu = torch.ops.x.add(a_npu, b_npu)

    # 同步确保计算完成
    torch.npu.synchronize()

    # 将结果移回CPU
    result = result_npu.cpu()

    # 验证结果
    if dtype in [torch.int32]:
        # 整数类型需要精确匹配
        assert torch.equal(result, expected), \
            f"Add failed for shape {shape}, dtype {dtype}. " \
            f"Expected {expected}, but got {result}"
    else:
        # 浮点类型使用统一容差比较
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), \
            f"Add failed for shape {shape}, dtype {dtype}. " \
            f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"

    # 可选：打印成功信息（调试用）
    print(f"✓ Test passed: shape={shape}, dtype={dtype}")
