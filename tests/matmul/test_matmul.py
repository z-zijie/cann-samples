import torch
import torch_npu
import x
import pytest


def test_matmul_interface_exist():
    """
    Test that the 'x.matmul' operator is present in torch.ops.
    This existence test asserts that the custom operator registered under the
    'x' namespace is discoverable from Python via torch.ops.x.matmul.
    It does not exercise operator functionality — only that the Python binding
    and registration are available.
    Rationale:
    The presence of this test guards against a common failure mode where an
    operator is implemented and registered in C++/ATen but is not exposed to
    the Python torch.ops namespace due to mismatches between the PyTorch
    operator schema and the C++ registration signature (argument names, types,
    or overloads). Such schema/signature inconsistencies can cause the
    operator to be hidden or not exported to Python, breaking consumers that
    expect to call torch.ops.x.matmul. This test will fail loudly if the
    binding is missing, prompting investigation into schema/registration issues.
    """
    # This test specifically protects against discrepancies between the
    # PyTorch operator schema and the C++ signature/registration that can
    # prevent the operator from being visible in torch.ops.x.
    print(torch.ops.x.matmul)
    assert hasattr(torch.ops.x, "matmul"), "The 'matmul' operator is not registered in the 'torch.ops.x' namespace."


def cpu_matmul(input_tensor, weight_tensor, trans_a, trans_b):
    output_dtype = input_tensor.dtype
    middle_dtype = torch.float32
    input_tensor = input_tensor.to(dtype=middle_dtype)
    weight_tensor = weight_tensor.to(dtype=middle_dtype)
    if trans_a:
        input_tensor = input_tensor.transpose(0, 1)
    if trans_b:
        weight_tensor = weight_tensor.transpose(0, 1)
    cpu_result = torch.matmul(input_tensor, weight_tensor).to(dtype=output_dtype)
    print(f"CPU Result shape: {cpu_result.shape}")
    print(f"CPU Result: {cpu_result}")
    return cpu_result


def npu_matmul(input_tensor, weight_tensor, trans_a, trans_b):
    npu_result = torch.ops.x.matmul(input_tensor.npu(), weight_tensor.npu(), trans_a, trans_b)
    print(f"NPU Result shape: {npu_result.shape}")
    print(f"NPU Result: {npu_result}")
    return npu_result.cpu()


SHAPES = [
    (1024, 2048, 4096)
]


DTYPES = [
    torch.float16,
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.skipif(not torch.npu.get_device_name().startswith('Ascend910_95') , reason="Only support Ascend910_95")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_operator(shape, dtype):
    """
    test torch.ops.x.matmul
    """
    # 创建输入张量
    m, k, n = shape
    input_tensor = torch.randn((m, k), dtype=dtype)
    weight_tensor = torch.randn((n, k), dtype=dtype)
    trans_a = False
    trans_b = True

    cpu_result = cpu_matmul(input_tensor, weight_tensor, trans_a, trans_b)
    npu_result = npu_matmul(input_tensor, weight_tensor, trans_a, trans_b)
    compare_result = torch.allclose(cpu_result, npu_result, rtol=1e-03, atol=1e-03, equal_nan=True)
    assert compare_result
