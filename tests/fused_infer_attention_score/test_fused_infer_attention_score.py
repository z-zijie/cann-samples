import torch
import torch_npu
import x
import math
import pytest
from compare import compare


def test_fused_infer_attention_score_interface_exist():
    """
    Test that the 'x.fused_infer_attention_score' operator is present in torch.ops.
    This existence test asserts that the custom operator registered under the
    'x' namespace is discoverable from Python via torch.ops.x.fused_infer_attention_score.
    It does not exercise operator functionality — only that the Python binding
    and registration are available.
    Rationale:
    The presence of this test guards against a common failure mode where an
    operator is implemented and registered in C++/ATen but is not exposed to
    the Python torch.ops namespace due to mismatches between the PyTorch
    operator schema and the C++ registration signature (argument names, types,
    or overloads). Such schema/signature inconsistencies can cause the
    operator to be hidden or not exported to Python, breaking consumers that
    expect to call torch.ops.x.fused_infer_attention_score. This test will fail loudly if the
    binding is missing, prompting investigation into schema/registration issues.
    """
    # This test specifically protects against discrepancies between the
    # PyTorch operator schema and the C++ signature/registration that can
    # prevent the operator from being visible in torch.ops.x.
    print(torch.ops.x.fused_infer_attention_score)
    assert hasattr(torch.ops.x, "fused_infer_attention_score"), "The 'fused_infer_attention_score' operator is not registered in the 'torch.ops.x' namespace."


ARGS = [
    (1, 1, 1, 128, 128, 128)
]


DTYPES = [
    torch.bfloat16,
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.skipif(not torch.npu.get_device_name().startswith('Ascend910_95') , reason="Only support Ascend910_95")
@pytest.mark.parametrize("args", ARGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_infer_attention_score_operator(args, dtype):
    """
    test torch.ops.x.fused_infer_attention_score
    """
    b, n1, n2, s1, s2, d = args
    torch.no_grad()
    q = torch.rand((b, n1, s1, d), dtype=torch.bfloat16)
    k = torch.rand((b, n2, s2, d), dtype=torch.bfloat16)
    v = torch.rand((b, n2, s2, d), dtype=torch.bfloat16)
    scaleValue = 1 / math.sqrt(d)
    mask = ~torch.tril(torch.ones(s1, s2)).to(torch.bool)
    enable_gqa = n1 != n2
    cpu_result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~mask, scale=scaleValue,
                                                                  enable_gqa=enable_gqa)

    q = q.to('npu')
    k = k.to('npu')
    v = v.to('npu')
    mask = mask.to('npu')
    sparseMode = 1
    numHeads = n1
    numKeyValueHeads = n2
    inputLayout = 'BNSD'
    npu_result = torch.ops.x.fused_infer_attention_score(q,
                                                         k,
                                                         v,
                                                         mask,
                                                         numHeads,
                                                         numKeyValueHeads,
                                                         scaleValue,
                                                         inputLayout,
                                                         sparseMode)
    compare(cpu_result.cpu().to(torch.float).numpy().flatten(), npu_result.cpu().to(torch.float).numpy().flatten())
    compare_result = torch.allclose(cpu_result, npu_result, rtol=1e-03, atol=1e-03, equal_nan=True)
    assert compare_result
