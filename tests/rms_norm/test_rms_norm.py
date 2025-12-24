import torch
import torch_npu
import x
import pytest


def test_rms_norm_interface_exist():
    """
    Test that the 'x.rms_norm' operator is present in torch.ops.
    This existence test asserts that the custom operator registered under the
    'x' namespace is discoverable from Python via torch.ops.x.rms_norm.
    It does not exercise operator functionality — only that the Python binding
    and registration are available.
    Rationale:
    The presence of this test guards against a common failure mode where an
    operator is implemented and registered in C++/ATen but is not exposed to
    the Python torch.ops namespace due to mismatches between the PyTorch
    operator schema and the C++ registration signature (argument names, types,
    or overloads). Such schema/signature inconsistencies can cause the
    operator to be hidden or not exported to Python, breaking consumers that
    expect to call torch.ops.x.rms_norm. This test will fail loudly if the
    binding is missing, prompting investigation into schema/registration issues.
    """
    # This test specifically protects against discrepancies between the
    # PyTorch operator schema and the C++ signature/registration that can
    # prevent the operator from being visible in torch.ops.x.
    print(torch.ops.x.rms_norm)
    assert hasattr(torch.ops.x, "rms_norm"), "The 'rms_norm' operator is not registered in the 'torch.ops.x' namespace."

def test_rms_norm_meta_basic():
    """
    Test basic meta function behavior of the 'x.rms_norm' operator.
    """
    x = torch.empty((2, 3, 4), dtype=torch.float32, device='meta')
    normalized_shape = [4]
    y, rstd = torch.ops.x.rms_norm(x, normalized_shape)
    assert y.device.type == 'meta', "Output tensor y should be on the 'meta' device."
    assert rstd.device.type == 'meta', "Output tensor rstd should be on the 'meta' device."
    assert y.shape == x.shape, "Output tensor y should have the same shape as input x."
    assert rstd.shape == (2, 3), "Output tensor rstd should have shape corresponding to normalized dimensions."


def test_rms_norm_meta_no_normalized_shape():
    """
    Test that the 'x.rms_norm' operator raises an error when normalized_shape is empty.
    """
    x = torch.empty((2, 3, 4), dtype=torch.float32, device='meta')
    normalized_shape = []
    with pytest.raises(RuntimeError, match="Expected normalized_shape to be at least 1-dimensional"):
        torch.ops.x.rms_norm(x, normalized_shape)


def test_rms_norm_meta_normalized_shape_dim_larger_than_input():
    """
    Test that the 'x.rms_norm' operator raises an error when normalized_shape has more dimensions than input.
    """
    x = torch.empty((2, 3, 4), dtype=torch.float32, device='meta')
    normalized_shape = [2, 3, 4, 5]  # More dimensions than input
    with pytest.raises(RuntimeError, match="Shape mismatch"):
        torch.ops.x.rms_norm(x, normalized_shape)


def test_rms_norm_meta_invalid_normalized_shape():
    """
    Test that the 'x.rms_norm' operator raises an error with invalid normalized_shape.
    """
    x = torch.empty((2, 3, 4), dtype=torch.float32, device='meta')
    normalized_shape = [5]  # Invalid shape
    with pytest.raises(RuntimeError, match="Shape mismatch: Expected input to have shape"):
        torch.ops.x.rms_norm(x, normalized_shape)


def test_rms_norm_meta_with_weight():
    """
    Test meta function behavior of the 'x.rms_norm' operator with weight tensor.
    """
    x = torch.empty((2, 3, 4), dtype=torch.float32, device='meta')
    weight = torch.empty((4,), dtype=torch.float32, device='meta')
    normalized_shape = [4]
    y, rstd = torch.ops.x.rms_norm(x, normalized_shape, weight)
    assert y.device.type == 'meta', "Output tensor y should be on the 'meta' device."
    assert rstd.device.type == 'meta', "Output tensor rstd should be on the 'meta' device."
    assert y.shape == x.shape, "Output tensor y should have the same shape as input x."
    assert rstd.shape == (2, 3), "Output tensor rstd should have shape corresponding to normalized dimensions."


def test_rms_norm_meta_invalid_weight_dim():
    """
    Test that the 'x.rms_norm' operator raises an error with invalid weight dim.
    """
    x = torch.empty((2, 3, 4), dtype=torch.float32, device='meta')
    weight = torch.empty((3, 4), dtype=torch.float32, device='meta')  # Invalid dim
    normalized_shape = [4]
    with pytest.raises(RuntimeError, match="Shape mismatch: weight should have the same number of dimensions as normalized_shape"):
        torch.ops.x.rms_norm(x, normalized_shape, weight)


def test_rms_norm_meta_invalid_weight_shape():
    """
    Test that the 'x.rms_norm' operator raises an error with invalid weight shape.
    """
    x = torch.empty((2, 3, 4), dtype=torch.float32, device='meta')
    weight = torch.empty((5,), dtype=torch.float32, device='meta')  # Invalid shape
    normalized_shape = [4]
    with pytest.raises(RuntimeError, match="Shape mismatch: Expected weight to have shape"):
        torch.ops.x.rms_norm(x, normalized_shape, weight)
