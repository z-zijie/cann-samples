import numpy
import argparse


def rms_norm_quant_numpy(x, gamma, beta, scale, offset, eps=1e-6):
    x = x.astype(numpy.float32)
    gamma = gamma.astype(numpy.float32)
    beta = beta.astype(numpy.float32)
    scale = scale.astype(numpy.float32)
    offset = offset.astype(numpy.float32)

    variance = numpy.mean(numpy.power(x, 2), axis=-1, keepdims=True)
    std = numpy.sqrt(variance + eps)
    rstd = 1 / std
    result_mid = x * rstd

    y = result_mid * gamma + beta
    y = y * scale + offset
    y_quant = y.clip(-128, 127)
    y_quant = y_quant.astype("int8")
    return y_quant


def gen_input_data(a, r, dtype):
    x_shape = (a, r)
    gamma_shape = (r, )
    beta_shape = (r, )
    scale_shape = (1, )
    offset_shape = (1, )

    x = numpy.random.uniform(low=-100, high=100, size=x_shape).astype(dtype)
    x.tofile("/mnt/workspace/gitCode/ops-samples-tpc/Samples/2_Performance/rms_norm_quant/input0.bin")

    gamma = numpy.random.uniform(low=-100, high=100, size=gamma_shape).astype(dtype)
    gamma.tofile("/mnt/workspace/gitCode/ops-samples-tpc/Samples/2_Performance/rms_norm_quant/input1.bin")

    beta = numpy.random.uniform(low=-100, high=100, size=beta_shape).astype(dtype)
    beta.tofile("/mnt/workspace/gitCode/ops-samples-tpc/Samples/2_Performance/rms_norm_quant/input2.bin")

    scale = numpy.random.uniform(low=-10, high=10, size=scale_shape).astype(dtype)
    scale.tofile("/mnt/workspace/gitCode/ops-samples-tpc/Samples/2_Performance/rms_norm_quant/input3.bin")

    offset = numpy.random.uniform(low=-10, high=10, size=offset_shape).astype(numpy.int8)
    offset.tofile("/mnt/workspace/gitCode/ops-samples-tpc/Samples/2_Performance/rms_norm_quant/input4.bin")

    y_out = rms_norm_quant_numpy(x, gamma, beta, scale, offset)
    y_out.tofile("/mnt/workspace/gitCode/ops-samples-tpc/Samples/2_Performance/rms_norm_quant/output0.bin")


if __name__ == "__main__":
    """使用argparse解析命令行参数"""
    parser = argparse.ArgumentParser(description='处理命令行参数')

    # 添加参数
    parser.add_argument('-r', '--rdim', type=int)
    parser.add_argument('-a', '--adim', type=int)
    parser.add_argument('-d', '--dtype', type=str)

    # 解析参数
    args = parser.parse_args()

    gen_input_data(args.adim, args.rdim, args.dtype)
