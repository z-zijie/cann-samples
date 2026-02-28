/*!
 * \file npu_ops_def.cpp
 * \brief
 */

#define Py_LIMITED_API_VERSION 0x03080000
#include <Python.h>
#include <torch/all.h>

extern "C" {
PyObject* PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
    };
    return PyModule_Create(&module_def);
}
}

namespace ascend_ops {

TORCH_LIBRARY(ascend_ops, m)
{
    m.def("matmul(Tensor input, Tensor weight, bool trans_a = False, bool trans_b = False) -> Tensor");
    m.def(R"(fused_infer_attention_score(Tensor q,
                                        Tensor k,
                                        Tensor v,
                                        Tensor mask,
                                        int numHeads,
                                        int numKeyValueHeads,
                                        float scaleValue = 0,
                                        str inputLayout = 'BSND',
                                        int sparseMode = 0) -> Tensor)");
}

} // namespace ascend_ops