# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test last-stage of codegen VM.

Restrictions: all shape lowered, explicit allocation.
"""
import tvm
import pytest
import numpy as np
from tvm import relax, TVMError
from tvm.script import relax as R, tir as T
from tvm.relax.testing.vm import check_saved_func
from tvm.relax.testing.runtime_builtin import MatchShapeCode, MakeShapeCode


def codegen(mod, target, exec_mode="bytecode"):
    builder = relax.ExecBuilder()
    tir_mod = relax.vm._vmcodegen(builder, mod, exec_mode=exec_mode)
    return relax.vm._vmlink(builder, target, tir_mod)


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_vm_copy(exec_mode):
    @tvm.script.ir_module
    class TestVMMove:
        @R.function
        def foo(x: R.Tensor((3, 4), "float32")):
            R.func_attr({"global_symbol": "foo"})
            z = R.call_packed("vm.builtin.copy", x, sinfo_args=(R.Tensor((3, 4), dtype="float32")))
            return z

    mod = TestVMMove
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    inp = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = check_saved_func(vm, "foo", inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_if_cond_const(exec_mode):
    @tvm.script.ir_module
    class TestVMIfCondConst:
        @R.function
        def main(x: R.Tensor(ndim=2, dtype="float32")) -> R.Tensor(ndim=2, dtype="float32"):
            R.func_attr({"global_symbol": "main"})
            if relax.const(True, dtype="bool"):
                ret = x
            else:
                ret = x
            return ret

    mod = TestVMIfCondConst
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(np.random.rand(3, 4))
    res = vm["main"](inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy())


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_vm_exec_serialize_export_library(exec_mode):
    @tvm.script.ir_module
    class TestVMMove:
        @R.function
        def foo(x: R.Tensor((3, 4), "float32")):
            R.func_attr({"global_symbol": "foo"})
            z = R.call_packed("vm.builtin.copy", x, sinfo_args=(R.Tensor((3, 4), dtype="float32")))
            return z

    mod = TestVMMove
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target)
    from tvm.contrib import utils

    temp_dir = utils.tempdir()
    path_exec = temp_dir.relpath("exec.so")
    ex.mod.export_library(path_exec)

    loaded_exec = relax.vm.Executable(tvm.runtime.load_module(path_exec))
    assert ex.as_text() == loaded_exec.as_text()


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_if_cond(exec_mode):
    @tvm.script.ir_module
    class TestVMCompileIf:
        @R.function
        def ife(cond: R.Tensor((), "bool"), x: R.Tensor((3, 4), "float32")) -> R.Tensor:
            R.func_attr({"global_symbol": "ife"})
            if cond:
                w = R.call_packed("test.vm.add", x, x, sinfo_args=(R.Tensor))
            else:
                w = R.call_packed("test.vm.mul", x, x, sinfo_args=(R.Tensor))
            return w

    mod = TestVMCompileIf
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(np.random.rand(3, 4))
    res = vm["ife"](tvm.nd.array(1), inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy() + inp.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["ife"](tvm.nd.array(True), inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy() + inp.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["ife"](tvm.nd.array(0), inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy() * inp.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["ife"](tvm.nd.array(False), inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy() * inp.numpy(), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_vm_return_const_tuple(exec_mode):
    @tvm.script.ir_module
    class ReturnConstTuple:
        @R.function
        def main(x: R.Tensor(ndim=2, dtype="float32")):
            R.func_attr({"global_symbol": "main"})
            y = R.const([1, 2])
            z = (y, R.const([3, 4]), x)
            return z

    mod = ReturnConstTuple
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(np.random.rand(2, 3))
    res0, res1, res2 = vm["main"](inp)
    tvm.testing.assert_allclose(res0.numpy(), np.array([1, 2]))
    tvm.testing.assert_allclose(res1.numpy(), np.array([3, 4]))
    tvm.testing.assert_allclose(res2.numpy(), inp.numpy())


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_vm_const_as_call_arg(exec_mode):
    @tvm.script.ir_module
    class TestVMConstAsCallArg:
        @R.function
        def main(x: R.Tensor(ndim=2, dtype="float32")):
            R.func_attr({"global_symbol": "main"})
            a = R.call_packed(
                "test.vm.add",
                relax.const([1, 2]),
                relax.const([3, 4]),
                sinfo_args=(R.Tensor(ndim=2, dtype="float32")),
            )
            b = R.call_packed(
                "test.vm.add",
                a,
                x,
                sinfo_args=(R.Tensor(ndim=2, dtype="float32")),
            )
            return b

    mod = TestVMConstAsCallArg
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(np.random.rand(1, 2))
    res = vm["main"](inp)
    tvm.testing.assert_allclose(res.numpy(), np.array([4, 6]) + inp.numpy())


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_shape_check_builtin(exec_mode):
    MS = MatchShapeCode
    MK = MakeShapeCode
    # slot assignment:
    # 0: n, 1: m
    sindex = {"n": 0, "m": 1}

    @tvm.script.ir_module
    class TestVMShapeCheck:
        @R.function
        def main(x: R.Tensor(["n", "m"], "float32")) -> R.Shape(ndim=3):
            R.func_attr({"global_symbol": "main"})
            n = T.Var("n", "int64")
            k = T.Var("k", "int64")
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(3)],
                sinfo_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info", x, 2, R.dtype("float32"), "", sinfo_args=[R.Tuple()]
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                2,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.STORE_TO_HEAP,
                sindex["m"],
                "",
                sinfo_args=[R.Tuple()],
            )
            # construct shape value for return
            s = R.call_packed(
                "vm.builtin.make_shape",
                shape_heap,
                3,
                MK.LOAD_SHAPE,
                sindex["m"],
                MK.LOAD_SHAPE,
                sindex["n"],
                MK.USE_IMM,
                2,
                sinfo_args=[R.Shape(ndim=3)],
            )
            return s

    mod = TestVMShapeCheck
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x = tvm.nd.array(np.zeros((1, 2)).astype("float32"))
    res = vm["main"](x)
    assert res == tvm.runtime.container.ShapeTuple([2, 1, 2])

    # wrong input type
    with pytest.raises(TypeError):
        vm["main"]([])

    # wrong ndim
    with pytest.raises(ValueError, match=r".*ndim.*"):
        vm["main"](tvm.nd.array(np.zeros(1).astype("float32")))

    # wrong dtype
    with pytest.raises(ValueError, match=r".*dtype.*"):
        vm["main"](tvm.nd.array(np.zeros((1, 2)).astype("int32")))


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_prim_value(exec_mode):
    @tvm.script.ir_module
    class TestVMPrimValue:
        @R.function
        def main():
            R.func_attr({"global_symbol": "main"})
            ret = R.prim_value(T.int64(1))
            return ret

    mod = TestVMPrimValue
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"]()
    assert res == 1


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_string_imm(exec_mode):
    @tvm.script.ir_module
    class TestVMStringImm:
        @R.function
        def main():
            R.func_attr({"global_symbol": "main"})
            ret = R.str("hello")
            return ret

    mod = TestVMStringImm
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"]()
    assert res == "hello"


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_datatype_imm(exec_mode):
    @tvm.script.ir_module
    class TestDataTypeImm:
        @R.function
        def main():
            R.func_attr({"global_symbol": "main"})
            ret = R.dtype("float32")
            return ret

    mod = TestDataTypeImm
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"]()
    assert res == "float32"


@pytest.mark.parametrize("exec_mode", ["bytecode", "compiled"])
def test_vm_builtin_reshape(exec_mode):
    @tvm.script.ir_module
    class TestVMBuiltinReshape:
        @R.function
        def main(x: R.Tensor((3, 4), "float32")):
            R.func_attr({"global_symbol": "main"})
            y = R.call_packed(
                "vm.builtin.reshape", x, (6, 2), sinfo_args=R.Tensor((6, 2), "float32")
            )
            return y

    mod = TestVMBuiltinReshape
    target = tvm.target.Target("llvm", host="llvm")
    ex = codegen(mod, target, exec_mode)
    dev = tvm.cpu()
    vm = relax.VirtualMachine(ex, dev)

    input_np = np.random.rand(3, 4).astype("float32")
    input = tvm.nd.array(input_np, dev)
    res = vm["main"](input)
    expected = input_np.reshape(6, 2)
    tvm.testing.assert_allclose(res.numpy(), expected, rtol=1e-7, atol=1e-7)
