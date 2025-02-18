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
import tvm
import tvm.testing
from tvm import relax as rx
from tvm.script import relax as R
from tvm.script import tir as T


@tvm.register_func("test.op.identity", override=True)
def identity_packed(a):
    return tvm.nd.array(a.asnumpy())


@T.prim_func
def identity_tir(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [54, 96])
    B = T.match_buffer(b, [54, 96])

    for i, j in T.grid(54, 96):
        with T.block("compute"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]


def test_call_tir() -> None:
    v0 = rx.Var("v0", R.Tensor([54, 96], "float32"))
    v1 = rx.call_tir(rx.extern("test.op.identity"), [v0], R.Tensor((54, 96), "float32"))
    v1 = rx.call_tir(identity_tir, [v0], R.Tensor((54, 96), "float32"))


def test_implicit_op():
    m, n = tvm.tir.Var("m", "int64"), tvm.tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float32"))
    y = rx.Var("y", R.Tensor([m, n], "float32"))

    def _check_call(expr, op_name: str):
        assert isinstance(expr, rx.Call)
        if not op_name.startswith("relax."):
            op_name = "relax." + op_name
        op = tvm.ir.Op.get(op_name)
        assert expr.op == op

    # Comparison operators
    _check_call(x > y, "greater")
    _check_call(x >= y, "greater_equal")
    _check_call(x < y, "less")
    _check_call(x <= y, "less_equal")

    # Arithmetic operators
    _check_call(x + y, "add")
    _check_call(x - y, "subtract")
    _check_call(x * y, "multiply")
    _check_call(x / y, "divide")
    _check_call(x // y, "floor_divide")
    # _check_call(x % y, "mod") <= relax.mod is not implemented yet

    # Cast
    _check_call(x.astype("float32"), "astype")

    # Call
    call_expr = x(y)(y)
    assert isinstance(call_expr.op, rx.Call)
    assert call_expr.op.op == x

    # GetTupleItem
    ## Eager get item for tuple
    tuple_expr = rx.Tuple((x, y))
    assert tuple_expr[0] == x
    assert tuple_expr[1] == y

    ## Eager get item for ShapeExpr
    shape_expr = rx.ShapeExpr((1, 2))
    assert shape_expr[0] == 1
    assert shape_expr[1] == 2

    ## Create TupleGetItem for other expr
    assert isinstance(x[0], rx.TupleGetItem)
    assert isinstance(x[1][0], rx.TupleGetItem)


if __name__ == "__main__":
    tvm.testing.main()
