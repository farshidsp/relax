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
import pytest
import numpy as np
import tvm
import tvm.testing

from tvm import relax, relay
from tvm.relax.testing import relay_translator

from .infrastructure import get_hexagon_target

def get_relay_conv2d_relu_x2(d_shape, w_shape):
    data = relay.var("data", shape=d_shape)
    weight1 = relay.var("weight1", shape=w_shape)
    weight2 = relay.var("weight2", shape=w_shape)
    conv1 = relay.nn.relu(
        relay.nn.conv2d(
            data=data,
            weight=weight1,
            kernel_size=w_shape[2:],
            padding=(1, 1),
        )
    )
    return relay.nn.relu(
        relay.nn.conv2d(
            data=conv1,
            weight=weight2,
            kernel_size=w_shape[2:],
            padding=(0, 0),
        )
    )


@tvm.testing.requires_hexagon
def test_conv2d_cpu(hexagon_launcher):
    data_np = np.random.randn(1, 64, 56, 56).astype("float32")
    weight1_np = np.random.randn(64, 64, 3, 3).astype("float32")
    weight2_np = np.random.randn(64, 64, 3, 3).astype("float32")

    relay_mod = tvm.IRModule.from_expr(get_relay_conv2d_relu_x2(data_np.shape, weight1_np.shape))
    params = {"weight1": weight1_np, "weight2": weight2_np}
    mod = relay_translator.from_relay(relay_mod["main"], get_hexagon_target("v69"), params)
    
    target_hexagon = tvm.target.hexagon("v69")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    ex = relax.vm.build(mod, target)
    with hexagon_launcher.create_session() as session:
        
        dev = session.device
        vm_mod = session.get_executor_from_factory(ex)
        vm_rt = relax.VirtualMachine(vm_mod, dev, profile=True)
        data = tvm.nd.array(data_np, dev)
        # vm_rt.set_input("main", data)
        # vm_rt.invoke_stateful("main")
        # hexagon_output = vm_rt.get_outputs("main").numpy()
        
        report = vm_rt.profile("main", data)
        
        print(report)
        # ex = relax.vm.build(mod, target)

        # vm = relax.VirtualMachine(ex, tvm.cpu(), profile=True)

        # report = vm.profile("main", tvm.nd.array(data_np))

        # print(report)


if __name__ == "__main__":
    tvm.testing.main()
