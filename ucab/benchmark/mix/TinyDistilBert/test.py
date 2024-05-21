from tvm.relay import ExprMutator
from tvm import relay
import tvm
from tvm.contrib import graph_executor
import numpy as np
from tvm.relay.expr import Call
import os

os.environ["TVM_BACKTRACE"] = "1"
class ReWriteInputs(ExprMutator):
    """This pass partitions the subgraph based on the if conditin

    """
    def __init__(self):
        super().__init__()
        self.inputs = []

    def visit_function(self, fn):
        new_params = []
        for x in range(len(fn.params)):
            if len(fn.params[x].type_annotation.shape)==1:
                d = fn.params[x].type_annotation.shape
                var_new = relay.var(fn.params[x].name_hint, shape=(1, 1, 1, d[0]), dtype=fn.params[x].type_annotation.dtype)
                new_params.append(var_new)
            else:
                new_params.append(fn.params[x])
        print(fn.params)
        print(fn.body.op)
        print(fn.body.args)
        print(fn.body.attrs)
        print(fn.body.type_args)
        print(fn.body.span)
        print(type(fn.body.op))
        print(fn.body)
        new_body = self.visit(fn.body)
        func = relay.Function(relay.analysis.free_vars(new_body), new_body)
        return func

    def visit_var(self, var):
        if len(var.type_annotation.shape) == 1:
            d = var.type_annotation.shape
            var_new = relay.var(var.name_hint, shape=(1, 1, 1, int(d[0])), dtype=var.type_annotation.dtype)
            return var_new
        else:
            return var
    
    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        print(type(call.args[0]))
        print(new_args)
        return Call(new_fn, new_args, call.attrs, call.type_args, call.span)
    
data_shape = (5,5,5)
data = relay.var("data", shape=data_shape, dtype="float32")
out = relay.take(data, relay.const(0, "int32"), axis=1)
out = relay.argmax(out, axis=1)
#out = relay.sum(out, axis=0)
mod = tvm.IRModule({})
mod["argmax"] = relay.Function([data], out)
print(mod)
argmax = relay.GlobalVar("argmax")
out = relay.sum(argmax(data), axis=0)
data2 = relay.var("data2", shape=data_shape, dtype="float32")
mod["main"] = relay.Function([data2], relay.sum(argmax(data2), axis=0))
print(mod)
"""
re_write_inputs = ReWriteInputs()
f = mod['main']
new_function = re_write_inputs.visit(f)
new_module = tvm.IRModule({"main": new_function})
print(new_module)
"""
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target)
    dev = tvm.device(str(target), 0)
    tvm_model = graph_executor.GraphModule(lib["default"](dev))

input_data = np.random.rand(*data_shape)
tvm_model.set_input("data", input_data)
tvm_model.run()
out = tvm_model.get_output(0)
print(out)
