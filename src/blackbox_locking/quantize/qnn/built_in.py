import inspect
from collections import OrderedDict
import types
import torch


def wrap_forward_with_dtypes(
    module: torch.nn.Module, compute_dtype: torch.dtype, cast_dtype: torch.dtype = torch.float32
):
    ori_forward = module.__class__.forward

    def auto_cast_forward(self, x):
        x = x.to(compute_dtype)
        output = ori_forward(self, x)
        output = output.to(cast_dtype)
        return output

    def auto_cast_forward_args_kwargs(self, *args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                continue
            if not arg.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
                continue

            args[i] = arg.to(compute_dtype)

        for k, v in kwargs.items():
            if not isinstance(v, torch.Tensor):
                continue
            if not v.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
                continue
            kwargs[k] = v.to(compute_dtype)

        output = ori_forward(self, *args, **kwargs)

        if isinstance(output, torch.Tensor):
            output = output.to(cast_dtype)
        elif isinstance(output, (list, tuple)):
            output = list(output)
            for i, out in enumerate(output):
                if not isinstance(out, torch.Tensor):
                    continue
                if not out.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
                    continue
                output[i] = out.to(cast_dtype)
        elif isinstance(output, (dict, OrderedDict)):
            for k, v in output.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if not v.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
                    continue
                output[k] = v.to(cast_dtype)
        else:
            raise RuntimeError(f"Unsupported output type: {type(output)}")

        return output

    if len(inspect.signature(ori_forward).parameters) == 2:
        module.forward = types.MethodType(auto_cast_forward, module)
    elif len(inspect.signature(ori_forward).parameters) > 2:
        module.forward = types.MethodType(auto_cast_forward_args_kwargs, module)
    else:
        raise RuntimeError(f"Unsupported forward signature: {inspect.signature(ori_forward)}")
