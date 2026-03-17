# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.native_layout.base import (
    NativeLayoutTensor,
    NATIVE_LAYOUT_TENSOR_PROPERTY_NAME,
)
from chitu.global_vars import get_global_args


def enable_native_layout_weight(
    key: str,
    native_layout_tensor_class: type,
    allow_missing: bool = False,
    *static_args,
    **static_kwargs,
) -> type:
    """
    Return a mix-in class that can be inherit by a Module class, which will enable the Module class to
    preprocess its weight and use it in a native layout.

    Example usage:
    ```
    class YourModule(enable_native_layout_weight("weight", MyNativeLayout), torch.nn.Module):
        ...
    your_module = YourModule()
    ```
    , where `MyNativeLayout` is a subclass of `NativeLayout`.

    After `your_module` loads state dict, the weight will automatically be transformed into the layout of
    `MyNativeLayout`, still stored in the original `Parameter`. Besides, `your_module.get_native_layout_weight()`
    will be available for getting an `MyNativeLayout` instance for the weight.

    `enable_native_layout_weight` also supports preprocessing the weight from another layout. To do this,
    you can set `_{key}_layout_class`, `_{key}_plain_shape`, `_{key}_layout_args` (optional) and
    `_{key}_layout_kwargs` (optional) attributes in the Module class so the layout can be recognized. For
    example:

    ```
    class YourModule(enable_native_layout_weight("weight", MyNativeLayout), torch.nn.Module):
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.randn(2, 5, 2, 5, 2))  # Native shape
            self._weight_layout_class = MyNativeLayout
            self._weight_plain_shape = (10, 20)  # Mathematical shape
    ```

    Args:
        key: The module will process `self.{key}` for its weight `Parameter`, and the native layout tensor
             getter will be named after `self.get_native_layout_{key}`.
        native_layout_tensor_class: A subclass of `NativeLayoutTensor` representing the layout.
        allow_missing: If True, elegantly skip the preprocessing if `self.{key}` does not exists.
        static_args: Other positional arguments passed to `NativeLayoutTensor`.
        static_kwargs: Other keyword arguments passed to `NativeLayoutTensor`.
    """

    class EnableNativeLayoutWeightMixIn:
        # NOTE: In Python, super().__init__ calls the next base class in the full inheritance graph
        # of the final class, so we can append a class to the base class, to make it act like a
        # further base class of the original base class.
        # See https://docs.python.org/3/tutorial/classes.html#multiple-inheritance

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # NOTE: Look up parameters (dynamically handled by torch.nn.Module) with __getattr__, but
            # look up real Python attributes with __getattribute__.

            def _get_native_layout_tensor():
                layout_args = (
                    self.__getattribute__(f"_{key}_layout_args")
                    if hasattr(self, f"_{key}_layout_args")
                    else ()
                )
                layout_kwargs = (
                    self.__getattribute__(f"_{key}_layout_kwargs")
                    if hasattr(self, f"_{key}_layout_kwargs")
                    else {}
                )
                return self.__getattribute__(f"_{key}_layout_class")(
                    self.__getattribute__(f"_{key}_plain_shape"),
                    self.__getattr__(key).data,
                    *layout_args,
                    **layout_kwargs,
                )

            def _preprocess_layout(module, incompatible_keys):
                if allow_missing and not hasattr(module, key):
                    return
                if hasattr(module, f"_{key}_layout_class"):
                    old_tensor = _get_native_layout_tensor()
                else:
                    old_tensor = module.__getattr__(key)

                inst_args = getattr(module, f"_{key}_layout_args", None)
                inst_kwargs = getattr(module, f"_{key}_layout_kwargs", None)
                other_args = inst_args if inst_args is not None else static_args
                other_kwargs = inst_kwargs if inst_kwargs is not None else static_kwargs

                def _eval(p):
                    if callable(p):
                        try:
                            return p(module)  # e.g. lambda m: m.in_features
                        except TypeError:
                            return p()  # e.g. torch.get_default_dtype
                    return p

                other_args = tuple(_eval(a) for a in other_args)
                other_kwargs = {k: _eval(v) for k, v in other_kwargs.items()}

                native_layout = native_layout_tensor_class.convert_from(
                    old_tensor,
                    *other_args,
                    **other_kwargs,
                )
                module.__setattr__(f"_{key}_layout_class", native_layout_tensor_class)
                module.__setattr__(f"_{key}_plain_shape", native_layout.plain_shape)
                module.__setattr__(f"_{key}_layout_args", other_args)
                module.__setattr__(f"_{key}_layout_kwargs", other_kwargs)
                module.__getattr__(key).data = native_layout.layout_tensor
                setattr(
                    module.__getattr__(key),
                    NATIVE_LAYOUT_TENSOR_PROPERTY_NAME,
                    type(native_layout),
                )

            # skip_model_load=True时立刻处理，否则加载后处理
            if get_global_args().debug.skip_model_load:
                _preprocess_layout(self, None)
            else:
                self.register_load_state_dict_post_hook(_preprocess_layout)

            # register get_native_layout_{key}
            self.__setattr__(f"get_native_layout_{key}", _get_native_layout_tensor)

    return EnableNativeLayoutWeightMixIn
