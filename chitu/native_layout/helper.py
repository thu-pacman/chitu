# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.native_layout.base import (
    NativeLayoutTensor,
    NativeLayoutTemplate,
    TensorWithNativeLayout,
)
from logging import getLogger

logger = getLogger(__name__)


class NativeLayoutMixin:
    """Mixin that provides :meth:`apply_native_layout` and
    :meth:`init_native_layout` for deferred native-layout conversion.

    Subclasses override :meth:`init_native_layout` to
    call :meth:`apply_native_layout` on each parameter that needs
    conversion::

        class MyLinear(NativeLayoutMixin, nn.Module):
            def __init__():
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(PLAIN_SHAPE))

            def init_native_layout(self):
                super().init_native_layout()
                self.apply_native_layout(self.weight, Layout1, state_dict_convert=False)
                self.apply_native_layout(self.weight, Layout2, state_dict_convert=False)
                self.apply_native_layout(self.weight, Layout3)
                self.apply_native_layout(self.weight, Layout4)

    When calling ``init_native_layout``, weight is converted as
    `Plain -> Layout1 -> Layout2 -> Layout3 -> Layout4`.

    When load_state_dict, state_dict["weight"] is assume to have shape and dtype
    as `Layout2`, and converted as `Layout2 -> Layout3 -> Layout4`.

    init_native_layout also build a getter `get_native_layout_weight()`, returns
    self.weight wrapped with Layout4.

    You can call Layout4.check_tensor(self.weight) to run an isinstance-like check.
    """

    def apply_native_layout(
        self,
        param: torch.nn.Parameter,
        layout_cls: type[NativeLayoutTensor],
        *args,
        state_dict_convert: bool = True,
        **kwargs,
    ) -> None:
        """Record (and eagerly apply) a native-layout conversion on *param*.

        Called from :meth:`init_native_layout`.  The conversion is always
        applied to ``param.data`` in-place via
        :meth:`NativeLayoutTemplate.convert`.

        Parameters
        ----------
        param :
            The parameter to convert.  Its initial shape is taken as the
            mathematical *plain_shape* for the first call in a chain.
        layout_cls :
            Target :class:`NativeLayoutTensor` subclass.
        *args, **kwargs :
            Forwarded to ``layout_cls.convert_from``.
        state_dict_convert :
            If ``True`` (default), this step participates in the
            ``load_state_dict`` pre-hook — the hook converts incoming
            ``state_dict`` data through this step on its way to the
            final layout.  If ``False``, the pre-hook assumes the
            ``state_dict`` data is already in this layout's format
            and skips conversion.  A ``False`` step must come before
            any ``True`` step on the same parameter.
        """

        # ---- walk existing chain ----
        existing: NativeLayoutTemplate | None = None
        if isinstance(param, TensorWithNativeLayout) and isinstance(
            param.native_layout, NativeLayoutTemplate
        ):
            existing = param.native_layout

        if (
            existing is not None
            and not state_dict_convert
            and existing.state_dict_convert
        ):
            raise ValueError(
                f"{type(self).__name__}: state_dict_convert=False "
                f"must come before state_dict_convert=True"
            )

        # ---- plain_shape ----
        plain_shape = existing.plain_shape if existing else param.shape

        # ---- state_dict_shape / state_dict_dtype ----
        if existing is not None:
            state_dict_shape = existing.state_dict_shape
            state_dict_dtype = existing.state_dict_dtype
        else:
            state_dict_shape = param.shape
            state_dict_dtype = param.dtype

        # ---- build template ----
        template = NativeLayoutTemplate(
            layout_cls=layout_cls,
            plain_shape=plain_shape,
            args=args,
            kwargs=kwargs,
            state_dict_convert=state_dict_convert,
            state_dict_shape=state_dict_shape,
            state_dict_dtype=state_dict_dtype,
            previous=existing,
        )

        # ---- eager conversion ----
        try:
            converted = template.convert(param.data)
        except Exception:
            if param.data.device.type == "meta":
                logger.error(
                    "convert failed for %s. Tip: %s must support meta " "tensor input.",
                    type(self).__name__,
                    layout_cls.__name__,
                )
            raise
        param.data = converted.layout_tensor

        # ---- update state_dict_shape/dtype after conversion ----
        if existing is None and not state_dict_convert:
            # First call, state_dict_convert=False: capture checkpoint
            # format after conversion (plain → packed).
            template.state_dict_shape = param.shape
            template.state_dict_dtype = param.dtype

        param.native_layout = template

    def init_native_layout(self):
        """Override in subclasses to call :meth:`apply_native_layout`.

        The default implementation is a no-op so that
        ``super().init_native_layout()`` chains end here.
        """
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_native_layout(model: torch.nn.Module):
    """Walk *model*, apply native-layout conversions and install hooks.

    For every sub-module that is a :class:`NativeLayoutMixin`:

    1. Call ``module.init_native_layout()`` — subclasses call
       :meth:`~NativeLayoutMixin.apply_native_layout`, which converts
       parameter data and attaches a :class:`NativeLayoutTemplate`.
    2. Iterate over *module*'s own parameters (non-recursive).  For each
       whose ``native_layout`` is a :class:`NativeLayoutTemplate`, install
       the ``get_native_layout_<name>`` getter and ``load_state_dict``
       hooks.

    Call this **once** after the full model tree is built but before the
    first ``load_state_dict``.
    """
    for module in model.modules():
        if not isinstance(module, NativeLayoutMixin):
            continue
        module.init_native_layout()
        for name, param in module.named_parameters(recurse=False):
            if not isinstance(param, TensorWithNativeLayout):
                continue
            template = param.native_layout
            if isinstance(template, NativeLayoutTemplate):
                _register_native_layout(module, name, template)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _register_native_layout(
    module: torch.nn.Module,
    name: str,
    template: NativeLayoutTemplate,
):
    """Install getter and state-dict hooks for a native-layout parameter.

    Raises :class:`RuntimeError` on duplicate registration.
    """
    if hasattr(module, f"get_native_layout_{name}"):
        raise RuntimeError(
            f"Duplicate native-layout registration for "
            f"{type(module).__name__}.{name}"
        )
    _install_getter(module, name, template)
    _install_state_dict_hooks(module, name, template)


def _install_getter(module: torch.nn.Module, name: str, template: NativeLayoutTemplate):
    """Install a ``get_native_layout_<name>`` method on *module*."""

    def getter(self):
        p = getattr(self, name)
        return template.build(p.data)

    setattr(module, f"get_native_layout_{name}", getter.__get__(module))


def _install_state_dict_hooks(
    module: torch.nn.Module, name: str, template: NativeLayoutTemplate
):
    """Install ``load_state_dict`` pre- and post-hooks.

    The pre-hook intercepts incoming ``state_dict`` entries, walks the
    template chain to find the first (deepest) template, and converts
    the data through all ``state_dict_convert=True`` steps.  If the
    deepest template has ``state_dict_convert=False`` the incoming data
    is assumed to already be in that layout's format and that step is
    skipped.

    The post-hook restores ``param.native_layout`` after
    ``load_state_dict(assign=True)`` replaces the ``Parameter`` object.
    """

    def pre_hook(
        module,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs: list[str],
    ):
        key = prefix + name
        if key not in state_dict:
            missing_keys.append(key)
            return
        try:
            data = state_dict[key]

            # Walk to the first template in the chain
            current = template
            chain = [current]
            while current.previous is not None:
                current = current.previous
                chain.insert(0, current)

            # If the first template has state_dict_convert=False,
            # the state_dict data is already in that intermediate
            # layout — wrap and skip the conversion step.
            if not chain[0].state_dict_convert:
                data = chain[0].build(data)
                chain = chain[1:]

            # Convert through all remaining (state_dict_convert=True)
            # templates.
            for t in chain:
                data = t.layout_cls.convert_from(data, *t.args, **t.kwargs)

            state_dict[key] = data.layout_tensor
        except Exception as e:
            logger.exception("Failed to convert native layout tensor")
            error_msgs.append(f"{key} convert failed: {e}")

    module.register_load_state_dict_pre_hook(pre_hook)

    def post_hook(module, _incompatible_keys):
        getattr(module, name).native_layout = template

    module.register_load_state_dict_post_hook(post_hook)
