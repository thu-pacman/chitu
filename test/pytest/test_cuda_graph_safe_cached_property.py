import torch

from chitu.cuda_graph import (
    cuda_graph_safe_cached_property,
    make_dispatched_graphed_callables,
)
from chitu.static_tensor import StaticTensor


class A:
    def __init__(self, x_val):
        self.out_of_graph_invoke_cnt = 0
        self.x = torch.tensor(x_val, device="cuda", dtype=torch.int32)
        self._y_static_tensor = StaticTensor(
            max_nelem=1, dtype=torch.int32, device="cuda"
        )
        self._y_up_to_date = False

    def upd(self, new_x_val):
        self.x[...] = new_x_val
        self._y_up_to_date = False

    @cuda_graph_safe_cached_property("_y_static_tensor", "_y_up_to_date")
    def y(self):
        self.out_of_graph_invoke_cnt += 1
        return self.x**2


def test_cache_out_of_graph_reuse_out_of_graph():
    a = A(3)
    assert a.out_of_graph_invoke_cnt == 0

    y1 = a.y
    assert y1.item() == 9
    assert a.out_of_graph_invoke_cnt == 1

    y2 = a.y
    assert y2.item() == 9
    assert a.out_of_graph_invoke_cnt == 1

    a.upd(4)

    y3 = a.y
    assert y3.item() == 16
    assert a.out_of_graph_invoke_cnt == 2

    y4 = a.y
    assert y4.item() == 16
    assert a.out_of_graph_invoke_cnt == 2


def test_cache_in_graph_reuse_in_graph():
    key = None

    a = A(3)
    assert a.out_of_graph_invoke_cnt == 0

    @make_dispatched_graphed_callables(
        args_max_nelem=[],
        kwargs_max_nelem={},
        output_max_nelem_callback=lambda key, sample_nelem: 1,
    )
    def f():
        y1 = a.y
        y2 = a.y
        return y2

    y2 = f(key)
    assert y2.item() == 9
    assert a.out_of_graph_invoke_cnt == 3  # 2 warmup (uncached) + 1 capture (cached)

    a.upd(4)

    y4 = f(key)
    assert y4.item() == 16
    assert a.out_of_graph_invoke_cnt == 3  # unchanged


def test_cache_out_of_graph_reuse_in_graph():
    key = None

    a = A(3)
    assert a.out_of_graph_invoke_cnt == 0

    y1 = a.y
    assert y1.item() == 9
    assert a.out_of_graph_invoke_cnt == 1

    @make_dispatched_graphed_callables(
        args_max_nelem=[],
        kwargs_max_nelem={},
        output_max_nelem_callback=lambda key, sample_nelem: 1,
    )
    def f():
        y2 = a.y
        return y2

    y2 = f(key)
    assert y2.item() == 9
    assert a.out_of_graph_invoke_cnt == 1  # unchanged

    a.upd(4)

    y3 = a.y
    assert y3.item() == 16
    assert a.out_of_graph_invoke_cnt == 2

    y4 = f(key)
    assert y4.item() == 16
    assert a.out_of_graph_invoke_cnt == 2  # unchanged


def test_cache_in_graph_reuse_out_of_graph():
    key = None

    a = A(3)
    assert a.out_of_graph_invoke_cnt == 0

    @make_dispatched_graphed_callables(
        args_max_nelem=[],
        kwargs_max_nelem={},
        output_max_nelem_callback=lambda key, sample_nelem: 1,
    )
    def f():
        y1 = a.y
        return y1

    y1 = f(key)
    assert y1.item() == 9
    assert a.out_of_graph_invoke_cnt == 2  # 1 warmup (uncached) + 1 capture (cached)

    y2 = a.y
    assert y2.item() == 9
    assert a.out_of_graph_invoke_cnt == 2  # unchanged

    a.upd(4)

    y3 = f(key)
    assert y3.item() == 16
    assert a.out_of_graph_invoke_cnt == 2  # unchanged

    y4 = a.y
    assert y4.item() == 16
    assert a.out_of_graph_invoke_cnt == 2  # unchanged


class B:
    def __init__(self, length):
        self.out_of_graph_invoke_cnt = 0
        self.length = length
        self._y_static_tensor = StaticTensor(
            max_nelem=100, dtype=torch.int32, device="cuda"
        )
        self._y_up_to_date = False

    def upd(self, length):
        self.length = length
        self._y_up_to_date = False

    @cuda_graph_safe_cached_property("_y_static_tensor", "_y_up_to_date")
    def y(self):
        self.out_of_graph_invoke_cnt += 1
        return torch.full((self.length,), self.length, device="cuda", dtype=torch.int32)


def test_cache_in_graph_reuse_out_of_graph_dynamic_shape():
    b = B(3)

    @make_dispatched_graphed_callables(
        args_max_nelem=[],
        kwargs_max_nelem={},
        output_max_nelem_callback=lambda key, sample_nelem: 100,
    )
    def f():
        y1 = b.y
        return y1

    y1 = f(key=3)
    assert y1.shape == (3,)
    assert torch.all(y1 == 3)
    assert b.out_of_graph_invoke_cnt == 2  # 1 warmup (uncached) + 1 capture (cached)

    y2 = b.y
    assert y2.shape == (3,)
    assert torch.all(y2 == 3)
    assert b.out_of_graph_invoke_cnt == 2  # unchanged

    b.upd(4)

    y3 = f(key=4)
    assert y3.shape == (4,)
    assert torch.all(y3 == 4)
    assert b.out_of_graph_invoke_cnt == 4  # 1 warmup (uncached) + 1 capture (cached)

    y4 = b.y
    assert y4.shape == (4,)
    assert torch.all(y4 == 4)
    assert b.out_of_graph_invoke_cnt == 4  # unchanged

    b.upd(3)  # Back to 3

    y5 = f(key=3)
    assert y5.shape == (3,)
    assert torch.all(y5 == 3)
    assert b.out_of_graph_invoke_cnt == 4  # unchanged

    y6 = b.y
    assert y6.shape == (3,)
    assert torch.all(y6 == 3)
    assert b.out_of_graph_invoke_cnt == 4  # unchanged
