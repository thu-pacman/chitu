import torch
import torchperf

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)
torch._dynamo.config.cache_size_limit = 102400


def run_sdp(iter, q, k, v):
    for i in range(iter):
        t = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return t


def verify_batch_mhca(b, l, e, s, ev, n_query, verify=False):
    # print(b, l, e, s, ev)
    # b, l, e, s, ev = 3, 13, 17, 19, 5
    # n_query = 2
    q = torch.randn((b, n_query, l, e), device="cuda")
    q_reshape = q.reshape([b, n_query * l, e])
    k = torch.randn((b, s, e), device="cuda")
    v = torch.randn((b, s, ev), device="cuda")
    # k_b = k.unsqueeze(1).broadcast_to([b, rt, s, e]).contiguous()
    # v_b = v.unsqueeze(1).broadcast_to([b, rt, s, ev]).contiguous()
    k_b = k.unsqueeze(1).broadcast_to([b, n_query, s, e]).contiguous()
    v_b = v.unsqueeze(1).broadcast_to([b, n_query, s, ev]).contiguous()

    # Check equivalence: only small tensors can pass the 1e-8 test
    if verify:
        t1 = torch.nn.functional.scaled_dot_product_attention(q, k_b, v_b)
        t2 = torch.nn.functional.scaled_dot_product_attention(q_reshape, k, v)
        print(t1, t2)
        print(torch.allclose(t1, t2.reshape_as(t1)))

    for t in (q_reshape, k, v):
        t.unsqueeze_(1)
    print([q.shape, k_b.shape, v_b.shape], [q_reshape.shape, k.shape, v.shape])
    f1 = lambda: torch.nn.functional.scaled_dot_product_attention(q, k_b, v_b)
    f2 = lambda: torch.nn.functional.scaled_dot_product_attention(q_reshape, k, v)
    # def run():
    #     torch.nn.functional.scaled_dot_product_attention(q, k_b, v_b)
    #     print('run')
    # f1 = run
    # # f1 = torch.compile(f1)
    # # f2 = torch.compile(f2)
    # s1 = torchperf.cuda_timeit(f1)
    # s2 = torchperf.cuda_timeit(f2)
    # print(f"Eager {s1*1000:.2f} {s2*1000:.2f} Speedup {s1/s2:.2f}")
    # f1 = torch.compile(f1)
    # f2 = torch.compile(f2)
    # s1 = torchperf.cuda_timeit(f1)
    # s2 = torchperf.cuda_timeit(f2)
    # print(f"Compile {s1*1000:.2f} {s2*1000:.2f} Speedup {s1/s2:.2f}")

    ret = []
    for compile in [
        False,
        True,
    ]:
        # torch.cuda.profiler.start()
        s1 = torchperf.cuda_timeit(f1, compile=compile)
        s2 = torchperf.cuda_timeit(f2, compile=compile)
        # s1 = 999
        # print(f"{compile} {s1*1000:.3f} {s2*1000:.3f} Speedup {s1/s2:.3f}")
        print(f"{compile} {s1*1000} {s2*1000} Speedup {s1/s2:.3f}")
        ret += [s1, s2]
    return ret


# QKV shape torch.Size([2, 10, 1024, 64]) torch.Size([2, 10, 1024, 64]) torch.Size([2, 10, 1024, 64])
# QKV shape torch.Size([2, 10, 1024, 64]) torch.Size([2, 10, 77, 64]) torch.Size([2, 10, 77, 64])
# print("b l s e ev Time(ms)")
# for n in {20, 2, 8, 40, 160}:
#     for l in {1024, 256}:
#         for s in {1024, 64, 77, 128}:
#             for e in {40, 64, 80, 128}:
#                 test_attn_time(n, l, s, e)

h, w = 1024, 1024
configs_sdxl = {
    # [b, l, e, s, ev]
    # "test": [
    #     [20, 1024, 64, 1024, 64],
    # ],
    "mhca": [
        ([20, 1024, 64, 77, 64], 16),
        ([40, 256, 64, 77, 64], 60),
    ],
    "mha": [
        ([20, 1024, 64, 1024, 64], 16),
        ([40, 256, 64, 256, 64], 60),
        ([1, 4096, 512, 4096, 512], 1),
    ],
}

# 20, 256, 64, 256, 64, 4
# 40, 64, 64, 64, 64, 60
# 20, 256, 64, 256, 64, 12

# 20, 256, 64, 77, 64, 4
# 40, 64, 64, 77, 64, 60
# 20, 256, 64, 77, 64, 12

# 1, 1024, 512, 1024, 512, 1


# for k,v in configs_sdxl.items():
#     for config in v:
#         verify_batch_mhca(*config)

times = []
for config, times in configs_sdxl["mhca"]:
    for batch in [1, 2, 4, 8, 16, 128]:
        # for batch in [2]:
        times.append(config + [batch] + verify_batch_mhca(*config, batch))
print(times)
