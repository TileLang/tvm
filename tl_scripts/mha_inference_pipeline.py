import torch
import torch.nn.functional as F
from tvm import tl
import tvm.tl.language as T
from functools import partial

def flashattn(batch, heads, seq_len, dim, is_casual, block_M, block_N):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    num_split = 1

    @T.macro
    def MMA0(
        K: T.Buffer(shape, dtype),
        Q_shared: T.Buffer([block_M, dim], dtype),
        K_shared: T.Buffer([block_N, dim], dtype),
        acc_s: T.Buffer([block_M, block_N], accum_dtype),
        k: T.int32,
        mid: T.int32,
        hid: T.int32,
        bid: T.int32,
        sid: T.int32,
    ):
        T.copy(K[bid, k * block_N : (k + 1) * block_N, hid, :], K_shared)
        if is_casual:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    mid * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                )
        else:
            T.clear(acc_s)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Buffer(shape, dtype),
        V_shared: T.Buffer([block_M, dim], dtype),
        acc_s_cast: T.Buffer([block_M, block_N], dtype),
        acc_o: T.Buffer([block_M, dim], accum_dtype),
        k: T.int32,
        hid: T.int32,
        bid: T.int32,
        sid: T.int32,
    ):
        T.copy(V[bid, k * block_N : (k + 1) * block_N, hid, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
        acc_s: T.Buffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.Buffer([block_M, block_N], dtype),
        scores_max: T.Buffer([block_M], accum_dtype),
        scores_max_prev: T.Buffer([block_M], accum_dtype),
        scores_scale: T.Buffer([block_M], accum_dtype),
        scores_sum: T.Buffer([block_M], accum_dtype),
        logsum: T.Buffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
        acc_o: T.Buffer([block_M, dim], accum_dtype),
        scores_scale: T.Buffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.macro
    def flash_attn_split(
        Q: T.Buffer(shape, dtype),
        K: T.Buffer(shape, dtype),
        V: T.Buffer(shape, dtype),
        Output_partial: T.Buffer(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads * batch, num_split, threads=128 * 2) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            mid = bx
            hid = by % heads
            bid = by // heads
            sid = bz

            T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bid, mid * block_M : (mid + 1) * block_M, hid, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv((mid + 1) * block_M, block_N)) if is_casual else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=2):
                MMA0(K, Q_shared, K_shared, acc_s, k, mid, hid, bid, sid)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, hid, bid, sid)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, mid * block_M : (mid + 1) * block_M, hid, :])

    @T.macro
    def combine(
        Output_partial: T.Buffer(shape, dtype),
        Output: T.Buffer(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128 * 2) as (bx, by, bz):
            pO_shared = T.alloc_shared([block_M, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            O_local = T.alloc_fragment([block_M, dim], dtype)

            T.copy(Output_partial[bz, bx * block_M : (bx + 1) * block_M, by, :], pO_shared)
            T.copy(pO_shared, O_local)
            T.copy(O_local, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype),
        K: T.Buffer(shape, dtype),
        V: T.Buffer(shape, dtype),
        Output_partial: T.Buffer(shape, dtype),
        Output: T.Buffer(shape, dtype),
    ):
        flash_attn_split(Q, K, V, Output_partial)
        combine(Output_partial, Output)

    return main


def ref_program(Q, K, V, Output_partial, casual):
    assert casual is False
    dim = Q.size(-1)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    return output


if __name__ == "__main__":
    BATCH, H, N_CTX, D_HEAD = 1, 1, 128, 64
    casual = False
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if casual:
        total_flops *= 0.5
    BLOCK_M = 128
    BLOCK_N = 128 # if D_HEAD <= 128 else 32
    program = flashattn(BATCH, H, N_CTX, D_HEAD, casual, BLOCK_M, BLOCK_N)
    ref_program = partial(ref_program, casual=casual)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [4], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)

    # latency = mod.do_bench(ref_program, warmup=500)
    # print("{:.2f} ms".format(latency))
    # print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    # latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="torch")
    # print("{:.2f} ms".format(latency))
    # print("{:.2f} TFlops".format(total_flops / latency * 1e-9))