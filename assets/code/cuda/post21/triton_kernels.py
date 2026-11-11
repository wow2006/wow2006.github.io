import torch
import triton
import triton.language as tl


# #region add
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
# #endregion


# #region matmul
# C (M x N, fp32) = A (M x K, fp16) @ B (K x N, fp16), all row-major.
# M, N, K must be multiples of the block sizes (no masks, to keep it short).
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    a_ptrs = a_ptr + rows[:, None] * K + ks[None, :]  # a BM x BK tile of pointers
    b_ptrs = b_ptr + ks[:, None] * N + cols[None, :]  # a BK x BN tile of pointers

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK
        b_ptrs += BK * N
    tl.store(c_ptr + rows[:, None] * N + cols[None, :], acc)
# #endregion


# Average GPU time of one call to `work`, in milliseconds (same scheme as post 03).
def time_ms(work, warmup=3, reps=20):
    for _ in range(warmup):
        work()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        work()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / reps


def main():
    # #region run-add
    n = 1 << 26
    a = torch.rand(n, device="cuda")
    b = torch.rand(n, device="cuda")
    c = torch.empty_like(a)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    ms = time_ms(lambda: add_kernel[grid](a, b, c, n, BLOCK=BLOCK))
    ok = torch.equal(c, a + b)
    print(f"vector add   n = {n}  {ms:7.3f} ms  {3 * n * 4 / ms / 1e6:6.1f} GB/s  "
          f"{'match' if ok else 'MISMATCH'}")
    # #endregion

    # #region run-matmul
    M = N = K = 4096
    A = torch.rand(M, K, device="cuda", dtype=torch.float16) * 2 - 1
    B = torch.rand(K, N, device="cuda", dtype=torch.float16) * 2 - 1
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = A.float() @ B.float()  # plain FP32 reference

    BM, BN, BK = 128, 128, 32
    grid = (M // BM, N // BN)
    ms = time_ms(lambda: matmul_kernel[grid](A, B, C, M, N, K, BM=BM, BN=BN, BK=BK,
                                             num_warps=4, num_stages=3))
    err = (C - ref).abs().max().item()
    print(f"matmul {M}x{N}x{K}, FP16 in, FP32 out")
    print(f"  Triton     {ms:7.3f} ms  {2 * M * N * K / ms / 1e9:6.1f} TFLOP/s  "
          f"max abs diff vs FP32 {err:.2e}")
    # #endregion
    print("PASSED" if ok and err < 1e-2 else "FAILED")


if __name__ == "__main__":
    main()
