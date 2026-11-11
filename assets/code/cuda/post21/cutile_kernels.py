import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


# #region add
@ct.kernel
def add_kernel(a, b, c, TILE: ConstInt):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(TILE,))
    b_tile = ct.load(b, index=(pid,), shape=(TILE,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)
# #endregion


# #region matmul
# C (M x N, fp32) = A (M x K, fp16) @ B (K x N, fp16), all row-major.
@ct.kernel
def matmul_kernel(A, B, C, BM: ConstInt, BN: ConstInt, BK: ConstInt):
    row, col = ct.bid(0), ct.bid(1)
    acc = ct.full((BM, BN), 0, dtype=ct.float32)
    for k in range(ct.num_tiles(A, axis=1, shape=(BM, BK))):
        a = ct.load(A, index=(row, k), shape=(BM, BK))
        b = ct.load(B, index=(k, col), shape=(BK, BN))
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(row, col), tile=acc)
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
    stream = torch.cuda.current_stream()

    # #region run-add
    n = 1 << 26
    a = torch.rand(n, device="cuda")
    b = torch.rand(n, device="cuda")
    c = torch.empty_like(a)
    TILE = 1024
    grid = (ct.cdiv(n, TILE), 1, 1)
    ms = time_ms(lambda: ct.launch(stream, grid, add_kernel, (a, b, c, TILE)))
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
    grid = (M // BM, N // BN, 1)
    ms = time_ms(lambda: ct.launch(stream, grid, matmul_kernel, (A, B, C, BM, BN, BK)))
    err = (C - ref).abs().max().item()
    print(f"matmul {M}x{N}x{K}, FP16 in, FP32 out")
    print(f"  cuTile     {ms:7.3f} ms  {2 * M * N * K / ms / 1e9:6.1f} TFLOP/s  "
          f"max abs diff vs FP32 {err:.2e}")
    # #endregion
    print("PASSED" if ok and err < 1e-2 else "FAILED")


if __name__ == "__main__":
    main()
