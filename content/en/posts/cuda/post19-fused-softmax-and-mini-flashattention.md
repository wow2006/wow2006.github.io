---
title: "Post 19: Fused softmax and a mini FlashAttention"
date: 2026-09-29T19:19:00+03:00
weight: 20
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

This post is **optional**. It is a second capstone that pulls together things from earlier posts: block reductions and warp shuffles (post 09), shared-memory tiles (posts 06 and 07), and bandwidth scores (post 03). If you only care about the basics, you can skip it.

In the SGEMM capstone (post 18), the goal was compute: keep the math units busy. Here the goal is memory. We take one operation that every transformer runs, *softmax*, and see how much faster it gets when we stop writing temporary results to global memory. Then we use the same idea, plus one small math trick, to build a tiny version of *FlashAttention*: an attention kernel that never stores the large N×N score matrix.

## What We Will Learn

In this article, we will cover:

* The numerically stable softmax (subtract the max first).
* Why three small kernels are slower than one fused kernel.
* The *online softmax*: max and sum in a single pass.
* Attention in plain terms, and why its N×N matrix is a memory problem.
* A readable FP32 FlashAttention forward pass that never writes that matrix.

## Table of Contents

1. The Full Program.
2. Softmax and the Max Trick.
3. Naive: Three Kernels.
4. Fused: One Kernel per Row.
5. Online Softmax.
6. Softmax Results.
7. Attention, the Naive Way.
8. A Mini FlashAttention.
9. Attention Results.
10. Build and Run.
11. Exercise.

---

## 1. The Full Program

```bash
mkdir flash-attention
cd flash-attention
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post19/main.cu" >}}

It is longer than usual because it has two parts. `run_softmax` compares three softmax kernels. `run_attention` compares a naive attention with our fused one, for three sequence lengths. `CUDA_CHECK`, `DeviceBuffer`, and `time_ms` are the same as in the earlier posts.

---

## 2. Softmax and the Max Trick

Softmax turns a row of numbers into probabilities: all positive, summing to 1.

```text
y[j] = exp(x[j]) / sum_k exp(x[k])
```

Written like this, it breaks easily. `expf(89)` is already larger than the biggest `float` (about 3.4 × 10^38), so any input above 88.7 becomes `inf`, and the result becomes `inf / inf = NaN`. The fix is to subtract the row maximum `m` first:

```text
y[j] = exp(x[j] - m) / sum_k exp(x[k] - m)
```

The `exp(-m)` factor appears on the top and the bottom, so it cancels and the answer is the same. But now the largest exponent is `exp(0) = 1`, so nothing overflows, and the sum is at least 1, so we never divide by zero. Every real softmax implementation does this.

The cost is that a row now needs three steps: find the max, sum the exponentials, then write the outputs. Each step needs the result of the one before it.

Our input is a matrix of 8192 rows × 4096 columns, filled with random values between -10 and 10. Each row is softmaxed on its own. That is 134 MB in and 134 MB out, well above the 72 MB L2 cache, so we really measure DRAM:

{{< code file="cuda/post19/main.cu" region="softmax-setup" link="false" >}}

The score is the same as in post 03: we count the bytes that *must* move (read `x` once, write `y` once) and divide by the time. A plain copy kernel of the same size gives us the ceiling.

---

## 3. Naive: Three Kernels

The straightforward version is one kernel per step. Each launches one block of 256 threads per row:

{{< code file="cuda/post19/main.cu" region="naive" link="false" >}}

Each thread walks its row with a stride of `blockDim.x`, so neighbouring threads read neighbouring floats and the loads are coalesced (post 05). At the end, the 256 per-thread results are combined with a block reduction. We wrote these in post 09; here is the version we use in this post:

{{< code file="cuda/post19/main.cu" region="reduce" link="false" >}}

Two small changes from post 09:

* We use `__shfl_xor_sync` instead of `__shfl_down_sync`. With XOR, lanes swap values in pairs (a "butterfly"), so after the 5 steps *every* lane has the full result, not just lane 0.
* `block_reduce` is a template over the value type and the operation. We need it for max, for sum, and in section 5 for something new. Every thread gets the final result, because every thread needs `m` and `s` for the next step.

The `__syncthreads()` at the start of `block_reduce` matters. A kernel may call it twice in a row, and without that barrier a fast warp could overwrite `partial` while a slow warp is still reading the previous result.

The problem with this version is not the reductions. It is the memory traffic. `row_max` reads `x`, `row_sum_exp` reads `x` again, and `row_normalize` reads `x` a third time and writes `y`. The matrix does not fit in L2, so each read comes from DRAM. That is 4 × 134 MB of traffic for a job that needs only 2 × 134 MB.

---

## 4. Fused: One Kernel per Row

The fix is to do all three steps in one kernel. The block that owns a row computes the max, then the sum, then writes the outputs:

{{< code file="cuda/post19/main.cu" region="fused" link="false" >}}

This still reads the row three times in the code. The difference is *when*. One row is only 16 KB, and the three passes run back to back in the same block. By the time the second and third pass read `row[j]`, it is still in the L1 or L2 cache. So from DRAM's point of view, `x` is read once and `y` written once.

No shared memory for the row, no clever tricks. Just keeping the work in one kernel so the data is still hot when we need it again.

---

## 5. Online Softmax

We can do better on paper: compute the max and the sum in the same pass. The trick [5] is to keep a *running* max `m` and a running sum `s` that is always measured against the current `m`. When a new value `x` arrives:

```text
m_new = max(m, x)
s_new = s * exp(m - m_new) + exp(x - m_new)
```

If `x` is not a new maximum, `m_new = m`, the factor `exp(m - m_new)` is 1, and we simply add `exp(x - m)`. If `x` is a new maximum, everything we summed so far was measured against the old, smaller max, so we scale it down by `exp(m - m_new)` before adding.

The same rule merges two partial states, which is exactly what a reduction needs:

{{< code file="cuda/post19/main.cu" region="maxsum" link="false" >}}

A single new value `x` is just the state `{x, 1}` (its own max, and `exp(x - x) = 1`). So the whole kernel is:

{{< code file="cuda/post19/main.cu" region="online" link="false" >}}

Each thread folds its elements into one `MaxSum`, then `block_reduce` merges the 256 states with `Combine`. For the warp step, we overload `shfl_xor` for `MaxSum`: it just shuffles both fields. This is why `block_reduce` is a template.

We start from `-FLT_MAX` rather than `-INFINITY` on purpose. Merging two empty states with `-INFINITY` would compute `exp(-inf - (-inf)) = exp(NaN)`, and the `NaN` would spread into the result.

---

## 6. Softmax Results

{{< code file="cuda/post19/main.cu" region="softmax-launch" link="false" >}}

Output: (RTX 4090)

```text
Softmax of 8192 rows x 4096 cols (134 MB in, 134 MB out)
copy (ceiling)     0.291 ms,  923 GB/s (91.5% of peak)
naive (3 kernels)  0.585 ms,  459 GB/s (45.6% of peak)  max err 1.3e-09
fused (3 passes)   0.294 ms,  913 GB/s (90.5% of peak)  max err 1.3e-09
online (2 passes)  0.295 ms,  909 GB/s (90.1% of peak)  max err 1.4e-09
```

* The naive version is almost exactly **2× slower** than the fused one. That matches the traffic: 4 × 134 MB versus 2 × 134 MB.
* The fused kernel runs at 99% of the copy kernel. Softmax is memory bound: the `expf` and the reductions are free, hidden behind the loads.
* The online version is *not* faster here. Because the fused kernel's extra passes hit the cache, both kernels move the same bytes from DRAM. The online version even does a bit more math (two `expf` per element in the first pass instead of one).

So why bother with the online softmax? Because it does not need to see the whole row before it starts. Every new value can be folded in and then forgotten. That is exactly the property we need for attention.

The errors are against a double-precision CPU reference. All three agree to about 1e-9.

---

## 7. Attention, the Naive Way

Attention [4] is the core of a transformer. For one head, we have three matrices of shape N × d: queries `Q`, keys `K`, and values `V`. Here N is the sequence length (1024 to 4096) and d = 64 is the head dimension. The output is:

```text
S = Q Kᵀ / sqrt(d)        N x N   (how well each query matches each key)
P = softmax(S), per row   N x N
O = P V                   N x d   (a weighted average of the values)
```

The naive version does exactly that, in three kernels, and stores `S` in global memory in between:

{{< code file="cuda/post19/main.cu" region="attn-naive" link="false" >}}

`attention_scores` is a naive matrix multiply, one thread per score, like the first version in post 07. `attention_pv` is another, one thread per output element. In between, we reuse `softmax_online` from part 1, in place on `S`.

{{< code file="cuda/post19/main.cu" region="attn-launch" link="false" >}}

The inputs are small: Q, K, V, and O together are 4 MB at N = 4096. But `S` is N × N floats. At N = 4096 that is 67 MB, 16 times more than everything else combined. And it grows with N²: at N = 32768 it would be 4.3 GB, and at N = 131072 it would be 69 GB, more than any consumer GPU has. Long sequences simply do not fit.

---

## 8. A Mini FlashAttention

FlashAttention [1] never stores `S`. The key observation: for one query row `i`, the output is

```text
O[i] = sum_j exp(S[i][j] - m) * V[j]  /  sum_j exp(S[i][j] - m)
```

The denominator is exactly the online softmax sum from section 5. The numerator is a sum of the same kind, just with a vector `V[j]` attached to each term. So when the running max grows, we rescale the output accumulator by the same factor `exp(m_old - m_new)` as the sum. We can walk over the keys one by one, compute each score, fold it in, and throw it away.

Here is the kernel:

{{< code file="cuda/post19/main.cu" region="flash" link="false" >}}

How the work is split:

* **One warp per query row.** A block has 8 warps, so it handles 8 query rows. With d = 64, each of the 32 lanes owns 2 of the 64 columns: `lane` and `lane + 32`. So a lane keeps 2 values of `q`, 2 values of the output `o`, and the `MaxSum` state, all in registers.
* **K and V in shared-memory tiles.** All 8 warps need every key and value row. So the block loads 64 keys and 64 values at a time into shared memory (2 × 16 KB), cooperatively and coalesced, and all 8 warps read them from there. This is the same tiling idea as the matrix multiply in post 07. The two `__syncthreads()` keep one warp from overwriting a tile another warp is still using.
* **The score.** Each lane multiplies its 2 columns of `q` with the same 2 columns of key `j`. `warp_reduce(s, Sum{})` adds the 32 partial products, and thanks to the XOR butterfly, every lane gets the full score.
* **The update.** The four lines after that are the online softmax from section 5, plus the same rescaling for `o0` and `o1`. At the very end, we divide by the sum once.

We fold `scale = 1/sqrt(d)` into `q` when we load it, so we do not multiply every score by it. The kernel assumes N is a multiple of 64 (the comment marks this); real code would handle the last partial tile.

The launch is one block of 256 threads per 8 query rows:

{{< code file="cuda/post19/main.cu" region="flash-launch" link="false" >}}

What goes to global memory now? Q, K, and V are read (K and V once per block, from L2 mostly), and O is written. That is it. No N × N buffer at all.

---

## 9. Attention Results

Output: (RTX 4090)

```text
Attention, single head, d = 64, FP32
n = 1024: S matrix 4 MB, Q/K/V/O 1.0 MB
  naive   0.255 ms (scores 0.220 + softmax 0.006 + PV 0.029)   1051 GFLOP/s  max err 2.7e-07
  flash   0.113 ms   2373 GFLOP/s  max err 2.9e-07
n = 2048: S matrix 17 MB, Q/K/V/O 2.1 MB
  naive   1.003 ms (scores 0.874 + softmax 0.016 + PV 0.114)   1070 GFLOP/s  max err 3.3e-07
  flash   0.307 ms   3503 GFLOP/s  max err 4.3e-07
n = 4096: S matrix 67 MB, Q/K/V/O 4.2 MB
  naive   3.919 ms (scores 3.443 + softmax 0.072 + PV 0.404)   1096 GFLOP/s  max err 3.3e-07
  flash   1.120 ms   3836 GFLOP/s  max err 6.0e-07
```

The FLOP count is the two matrix multiplies, 4 × N² × d. The errors are against a double-precision CPU reference on every 64th query row.

**Memory.** This is the result that does not depend on how well we tuned anything. The fused kernel needs no extra buffer. The naive one needs 67 MB extra at N = 4096, and that number grows 4× every time N doubles.

**Correctness.** Both versions agree with the CPU to about 1e-7 or better, which is normal FP32 rounding over sums of thousands of terms. The online rescaling costs no accuracy.

**Time.** The fused kernel is 3.5× faster at N = 4096. Be careful with that number, though. Look at the breakdown: most of the naive time is our naive `attention_scores` kernel. With a good matrix multiply (cuBLAS, or the SGEMM from post 18) that part would shrink a lot, and the gap would be much smaller. What cannot be tuned away is the traffic for `S`: it is written once, read and written by the softmax, and read again by `P·V`. At larger N, when `S` no longer fits in the L2 cache, that alone is about 4 × N² × 4 bytes through DRAM. (At N = 4096, `S` is 67 MB and just fits in the 72 MB L2, which is why the softmax step is so quick here.)

**Honest limits.** 3.8 TFLOP/s is under 5% of the 4090's 82.6 TFLOP/s FP32 peak. For every key, each lane does 2 useful multiply-adds, but also 5 shuffles, 2 `expf`, and a rescale of its output. We chose readable over fast. Real FlashAttention [1][2][3] is a very different amount of engineering:

* It uses tensor cores (post 17) in FP16/BF16 for both matrix multiplies, not FP32 CUDA cores.
* Each thread computes many scores at once (register tiling, like post 18), and the rescaling happens once per tile of keys, not once per key.
* It splits the work across heads and batches, overlaps loads with compute (`cp.async`, and TMA on Hopper), and on Hopper and Blackwell uses warp specialization.
* It also has a backward pass, which is harder than the forward pass.

The idea, though, is exactly the one in our short kernel: online softmax, so the N × N matrix never has to exist.

---

## 10. Build and Run

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o flash-attention
./flash-attention
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post19/CMakeLists.txt" >}}

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/flash-attention
```
{{< /tab >}}
{{< /tabs >}}

The program prints both parts shown above, in about 2 seconds. Most of that is the CPU reference.

---

## 11. Exercise

Language models use *causal* attention: query `i` may only look at keys `j <= i` (a token cannot see the future). Add this to `flash_attention`:

1. Inside the `j` loop, skip keys with `tile + j > row`. Since the whole warp works on the same row, the whole warp skips together, so `warp_reduce` stays safe.
2. Better: stop the `tile` loop early. The last query row of a block is `blockIdx.x * kRowsPerBlock + kRowsPerBlock - 1`, so no tile after that is needed. Be careful: the loop contains `__syncthreads()`, so every warp in the block must run the same number of iterations.
3. Update the CPU reference in `max_error` to only sum over `j <= i`, and check the error is still around 1e-7.

How much faster is the causal kernel, and why is it close to the number you would guess from counting the work?

In summary, softmax is memory bound, so the win comes from moving fewer bytes: fusing three kernels into one halved the traffic and doubled the speed, reaching 99% of a plain copy. The online softmax keeps a running max and a rescaled sum, so it can consume values one at a time and forget them. That is what lets FlashAttention compute attention without ever writing the N × N score matrix, saving memory that grows with N². Our readable FP32 version is correct and already faster than the naive one, but real FlashAttention adds tensor cores and many more tricks on top of the same idea.

[1]: https://arxiv.org/abs/2205.14135 "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)"
[2]: https://arxiv.org/abs/2307.08691 "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)"
[3]: https://arxiv.org/abs/2407.08608 "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Shah et al., 2024)"
[4]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need (Vaswani et al., 2017)"
[5]: https://arxiv.org/abs/1805.02867 "Online normalizer calculation for softmax (Milakov and Gimelshein, 2018)"
[6]: https://github.com/Dao-AILab/flash-attention "FlashAttention source code"
