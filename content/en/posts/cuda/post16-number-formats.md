---
title: "Post 16: Number formats"
date: 2026-09-29T19:16:00+03:00
weight: 17
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

In post 03, we learned to score a kernel by its bandwidth, and most of the kernels in this series ended up memory bound: the GPU spends its time waiting for bytes, not doing math. So far every number we moved was a 4-byte `float`. If we use a 2-byte number instead, we move half the bytes, and a memory-bound kernel runs almost twice as fast. This is why machine learning moved to FP16, BF16, and now FP8. But a smaller number has fewer bits, and those bits have to come from somewhere. In this post, we will see exactly what we give up, how to avoid the classic mistake, and what `--use_fast_math` really does to our results.

## What We Will Learn

In this article, we will cover:

* How FP32, FP16, BF16, and FP8 lay out their bits, and what that means for range and precision.
* The CUDA types `__half`, `__half2`, `__nv_bfloat16`, `__nv_fp8_e4m3`, and `__nv_fp8_e5m2`.
* Why we store in FP16 but accumulate in FP32 (mixed precision).
* How much bandwidth we win by moving half-size numbers.
* What `--use_fast_math` changes, and how to measure its error against a double-precision reference.

## Table of Contents

1. Bits, Range, and Precision.
2. The Full Program.
3. Round Trip: What Each Format Keeps.
4. Summing Many Small Numbers.
5. The Bandwidth Win.
6. Fast Math.
7. Build and Run.
8. Exercise.

---

## 1. Bits, Range, and Precision

Every floating-point number is stored as three fields: a sign bit, an exponent, and a mantissa (also called the fraction). The value is roughly `±1.mantissa × 2^exponent`.

* The **exponent** bits decide the *range*: how large and how small a number can get.
* The **mantissa** bits decide the *precision*: how close two neighbouring numbers are.

With a fixed number of bits, every format picks a trade-off between the two:

| Format   | CUDA type        | Sign | Exponent | Mantissa | Largest value | Smallest normal | Epsilon (gap after 1.0) |
| -------- | ---------------- | ---- | -------- | -------- | ------------- | --------------- | ----------------------- |
| FP32     | `float`          | 1    | 8        | 23       | 3.4 × 10^38   | 1.2 × 10^-38    | 1.2 × 10^-7             |
| FP16     | `__half`         | 1    | 5        | 10       | 65504         | 6.1 × 10^-5     | 9.8 × 10^-4             |
| BF16     | `__nv_bfloat16`  | 1    | 8        | 7        | 3.4 × 10^38   | 1.2 × 10^-38    | 7.8 × 10^-3             |
| FP8 E4M3 | `__nv_fp8_e4m3`  | 1    | 4        | 3        | 448           | 1.6 × 10^-2     | 0.125                   |
| FP8 E5M2 | `__nv_fp8_e5m2`  | 1    | 5        | 2        | 57344         | 6.1 × 10^-5     | 0.25                    |

A few things stand out:

* **FP16** keeps decent precision (about 3 decimal digits) but has a tiny range. Anything above 65504 becomes infinity.
* **BF16** ("brain float", from Google) is just the top 16 bits of a `float`. It has the same 8-bit exponent, so the same range as FP32, but only 7 mantissa bits (about 2 decimal digits). Converting from `float` cannot overflow, which is why training prefers it.
* **FP8** comes in two flavours. E4M3 has more precision and is used for weights and activations. E5M2 has more range and is used for gradients. Neither is precise: with 3 mantissa bits, E4M3 can only store 8 values between 1 and 2.

The *epsilon* column is the one to remember. It tells us the relative gap between neighbouring numbers. Any result is rounded to that gap, so a single FP16 operation can be off by about 0.05%, and a BF16 operation by about 0.4%.

Each type lives in its own header:

{{< code file="cuda/post16/main.cu" region="headers" link="false" >}}

All of them work in both host and device code, so we can create and convert them on the CPU too.

---

## 2. The Full Program

```bash
mkdir number-formats
cd number-formats
code main.cu
```

You can write the following in `main.cu`

{{< code file="cuda/post16/main.cu" >}}

The program has four independent parts. We keep `CUDA_CHECK` and the `unique_ptr` buffer from post 01, but `make_device_buffer` is now a template, because we allocate `float`, `__half`, and `__nv_bfloat16` buffers. `time_ms` is the cudaEvent helper from post 03: 3 warm-up calls, then the average of 20.

---

## 3. Round Trip: What Each Format Keeps

The quickest way to feel the table above is to convert a few numbers to each format and back:

{{< code file="cuda/post16/main.cu" region="round-trip" link="false" >}}

`__float2half` and `__half2float` convert to and from FP16. BF16 has the same pair with `bfloat16` in the name. The FP8 types are simpler: we construct them from a `float` and cast back with `float(...)`.

Output:

```text
== Round trip float -> format -> float ==
       float         fp16         bf16     fp8 e4m3     fp8 e5m2
         0.1   0.09997559    0.1000977    0.1015625      0.09375
   0.3333333     0.333252    0.3339844      0.34375       0.3125
    3.141593     3.140625     3.140625         3.25            3
         300          300          300          288          320
       70000          inf        70144          448        57344
       1e-05 1.001358e-05 1.001358e-05            0 1.525879e-05
```

Every row matches the table:

* FP16 keeps about 4 digits of 0.1 and 1/3. BF16 keeps about 3.
* FP8 E4M3 turns π into 3.25, and E5M2 turns it into 3. Near 300, E4M3 can only represent steps of 32, and E5M2 steps of 64.
* 70000 is above FP16's 65504, so it becomes `inf`. BF16 keeps it (rounded to 70144). The FP8 constructors *saturate*: instead of infinity, they clamp to the largest finite value, 448 and 57344. That is the `__NV_SATFINITE` mode in `cuda_fp8.h`.
* 1e-5 is below E4M3's smallest value, so it becomes 0. E5M2 has FP16's exponent, so it still keeps a (very rough) 1.5e-5.

On the RTX 4090 (sm_89), the conversions to and from FP8 are real hardware instructions. We can see them in the compiled code:

```bash
cuobjdump -sass number-formats | grep F2FP
```

```text
        /*0090*/                   F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R5, RZ, R2.reuse, RZ ;  /* 0x00000002ff05723e */
        /*00a0*/                   F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C R6, RZ, R2.reuse, RZ ;  /* 0x00000002ff06723e */
        /*00b0*/                   F2FP.F16.F32.PACK_AB R0, RZ, R2 ;                               /* 0x00000002ff00723e */
        /*00f0*/                   F2FP.F16.E4M3.UNPACK_B R0, R5 ;                                 /* 0x00000005ff00723e */
        /*0110*/                   F2FP.F16.E5M2.UNPACK_B R6, R6 ;                                 /* 0x00000006ff06723e */
        /*0120*/                   F2FP.BF16.F32.PACK_AB R2, RZ, R2 ;                              /* 0x00000002ff02723e */
```

Each conversion in `round_trip` is a single `F2FP` instruction. Converting FP8 back goes through FP16 (`F2FP.F16.E4M3.UNPACK_B`). On older GPUs, `cuda_fp8.h` falls back to a slower software conversion. Note that sm_89 has no FP8 add or multiply for normal code. FP8 here is a *storage* format: we convert it to FP16 or FP32 to compute, or feed it to the Tensor Cores, which we will meet in the next post.

---

## 4. Summing Many Small Numbers

Here is the classic mistake. We store 10,000 copies of 0.001 and add them up. The answer should be 10.

{{< code file="cuda/post16/main.cu" region="sum-host" link="false" >}}

The kernel is a template over two types: `T`, the type the data is stored in, and `Acc`, the type we add in. Each thread sums one column of a `rows × cols` matrix, so neighbouring threads read neighbouring elements, which keeps the loads coalesced (post 05).

{{< code file="cuda/post16/main.cu" region="column-sum" link="false" >}}

We launch it once with an FP16 accumulator and once with an FP32 accumulator, reading the same FP16 data:

{{< code file="cuda/post16/main.cu" region="sum-launch" link="false" >}}

The BF16 launches are the same with `__nv_bfloat16`. Output:

```text
== Sum of 10000 x 0.001 (exact answer: 10) ==
store fp16, accumulate fp16:    4.0000
store fp16, accumulate fp32:   10.0040
store bf16, accumulate bf16:    0.5000
store bf16, accumulate fp32:    9.9945
```

The FP16 sum gets stuck at 4, and the BF16 sum at 0.5. Nothing crashed, and there is no warning. Why?

Between 4 and 8, the gap between neighbouring FP16 numbers is 2^-8 ≈ 0.0039. When we compute `4 + 0.001`, the exact result 4.001 is not an FP16 number, so it is rounded to the nearest one, and the nearest one is 4 itself. From then on, every addition changes nothing. BF16 has 3 fewer mantissa bits, so its gaps are 8 times larger, and it gets stuck much earlier, at 0.5.

The FP32 accumulator gives 10.004 and 9.9945. They are not exactly 10 because 0.001 itself cannot be stored exactly in FP16 or BF16: it becomes 0.0010004 and 0.00099945. That is a small, fixed error in each input. The FP16 accumulator was a different kind of error: it grew with every addition until the answer was wrong by 60%.

This is the rule behind *mixed precision*:

> Store in FP16 or BF16 to save bandwidth. Accumulate in FP32.

The data stays small in memory, and each thread keeps its running sum in a 32-bit register, where it costs nothing extra. The Tensor Cores in the next post are usually used the same way: they multiply FP16 inputs and add the products into an FP32 accumulator.

---

## 5. The Bandwidth Win

Now the reason we bother with small formats at all. We compute `out = a * x + y` on 32M elements, in three versions:

{{< code file="cuda/post16/main.cu" region="axpy" link="false" >}}

* `axpy_float` is our usual kernel.
* `axpy_half` does the same with `__half`. `__hfma(a, b, c)` computes `a * b + c` in FP16.
* `axpy_half2` uses `__half2`, a pair of two FP16 values packed in 4 bytes. `__hfma2` computes on both halves with a single instruction, so each thread handles two elements, and we launch half as many threads.

Since `cudaMalloc` returns memory aligned to at least 256 bytes, we can safely reinterpret an `__half` buffer as an `__half2` buffer, as long as `n` is even:

{{< code file="cuda/post16/main.cu" region="axpy-launch" link="false" >}}

`__half2half2` copies one `__half` into both lanes of a `__half2`. Each element reads `x` and `y` and writes `out`, so the kernel moves 3 values per element. That gives us the usual score line from post 03 (the 1008 GB/s peak of the RTX 4090 is hard-coded here):

{{< code file="cuda/post16/main.cu" region="axpy-score" link="false" >}}

Output:

```text
== out = a * x + y, 33554432 elements ==
float : 0.432 ms, 932 GB/s (92.4% of peak)
half  : 0.216 ms, 932 GB/s (92.4% of peak)
half2 : 0.216 ms, 932 GB/s (92.5% of peak)
max |half2 - float| = 0.000854373
```

All three kernels run at the same 92% of peak bandwidth, because they are all limited by memory. But the FP16 versions move half the bytes, so they finish in half the time. The buffers are 128 MB (float) and 64 MB (half), larger than the 72 MB L2 cache, so this is real DRAM traffic.

Notice that `half` and `half2` are equally fast here. The kernel does one multiply-add per 6 bytes, so the arithmetic is almost free, and doing it two at a time does not help. `__half2` matters when a kernel is closer to compute bound: it halves the number of arithmetic instructions, and each thread loads 4 bytes instead of 2. It is a habit worth having, but do not expect it to speed up every kernel.

The price is the last line. The FP16 result differs from the float result by up to 0.00085. That comes from rounding `x`, `y`, and the result to FP16, whose gap between neighbours is about 0.001 for values between 1 and 2. For many workloads, such as images, audio, or neural network activations, that is fine. For a physics simulation that runs for a million steps, it may not be.

---

## 6. Fast Math

The GPU has special hardware, the SFU (special function unit), that computes approximate `sin`, `cos`, `exp2`, `log2`, and reciprocal in a single instruction. The normal `sinf` and `expf` do extra work around it, such as careful range reduction, to keep the error within 1 or 2 ULP (units in the last place). CUDA gives us two ways to use the fast version:

1. Call the intrinsics by hand: `__sinf`, `__cosf`, `__expf`, `__logf`, `__fdividef`, and more.
2. Compile with `--use_fast_math`, which does it for the whole file.

`--use_fast_math` is a shortcut for four flags:

| Flag               | Effect                                                                     |
| ------------------ | -------------------------------------------------------------------------- |
| `-ftz=true`        | Subnormal numbers (below 1.2 × 10^-38) are flushed to zero.                |
| `-prec-div=false`  | `a / b` uses a faster, less accurate division (like `__fdividef`).         |
| `-prec-sqrt=false` | `sqrtf` uses a faster, less accurate square root.                          |
| `-fmad=true`       | `a * b + c` may be fused into one FMA instruction. This is already the default. |

On top of that, it replaces `sinf`, `cosf`, `expf`, `logf`, and friends with their intrinsic versions.

To measure the trade-off, we need a kernel that is limited by math, not memory. For each `x`, ours adds 32 terms, each with a sine, an exponential, and a division. We write it once as a template, and `if constexpr` picks the accurate functions or the intrinsics at compile time:

{{< code file="cuda/post16/main.cu" region="math" link="false" >}}

The CPU function computes the same formula in `double`. Its error is far below FP32's, so we can treat it as the exact answer. All the terms are positive, so the sum never cancels to near zero, and the relative error `|gpu - ref| / |ref|` is meaningful for every element:

{{< code file="cuda/post16/main.cu" region="accuracy" link="false" >}}

The inputs are 4M values of `x` spread over [0, 100). Output:

```text
== 32 terms of sin/exp/divide on 4194304 values ==
accurate  0.265 ms  max abs err 1.76e-05  max rel err 8.57e-07
intrinsic 0.084 ms  max abs err 1.89e-05  max rel err 6.48e-06
```

The intrinsic version is 3.2 times faster. Its worst relative error is 7.6 times larger, but still 6.5 × 10^-6, which is around 50 ULP of a `float`. The accurate version stays within about 7 ULP, and most of that comes from the 32 additions, not from `sinf` and `expf`.

Why is the intrinsic error larger for this input range? The documented error of `__sinf` is only small for arguments in [-π, π], and the error of `__expf` grows with the size of its argument. Our `t = k * x` goes up to 3200. If we shrink the input range to [0, 10), the two versions are almost equally accurate. So the answer to "is fast math safe?" depends on your input range, and the only way to know is to measure against a reference, like we did here.

### What the compiler flag does to the "accurate" kernel

Now build the same program with `--use_fast_math` and look at the same section:

```text
== 32 terms of sin/exp/divide on 4194304 values ==
accurate  0.082 ms  max abs err 1.89e-05  max rel err 6.48e-06
intrinsic 0.082 ms  max abs err 1.89e-05  max rel err 6.48e-06
1e-39 * 0.5 on the GPU = 0
```

Our "accurate" kernel now has exactly the same speed and error as the intrinsic one. The flag silently replaced `sinf`, `expf`, and `/` everywhere in the file. That is the main danger of `--use_fast_math`: it is a global switch, and it also applies to code that was written expecting accurate results.

The last line is flush-to-zero. This tiny kernel halves a subnormal number:

{{< code file="cuda/post16/main.cu" region="ftz" link="false" >}}

{{< code file="cuda/post16/main.cu" region="ftz-run" link="false" >}}

The normal build prints `5e-40`. With `--use_fast_math`, subnormals are treated as zero, so it prints `0`. Most code never sees numbers that small, but code that divides by a tiny value, or checks `x > 0`, can suddenly behave differently.

A safer habit is to leave the flag off and call intrinsics by hand in the few hot spots where you have measured that the error is acceptable. You can also enable only part of it, for example `-ftz=true`, instead of all four flags.

---

## 7. Build and Run

We build the program twice: once normally and once with `--use_fast_math`.

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE -arch=native -O3 main.cu -o number-formats
nvcc -U_GNU_SOURCE -arch=native -O3 --use_fast_math main.cu -o number-formats-fast
./number-formats
./number-formats-fast
```
{{< /tab >}}
{{< tab "CMake" >}}
Create `CMakeLists.txt` next to `main.cu`

{{< code file="cuda/post16/CMakeLists.txt" >}}

The generator expression `$<$<COMPILE_LANGUAGE:CUDA>:...>` passes the flag to `nvcc` only, not to the host C++ compiler.

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/number-formats
./build/number-formats-fast
```

Without `-DCMAKE_BUILD_TYPE=Release`, the host code (here, the double-precision reference) is built without optimization and takes longer to run. The GPU timings are not affected.
{{< /tab >}}
{{< /tabs >}}

Output of `./number-formats` on an RTX 4090:

```text
== Round trip float -> format -> float ==
       float         fp16         bf16     fp8 e4m3     fp8 e5m2
         0.1   0.09997559    0.1000977    0.1015625      0.09375
   0.3333333     0.333252    0.3339844      0.34375       0.3125
    3.141593     3.140625     3.140625         3.25            3
         300          300          300          288          320
       70000          inf        70144          448        57344
       1e-05 1.001358e-05 1.001358e-05            0 1.525879e-05

== Sum of 10000 x 0.001 (exact answer: 10) ==
store fp16, accumulate fp16:    4.0000
store fp16, accumulate fp32:   10.0040
store bf16, accumulate bf16:    0.5000
store bf16, accumulate fp32:    9.9945

== out = a * x + y, 33554432 elements ==
float : 0.432 ms, 932 GB/s (92.4% of peak)
half  : 0.216 ms, 932 GB/s (92.4% of peak)
half2 : 0.216 ms, 932 GB/s (92.5% of peak)
max |half2 - float| = 0.000854373

== 32 terms of sin/exp/divide on 4194304 values ==
accurate  0.265 ms  max abs err 1.76e-05  max rel err 8.57e-07
intrinsic 0.084 ms  max abs err 1.89e-05  max rel err 6.48e-06
1e-39 * 0.5 on the GPU = 5e-40
```

The first three sections of `./number-formats-fast` are the same (up to timing noise); the last one is shown in section 6.

---

## 8. Exercise

Add a BF16 version of the elementwise kernel, `axpy_bf162`, using `__nv_bfloat162` and `__hfma2` from `cuda_bf16.h`. Convert the inputs with `__float2bfloat16` and the scalar `a` with `__float2bfloat162_rn`.

Before running it, predict two numbers:

1. Its time, compared with `axpy_half2`.
2. Its maximum difference from the float result, compared with `half2`'s 0.00085.

Hint: look at the bytes per element, and at the mantissa bits in the table of section 1. BF16 has 3 fewer mantissa bits than FP16, and 2^3 = 8.

In summary, a smaller number format moves fewer bytes, and for a memory-bound kernel that directly means less time: FP16 halved our elementwise kernel from 0.432 ms to 0.216 ms at the same 92% of peak bandwidth. The price is range and precision, and the table of sign, exponent, and mantissa bits tells us how much. The dangerous case is accumulation: an FP16 sum of 10,000 × 0.001 got stuck at 4, while the same FP16 data summed in an FP32 register gave 10.004. So we store small and accumulate in FP32. `--use_fast_math` made our math-heavy kernel 3.2 times faster, at 7.6 times the worst relative error for our input range, but it also silently changes every kernel in the file and flushes subnormals to zero. Whichever format or flag you choose, measure the error against a double-precision reference on your real input range. In the next post, we will use these FP16 inputs and FP32 accumulators where they matter most: on the Tensor Cores.

[1]: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF.html "CUDA Math API: Half Precision Intrinsics"
[2]: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP8.html "CUDA Math API: FP8 Intrinsics"
[3]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix "CUDA Programming Guide: Mathematical Functions (error bounds of each function and intrinsic)"
[4]: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math "NVCC: --use_fast_math"
[5]: https://docs.nvidia.com/cuda/floating-point/index.html "Floating Point and IEEE 754 Compliance for NVIDIA GPUs"
[6]: https://arxiv.org/abs/2209.05433 "FP8 Formats for Deep Learning (Micikevicius et al., 2022)"
