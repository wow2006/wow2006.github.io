import cutlass
import cutlass.cute as cute


@cute.kernel
def hello_kernel():
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if tidx == 0:
        cute.printf("Hello from block {}", bidx)


@cute.jit
def hello():
    hello_kernel().launch(grid=(2, 1, 1), block=(32, 1, 1))


cutlass.cuda.initialize_cuda_context()
hello()
