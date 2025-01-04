#ifndef MACROS_H__
#define MACROS_H__

#ifdef CUDA_COMPILE
#define PRJ_ALLOC(src, s, c) cudaMalloc((void**)&src, sizeof(s)*c);
#define PRJ_FREE(src) cudaFree(src);
#define PRJ_MEMCPY(dest, src, s, kind) cudaMemcpy(dest, src, s, kind);
#define PRJ_FUNC_CALL(f, ...) f<<<CUDA_BLOCK_NUM, THREADS_PER_BLOCK>>>(__VA_ARGS__)
#define PRJ_FUNC_CALL_SINGLE(f, ...) f<<<1, 1>>>(__VA_ARGS__)
#define PRJ_CUDA_WAIT() cudaDeviceSynchronize();
#else
#define PRJ_ALLOC(src, s, c) src = new s[c];
#define PRJ_FREE(src) delete[] src;
#define PRJ_MEMCPY(dest, src, s, kind) memcpy(dest, src, s);
#define PRJ_FUNC_CALL(f, ...)   for (uint64_cu partIndex=0;partIndex<CUDA_BLOCK_NUM*THREADS_PER_BLOCK;partIndex++) { \
                                    f(__VA_ARGS__, partIndex); \
                                }
#define PRJ_FUNC_CALL_SINGLE(f, ...) f(__VA_ARGS__)
#define PRJ_CUDA_WAIT()
#endif

#endif
