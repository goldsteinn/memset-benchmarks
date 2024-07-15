#include <assert.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>
#define BENCH_FUNC __attribute__((noinline, aligned(4096)))
#define COMPILER_DO_NOT_OPTIMIZE_OUT(X)                                        \
    asm volatile("" : : "r,m,v"(X) : "memory")
static struct timespec
get_ts() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

static uint64_t
to_ns(struct timespec ts) {
    return ts.tv_sec * 1000 * 1000 * 1000 + ts.tv_nsec;
}
static uint64_t
get_ns() {
    struct timespec ts = get_ts();
    return to_ns(ts);
}


static uint64_t
get_ns_dif(struct timespec ts_start, struct timespec ts_end) {
    return to_ns(ts_end) - to_ns(ts_start);
}

static double
gb_sec(uint64_t bytes, uint64_t ns) {
    double d_bytes = bytes;
    double d_ns    = ns;
    double d_gb    = d_bytes / ((double)1024 * 1024 * 1024);
    double d_sec   = d_ns / (double)(1000 * 1000 * 1000);
    return d_gb / d_sec;
}


static uint32_t          g_iter;
static uint32_t          g_init_thread;
static uint32_t          g_reuse;
static uint32_t          g_align;
static size_t            g_size;
static const char *      g_init_method;
static pthread_barrier_t g_barrier;


#define NAME V_TO_STR(FUNC)
typedef void (*memset_func_t)(uint8_t *, int, size_t);
typedef void * (*bench_func_t)(void *);

typedef struct benchmark {
    const char *  name;
    bench_func_t  bench;
    memset_func_t func;
} benchmark_t;

typedef struct targs {
    uint8_t * dst;

    pthread_t tid;
    uint64_t  ns_out;
    uint32_t  val;
} targs_t;


static void
memset_t(uint8_t * dst, int val, size_t len) {
    __m256i v0 = _mm256_set1_epi8((char)val);
#define VEC_SIZE "32"
    // clang-format off
    __asm__ volatile(
        ".p2align 6\n"
        "1:\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 0)(%[dst])\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 1)(%[dst])\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 2)(%[dst])\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 3)(%[dst])\n"
        "subq $(" VEC_SIZE " * -4), %[dst]\n"
        "addq $(" VEC_SIZE " * -4), %[len]\n"
        "jg 1b"
        : [dst] "+r"(dst), [len] "+r"(len), [v0] "=&v"(v0)
        : [val] "r" (val)
        : "cc", "memory");
    // clang-format on
#undef VEC_SIZE
}


static void
memset_nt(uint8_t * dst, int val, size_t len) {
    __m256i v0 = _mm256_set1_epi8((char)val);
#define VEC_SIZE "32"
    // clang-format off
    __asm__ volatile(
        ".p2align 6\n"
        "1:\n"
        "vmovntdq %[v0], (" VEC_SIZE " * 0)(%[dst])\n"
        "vmovntdq %[v0], (" VEC_SIZE " * 1)(%[dst])\n"
        "vmovntdq %[v0], (" VEC_SIZE " * 2)(%[dst])\n"
        "vmovntdq %[v0], (" VEC_SIZE " * 3)(%[dst])\n"
        "subq $(" VEC_SIZE " * -4), %[dst]\n"
        "addq $(" VEC_SIZE " * -4), %[len]\n"
        "jg 1b\n"
        "sfence\n"
        : [dst] "+r"(dst), [len] "+r"(len), [v0] "=&v"(v0)
        : [val] "r" (val)
        : "cc", "memory");
    // clang-format on
#undef VEC_SIZE
}


static void
memset_cd(uint8_t * dst, int val, size_t len) {
    __m256i  v0 = _mm256_set1_epi8((char)val);
    uint8_t *begin, *begin_save;
#define VEC_SIZE "32"
    // clang-format off
    __asm__ volatile(
        "vmovdqa %[v0], (" VEC_SIZE " * 0)(%[dst])\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 1)(%[dst])\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 2)(%[dst])\n"
        "vmovdqa %[v0], (" VEC_SIZE " * 3)(%[dst])\n"
        "movq %[dst], %[begin_save]\n"
        "subq $(" VEC_SIZE " * -4), %[dst]\n"
        "addq $(" VEC_SIZE " * -4), %[len]\n"
        ".p2align 6\n"
        "1:\n"
        "movq %[begin_save], %[begin]\n"
        "movdir64b  (%[begin]), %[dst]\n"
        "subq $-64, %[dst]\n"
        "movdir64b  (%[begin]), %[dst]\n"
        "subq $-64, %[dst]\n"
        "addq $(" VEC_SIZE " * -4), %[len]\n"
        "jg 1b"
        : [dst] "+r"(dst), [len] "+r"(len), [v0] "=&v"(v0),
          [begin] "=&r"(begin), [begin_save] "=&r"(begin_save)
        : [val] "r" (val)
        : "cc", "memory");
    // clang-format on
#undef VEC_SIZE
}

static size_t
use(uint8_t * dst, size_t len) {
    __m256i v0, v1, v2, v3;
#define VEC_SIZE "32"
    // clang-format off
    __asm__ volatile(
        "vpxor %[v0], %[v0], %[v0]\n"
        "vpxor %[v1], %[v1], %[v1]\n"
        "vpxor %[v2], %[v2], %[v2]\n"
        "vpxor %[v3], %[v3], %[v3]\n"
        ".p2align 6\n"
        "1:\n"
        "vpxor (" VEC_SIZE " * 0)(%[dst]), %[v0]\n"
        "vpxor (" VEC_SIZE " * 1)(%[dst]), %[v1]\n"
        "vpxor (" VEC_SIZE " * 2)(%[dst]), %[v2]\n"
        "vpxor (" VEC_SIZE " * 3)(%[dst]), %[v3]\n"
        "subq $(" VEC_SIZE " * -4), %[dst]\n"
        "addq $(" VEC_SIZE " * -4), %[len]\n"
        "jg 1b"
        "vpxor %[v0], %[v1], %[v1]\n"
        "vpxor %[v2], %[v3], %[v3]\n"
        "vpxor %[v1], %[v3], %[v3]\n"
        "vpcmpeqb %[v0], %[v0], %[v0]\n"
        "vpor %[v0], %[v3], %[v3]\n"
        "vmovq %[v3], %[len]\n"
        "incq %[len]\n"
        : [dst] "+r"(dst),  [len] "+r"(len), [v0] "=&v"(v0),
          [v1] "=&v"(v1), [v2] "=&v"(v2), [v3] "=&v"(v3)
        :
        : "cc", "memory");
    // clang-format on
#undef VEC_SIZE
    return len;
};

static void
memset_erms(uint8_t * dst, int val, size_t len) {
    __asm__ volatile("rep stosb" : "+D"(dst), "+c"(len) : "a"(val) : "memory");
}


static int
check_memset(const uint8_t * p,
             size_t          align,
             size_t          sz,
             size_t          sz_end,
             int             val) {
    size_t  i    = 0;
    int     okay = 0;
    uint8_t val8 = val & 0xff;
    for (; i < align; ++i) {
        okay |= (p[i] != 0);
    }
    if (okay) {
        fprintf(stderr, "Bad0\n");
    }

    for (; i < (sz + align) && i < sz_end; ++i) {
        okay |= (p[i] != val8);
    }
    if (okay) {
        fprintf(stderr, "Bad1\n");
    }
    for (; i < sz_end; ++i) {
        okay |= (p[i] != 0);
    }
    if (okay) {
        fprintf(stderr, "Bad2\n");
    }
    return okay == 0;
}

static void
test(size_t incr, memset_func_t func) {
    size_t    test_end = 1UL << 20;
    uint8_t * dst      = (uint8_t *)mmap(NULL, test_end + 8192, PROT_NONE,
                                         MAP_ANONYMOUS | MAP_PRIVATE, -1, 0L);

    assert(dst != MAP_FAILED);
    dst += 4096;

    size_t size = incr;

    for (; size < test_end; size += incr) {
        size_t size_end = size + 4095;
        size_end &= -4096;
        size_t align;
        assert(mprotect(dst, size_end, PROT_WRITE | PROT_READ) == 0);
        for (align = 0; align < 4096; align += 64) {
            if ((align + size) > size_end) {
                break;
            }

            memset(dst + align, -1, size);
            assert(check_memset(dst, align, size, size_end, -1));
            memset(dst + align, 55, size);
            assert(check_memset(dst, align, size, size_end, 55));
            memset(dst + align, 0, size);
            assert(check_memset(dst, align, size, size_end, 0));
        }
        assert(mprotect(dst, size_end, PROT_NONE) == 0);
    }

    munmap(dst - 4096, test_end + 8192);
}

static uint8_t *
init_mem(const char * method, uint64_t size, uint64_t align) {
    uint8_t * dst = (uint8_t *)mmap(NULL, size + align, PROT_READ | PROT_WRITE,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0L);
    assert(dst != MAP_FAILED);
    if (strcmp(method, "atomic_rwn") == 0) {
        for (size_t j = 0; j < (size + align); j += 4096) {
            __atomic_fetch_and(dst + j, 0xff, __ATOMIC_RELAXED);
        }
    }
    else if (strcmp(method, "atomic_rw") == 0) {
        for (size_t j = 0; j < (size + align); j += 4096) {
            __atomic_fetch_or(dst + j, 0xff, __ATOMIC_RELAXED);
        }
    }
    else if (strcmp(method, "rwn") == 0) {
        for (size_t j = 0; j < (size + align); j += 4096) {
            dst[j] &= 0xff;
        }
    }
    else if (strcmp(method, "rw") == 0) {
        for (size_t j = 0; j < (size + align); j += 4096) {
            dst[j] |= 0xff;
        }
    }
    else if (strcmp(method, "w") == 0) {
        for (size_t j = 0; j < (size + align); j += 4096) {
            dst[j] = 0xff;
        }
    }
    else if (strcmp(method, "wz") == 0) {
        for (size_t j = 0; j < (size + align); j += 4096) {
            dst[j] = 0x0;
        }
    }
    else if (strcmp(method, "r") == 0) {
        for (size_t j = 0; j < (size + align); j += 4096) {
            COMPILER_DO_NOT_OPTIMIZE_OUT(dst[j]);
        }
    }
    else if (strcmp(method, "none") == 0) {
        /* Do nothing.  */
    }
    else {
        printf("Missing init stategy\n");
        printf(
            "Available\n"
            "\t'atomic_rwn': atomic read+write but non-setting\n"
            "\t'atomic_rw': atomic read+write\n"
            "\t'rwn': read+write but non-setting\n"
            "\t'rw': read+write\n"
            "\t'wz': write zero\n"
            "\t'r': read only\n");
        return NULL;
    }
    return dst + align;
}


#define make_bench_func(name, func)                                            \
    static void * BENCH_FUNC bench_impl_##name(void * arg) {                   \
        struct timespec start, end;                                            \
        uint32_t        bench_iter = g_iter;                                   \
        uint32_t        reuse      = g_reuse;                                  \
        size_t          sz         = g_size;                                   \
        bench_iter &= -1;                                                      \
        assert(reuse <= 1);                                                    \
                                                                               \
        uint8_t * dst = ((targs_t *)arg)->dst;                                 \
        int       val = ((targs_t *)arg)->val;                                 \
        assert((dst == NULL) == (g_init_thread == 1));                         \
        if (g_init_thread == 1) {                                              \
            dst = init_mem(g_init_method, g_size, g_align);                    \
            assert(dst != NULL);                                               \
        }                                                                      \
        pthread_barrier_wait(&g_barrier);                                      \
                                                                               \
        if (reuse) {                                                           \
            start = get_ts();                                                  \
            __asm__ volatile(".p2align 7\n" : : :);                            \
            for (; bench_iter; --bench_iter) {                                 \
                func(dst, val, sz);                                            \
                COMPILER_DO_NOT_OPTIMIZE_OUT(use(dst, sz));                    \
            }                                                                  \
            end = get_ts();                                                    \
        }                                                                      \
        else {                                                                 \
            start = get_ts();                                                  \
            __asm__ volatile(".p2align 7\n" : : :);                            \
            for (; bench_iter; --bench_iter) {                                 \
                func(dst, val, sz);                                            \
            }                                                                  \
            end = get_ts();                                                    \
        }                                                                      \
        ((targs_t *)arg)->ns_out = get_ns_dif(start, end);                     \
        return NULL;                                                           \
    }

make_bench_func(erms, memset_erms);
make_bench_func(movdir64, memset_cd);
make_bench_func(temporal, memset_t);
make_bench_func(non_temporal, memset_nt);

benchmark_t G_benchmarks[] = { { "erms", bench_impl_erms, memset_erms },
                               { "movdir64", bench_impl_movdir64, memset_cd },
                               { "temporal", bench_impl_temporal, memset_t },
                               { "non_temporal", bench_impl_non_temporal,
                                 memset_nt } };
int
main(int argc, char ** argv) {
    if (argc < 9) {
        if (strcmp(argv[1], "test") == 0) {

            for (size_t i = 0; i < sizeof(G_benchmarks) / sizeof(benchmark_t);
                 ++i) {
                printf("Testing: %s\n", G_benchmarks[i].name);
                test(1024, G_benchmarks[i].func);
            }
            return 0;
        }


        printf(
            "Usage: %s <'test'/nthreads> <size> <iter> <align> <reuse> <val> <init> <impl>\n",
            argv[0]);
        return 0;
    }


    long   nthreads = strtol(argv[1], NULL, 10);
    char * end;
    size_t size = strtoul(argv[2], &end, 10);
    if (strncasecmp(end, "kb", 2) == 0) {
        size *= 1024;
    }
    else if (strncasecmp(end, "mb", 2) == 0) {
        size *= 1024 * 1024;
    }
    else if (strncasecmp(end, "gb", 2) == 0) {
        size *= 1024 * 1024 * 1024;
    }
    uint32_t iter = strtoul(argv[3], NULL, 10);
    iter &= -1;
    uint32_t align = strtoul(argv[4], NULL, 10);
    uint32_t reuse = strtoul(argv[5], NULL, 10);

    assert(reuse <= 1);
    align %= 4096;
    assert(nthreads != 0 && size != 0 && iter != 0);

    int val = atoi(argv[6]);

    uint32_t init_thread;
    if (!strcmp(argv[8], "main")) {
        init_thread = 0;
    }
    else if (!strcmp(argv[8], "thread")) {
        init_thread = 1;
    }
    else {
        printf(
            "Missing init location\nAvailbe\n"
            "\t'main'  : Initialize memory in main\n"
            "\t'thread': Initialize memory in the thread\n");
        return 2;
    }

    const benchmark_t * bm = NULL;
    for (size_t i = 0; i < sizeof(G_benchmarks) / sizeof(benchmark_t); ++i) {
        if (!strcmp(argv[9], G_benchmarks[i].name)) {
            bm = &G_benchmarks[i];
            break;
        }
    }
    if (bm == NULL) {
        printf("Unknown benchmark: %s\n", argv[9]);
        printf("Available:\n");
        for (size_t i = 0; i < sizeof(G_benchmarks) / sizeof(benchmark_t);
             ++i) {
            printf("\t%s\n", G_benchmarks[i].name);
        }
        return 1;
    }
    g_init_thread = init_thread;
    g_iter        = iter;
    g_size        = size;
    g_align       = align;
    g_init_method = argv[7];
    assert(pthread_barrier_init(&g_barrier, NULL, nthreads) == 0);
    pthread_attr_t attr;

    assert(pthread_attr_init(&attr) == 0);
    assert(pthread_attr_setstacksize(&attr, 524288) == 0);

    targs_t targs[nthreads];
    for (long i = 0; i < nthreads; ++i) {
        uint8_t * dst = NULL;
        if (init_thread == 0) {
            dst = init_mem(g_init_method, g_size, g_align);
            assert(dst != NULL);
        }

        targs[i].dst = dst;
        targs[i].val = val;
        assert(pthread_create(&(targs[i].tid), &attr, bm->bench,
                              (void *)(targs + i)) == 0);
    }
    for (long i = 0; i < nthreads; ++i) {
        assert(pthread_join(targs[i].tid, NULL) == 0);
    }
    for (long i = 0; i < nthreads; ++i) {
        munmap(targs[i].dst - align, size + align);
    }
    printf("func,nthreads,iter,size,align,reuse,val,init,init_loc,time_ns\n");
    for (long i = 0; i < nthreads; ++i) {
        printf("%s,%ld,%u,%zu,%u,%u,%d,%s,%s,%lu\n", bm->name, nthreads, iter,
               size, align, reuse, val, argv[7], argv[8], targs[i].ns_out);
    }
}
