import os
import multiprocessing
import datetime
import sys

assert len(sys.argv) > 1, "Usage: <benchmark file> <opt: outfile>"

nthreads = []
NCPUS = multiprocessing.cpu_count()
for i in range(0, 6):
    cpus = 1 << i
    if cpus > NCPUS:
        continue
    nthreads.append(cpus)

for i in range(1, 30):
    cpus = 32 * i
    if cpus > NCPUS:
        continue
    nthreads.append(cpus)

nthreads.append(NCPUS)
nthreads = sorted(list(set(nthreads)))
#nthreads = nthreads[:-1]
sizes = [(4096 << x) for x in range(8, 17)]
iters = (500, 2**33)
aligns = (0) #, 64, 4032)
reuses = (0, 1)
vals = (0, 1)
inits = ("w", "wz", "r")
impls = ["erms", "movdir64", "temporal", "non_temporal"]

impls = ["erms", "temporal", "non_temporal"]

BM = sys.argv[1]

date_uid = str(datetime.datetime.now()).replace(" ", "-").replace(":",
                                                                  "-").replace(
                                                                      ".", "-")
DST_FILE = "results-{}.txt".format(date_uid)
if len(sys.argv) > 2:
    DST_FILE = sys.argv[2]


def os_do(cmd):
    print(cmd)
    return os.system(cmd) == 0


for nthread in nthreads:
    for size in sizes:
        it = max(int(iters[1] / size), iters[0])
        for align in aligns:
            for reuse in reuses:
                for val in vals:
                    for init in inits:
                        for impl in impls:
                            assert os_do(
                                "./{} {} {} {} {} {} {} {} {} >> {}".format(
                                    BM, nthread, size, it, align, reuse, val,
                                    init, impl, DST_FILE))
