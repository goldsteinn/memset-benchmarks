import csv
import sys
import statistics

KEY_ORDER = ["val", "align", "reuse", "init", "size", "nthreads"]
KILLS = ["iter"]
CATEGORY = "func"
OUTPUT = "time_ns"

results = {}

uniques = []
last = []

key_details = {}
for key in KEY_ORDER:
    key_details[key] = set()

with open(sys.argv[1], newline='\n') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        keys = []
        assert (len(row) - 2) == (len(KEY_ORDER) + len(KILLS))
        for key in KEY_ORDER:
            keys.append(row[key])
            key_details[key].add(row[key])

        tmp = results
        for i in range(0, len(keys)):
            tmp.setdefault(keys[i], {})
            tmp = tmp[keys[i]]

        if row[KEY_ORDER[-1]] not in last:
            last.append(row[KEY_ORDER[-1]])

        if row[CATEGORY] not in uniques:
            uniques.append(row[CATEGORY])

        tmp.setdefault(row[CATEGORY], [])
        tmp[row[CATEGORY]].append(row[OUTPUT])


def output(idx, res, avg):
    if (idx + 2) < len(KEY_ORDER):
        for val in res:
            print("{}={}".format(KEY_ORDER[idx], val), end=",\n")
            output(idx + 1, res[val], avg)
        return

    if (idx + 1) < len(KEY_ORDER):
        print(KEY_ORDER[-1], end=",")
        print(("," * (len(uniques) + 1)).join(last), end=",\n")
        for _ in last:
            print("", end=",")
            for cat in uniques:
                print(cat, end=",")
        print("", end="\n")
        for val in res:
            print("{}={}".format(KEY_ORDER[idx], val), end=",")
            output(idx + 1, res[val], avg)
        return

    for lkey in last:
        total = 0.0
        for cat in uniques:
            assert len(res[lkey][cat]) == int(lkey)
            total += statistics.median(map(float, res[lkey][cat]))
        total = total / float(len(uniques))
        for cat in uniques:
            val = statistics.median(map(float, res[lkey][cat]))
            if avg:
                val = val / total
            print("{:.2f}".format(val), end=",")
        print("", end=",")
    print("", end="\n")


for val in results:
    break
    print(val)
    for vval in results[val]:
        print("\t{}".format(vval))
        for vvval in results[val][vval]:
            print("\t\t{}".format(vvval))
            for vvvval in results[val][vval][vvval]:
                print("\t\t\t{}".format(vvvval))
                for vvvvval in results[val][vval][vvval][vvvval]:
                    print("\t\t\t\t{}".format(vvvvval))
                    for vvvvvval in results[val][vval][vvval][vvvval][vvvvval]:
                        print("\t\t\t\t\t{}".format(vvvvvval))

output(0, results, True)
