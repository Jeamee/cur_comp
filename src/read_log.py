from argparse import ArgumentParser
from collections import defaultdict
import pandas as pd
import re


def main():
    parser = ArgumentParser()
    parser.add_argument("log", type=str)
    parser.add_argument("csv", type=str)
    args = parser.parse_args()

    val_extractor = re.compile("\d.\d+")
    key_extractor = re.compile("\'[A-Za-z]+\'?")

    index = defaultdict(list)

    with open(args.log, "r") as reader:
        for line in reader.readlines():
            if "total" in line:
                vals = val_extractor.findall(line)
                keys = key_extractor.findall(line)
                index["total"].append(float(vals[0]))
                for key, val in zip(keys, vals[1:]):
                    index[key].append(val)

    df = pd.DataFrame(index)
    df.to_csv(args.csv)


if __name__ == "__main__":
    main()

