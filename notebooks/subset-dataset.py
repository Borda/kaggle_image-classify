"""
Create a subset with more frequent labels
> python notebooks/subset-dataset.py TEMP/train-from-kaggle.csv 1500
"""

import itertools
import os.path
import sys

import numpy as np
import pandas as pd

COUNT_THR = 1000
CSV_NAME = "train-from-kaggle.csv"
COL_LABELS = 'attribute_ids'


def main(path_csv: str = CSV_NAME, count_thr: int = COUNT_THR):
    print(f"Loafing: {path_csv}")
    df_train = pd.read_csv(path_csv)
    print(f"Samples: {len(df_train)}")
    labels_all = list(itertools.chain(*[[int(lb) for lb in lbs.split(" ")] for lbs in df_train[COL_LABELS]]))
    lb_hist = dict(zip(range(max(labels_all) + 1), np.bincount(labels_all)))
    print(f"Filter: {count_thr}")
    df_hist = pd.DataFrame([dict(lb=lb, count=count) for lb, count in lb_hist.items()
                            if count > count_thr]).set_index("lb")
    print(f"Reductions: {len(lb_hist)} >> {len(df_hist)}")

    allowed_lbs = set(list(df_hist.index))
    df_train[COL_LABELS] = [
        " ".join([lb for lb in lbs.split() if int(lb) in allowed_lbs]) for lbs in df_train[COL_LABELS]
    ]
    df_train[COL_LABELS].replace('', np.nan, inplace=True)
    df_train.dropna(subset=[COL_LABELS], inplace=True)
    print(f"Samples: {len(df_train)}")
    name_csv, _ = os.path.splitext(os.path.basename(path_csv))
    path_csv = os.path.join(os.path.dirname(path_csv), f"{name_csv}_min-lb-sample-{count_thr}.csv")
    df_train.to_csv(path_csv)

    labels_all = list(itertools.chain(*[[int(lb) for lb in lbs.split(" ")] for lbs in df_train[COL_LABELS]]))
    print(f"sanity check - nb labels: {len(set(labels_all))}")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
