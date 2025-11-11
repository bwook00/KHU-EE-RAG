import math
import os
import pathlib
from typing import List
import pytest

from langchain.vectorstores import Chroma

def main():
    retrieval_gt =  [['test-1', 'test-2'], ['test-3']]
    pred = ['test-1', 'pred-1', 'test-2', 'pred-3']  # recall: 0.5, precision: 0.5, f1: 0.5

    result = retrieval_ndcg(gt=retrieval_gt, pred=pred)
    return  result


def retrieval_ndcg(gt: List[List[str]], pred: List[str]):
    """
    Compute nDCG (Normalized Discounted Cumulative Gain) for retrieval.

    :param gt: 2-d list of ground truth ids.
        It contains and/or connections between ids.
    :param pred: Prediction ids.
    :return: The nDCG score.
    """
    if not pred:
        return 0.0

    gt_all = set()
    for gt_group in gt:
        gt_all.update(gt_group)

    if not gt_all:
        return 0.0

    dcg = 0.0
    for i, pred_id in enumerate(pred):
        if pred_id in gt_all:
            dcg += 1.0 / math.log2(i + 2)

    ideal_length = min(len(gt_all), len(pred))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_length))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg

if __name__ == "__main__":
    result = main()
    print(0.7039180890341347 == pytest.approx(result, rel=1e-4))

