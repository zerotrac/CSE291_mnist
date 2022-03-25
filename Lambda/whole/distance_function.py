from typing import List


def count_one(x: int) -> int:
    """
    计算 x 二进制表示中 1 的个数
    """
    return bin(x).count("1")

def compressed_iou(x: List[int], y: List[int]) -> float:
    """
    用于 KNN 的相似度函数
    x 和 y 是两个压缩好的一维列表，相似度就是交并比
    """
    numerator = denominator = 0
    for xk, yk in zip(x, y):
        numerator += count_one(xk & yk)
        denominator += count_one(xk | yk)
    return numerator / denominator