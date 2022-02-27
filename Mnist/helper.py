import const

from typing import List


def helper_print_image(img: List[List[int]]) -> None:
    """
    输出给定的一张 const.SIZE * const.SIZE 大小的二值化图片
    调试用，pipeline 中用不到
    """
    for i in range(const.SIZE):
        for j in range(const.SIZE):
            print(img[i][j], end=" ")
        print()