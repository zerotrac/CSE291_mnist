from typing import List

import const


def binarization(img: List[List[int]]) -> List[List[int]]:
    """
    对于给定的一张 const.SIZE * const.SIZE 大小的灰度图片
    将其二值化：大于等于平均灰度的置为 1，其它置为 0
    """
    average = sum(sum(row) for row in img) / (const.SIZE * const.SIZE)
    for i in range(const.SIZE):
        for j in range(const.SIZE):
            img[i][j] = (1 if img[i][j] >= average else 0)
    
    return img

def compression(img: List[List[int]], group_size = 16) -> List[int]:
    """
    对于给定的一张 const.SIZE * const.SIZE 大小的二值化图片
    将其按照行优先的顺序，每 group_size 个 bit 压缩成一个数，得到一个一维列表，便于存储和相似度计算
    """
    span = sum(img, [])
    compressed_img = list()
    for i in range(0, const.SIZE * const.SIZE, group_size):
        segment = "".join(str(d) for d in span[i:i+group_size])
        compressed_img.append(int(segment, 2))
        
    return compressed_img

def decompression(compressed_img: List[int], group_size = 16) -> List[List[int]]:
    """
    compression 操作的逆操作
    调试用，pipeline 中用不到
    """
    img = [[0] * const.SIZE for _ in range(const.SIZE)]
    idx, bit = 0, group_size - 1
    for i in range(const.SIZE):
        for j in range(const.SIZE):
            img[i][j] = (compressed_img[idx] >> bit) & 1
            if bit == 0:
                idx += 1
                bit = (const.SIZE * const.SIZE - group_size * (len(compressed_img) - 1) if idx == len(compressed_img) - 1 else group_size)
            bit -= 1

    return img