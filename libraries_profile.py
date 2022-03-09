# import cv2
# import numpy as np
# import pickle

import matplotlib.pyplot as plt


# while True:
    # pass


x = ["raw Python", "opencv", "numpy", "pickle"]
y1 = [6.3, 32.1, 26.1, 6.5]
y2 = [num - y1[0] for num in y1[1:]]

x1_pos = [0.5, 1.3, 2.3, 3.3]
x2_pos = [1.7, 2.7, 3.7]
xtick_pos = [0.5, 1.5, 2.5, 3.5]

plt.bar([x1_pos[0]], [y1[0]], width=0.4, label="Raw", color="orange")
plt.bar(x1_pos[1:], y1[1:], width=0.4, label="Total", color="blue")
plt.bar(x2_pos, y2, width=0.4, label="Net", color="green")

plt.xlabel("Python Libraries")
plt.ylabel("Memory Comsumption (MB)")

plt.xticks(xtick_pos, x)
plt.legend(loc="best")

plt.show()


""" Results:
Nothing: 6.3 MB
cv2: 32.1 MB
numpy: 26.1 MB
pickle: 6.5 MB
"""
