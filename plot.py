from typing import List

import matplotlib.pyplot as plt


lines = [[int(x) for x in line.strip().split()] for line in open("memory.dat", "r")]

def get(idx: int) -> List[int]:
    return [line[idx] for line in lines]

train_images = get(0)
train_labels = get(1)
boxes = get(2)
img = get(3)
predictions = get(4)

ts = [0.1 * i for i in range(len(lines))]

plt.yscale("log")

plt.xticks([i for i in range(0, int(len(lines) * 0.1) + 2, 3)])
plt.yticks([10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8], ["100B", "1KB", "10KB", "100KB", "1MB", "10MB", "100MB"])

plt.plot(ts, train_images, color="red")
plt.plot(ts, train_labels, color="orange")
plt.plot(ts, boxes, color="blue")
plt.plot(ts, img, color="green")
plt.plot(ts, predictions, color="purple")

plt.legend(["Training Images", "Training Labels", "Digit Boxes", "OCR Image", "Results"], loc="upper left")
plt.xlabel("Time (s)")
plt.ylabel("Memory Consumption")
plt.grid(True)

plt.show()