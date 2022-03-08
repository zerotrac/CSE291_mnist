from cv2 import line
from typing import List

import matplotlib.pyplot as plt

# Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
# Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
  
# plt.plot(Year, Unemployment_Rate, color='red', marker='o')
# plt.title('Unemployment Rate Vs Year', fontsize=14)
# plt.xlabel('Year', fontsize=14)
# plt.ylabel('Unemployment Rate', fontsize=14)
# plt.grid(True)
# plt.show()

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
# print(train_images)
# print(ts)

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