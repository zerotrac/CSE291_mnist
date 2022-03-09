from matplotlib import markers
import matplotlib.pyplot as plt


x = ["100", "500", "1000", "5000", "10000", "20000", "60000"]
y1 = [18.542410, 19.039062, 19.886719, 24.828125, 30.089844, 40.753904, 84.394531]
y2 = [84.87952, 82.27218, 81.76144, 81.05438, 81.21542, 81.37120, 81.35784]


def fig1():
    plt.plot(x, y1, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Peak Memory (MB)")
    plt.show()


def fig2():
    plt.plot(x, y2, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (s)")
    plt.show()


fig1()
fig2()