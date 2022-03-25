# CSE291_mnist

The original Scad code for our project is in the Mnist directory, and the improved Scad code (using small batches as stated in our report) is in the Minist_Improved directory, and to change the batch size, just modify the global variable BATCH_SIZE in each function. Code in Minist_test directory is used for memory profiling.

To run the whole project on Scad, just push the Mnist or Mnist_Improved directory under the path runtime/test/src/, then follow the steps in Scad README. And you should set up two memory server, for func1.o.py and func2.o.py, they will store the intermediate data in memory server 1. For func3.o.py, it will store the intermediate data in the memory server 2. For func4.o.py, it will fetch data from memory server 1 and memory server 2, then store the results in memory server 2. For func5.o.py, it will fetch data from memory server 2, and output the final figure in /data directory.
