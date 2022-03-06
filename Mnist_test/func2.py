#@ type: compute
#@ parents:
#@   - func1
#@ dependents:
#@   - func3
#@ corunning:
#@   mem1:
#@     trans: mem1
#@     type: rdma

import struct
import pickle
import sys
import pickle

TRAIN_SIZE = 60000
SIZE = 28

BATCH_SIZE = 1000

OUTPUT = "data/labels_store"

def sendToServer(output, data, remoteIndex):
    with open(output, "r+b") as fin:
         dataBytes = pickle.dumps(data)
         length = len(dataBytes)
         lenBytes = struct.pack('@I', length)
         fin.seek(remoteIndex, 0)
         print(fin.tell())
         fin.write(lenBytes)
         print(fin.tell())
         remoteIndex += 4
         fin.seek(remoteIndex, 0)
         fin.write(dataBytes)
         print(fin.tell())
         remoteIndex += length
    return remoteIndex

def test():

    labels = list()

    filename = "data/mnist_labels"

    item_count = TRAIN_SIZE

    remoteIndex = 0
    count = 0
    with open(OUTPUT, "r+b") as fin:
         fin.seek(remoteIndex, 0)
         countBytes = struct.pack('@I', count)
         fin.write(countBytes)
         remoteIndex += 4
    

    with open(filename, "rb") as fin:
        assert int.from_bytes(fin.read(4), byteorder="big") == 0x00000801
        N = int.from_bytes(fin.read(4), byteorder="big")
        assert N == item_count
        #assert (N := int.from_bytes(fin.read(4), byteorder="big")) == item_count

        for t in range(N):
            if t % 1000 == 0:
                print(f"read_mnist_label() index = {t}")
            if t % BATCH_SIZE == 0 and t != 0:
               remoteIndex = sendToServer(OUTPUT, labels, remoteIndex)
               count += 1
               labels.clear()
                

            label = int.from_bytes(fin.read(1), byteorder="big")
            labels.append(label)
        
        if len(labels) != 0:
            remoteIndex = sendToServer(OUTPUT, labels, remoteIndex)
            count += 1
            labels.clear()

        with open(OUTPUT, "r+b") as fin:
             fin.seek(0, 0)
             fin.write(struct.pack('@I', count))
        return {}


if __name__ == "__main__":
     test()
