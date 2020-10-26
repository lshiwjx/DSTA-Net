# import torchvision
import torch
import numpy as np
# import matplotlib.pyplot as plt
import time


class TimerBlock:
    """
    with TimerBlock(title) as block:
        block.log(msg)
        block.log2file(addr,msg)
    """

    def __init__(self, title):
        print("{}".format(title))
        self.content = []
        self.addr = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.time() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        s = "  [{:.3f}{}] {}".format(duration, units, string)
        print(s)
        self.content.append(s + '\n')
        fid = open(self.addr, 'a')
        fid.write("%s\n" % (s))
        fid.close()

    def save(self, fid):
        f = open(fid, 'a')
        f.writelines(self.content)
        f.close()

    def log2file(self, fid, string):
        fid = open(fid, 'a')
        fid.write("%s\n" % (string))
        fid.close()


class IteratorTimer():
    """
    An iterator to produce duration. self.last_duration
    """

    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = self.iterator.__next__()
        self.last_duration = (time.time() - start)
        return n

    next = __next__


if __name__ == '__main__':
    with TimerBlock('Test') as block:
        block.log('1')
        block.log('2')
        block.save('../train_val_test/runs/test.txt')
