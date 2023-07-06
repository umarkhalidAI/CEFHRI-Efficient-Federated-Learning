import multiprocessing
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
import multiprocessing.pool
import time

from random import randint
from typing import Callable, Iterable, Any

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
# class MyPool(multiprocessing.pool.Pool):
#     Process = NoDaemonProcess

# NOTE: For CUDA 11, use this definition in MyPool instead
class MyPool(multiprocessing.pool.Pool):

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None):
        super().__init__(processes,initializer,initargs,maxtasksperchild,context)

    def Process(self, *args, **kwds):
        proc = super(MyPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc