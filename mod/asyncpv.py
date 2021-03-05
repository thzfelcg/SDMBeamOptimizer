import numpy as np
import epics as ep
from epics.ca import CAThread
from queue import Queue

class listCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def put(self, newval):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(newval)
        
    def __getitem__(self, key):
        if self.memory:
            return self.memory[key]
        else:return 0
    
    def clear(self):
        self.memory.clear()
        
    def isFull(self):
        return len(self.memory) == self.capacity

    @property
    def mean(self):
        if self.memory:
            return np.mean(self.memory)
        else:return 0
        
    @property
    def std(self):
        if self.memory:
            return np.std(self.memory)
        else:return np.inf
        
    def __len__(self):
        return len(self.memory)
    def __repr__(self):
        return str(self.memory)

class stableList(listCache):
    def __init__(self, capacity, threshold):
        super().__init__(capacity)
        self.std_threshold = threshold

    @property
    def isStable(self):
        return len(self.memory) >= self.capacity and self.std < self.std_threshold
    
class AsyncPV(ep.PV):
    def __init__(self, pvname, cacheSize=1):
        super().__init__(pvname)
        if cacheSize > 0 : self.makeCache(cacheSize)
        self._q = Queue(1)
        self._th = CAThread(target = self._put, daemon=True)
        self._th.start()
        
    def makeCache(self, cacheSize=1):
        self.cache = listCache(cacheSize)
        self.add_callback(self._update)
        
    def _update(self, pvname=None, value=None, char_value=None, **kw):
        self.cache.put(value)
        
    def _put(self):
        while True:
            self.put(self._q.get())
            
#     def terminate(self):
#         self._q = None
#         self.async_put = None
    
    def async_put(self, value):
        '''directly use pv.put inside a CA callback function will cause problems,
        thus async_put can workaround this by invoking pv.put in another thread'''
        self._q.put(value)
        
    def __getitem__(self, key):
        if self.cache.memory:
            return self.cache.memory[key]
        else:return self.get()