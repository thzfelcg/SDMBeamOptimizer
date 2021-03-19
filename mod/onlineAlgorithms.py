class AlgorithmInterface:
    def __init__(self, obj_f, inp_bounds):
        pass
    
    def suggest(self) ->dict :
        pass
    
    def register(self, X, Y):
        pass
    
    @property
    def isRunning(self):
        return True
    

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
class OnlineBayesian(BayesianOptimization):
    def __init__(self, obj_f, pbounds,
                kind='ucb', kappa=2.567, kappa_decay=1, xi=0.0, kappa_decay_delay=0,
                random_steps=5, max_iter=1500):
        super().__init__(obj_f, pbounds)
        self.f = obj_f
        self.utility = UtilityFunction(kind=kind,
                                    kappa=kappa,
                                    xi=xi,
                                    kappa_decay=kappa_decay,
                                    kappa_decay_delay=kappa_decay_delay)
        self._isRunning = True
        self._nstep = 0
        self._random_steps = max(random_steps, 2)
        self.max_iter = max_iter
        
    def suggest(self):
        self._nstep += 1
        if self._nstep <= self._random_steps:
            return self._space.array_to_params(self._space.random_sample())
        elif self._nstep < self.max_iter:
            return super().suggest(self.utility)
        else:
            return None
    
    def register(self, X, Y):
        super().register(
            params = X,
            target = self.f(**dict(X,**Y))
        )
    
    @property
    def isRunning(self):
        return self._isRunning
        
    def destroy(self):
        self._isRunning = False
        

from queue import Queue
from threading import Thread
import numpy as np
from algo.rcds import RCDS
class OnlineRCDS(RCDS):
    def __init__(self, obj_f, inp_bounds, noise_level=0.1, step = 0.01, max_iter=1500):
        Nvar = len(inp_bounds)
        g_vrange = np.array(list(inp_bounds.values())) # require that inp_bounds is ordered(py>=3.6)
        super().__init__(obj_f, noise_level, Nvar, g_vrange, max_iter)
        self.step = step
        self.pb = inp_bounds # paramater bounds
        self.Xsuggestion = Queue(1)
        self.Yobj = Queue(1)
        self._th = None # the thread that actually runs the algorithm
        self._thc = None # the _th creator that will do the pre&post process before&after _th runs
        
    def _observer(self, X):
        x0 = np.array([
                (X[pn]-self.pb[pn][0])/(self.pb[pn][1]-self.pb[pn][0]) for pn in self.pb.keys()
            ]) # min-max normalization
        self._th = Thread(name='main optimizer',
                          target=self.powellmain,
                          args=(x0, self.step),
                          daemon=True)
        self._th.start()
        self._th.join() # wait until main rcds return
        assert self.Xsuggestion.empty()
        self.Xsuggestion.put(None) # terminate
        
    @property
    def isRunning(self):
        if self._th:
            return self._th.isAlive()
        else:
            return False
    
    def suggest(self) ->dict :
        return self.Xsuggestion.get()
    
    def register(self, X, Y):
        # todo : better start & terminate
        if self._thc is None: # the first time get the X,Y pair
            self._thc = Thread(name='observer',
                               target=self._observer,
                               args=(X,),
                               daemon=True)
            self._thc.start()
            self.Xsuggestion.get() # remove the initial x
        self.Yobj.put(self.objfunc(**dict(X,**Y)))
        
    def func_obj(self,x):
        '''x : a column vector
           return a float
        '''
        # un-normalize
        p = self.g_vrange[:,0]+np.multiply((self.g_vrange[:,1]-self.g_vrange[:,0]),x)
        
        if any(p<self.g_vrange[:,0]) or any(p>self.g_vrange[:,1]):
        # if min(x)<0 or max(x)>1:
            # self.Xsuggestion.put(None) # jump over invalid input
            # self.Yobj.get()
            obj = float('NaN')
        else:
            self.Xsuggestion.put({key:val for key,val in zip(self.pb.keys(), p)})
            obj = self.Yobj.get()
            self.g_cnt +=1
        
        return obj
    
    
from algo.esmin import ES_min
class OnlineES(ES_min):
    def __init__(self, obj_f, inp_bounds, norm_coef=0.05,
                 kES=0.5, alphaES=1.0, w0=500.0, max_iter = 1500):
        self.objf = obj_f # objective function to be minimized
        self.norm_coef = norm_coef
        self.kES = kES
        self.alphaES = alphaES
        self.w0 = w0
        self.dtES = 2*np.pi/(10*1.75*w0)
        self.max_iter = max_iter
        self.bounds = list(inp_bounds.values())

        self.pb = inp_bounds # paramater bounds
        self.Xsuggestion = Queue(1)
        self.Yobj = Queue(1)
        self._th = None # the thread that actually run the algorithm
        self._thc = None # the _th creator that will do the pre&post process before&after _th runs
        
    def _observer(self, X):
        # convert dict to list
        x0 = [X[pn] for pn in self.pb.keys()]
        self._th = Thread(name='main optimizer',
                          target=self.minimize,
                          args=(x0,),
                          daemon=True)
        self._th.start()
        self._th.join() # wait until esmin return
        assert self.Xsuggestion.empty()
        self.Xsuggestion.put(None) # terminate
        
    @property
    def isRunning(self):
        if self._th:
            return self._th.isAlive()
        else:
            return False
    
    def suggest(self) ->dict :
        return self.Xsuggestion.get()
    
    def register(self, X, Y):
        if self._thc is None: # the first time get the X,Y pair
            self._thc = Thread(name='observer',
                               target=self._observer,
                               args=(X,),
                               daemon=True)
            self._thc.start()
            self.Xsuggestion.get() # remove the initial x
        self.Yobj.put(self.objf(**dict(X,**Y)))
        
    def error_func(self,x):
        '''x : a column vector
           return a float
        '''
        # unlike RCDS, ES_min will do normalization/un-normalization inside
        self.Xsuggestion.put({key:val for key,val in zip(self.pb.keys(), x)})
        obj = self.Yobj.get()
        
        return obj