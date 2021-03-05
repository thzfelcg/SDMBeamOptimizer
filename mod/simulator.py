from collections import Iterable
class Simulator:
    def __init__(self, in_pvs, out_pvs, sig_pv, model):
        '''
        in_pvs, out_pvs : list of AsyncPV
        sig_pv : PV
        model : callable, recive values in in_pvs
        eg :
        sim = Simulator([x1,x2], [y], sigmodel, model)
        the result of model(x1, x2) will be put into y when sigmodel changes to True
        '''
        self.i = in_pvs
        self.o = out_pvs
        self.s = sig_pv
        self.f = model
        self.sig_cbk = sig_pv.add_callback(self._response)
        
    def _response(self, pvname=None, value=None, char_value=None, **kw):
        if value:
            X = {pv.pvname:pv[-1] for pv in self.i}
            Y = self.f(**X)
            if not isinstance(Y, Iterable):
                Y = [Y,]
            for pv,y in zip(self.o,Y):
                pv.async_put(y)
        
    def destroy(self):
        self.s.remove_callback(self.sig_cbk)
        

class Optimizer:
    def __init__(self, in_pvs, out_pvs, sig_pv, algo):
        self.i = in_pvs
        self.o = out_pvs
        self.s = sig_pv
        self.sig_cbk = sig_pv.add_callback(self._optim_step)
        
        self.algo = algo
    
    def _optim_step(self, pvname=None, value=None, char_value=None, **kw):
        if not value:
            X = {pv.pvname:pv[-1] for pv in self.i}
            Y = {pv.pvname:pv[-1] for pv in self.o}
            self.algo.register(X, Y)
            X_suggestion:dict = self.algo.suggest()
            #X_suggestion.setdefault()
            if X_suggestion is None:
                if not self.algo.isRunning():
                    print('new input is None, optimizing algorithm has terminated')
            else:
                for pv in self.i:
                    pv.async_put(X_suggestion[pv.pvname])
       
    def destroy(self):
        self.s.remove_callback(self.sig_cbk)