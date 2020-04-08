import numpy as np
import scipy.special as sp

class TPSA_meta(type):
    '''Meta class allowing the truncation order to be changed at runtime'''
    def __init__(cls,*args,**kwargs):
        cls._ORDER = 2
        cls._binom_coeffs = [[sp.binom(n, k) for k in range(cls._ORDER+1)]
                             for n in range(cls._ORDER+1)]

    @property
    def order(cls):
        return cls._ORDER

    @order.setter
    def order(cls,order):
        if isinstance(order,int):
            if order>1:
                cls._ORDER=order
                cls._binom_coeffs = [[sp.binom(n, k) for k in range(order+1)] for n in range(order+1)]
            else:
                raise ValueError("Order must be greater than 1")
        else:
            raise TypeError("Truncation order must be integer")

class TPSA(metaclass=TPSA_meta):
    '''Truncated power-series algebra

    _ORDER determines where to truncate, can reset with TPSA.order
    _binom_coeffs are computed when _ORDER is set, used in __mul__
    '''

    def __init__(self,value):
        '''Generate a truncated power-series

        value: should be either a single float to specify
        the value of a function or list of floats to specify
        the function value and derivatives up to some order
        '''
        if isinstance(value,(list,tuple,np.ndarray)):
            if len(value)!=TPSA._ORDER+1:
                raise ValueError(f"Input is length {len(value)}, should be {TPSA._ORDER+1}")
            else:
                self.fx = np.array(value).astype(float)
        elif isinstance(value,(int,float)):
            self.fx = np.zeros(TPSA._ORDER + 1)
            self.fx[0] = float(value)
            self.fx[1] = 1.0
        else:
            raise TypeError(f"Value is invalid type, must be list/tuple or single float")

    def __str__(self):
        return str(self.fx)

    #Addition
    def __add__(self,other):
        '''self is first operand, other is second'''
        try:
            return TPSA(self.fx+other.fx,len(self.fx))
        except AttributeError:
            temp=self.fx.copy()
            temp[0]+=other
            return TPSA(temp)

    def __radd__(self, other):
        '''other is first operand, self is second'''
        return self.__add__(other)

    def __iadd__(self, other):
        try:
            self.fx+=other.fx
            return self
        except AttributeError:
            self.fx[0]+=other
            return self

    #Subtraction
    def __neg__(self):
        val=-self.fx
        return TPSA(val)

    def __sub__(self,other):
        '''self is first, other second'''
        return self.__add__(-other)

    def __rsub__(self, other):
        '''other is first, self is second'''
        temp=-self
        return temp.__add__(other)

    #Multiplication
    def __mul__(self, other):
        try:
            length=len(self.fx)
            val=np.zeros(length)
            for j in range(length):
                temp=np.zeros(other.fx.shape)
                temp[:j+1]=other.fx[j::-1]
                val[j]+=np.einsum('i,i->',TPSA._binom_coeffs[j],
                        np.einsum('i,i->i',self.fx,temp))
            return TPSA(val)
        except AttributeError:
            temp=self.fx.copy()
            temp*=other
            return TPSA(temp)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        try:
            length=len(self.fx)
            val=np.zeros(length)
            for j in range(length):
                temp=np.zeros(other.fx.shape)
                temp[:j+1]=other.fx[j::-1]
                val[j]+=np.einsum('i,i->',TPSA._binom_coeffs[j],
                        np.einsum('i,i->i',self.fx,temp))
            self.fx=val
            return self
        except AttributeError:
            temp = self.fx.copy()
            temp *= other
            return TPSA(temp)

    #Division
    def __truediv__(self,other):
        try:
            temp=TPSA._series(other,inv=True)
            temp+=1/other.fx[0]
            return self*temp
        except AttributeError:
            temp = self.fx.copy()
            temp *= 1/other
            return TPSA(temp)

    def __rtruediv__(self, other):
        temp=TPSA._series(self,inv=True)
        temp+=1/self.fx[0]
        return other*temp

    #Functions
    @staticmethod
    def _Taylor(trunc, alt=1):
        '''Helper function used to compute exp and trig expansions'''
        store = np.zeros(TPSA.order + 1)
        store[1:] = trunc.fx[1:]
        store = TPSA(store)
        accumulate = 1
        new = 1
        for k in range(1, TPSA.order + 1):
            accumulate *= store
            new += (alt**(k//2) / np.math.factorial(k)) * accumulate
        return new

    @staticmethod
    def _series(trunc,inv=False):
        '''Helper function for natural log/inverse series'''
        store = np.zeros(TPSA.order + 1)
        store[1:] = trunc.fx[1:] / trunc.fx[0]
        store = TPSA(store)
        accumulate = 1
        new = 0
        for k in range(1, TPSA.order + 1):
            accumulate *= store
            if inv:
                pre=((-1)**k)/trunc.fx[0]
            else:
                pre=(((-1)**(k + 1))/k)
            new += pre * accumulate
        return new

    @staticmethod
    def exp(trunc):
        try:
            pre=np.exp(trunc.fx[0])
            new=TPSA._Taylor(trunc)
            return pre*new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")

    @staticmethod
    def sin(trunc):
        try:
            pre=[np.sin(trunc.fx[0]),np.cos(trunc.fx[0])]
            new=TPSA._Taylor(trunc, alt=-1)
            for i in range(2):
                new.fx[i::2]*=pre[i]
            return new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")

    @staticmethod
    def cos(trunc):
        try:
            pre = [np.cos(trunc.fx[0]), -np.sin(trunc.fx[0])]
            new = TPSA._Taylor(trunc, alt=-1)
            for i in range(2):
                new.fx[i::2] *= pre[i]
            return new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")

    @staticmethod
    def tan(trunc):
        return TPSA.sin(trunc)/TPSA.cos(trunc)

    @staticmethod
    def ln(trunc):
        try:
            new = TPSA._series(trunc)
            new += np.log(trunc.fx[0])
            return new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")
