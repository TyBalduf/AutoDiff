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
                self._fx = np.array(value).astype(float)
        elif isinstance(value,(int,float)):
            self._fx = np.zeros(TPSA._ORDER + 1)
            self._fx[0] = float(value)
            self._fx[1] = 1.0
        else:
            raise TypeError(f"Value is invalid type, must be list/tuple or single float")


    #Access/Formatting
    def __str__(self):
        return str(self._fx)

    def __getitem__(self, item):
        return self._fx[item].copy()

    #Comparison/equality
    def __eq__(self, other):
        try:
            return np.allclose(self._fx, other._fx)
        except AttributeError:
            return False

    #Addition
    def __add__(self,other):
        '''self is first operand, other is second'''
        try:
            return TPSA(self._fx + other._fx)
        except AttributeError:
            temp=self._fx.copy()
            temp[0]+=other
            return TPSA(temp)

    def __radd__(self, other):
        '''other is first operand, self is second'''
        return self+other

    def __iadd__(self, other):
        try:
            self._fx+=other._fx
            return self
        except AttributeError:
            self._fx[0]+=other
            return self

    def __abs__(self):
        if self._fx[0]==0:
            raise ValueError("Absolute value of derivatives is ambiguous for f(x)=0")
        elif self._fx[0]>0:
            return -self
        else:
            return self

    #Subtraction
    def __neg__(self):
        val=-self._fx
        return TPSA(val)

    def __sub__(self,other):
        '''self is first, other second'''
        return self+(-other)

    def __rsub__(self, other):
        '''other is first, self is second'''
        temp=-self
        return temp+other

    #Multiplication
    def __mul__(self, other):
        try:
            length=len(self._fx)
            val=np.zeros(length)
            for j in range(length):
                temp=np.zeros(other._fx.shape)
                temp[:j+1]=other._fx[j::-1]
                val[j]+=np.einsum('i,i->', TPSA._binom_coeffs[j],
                                  np.einsum('i,i->i', self._fx, temp))
            return TPSA(val)
        except AttributeError:
            temp=self._fx.copy()
            temp*=other
            return TPSA(temp)

    def __rmul__(self, other):
        return self*other

    def __imul__(self, other):
        try:
            length=len(self._fx)
            val=np.zeros(length)
            for j in range(length):
                temp=np.zeros(other._fx.shape)
                temp[:j+1]=other._fx[j::-1]
                val[j]+=np.einsum('i,i->', TPSA._binom_coeffs[j],
                                  np.einsum('i,i->i', self._fx, temp))
            self._fx=val
            return self
        except AttributeError:
            temp = self._fx.copy()
            temp *= other
            return TPSA(temp)

    def __pow__(self, power, modulo=None):
        ##Should optimize integer power at some point
        return TPSA.exp(power * TPSA.ln(self))

    #Division
    def __truediv__(self,other):
        '''Division done via expansion in _series function'''
        try:
            factors=((-1)**(k+1)/other._fx[0] for k in range(TPSA.order))
            temp=TPSA._series(other/other._fx[0],factors)
            temp._fx[0]*=1/other._fx[0]
            return self*temp
        except AttributeError:
            temp = self._fx.copy()
            temp *= 1/other
            return TPSA(temp)

    def __rtruediv__(self, other):
        factors = ((-1) ** (k + 1)/self._fx[0] for k in range(TPSA.order))
        temp=TPSA._series(self/self._fx[0],factors)
        temp._fx[0]*=1/self._fx[0]
        return other*temp


    #Functions
    @staticmethod
    def _series(trunc,factors):
        '''Helper function for natural log/inverse series'''
        store = np.zeros(TPSA.order + 1)
        store[1:] = trunc._fx[1:]
        store = TPSA(store)
        accumulate = 1
        new = 1
        for f in factors:
            accumulate *= store
            new += f * accumulate
        return new

    @staticmethod
    def exp(trunc):
        try:
            factors=(1/(np.math.factorial(k+1)) for k in range(TPSA.order))
            pre=np.exp(trunc._fx[0])
            new=TPSA._series(trunc,factors)
            return pre*new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")

    @staticmethod
    def ln(trunc):
        try:
            factors=((-1)**k/(k+1) for k in range(TPSA.order))
            new = TPSA._series(trunc/trunc._fx[0],factors)
            new._fx[0] *= np.log(trunc._fx[0])
            return new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")

    @staticmethod
    def sin(trunc):
        try:
            factors=((-1)**((k+1)//2) / np.math.factorial(k+1) for k in range(TPSA.order))
            pre=[np.sin(trunc._fx[0]),np.cos(trunc._fx[0])]
            new=TPSA._series(trunc, factors)
            for i in range(2):
                new._fx[i::2]*=pre[i]
            return new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")

    @staticmethod
    def cos(trunc):
        try:
            factors = ((-1)**((k+1)// 2) / np.math.factorial(k+1) for k in range(TPSA.order))
            pre = [np.cos(trunc._fx[0]), -np.sin(trunc._fx[0])]
            new = TPSA._series(trunc, factors)
            for i in range(2):
                new._fx[i::2] *= pre[i]
            return new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")

    @staticmethod
    def tan(trunc):
        return TPSA.sin(trunc)/TPSA.cos(trunc)

    @staticmethod
    def sec(trunc):
        return 1.0 / TPSA.cos(trunc)

    @staticmethod
    def csc(trunc):
        return 1.0 / TPSA.sin(trunc)

    @staticmethod
    def cot(trunc):
        return TPSA.cos(trunc) / TPSA.sin(trunc)

    @staticmethod
    def logistic(trunc,pow=1.0):
        return (1+TPSA(-trunc))**pow

    @staticmethod
    def tanh(trunc):
        temp=TPSA.exp(2*trunc)
        return (temp-1)/(temp+1)

    @staticmethod
    def heaviside(trunc,smooth=10):
        return TPSA.logistic(2*smooth*trunc)