import numpy as np
import scipy.special as sp
import DelegateFunc as df

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

    Use var in a TPSA to differentiate functions rather than values
    '''

    #Allow functions to be combined before being called
    var=df.DelegateFunc(lambda x: x,'x')
    _cosFunc=df.DelegateFunc(np.cos,'cos')
    _sinFunc=df.DelegateFunc(np.sin,'sin')
    _expFunc=df.DelegateFunc(np.exp,'exp')
    _lnFunc=df.DelegateFunc(np.log,'ln')


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
                self._fx = np.array(value)
        elif isinstance(value,(int,float)):
            self._fx = np.zeros(TPSA._ORDER + 1)
            self._fx[0] = value
            self._fx[1] = 1.0
        elif isinstance(value,df.DelegateFunc):
            self._fx = np.zeros(TPSA._ORDER + 1,dtype=object)
            self._fx[0] = value
            self._fx[1] = 1.0
        else:
            raise TypeError(f"Value is invalid type, must be list/tuple or single float")


    #Access/Formatting
    def __str__(self):
        return str(self._fx)

    def __getitem__(self, item):
        try:
            return self._fx[item].copy()
        except AttributeError:
            return self._fx[item]

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
            if self._fx.dtype==object or other._fx.dtype==object:
                val=np.zeros(length,dtype=object)
                for i in range(length):
                    for j in range(i+1):
                        val[i]+=TPSA._binom_coeffs[i][j]*self._fx[j]*other._fx[i-j]
            else:
                val=np.zeros(length)
                for j in range(length):
                    temp=np.zeros(other._fx.shape)
                    temp[:j+1]=other._fx[j::-1]
                    val[j]=np.einsum('i,i,i->',TPSA._binom_coeffs[j],self._fx,temp)
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
            if self._fx.dtype==object or other._fx.dtype==object:
                val=np.zeros(length,dtype=object)
                for i in range(length):
                    for j in range(i+1):
                        val[i]+=TPSA._binom_coeffs[i][j]*self._fx[j]*other._fx[i-j]
            else:
                val=np.zeros(length)
                for j in range(length):
                    temp=np.zeros(other._fx.shape)
                    temp[:j+1]=other._fx[j::-1]
                    val[j]=np.einsum('i,i,i->', TPSA._binom_coeffs[j], self._fx, temp)
            self._fx=val
            return self
        except AttributeError:
            self._fx *= other
            return self

    def __pow__(self, power, modulo=None):
        ##Should optimize integer power at some point
        return TPSA.exp(power * TPSA.ln(self))

    #Division
    def __truediv__(self,other):
        '''Division done via expansion in _series function'''
        try:
            factors=((-1)**(k+1)/other._fx[0] for k in range(TPSA.order))
            temp=TPSA._series(other/other._fx[0],factors)+(1/other._fx[0])
            return self*temp
        except AttributeError:
            temp = self._fx.copy()
            temp *= 1/other
            return TPSA(temp)

    def __rtruediv__(self, other):
        factors = ((-1) ** (k + 1)/self._fx[0] for k in range(TPSA.order))
        temp=TPSA._series(self/self._fx[0],factors)+(1/self._fx[0])
        return other*temp


    #Functions
    @staticmethod
    def _series(trunc,factors):
        '''Helper function to produce series
        1. Raises power of trunc[:1], accumulates
        2. At each step, adds factor[i]*accumulate to new
        '''
        if isinstance(trunc._fx[0],df.DelegateFunc):
            store = np.zeros(TPSA.order + 1,dtype=object)
            temp=np.zeros(TPSA.order+1,dtype=object)
            new = TPSA(temp)
            temp[0]=1.0
            accumulate= TPSA(temp)
        else:
            store = np.zeros(TPSA.order + 1)
            new=0
            accumulate = 1
        store[1:] = trunc._fx[1:]
        store = TPSA(store)
        for f in factors:
            accumulate *= store
            ##having accumulate first ensures TPSA.__mul__ used
            ##rather than DelegateFunc.__mul__
            new += accumulate*f
        return new

    @staticmethod
    def exp(trunc):
        try:
            factors=(1/(np.math.factorial(k+1)) for k in range(TPSA.order))
            pre=np.exp(trunc._fx[0])
            new=TPSA._series(trunc,factors)+1
            return pre*new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")

    @staticmethod
    def ln(trunc):
        try:
            factors=((-1)**k/(k+1) for k in range(TPSA.order))
            new = TPSA._series(trunc/trunc._fx[0],factors)+np.log(trunc._fx[0])
            return new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")

    @staticmethod
    def sin(trunc):
        try:
            pre = [TPSA._cosFunc(trunc._fx[0]), TPSA._sinFunc(trunc._fx[0])]
            factors=(pre[k%2]*(-1)**((k+1)//2) * (1/np.math.factorial(k+1)) for k in range(TPSA.order))
            new=TPSA._series(trunc, factors)+TPSA._sinFunc(trunc._fx[0])
            return new
        except AttributeError:
            raise AttributeError("Function only accepts TPSA object")

    @staticmethod
    def cos(trunc):
        try:
            pre = [TPSA._sinFunc(trunc._fx[0]),TPSA._cosFunc(trunc._fx[0])]
            factors = [pre[k%2]*(-1)**((k+2)// 2) * (1/np.math.factorial(k+1)) for k in range(TPSA.order)]
            new = TPSA._series(trunc, factors)+TPSA._cosFunc(trunc._fx[0])
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
    def heaviside(trunc,sharp=10):
        '''Larger sharp is better approximation'''
        return TPSA.logistic(2*sharp*trunc)