class DelegateFunc:
    def __init__(self,func,string):
        self.func=func
        self.string=string

    def __call__(self, val):
        try:
           return self.func(val)
        except AttributeError:
            newfunc= lambda x: self.func(val(x))
            newstring=self.string +'('+val.string+')'
            return DelegateFunc(newfunc,newstring)

    def __str__(self):
        return self.string

    def __mul__(self, other):
        try:
            newfunc=lambda x: self.func(x)*other.func(x)
            if len(self.string)<=len(other.string):
                newstring=self.string+'*('+other.string+')'
                if any((s == str(self) for s in ('0.0', '0', '-0.0', '-0'))):
                    newstring='0'
            else:
                newstring=other.string+'*('+self.string+')'
                if any((s == str(other) for s in ('0.0', '0', '-0.0', '-0'))):
                    newstring='0'
            return DelegateFunc(newfunc,newstring)
        except AttributeError:
            newfunc = lambda x: other*self.func(x)
            if any((s == str(self) for s in ('0.0', '0', '-0.0', '-0'))):
                newstring = '0'
            elif str(other)=='1.0' or str(other)=='1':
                newstring=self.string
            elif str(other)=='-1.0' or str(other)=='-1':
                newstring='-('+self.string+')'
            elif any((s==str(other) for s in ('0.0','0','-0.0','-0'))):
                newstring='0'
            else:
                newstring = str(other)+"*("+self.string+")"
            return DelegateFunc(newfunc, newstring)

    def __rmul__(self, other):
        return self*other

    def __add__(self, other):
        try:
            newfunc=lambda x: self.func(x)+other.func(x)
            if other.string=='0.0' or other.string=='0':
                if self.string=='0.0' or self.string=='0':
                    newstring='0'
                else:
                    newstring=self.string
            elif self.string=='0.0' or self.string=='0':
                newstring=other.string
            else:
                newstring=self.string+"+"+other.string
            return DelegateFunc(newfunc,newstring)
        except AttributeError:
            newfunc = lambda x: other+self.func(x)
            if str(other)=='0' or str(other)=='0.0':
                newstring = self.string
            elif self.string=='0' or self.string=='0.0':
                newstring=str(other)
            else:
                newstring=self.string+"+"+str(other)
            return DelegateFunc(newfunc, newstring)

    def __radd__(self, other):
        return self+other

    def __neg__(self):
        newfunc=lambda x: -self.func(x)
        newstring='-('+self.string+')'
        return DelegateFunc(newfunc,newstring)

