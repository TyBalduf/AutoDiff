from TPSA import TPSA
import math

#Multiplication
print(f"current order={TPSA.order}")
x=TPSA(10)
print(f"x={x}")
print(f"x*x*x*x={x*x*x*x}")
print(f"x**4={x**4}")
print(f"x*-1={x**-1}")

print()
TPSA.order=3
print(f"new order={TPSA.order}")
y=TPSA([0.0,1,2,4])
print(f"y={y}")
print(f"3*y={3*y}")

#Division

inv=y/3
print(f"y/3={inv}")

print()
z=TPSA([3.0,1,0,0])
print(f"z={z}")
inv=1/z
print(f"1/z={inv}")
print(f"1/exp(z)={1/TPSA.exp(z)}")
print(f"exp(-z)={TPSA.exp(-z)}")
zsqr=z*z
exp=TPSA.exp(zsqr)
print(f"z={z}")
print(f"z*z={zsqr}")
print(f"exp(z*z)={exp}")

print()
w=TPSA(math.pi+1)
print(f"w={w}")
cosine=TPSA.cos(w)
sine=TPSA.sin(w)
print(f"cos(w)={cosine}")
print(f"sin(w)={sine}")
print(f"tan(w)={TPSA.tan(w)}")


e=TPSA(math.e)
ln=TPSA.ln(e)
print(f"ln(e)={ln}")
print(f"1/e={1/e}")

print("Iterate over w")
new=w[1:3]
new[0]=15
print(w)
for i in w:
    print(i)

TPSA.order=8
b=TPSA(5)
a=TPSA.ln(TPSA.tanh(b))
print(f"b={b}")
print(f"a=ln(tanh(b))={a}")
print(a+b)

orig=TPSA.cos(TPSA.cos(TPSA(math.e)))+TPSA(math.e)
print(f"cos(cos(e))+e={orig}")
delay=TPSA(TPSA.var)
print(delay[0])
# A=TPSA.cos(TPSA.cos(TPSA(TPSA.var)))
# for func in A:
#     print(func(math.pi+1))
# B=TPSA.exp(-TPSA(TPSA.var))
# print("Series for exp(-x)")
# for func in B:
#     print(func)
# print("Series for ln(-x)")
# C=TPSA.ln(-TPSA(TPSA.var))
# for func in C:
#     print(func)
x=TPSA(1)
print(x**-0.2)