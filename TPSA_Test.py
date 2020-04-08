from TPSA import TPSA
import math

#Multiplication
print(TPSA.order)
x=TPSA(10)
print(f"x={x}")
print(f"x*x*x*x={x*x*x*x}")

#Mixed mult
TPSA.order=3
y=TPSA([0,1,2,4])
print(f"y={y}")
print(f"3*y={3*y}")

#Division

inv=y/3
print(f"y/3={inv}")
z=TPSA([3,1,0,0])
inv=1/z
print(f"1/z={inv}")
print(f"1/exp(z)={1/TPSA.exp(z)}")
print(f"exp(-z)={TPSA.exp(-z)}")

#Functions

zsqr=z*z
exp=TPSA.exp(zsqr)
print(f"z={z}")
print(f"z*z={zsqr}")
print(f"exp(z*z)={exp}")

w=TPSA(math.pi)
cosine=TPSA.cos(w+1)
sine=TPSA.sin(w+1)
print(f"cos(pi+1)={cosine}")
print(f"sin(pi+1)={sine}")
print(f"tan(pi+1)={TPSA.tan(w+1)}")

e=TPSA(math.e)
ln=TPSA.ln(e)
print(f"ln(e)={ln}")
print(f"1/e={1/e}")

