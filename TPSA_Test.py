from TPSA import TPSA

print(TPSA.order)
x=TPSA(10)
y=TPSA([0,1,2])
print((x*x*x*x))
TPSA.order=3
z=TPSA([0,1,2,4])
print((z*3))