"""
print(2 in [1,2,3,4,5])
print(6 in [1,2,3,4,5])
print("p" in "pyhton")

liste =[1,2,3,4,5]
toplam =0

for i in liste:
    print(i)
    toplam =toplam + i
print(toplam)

for i in liste:
    if i % 2 == 0:
        print(i)

demet = ((1,2),(4,5),(6,7),(8,9))
for (i,j) in demet:
    print(i,j)

sozluk = {"bir":1,"iki":2,"uc":3,"dort":4}
for (i,j) in sozluk.items():
    print(i,j)
"""
from operator import index

i = 0
while (i<10):
    print(i)
    i +=1

print("*************************")
liste =[1,2,3,4,5,9,7]
index = 0
while(index<len(liste)):
    print(index,liste[index])
    index +=1
