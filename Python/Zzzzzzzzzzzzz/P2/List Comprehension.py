
"""
liste1 = [1,2,3,4,5,6,7,8,9]
liste2 = []

for i in liste1:
    liste2.append(i)
print(liste2)
"""

liste3 = [1,2,3,4,5,6]
liste4 = [i for i in liste3]   # ÜSTTEKİ İLE AYNI İŞLEM
print(liste4)

print("***************************")

liste5 = [1,2,3,4,5,6]
liste6 = [i*2 for i in liste5]
print(liste6)

print("***************************")

liste7 = [(1,2),(3,4),(5,6)]
liste8 = [i*j for i,j in liste7]
print(liste8)

print("***************************")

s = "Pythın"

list1 = [i for i in s]
print(list1)

print("***************************")

list = [(1,2,3),(4,5,6,7),(8,9)]

for i in list:
    for x in i:
        print(x)

print("***************************")

list2 = [x for i in list for x in i]
print(list2)


