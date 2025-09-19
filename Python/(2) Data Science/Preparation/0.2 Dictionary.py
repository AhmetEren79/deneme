dictionary ={"Spain":"PutaMadre","Usa":"NewYork"}

dictionary["Spain"]= "barcelona"
print(dictionary)
dictionary["France"] = "Paris"
print(dictionary)
del dictionary["Spain"]
print(dictionary)

print("**************")

for key,value in dictionary.items():         # ********************************
    print(key," : ",value)
print(" ")

print("France" in dictionary)
# dictionary.clear()      # İÇİNDEKİ DEĞERLERİ SİLER MEMORY DE YER KAPLAR
# del dictionary          # KOMPLE SİLER