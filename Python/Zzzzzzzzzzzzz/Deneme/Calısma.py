
"""
ad = input("Lütfen adınızı giriniz.")
soyad = input("Lütfen soyadınızı giriniz.")
takım = input("Lütfen takım adınızı giriniz")

bilgiler = [ad,soyad,takım]

print("Adınız = ", ad,"\n Soyad = ",soyad,"\nTakım = ",takım )
"""

sayi1 = int(input("sayi1 = "))
sayi2 = int(input("sayi2 = "))
sayi3 = int(input("sayi3 = "))

delta = sayi2 ** 2 - 4 * sayi1*sayi3

x1 = (-sayi2 -delta**0.5)/(2*sayi1)
x2 = (-sayi2 +delta**0.5)/(2*sayi1)

print(x1,"  ",x2)
