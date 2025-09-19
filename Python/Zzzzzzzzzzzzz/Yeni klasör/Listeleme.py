liste = list("Merhaba")
print(liste)

liste2 = [1,2,3,4,5,6,7,8,9]
print(liste2[2])

print(liste[-1])   #  SON ELEMAN I YAZDIRIR

print(len(liste))   # UZUNLUĞU YAZDIRIR

print(liste2[:5])   #   0 DAN 4. İNDEKSE KADAR YAZDIRIR.
print(liste2[5:])   #   5.İNDEKS DAHİL SONA KADAR YAZDIRIR

liste3 = liste+liste2
print(2*liste3)

liste[1] = 12   # Dizilerde olmaz
print(liste)

liste.append("SLM")    # Ekleme yapar
print(liste)

liste.pop()  # Son elemanı yok eder , İndeks numarası verilebir.
print(liste)