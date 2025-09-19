"""
def selamla():
    print("slm moruk")
print(type(selamla))
selamla()

def selamla(isim):
    print("isminiz : ",isim)
selamla("Ahmet")

def toplama(a,b,c):
    print("Toplam : ",(a+b+c))
toplama(2,3,4)

def fakt(sayi):
    fak =1
    if(sayi<=0):
        print("Lütfen geçerli bir sayi seçiniz")
    else:
        for i in range(1,sayi+1):
           fak *=i
           print(fak)
fakt(4)
          """


def bilgilerigoster(isim, soyisim,numara):
    print(isim,soyisim,numara)

def bilgilerigoster(isim = "Bilgi Yok", soyisim = "Bilgi Yok",numara = "Bilgi Yok"):
    print(isim,soyisim,numara)

bilgilerigoster("Ahmet")
bilgilerigoster(numara = "12")

#  *****************************************************************
def toplama(a,b,c):
    return a+b+c
def toplama(*a):
    return a

def toplama(*a):
    toplama = 0
    for i in a:
        toplama += i
    print(toplama)

toplama(5,6,7,8,94)


