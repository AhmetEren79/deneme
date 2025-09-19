

i = 0
while(i<10):
    if(i==5):
       break
    print(i)
    i+=1

while (True):
    deneme = input("İsminizi girin (Çıkmak için q ya basın) ")
    if(deneme =="q"):
        print("Sistemden çıkılıyor")
        break
    print("İsminiz = ",deneme)
