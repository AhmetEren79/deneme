print("""
           ****************************

           1.Bakiye kaç

           2.Para Yatır

           3.Para Çek

           4.Programdan Çıkmak için 'q' ya basın

           *****************************
    """)

bakiye = 1000


while(True):

    islem = input("Lütfen yapmak istediğiniz işlemi seçiniz")

    if(islem == "q"):
        print("Sistemden Çıkılıyor...")
        break
    elif(islem == "1"):
        print("Bakiyeniz = ",bakiye)
    elif(islem == "2"):
        miktar = int(input("Para Yatıralacak Miktarı seçin..."))
        bakiye +=miktar
        print("Yeni bakiyeniz = ",bakiye)
    elif(islem =="3"):
        miktar = int(input("Çekilecek Miktarı giriniz..."))
        if(bakiye<miktar):
            print("Çekmek istediğiniz miktar bakiyenizden fazladır...")
            continue
        bakiye -= miktar
        print("Toplam bakiyeniz = ",bakiye)
    else:
        print("Lütfen geçerli bir değer giriniz.")


