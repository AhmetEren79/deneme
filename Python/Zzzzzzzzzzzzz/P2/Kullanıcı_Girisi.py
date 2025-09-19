print("Lütfen Kullanıcı adı ve şifrenizi giriniz.")

sys_kullaniciadi = "ahmet"
sys_sifre = 123
deneme_hak = 2

while(True):
    kullanici = input("Kullanıcı adınızı giriniz...")
    sifre = int(input("Şifrenizi giriniz..."))
    if(deneme_hak == 0):
        print("Deneme hakkınız kalmadı sonradan tekrar deneyiniz...")
        break
    if(kullanici != sys_kullaniciadi or sifre != sys_sifre ):
        print("Şifre ya da kullanıcı adınız yanlış")
        deneme_hak -=1
    else:
        print("Başarıyla giriş yaptınız...")
        break

