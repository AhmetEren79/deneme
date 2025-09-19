def cemberAlanhesabi(r,pi=3.14):
    output =2*pi*r
    return output
def hesapla(boy,kilo,*args):
    print(args)
    output=(boy+kilo)*args[0]
    return output
#def hesapla(boy,kilo,yas):
  # print(args)
  # output=(boy+kilo)*yas
  # return output

print(cemberAlanhesabi(2))
print(hesapla(170,55))