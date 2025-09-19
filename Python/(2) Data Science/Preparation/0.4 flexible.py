def deneme(a,b=1,c=2):
    return a+b+c

print(deneme(5))
print(deneme(5,4,3))
# ---------------------  ---------------------
def deneme2(*args):
    for i in args:
        print(i)
print(deneme2(1,2,3,4,5))

# ---------------------  dictionary için ---------------------
def deneme3(**kwargs):
    for key,value in kwargs.items():
        print(key," ",value)
my_dic = {"Tr":"Adana","Abd":"Newyork","Population":123456}
deneme3(**my_dic)

# ---------------------  Lambda function için ---------------------
square = lambda x:x**2
print(square(2))

top = lambda  x,y,z: x+y+z
print(top(3,4,5))

nbr_list = [1,2,3]
carp = map(lambda x:x*2,nbr_list)
print(list(carp))

# def carp(*args):
#     for each in args:             # ARGS İLE
#         print(each*2)
# print(carp(1,2,3))

# def carp(l):                      # ÜSTTEKİNİN AYNISI AMA FARKLISI
#     for each in l:
#         print(each*2)
# print(carp([1,2,3]))