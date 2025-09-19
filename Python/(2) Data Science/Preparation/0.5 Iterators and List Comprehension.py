import pandas as pd
# --------------------- ZİP'LEME ---------------------
list1=[1,2,3,4]
list2 =[5,6,7,8]
z=zip(list1,list2)
print(z)
z_list=list(z)
print(z_list)
# --------------------- UN ZİP'LEME ---------------------
un_zip = zip(*z_list)
un_list1,un_list2=list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list2))
print(type(list(un_list2)))
print(type(un_list2))
print("*****************")
# --------------------- LİST COMPREHENSİON ---------------------
num1 = [1,2,3]
num2 = [i+1 for i in num1]
print(num2)

num3 = [5,10,15]
num4 = [each*5 if each==5 else each*2 for each in num3]
print(num4)
print("******************")

dataFrame = pd.read_csv("../pokemon.csv")
ort_Speed = sum(dataFrame.Speed)/len(dataFrame.Speed)
dataFrame["speed_level"] = ["Low" if ort_Speed>each else "High" for each in dataFrame.Speed]
print(ort_Speed)
print(dataFrame.loc[:10,["speed_level","Speed"]])