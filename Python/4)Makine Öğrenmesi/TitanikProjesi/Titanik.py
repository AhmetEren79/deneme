from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

pd.set_option("display.max_columns", None)

train_df=pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_PassengerId=test_df["PassengerId"]

def detect_outlier(df,features):
    outlier_indices=[]

    for each in features:
        Q1 = np.percentile(df[each], 25)
        Q3 = np.percentile(df[each],75)
        IQR = Q3-Q1

        outlier_step = IQR*1.5

        outlier_list_col = df[(df[each] < Q1 - outlier_step) | (df[each]>Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i,v in outlier_indices.items() if v > 2)
    return multiple_outliers

print(train_df.loc[detect_outlier(train_df,["Age","SibSp","Parch","Fare"])])
train_df = train_df.drop(detect_outlier(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)

# --------------- Find the Missing Data-----------------

train_df_len=len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)   # *******

print(train_df.head())
print("**************")
print(train_df.columns[train_df.isnull().any()])    # Hangi features de null var bakar
print("**************")
print(train_df.isnull().sum())    # kaç null var bakar
print("**************")

# --------------- Fİll the Missing Data-----------------

print(train_df[train_df["Embarked"].isnull()])                # Boş olan Embarklar neler bakar

# train_df.boxplot(column="Fare",by="Embarked")
# plt.show()

train_df["Embarked"] = train_df["Embarked"].fillna("C")        # Boş Embarkedi  C ile değiştirdik
print("**************")
print(train_df[train_df["Embarked"].isnull()])
print("******************************************")    #   Fare için ise
print(train_df[train_df["Fare"].isnull()])

train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))    #  Fare Doldurduk
print(train_df[train_df["Fare"].isnull()])
print("******************************************")
                # Isı Haritası
list1 = ["SibSp","Parch","Age","Fare","Survived"]
# sns.heatmap(train_df[list1].corr(),annot=True,fmt = ".2f")
# plt.show()

print("******************************************")
                #SibSP - Survived
# g = sns.catplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", height = 6)
# g.set_ylabels("Survived Probability")
# # plt.show()
# SibSp değerine göre hayatta kalma oranı
survival_prob = train_df.groupby("SibSp")["Survived"].mean()
print(survival_prob)

print("******************************************")
                #Parch - Survived
# g = sns.catplot(x = "Parch", y = "Survived", data = train_df, kind = "bar", height = 6)
# g.set_ylabels("Survived Probability")
# # plt.show()
SibSurvivor_prob = train_df.groupby("Parch")["Survived"].mean()
print(SibSurvivor_prob)

print("******************************************")
                #Pclass - Survived
# g = sns.catplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", height = 6)
# g.set_ylabels("Survived Probability")
# # plt.show()
PclSur_prob = train_df.groupby("Pclass")["Survived"].mean()
print(PclSur_prob)

print("******************************************")
                #Age - Survived
# sns.displot(data=train_df, x="Age", col="Survived", bins=25)
# plt.show()
ageSur_prob = train_df.groupby("Survived")["Age"].mean()
print(ageSur_prob)
print("******************************************")
                #Age - Survived - Pclass
# g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", height = 2)
# g.map(plt.hist, "Age", bins = 25)
# g.add_legend()
# plt.show()
print("******************************************")
                #Embark - Sex - Pclass - Survived
# g = sns.FacetGrid(train_df, row='Embarked', height=2, aspect=1.3)
# g.map_dataframe(sns.pointplot,x='Pclass', y='Survived', hue='Sex', errorbar=None,palette='dark:#1f77b4')   # İsterseniz hata çubuklarını kapatın
# g.add_legend()
# plt.show()
print("******************************************")
                #Embark - Sex - Fare - Survived
# g = sns.FacetGrid(train_df, row='Embarked', col='Survived',height=2.3, aspect=1.2)
# g.map_dataframe(sns.barplot, x='Sex', y='Fare', errorbar=None)
# g.set_axis_labels("Sex", "Fare")
# g.add_legend()
# plt.show()
print("******************************************")
# -----------------FİLL MİSSİNG AGE VALUES-------------------
print(train_df[train_df["Age"].isnull()])

# (Opsiyonel ama iyi pratik) Eğer train_df bir filtreden geldiyse kopya al:
# train_df = train_df.copy()

# Yaşı NaN olan satırların indexleri (etiket indexi)
# 1) Önce boş olan Age satırlarının indexlerini kaydet
nan_idx = train_df.index[train_df["Age"].isna()]

# 2) Doldurma işlemini yap
global_med = train_df["Age"].median()
for idx in nan_idx:
    sibsp  = train_df.loc[idx, "SibSp"]
    parch  = train_df.loc[idx, "Parch"]
    pclass = train_df.loc[idx, "Pclass"]

    mask = (
        (train_df["SibSp"]  == sibsp) &
        (train_df["Parch"]  == parch) &
        (train_df["Pclass"] == pclass)
    )
    age_pred = train_df.loc[mask, "Age"].median()

    train_df.loc[idx, "Age"] = age_pred if pd.notna(age_pred) else global_med

# 3) Önceden boş olan satırları ve yeni Age değerlerini gör
print(train_df.loc[nan_idx, ["SibSp", "Parch", "Pclass", "Age"]])
print(train_df.loc[172, "Age"])

#  FEATURE ENGİNEERİNG
#    Name -Title
name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train_df["Title"].head(10)

# sns.countplot(x="Title", data = train_df)
# plt.xticks(rotation = 60)
# plt.show()

train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
train_df["Title"].head(20)

# sns.countplot(x="Title", data = train_df)
# plt.xticks(rotation = 60)
# # plt.show()

# g = sns.catplot(x="Title", y="Survived", data=train_df, kind="bar")
# g.set_xticklabels(["Master", "Mrs", "Mr", "Other"])
# g.set_ylabels("Survival Probability")
# plt.show()

train_df.drop(labels = ["Name"], axis = 1, inplace = True)
train_df = pd.get_dummies(train_df,columns=["Title"])
train_df[["Title_0", "Title_1", "Title_2", "Title_3"]] = train_df[["Title_0", "Title_1", "Title_2", "Title_3"]].astype(int)
# print(train_df.head())


#      Family Size

train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df.head()

# g = sns.catplot(x = "Fsize", y = "Survived", data = train_df, kind = "bar")
# g.set_ylabels("Survival")
# plt.show()

train_df["family_size"] = [1 if i in [2, 3, 4] else   2 if i in [1, 7] else  3 if i in [5, 6] else 0  for i in train_df["Fsize"]]

print(train_df["family_size"].value_counts())

# sns.countplot(x = "family_size", data = train_df)
# plt.show()

# g = sns.catplot(x = "family_size", y = "Survived", data = train_df, kind = "bar")
# g.set_ylabels("Survival")
# plt.show()
train_df = pd.get_dummies(train_df, columns= ["family_size"])
print(train_df.head())
print("**************************")


#       EMBARK
train_df = pd.get_dummies(train_df, columns=["Embarked"])
print(train_df.head())
print("**************************")

#       TİCKET
tickets = []
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"] = tickets

# print(train_df["Ticket"].head(20))
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")
# print(train_df.head(10))
# print(train_df.head(10).shape)
print("**************************")

#        PCLASS
train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns= ["Pclass"])
# print(train_df.head())
print("**************************")
#       SEX
train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns=["Sex"])
# print(train_df.head())
print("**************************")
#       DROP ID and CABİN
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
print(train_df.head())
print("**************************")
#       MODELİNG
print(train_df_len)
test = train_df[train_df_len:]
test = test.drop(labels=["Survived"], axis=1)
print(test.head())
print(len(test))

train = train_df[:train_df_len]
X_train = train.drop(labels = "Survived", axis = 1)
y_train = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train)*100,2)
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print("Logistic Regressin Training Accuracy: % {}".format(acc_log_train))
print("Logistic Regressin Testing Accuracy: % {}".format(acc_log_test))

random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             # LogisticRegression(random_state = random_state,max_iter=1000),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

# logreg_param_grid = {"C":np.logspace(-3,3,7),
#                     "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   # logreg_param_grid,
                   knn_param_grid]

cv_result = []
best_estimators = []

# 1. Önce TÜM modelleri eğitmek için döngüyü çalıştırın
for i in range(len(classifier)):
    clf = GridSearchCV(
        classifier[i],
        param_grid=classifier_param[i],
        cv=StratifiedKFold(n_splits=10),
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_train, y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)

    # Model adını ve o anki en iyi skoru yazdır
    print(f"{classifier[i].__class__.__name__}: {cv_result[i]:.4f}")

# --- DÖNGÜNÜN SONU ---

# 2. Döngü bittikten ve tüm modeller eğitildikten sonra tahmin ve kaydetme işlemini yapın
print("\nTüm modeller eğitildi. En iyi RandomForest modeli ile tahmin yapılıyor...")

# En iyi RandomForest modeli listenin 3. elemanıdır (index 2)
# DecisionTree (0), SVC (1), RandomForest (2), KNeighbors (3)
rf_best = best_estimators[2]

# Test setinde tahmin yap
test_survived = pd.Series(rf_best.predict(test), name="Survived").astype(int)

# PassengerId ile birleştir
results = pd.concat([test_PassengerId, test_survived], axis=1)

# CSV olarak kaydet
results.to_csv("titanic.csv", index=False)

print("Tahminler 'titanic.csv' dosyasına başarıyla kaydedildi.")

aa = pd.read_csv("titanic.csv")
print(aa)
