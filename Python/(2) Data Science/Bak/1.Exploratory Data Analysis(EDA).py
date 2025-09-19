import pandas as pd

data = pd.read_csv("../pokemon.csv")

print(data["Type 1"].value_counts())       # Tip 1 türündeki pokemonların kaçar tane oldukları   ********************
print(data["Type 2"].value_counts())       # Çok kullanılır