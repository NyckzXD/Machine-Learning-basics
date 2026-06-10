 # %%

import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
df

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)

# %%

y = df['Fruta'] #Variavel resposta

caracteristicas = ["Arredondada","Suculenta","Vermelha","Doce"]
x = df[caracteristicas] #Caracteristicas

# %%
#ISSO É MACHINE LEARNING!!!
arvore.fit(x,y)

# %%
arvore.predict([[0,0,0,0]])