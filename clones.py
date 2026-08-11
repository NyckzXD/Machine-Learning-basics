 # %%

import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")
df

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)
# %%

y = df['p2o_master_id'] #Variavel resposta

caracteristicas = ["Massa(em kilos)","Estatura(cm)","Distância Ombro a ombro","Tamanho do crânio","Tamanho dos pés","Tempo de existência(em meses)","Status "]
x = df[caracteristicas] #Caracteristicas

# %%
#ISSO É MACHINE LEARNING!!!
arvore.fit(x,y)

# %%
arvore.predict([[0,0,0,0]])
# %%
import matplotlib.pyplot as plt

tree.plot_tree(arvore, feature_names=caracteristicas, class_names= arvore.classes_, filled= True, max_depth=3)
# %%
proba = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=arvore.classes_)
# %%