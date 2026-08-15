# %%

import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()
# %%
from sklearn import linear_model

X = df[['cerveja']] # dataframe
y = df['nota']  # Series

# MACHINE LEARNING
reg = linear_model.LinearRegression()
reg.fit(X, y)

# %%
a,b = reg.intercept_, reg.coef_[0]
print(a, b)

# %%
predict = reg.predict(X.drop_duplicates())
# %%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(1)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'], predict)
