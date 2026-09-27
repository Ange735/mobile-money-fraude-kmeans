# %% [markdown]
# TP 2 (autonome) : Détection de fraude Mobile Money avec K-means
# ----------------------------------------------------------------
# CONTEXTE
# Tu es data scientist chez un opérateur de Mobile Money en Afrique de l'Ouest.
# Tu reçois un extrait de 8 130 transactions SANS étiquettes. Le service
# conformité soupçonne que quelques fraudes s'y cachent (moins de 2 %),
# mais personne ne sait lesquelles ni de quel type.
#
# TA MISSION
# 1. Explorer les données : types de variables, distributions, asymétries.
# 2. Préparer les données pour K-means. Attention, c'est moins direct que
#    dans le TP 1. Réfléchis à ces trois points avant de coder :
#      - `type_transaction` est une variable catégorielle ;
#      - `heure` est cyclique : 23h et 0h sont proches, mais pas numériquement ;
#      - un montant de 50 000 FCFA n'a pas le même sens selon le solde du compte.
#        Peux-tu créer une variable plus parlante ?
# 3. Choisir k, entraîner K-means, décrire chaque cluster en langage métier.
# 4. Construire un score d'anomalie et une règle d'alerte (tu peux combiner
#    plusieurs signaux, comme dans le TP 1).
# 5. Sortir la liste des 20 transactions les plus suspectes, avec pour chacune
#    une phrase qui explique POURQUOI elle est suspecte.
# 6. SEULEMENT À LA FIN : charger labels_A_OUVRIR_A_LA_FIN.csv, calculer
#    précision et rappel, et analyser quels types de fraude tu rates et pourquoi.
#
# RÈGLES
# - Pas d'ouverture du fichier de labels avant l'étape 6.
# - Tu peux t'appuyer sur le TP 1 comme référence, mais pas copier-coller :
#   les variables et les pièges ne sont pas les mêmes.
# - Garde une trace de tes choix en commentaires (pourquoi ce k ? ce seuil ?).
#
# OBJECTIF CHIFFRÉ (indicatif) : rappel > 0.90 avec une précision > 0.50.
# C'est atteignable avec K-means seul et un bon prétraitement.

# %% Import des données
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("transactions_mobile_money.csv")

# %%
#Dans cette artie j'essaie de visualiser les données pour voir la structure du dataset
print(df.shape)
print(df.head())
print(df.describe())
print(df.dtypes)
# %%
col_id = ["id_transaction"] #j'isole la colonne id pour ne pas l'utiliser dans le clustering
X=df.drop(columns=col_id)
print(df.nunique()) # pour voir le nombre de valeurs uniques par variable
X["montant_norm"] = (
    X["montant_fcfa"].astype(float) /
    X["solde_avant_fcfa"].astype(float)
)#feature ingeniering: je creer une new variable pour obtenir plus d'iinfos sur le montant des transactions
col_bin = X.loc[:, X.nunique() == 2].columns.tolist()
col_num=X.select_dtypes(include=np.number).columns.difference(col_bin).tolist()
col_cat=X.select_dtypes(include="object").columns.difference(col_bin).tolist()
#j'isole les différents type de colonnes pour pouvoir les traiter separements
print("Colonnes binaires :", col_bin)
print("Colonnes numériques :", col_num)
print("Colonnes catégorielles :", col_cat)

# %%
print(X[col_cat].head())
x=pd.get_dummies(X[col_cat], dtype=int)
X=X.drop(columns=col_cat)
X=pd.concat([X,x],axis=1)
col_cat = x.columns.tolist()
print("Colonnes catégorielles après one-hot encoding :", col_cat)
print(X[col_cat].head())
# ici j'encode la variable catégorielle avec one hot encoding (vu que ya pas d'ordre logique)
# %%
print(X[col_num].describe())
for col in col_num:
    plt.figure()
    plt.hist(X[col], bins=50)
    plt.title(col)
    plt.show()
    print(X[col].skew())
#ici je visualise les distributions et le skewness afin de p ouvoir determiner les transformations que je peux apporter 

# %%
W = X.copy()
#je travail sur une copie pour ne pas alterer les données originales
# %%
W["distance_habituelle_km"]=np.log(1+W["distance_habituelle_km"])
fig, axs=plt.subplots(1,2,figsize=(10,5))
axs[1].hist(W["distance_habituelle_km"], bins=50)
axs[1].set_title(f"distance_habituelle_km (log(1+x))) skew={W['distance_habituelle_km'].skew():.2f}")
axs[0].hist(X["distance_habituelle_km"], bins=50)
axs[0].set_title(f"distance_habituelle_km (original) skew={X['distance_habituelle_km'].skew():.2f}")
plt.show()
# ici la distribution de cette variable etait tres asymetrique a droite, donc j'ai appliqué un log(1+x)  pour la rendre plus symétrique

# %%
W["sin_heure"]=np.sin(W["heure"]*np.pi/12)
W["cos_heure"]=(np.cos(W["heure"]*np.pi/12))
fig, axs=plt.subplots(1,2,figsize=(10,5))
axs[1].hist(W["sin_heure"], bins=50)
axs[1].set_title(f"sin_heure .skew={W['sin_heure'].skew():.2f}")
axs[0].hist(W["cos_heure"], bins=50)
axs[0].set_title(f"cos_heure. skew={W['cos_heure'].skew():.2f}")
plt.show()
# ici l'heure est une variable cyclique. donc si je la garde tel quelle, 23h et 00h vont semblé tres eloigné alors que ce n'est pas le cas
# donc j'ai appliqué cette transformation en faisant une regle de trois 
# %%
W["montant_norm"]=np.log(W["montant_norm"]+1)
fig, axs=plt.subplots(1,2,figsize=(10,5))
axs[1].hist(W["montant_norm"], bins=50)
axs[1].set_title(f"montant_norm (log(1+x))) skew={W['montant_norm'].skew():.2f}")
axs[0].hist(X["montant_norm"], bins=50)
axs[0].set_title(f"montant_norm (original) skew={X['montant_norm'].skew():.2f}")
plt.show()
# ici la distribution de cette variable etait asymetrique avec une queue assez longue donc j'ai appliquer cette transformation
# pour reduire un peu le skew
# %%
W["nb_transactions_24h"]=np.log(W["nb_transactions_24h"]+1)
fig, axs=plt.subplots(1,2,figsize=(10,5))
axs[1].hist(W["nb_transactions_24h"], bins=50)
axs[1].set_title(f"nb_transactions_24h (log(1+x))) skew={W['nb_transactions_24h'].skew():.2f}")
axs[0].hist(X["nb_transactions_24h"], bins=50)
axs[0].set_title(f"nb_transactions_24h (original) skew={X['nb_transactions_24h'].skew():.2f}")
plt.show()
# ici la distribution de cette variable etait tres asymetrique a droite, donc j'ai appliqué un log(1+x)  pour la rendre plus symétrique

# %%
W["montant_fcfa"]=np.log(W["montant_fcfa"]+1)
fig, axs=plt.subplots(1,2,figsize=(10,5))
axs[1].hist(W["montant_fcfa"], bins=50)
axs[1].set_title(f"montant_fcfa (log(1+x))) skew={W['montant_fcfa'].skew():.2f}")
axs[0].hist(X["montant_fcfa"], bins=50)
axs[0].set_title(f"montant_fcfa (original) skew={X['montant_fcfa'].skew():.2f}")
plt.show()
# ici la distribution de cette variable etait tres asymetrique a droite, donc j'ai appliqué un log(1+x)  pour la rendre plus symétrique


# %%
W["solde_avant_fcfa"]=np.log(W["solde_avant_fcfa"]+1)
fig, axs=plt.subplots(1,2,figsize=(10,5))
axs[1].hist(W["solde_avant_fcfa"], bins=50)
axs[1].set_title(f"solde_avant_fcfa (log(1+x))) skew={W['solde_avant_fcfa'].skew():.2f}")
axs[0].hist(X["solde_avant_fcfa"], bins=50)
axs[0].set_title(f"solde_avant_fcfa (original) skew={X['solde_avant_fcfa'].skew():.2f}")
plt.show()
# ici la distribution de cette variable etait tres asymetrique a droite, donc j'ai appliqué un log(1+x)  pour la rendre plus symétrique

# %%
col_num.append("sin_heure")
col_num.append("cos_heure")
col_num.remove("heure")
# ici je supprime la colone heure dans col_num car elle ne sert plus vu que j'ai les sin et cos 
W.drop(columns=["heure"], inplace=True)
X.drop(columns=["heure"], inplace=True)
#ici je supprime carrément la colonne heure dans W et X pour ne pas l'utiliser par erreur 
# %%
print(W[col_num].describe())
for col in col_num:
    plt.figure()
    plt.hist(W[col], bins=50)
    plt.title(col)
    plt.show()
    print(W[col].skew())
# %%
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
W[col_num] = scaler.fit_transform(W[col_num])

# %%
print(W[col_num].describe())

# %% K-means
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

res=[]
for k in range(2,11):
    kmeans = KMeans(n_clusters=k,n_init=10, random_state=42)
    kmeans.fit(W)
    s = silhouette_score(W, kmeans.labels_,sample_size=3000)
    c = kmeans.inertia_
    p = np.bincount(kmeans.labels_) #compte les points par cluster.
    res.append([k,s,c,p])
df_res=pd.DataFrame(res, columns=["k","silhouette","inertia","points_par_cluster"])    

fig, axs = plt.subplots(1,2,figsize=(10,5))
axs[0].plot(df_res["k"], df_res["silhouette"], marker="o")
axs[0].set_xlabel("k")
axs[0].set_ylabel("silhouette")
axs[1].plot(df_res["k"], df_res["inertia"], marker="o")
axs[1].set_xlabel("k")
axs[1].set_ylabel("inertia")
plt.show()

# %%
print(df_res.round(2))
# Apres visualisation, je constate que 
# %% # dans cette partie j'essaie de trouver le k optimal en utilisant la methode de la silhouette et du coude
# apres visualisation, je vois que avec la methode de laa silhouette, ya pas vraiment de pic, donc cette methode n'est pas concluente
# Aussi en observant la methode du coude, je vois qu'il n'ya pas vraiment de coude ou plutot le coude est lisse donc ne peux pas me donner une reel information
# Ainsi donc en observant le nmbre de points dans chaque cluster pour chaque k je constate que c'est a k =8 que ya 2 petits clusters qui se forme et qui ont une 
# vleur a peu pres stable quand k augmente, Au dela de k=8, les gros clusters se divisent juste succesivement sans rien changer de nouveau
# en conclusion on prendra k = 8 pour le clustering
k = 8
print(f"k optimal: {k}")
# %%
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(W)
df["cluster"] = kmeans.labels_
#j'enregistre les labels des clusters que j'ai trouver dans le dataframe original pour pouvoir les utiliser plus tard

# %%
print(df["cluster"].value_counts().sort_index())
# j'affiche chaque cluster avec le nombre de point qu'il contiznt
# %%
print(col_num)
# %%
df["montant_norm"] = (
    df["montant_fcfa"].astype(float) /
    df["solde_avant_fcfa"].astype(float)
)
# je rajoute la colone montant_norm dans le dataframe depart pour pouvoir faire les comparaisons

# %%
col_num_df = df.select_dtypes(include=np.number).columns.difference(col_bin,).tolist()
col_num_df.remove("cluster")
print(col_num_df)
# je forme les colonnes numeriques pour le dataframe de depart pour pouvoir faire les comparaisons avec le dataframe W


# %%
profils=df.groupby("cluster")[col_num_df].median().round(2)
profils.head(8)
# %%
profils["portion_nouvel_appareil"] = df.groupby("cluster")["nouvel_appareil"].mean()
profils["nb_transactions"] = df["cluster"].value_counts()
profils["part_trafic"] = df["cluster"].value_counts(normalize=True)
globale = df[col_num_df].median()
globale["portion_nouvel_appareil"] = df["nouvel_appareil"].mean()
globale["nb_transactions"] = len(df)
globale["part_trafic"] = 1.0
profils.loc["GLOBAL"] = globale

print(profils.round(2))
# avec cette partie j'arrive a voir chaque cluster avec ces statistiques. Ainsi je constate que les clusters 7 et 5 sont anormals car deja ce sont les plus petits et aussi par example le cluster 5
# a une distance habituelle tres grande comparé aux autres clusters, se it a des heures inhabituel avec des montants enormes et utilise le plus de nouvel appareil avec le cluster 7
# et que le clusters 7 a qant a lui une ancienneté tres faible par rapport aux autres clusters, utilise aussi des montants enormes, a le plus de nombre de transaction en 1 heure et aussi part generalement d'un nouvel appareil
# Pour les nouvel appareil j'i utilisé le mean parce qu'il etait plus parlant que la median ici, vu ue c'est une variable binaire
# %%

print(pd.crosstab(df["cluster"], df["type_transaction"], normalize="index").round(2))
# Ici je compare la vriable categorielle pour voir la repartion dans les différents clusters. Ici aussi le constat est bizarre dans le cluster 7 on remrque qu'il n'ya que des transferts, ce qui esr etrange 
# Aussi pour le cluster 5 ont qu'il ya tres peu de depot mais beaucoup plus de paiement de retrait et de transfert 

# %%
# ici je vais maintenant determiner le score d'anomalie 
centres = kmeans.cluster_centers_[kmeans.labels_]
df["distance"] = np.linalg.norm(W.values - centres, axis=1)
df["score"] = df["distance"] / df.groupby("cluster")["distance"].transform("median")
# On divise par la médiane du cluster parce que certains clusters sont plus étalés que d'autres : une même distance peut être banale dans l'un et suspecte dans l'autre
# %%
#ICI on defini les critères d'alertes 
TAUX_ALERTE = 0.02 #d'apres l'enoncé
# Signal 1 : loin du centre de son cluster
seuil = df["score"].quantile(1 - TAUX_ALERTE)
df["alerte_score"] = df["score"] > seuil

# Signal 2 : appartenir à un cluster de moins de 2 % du trafic
part = df["cluster"].map(df["cluster"].value_counts(normalize=True))
df["alerte_petit_cluster"] = part < TAUX_ALERTE

# Combinaison : au moins un des deux signaux
# df["alerte"] = df["alerte_score"] | df["alerte_petit_cluster"]
df["alerte"] = df["alerte_petit_cluster"]


print(df[["alerte_score", "alerte_petit_cluster", "alerte"]].sum())
# au final j'ai 295 alertes que j'ai reussi a detecter ! ce qui  est effectivement inferieur au 2% mentionner dans l'enoncé 
# %%
from sklearn.metrics import precision_score, recall_score

labels = pd.read_csv("labels_A_OUVRIR_A_LA_FIN.csv")
df = df.merge(labels, on="id_transaction", how="left")
y_true = df["label"] != "normal"

# Performance globale de chaque règle
for col in ["alerte_petit_cluster", "alerte"]:
    print(f"{col:22s} précision={precision_score(y_true, df[col]):.2f} "
          f"rappel={recall_score(y_true, df[col]):.2f}")

# Détail par type de fraude
print(pd.crosstab(df["label"], df["alerte"], margins=True))
print(df[y_true].groupby("label")["alerte"].mean().round(2))

# Dans quels clusters sont tombées les fraudes ?
print(pd.crosstab(df["label"], df["cluster"]))
# %%

# %%
