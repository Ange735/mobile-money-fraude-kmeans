# TP 2 : Détection de fraude Mobile Money avec K-means

Détection non supervisée de fraudes dans 8 130 transactions Mobile Money (Afrique de l'Ouest), sans étiquettes, avec K-means.

👉 **Pour voir les résultats sans rien exécuter, ouvrir [`tp2_mobile_money_fraude.ipynb`](tp2_mobile_money_fraude.ipynb)** : GitHub affiche le notebook avec toutes les sorties et tous les graphiques.

## Contenu

| Fichier | Description |
|---|---|
| `tp2_mobile_money_fraude.py` | Code source du TP (cellules `# %%`) |
| `tp2_mobile_money_fraude.ipynb` | Le même code, déjà exécuté, avec les résultats |
| `transactions_mobile_money.csv` | 8 130 transactions (sans labels) |
| `labels_A_OUVRIR_A_LA_FIN.csv` | Labels réels, utilisés seulement pour l'évaluation finale |
| `top20_suspects.csv` | Les 20 transactions les plus suspectes, avec une explication pour chacune |

## Démarche

1. **Préparation** : one-hot encoding de `type_transaction` ; encodage cyclique de l'heure (sin/cos) ; nouvelle variable `montant_norm = montant / solde` ; transformation `log(1+x)` des variables très asymétriques ; standardisation.
2. **Choix de k = 8** : ni la silhouette ni la méthode du coude ne donnent de réponse nette. À partir de k = 8, deux petits clusters stables apparaissent.
3. **Profil des clusters** : deux petits clusters sortent du lot.
   - **Cluster 5** : distances inhabituelles, heures atypiques, gros montants, beaucoup de nouveaux appareils.
   - **Cluster 7** : comptes très récents, uniquement des transferts, beaucoup de transactions sur 24 h.
4. **Règle d'alerte** : une transaction est signalée si elle appartient à un cluster qui représente moins de 2 % du trafic.

## Résultats

| Règle | Précision | Rappel |
|---|---|---|
| Petit cluster (< 2 % du trafic) | **0,97** | **1,00** |

Les 130 fraudes sont toutes détectées (134 alertes, dont 4 faux positifs) :

| Type de fraude | Détectées | Cluster |
|---|---|---|
| compte_mule | 50 / 50 | 7 |
| sim_swap | 45 / 45 | 5 |
| faux_marchand | 35 / 35 | 5 |

## Lancer le code

```bash
pip install pandas numpy matplotlib scikit-learn
python tp2_mobile_money_fraude.py
```
