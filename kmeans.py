from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


data = pd.read_csv('IMDB_embeddings_pool.csv', header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X)
nmi = normalized_mutual_info_score(y, y_pred)
ari = adjusted_rand_score(y, y_pred)
print('NMI {:.4f} | ARI {:.4f}'.format(nmi, ari))

