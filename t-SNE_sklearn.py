# https://lovit.github.io/nlp/representation/2018/09/28/tsne/
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import time
import glob
import os

iframe_path = glob.glob('./labeled_i-frames/**/*.jpg', recursive=True)
num_iframes = len(iframe_path)

width, height, channels = (480, 270, 3)
# X = np.zeros((num_iframes, width, height, channels))
iframes = np.array([np.array(Image.open(path).resize((width, height))) for path in iframe_path])
iframes = np.reshape(iframes, (iframes.shape[0], -1))

iframe_labels = [os.path.basename(os.path.dirname(path)) for path in iframe_path]

# n_components: dimension of the embedding space
# perplexity: the sigma_i, number of influential points in the space
# metric: standard of the distance between two points in the original space
# method: data indexing method in the original space
# tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,
#      learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
#      min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
#      random_state=None, method='barnes_hut', angle=0.5)
t_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_result = tsne.fit_transform(iframes)

figure = plt.gcf()
figure.set_size_inches(24, 18)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], edgecolors="black")
plt.show()
