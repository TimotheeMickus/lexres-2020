import gensim # for loading embeddings
from sklearn.decomposition import PCA # for dimensionality reduction
import numpy as np  # for estimates computation
import random # for random sampling
import tqdm # for neat logging

# support, i.e., number of random sample points to compute estimates
SUPPORT=100000

def sample_vecs(vecs, support=SUPPORT):
  """
  Randomly sample vectors, format as matrix
  """
  return np.stack(random.sample(vecs, support), axis=0)

def reduce(vectors, dim):
  """
  Perform dimensionality reduction
  """
  assert dim < vectors.shape[1]
  return PCA(n_components=dim).fit_transform(vectors)
    
def cos(vectors1, vectors2):
  """
  Compute distribution of cosine over sample
  """
  return np.einsum('ij,ij->i', vectors1, vectors2) / (np.linalg.norm(vectors1, axis=1) * np.linalg.norm(vectors2, axis=1))

def dist(vectors1, vectors2):
  """
  Compute distribution of Euclidean distance over sample
  """
  return np.linalg.norm(vectors1 - vectors2, axis=1)
  
def norm(vectors1):
  """
  Compute distribution of Euclidean norm over sample
  """
  return np.linalg.norm(vectors1, axis=1)

if __name__ == "__main__":

  #  minimal CLI
  import argparse
  parser = argparse.ArgumentParser("Evaluates cos, dist, norm and cdist according to dimensionality")
  parser.add_argument("--embs", type=str, required=True, help="path to model")
  parser.add_argument("--output", type=str, required=True, help="path to output TSV")
  parser.add_argument("--binary", action="store_true", help="if path correspond to .bin")
  ars = parser.parse_args()
  
  # retrieve embeddings
  model = gensim.models.KeyedVectors.load_word2vec_format(args.embs, binary=args.binary)
  # we don't really care about which word they correspond to
  vecs = [model[w].reshape(-1) for w in model.vocab]
  
  with open(args.output, "w") as ostr:
    # file header
    print("id", "ne", "nv", "nr", "de", "dv", "dr", "ce", "cv", "cr", sep="\t", file=ostr)
    # for each dim value from 1 to 256
    for dim in tqdm.trange(1, 257):
      # we need two samples for cosine and distance
      vectors1, vectors2 = sample_vecs(vecs), sample_vecs(vecs)
      # PCA reduction should be done over all vectors
      all_vectors = reduce(np.concatenate((vectors1, vectors2), axis=0), dim)
      # re-split
      vectors1, vectors2 = all_vectors[:SUPPORT], all_vectors[SUPPORT:]
      
      # get distributions, means and variances
      norm_distrib = norm(vectors1)
      ne, nv = norm_distrib.mean(), norm_distrib.std() ** 2
      distance_distrib = dist(vectors1, vectors2)
      de, dv = distance_distrib.mean(), distance_distrib.std() ** 2
      cos_distrib = cos(vectors1, vectors2)
      ce, cv = cos_distrib.mean(), cos_distrib.std() ** 2
      
      # to file
      print(dim, ne, nv, nv/ne, de, dv, dv/de, ce, cv, cv/ce, sep="\t", file=ostr, flush=True)
    

  
