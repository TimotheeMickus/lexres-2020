import gensim
import numpy as np
import pandas as pd
import scipy.stats


if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser("Knock-off replication of experiments from Bonami & Paperno (2018)")
  parser.add_argument("--triples_csv", type=str, required=True, help="path to triples.csv")
  parser.add_argument("--embeddings", type=str, required=True, help="path to W2V model")
  parser.add_argument("--use_binary", action="store_true", help="if embeddings in word2vec .bin format")
  args = parser.parse_args()
  
  print("reading data from %s and embs from %s" % (args.triples_csv, args.embeddings))
  
  embs = gensim.models.KeyedVectors.load_word2vec_format(args.embeddings, binary=args.use_binary) 
  df = pd.read_csv(args.triples_csv)
  
  # load the embeddings for all words in the CSV
  for paradigm_cell in ["agent", "verb", "3sg"]:
    df["%s_vec" % paradigm_cell] = df[paradigm_cell].apply(lambda v: embs[v] if v in embs else None)
  
  # NB: we need to remove values for which we found no vector
  df.dropna(inplace=True)
  
  # compute the offset between agent noun and bare verb
  df["deriv_offset"] = df["agent_vec"] - df["verb_vec"]
  # compute the offset between 3sg and bare verb
  df["infl_offset"] = df["3sg_vec"] - df["verb_vec"]
  
  for rel_type in ["infl", "deriv"]:
    offset_colname = "%s_offset" % rel_type
    # compute the average offset
    mean_offset = df[offset_colname].mean()
    # compute Euclidean distance to the average offset
    df["%s_euclidean" % rel_type] = df[offset_colname].apply(lambda v: np.linalg.norm(mean_offset - v))
    # compute cosine distance to the average offset
    df["%s_cos"  % rel_type] = df[offset_colname].apply(lambda v: 1 - (mean_offset.dot(v)/ (np.linalg.norm(mean_offset) * np.linalg.norm(v))))
  
  for method in ["euclidean", "cos"]:
    infl, deriv = df["infl_%s" % method].to_numpy(), df["deriv_%s" % method].to_numpy(),
    print("method %s:" % method, *scipy.stats.ttest_rel(infl, deriv))
    
  
