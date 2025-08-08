# Transformer-from-scratch

The objective is to buid a transformer (same ai structure as chatGPT for example) from scratch. What we mean by "from scratch" in this project is that we will code everything using only classic modules such as numpy. The objective is also to try not to use torch. 

status of the project: just started

tokenizer:
We use the algorithm BPE with N=1000 on data.txt (some generated french text), resulting in vocab.pkl the vocabulary dictionary.
