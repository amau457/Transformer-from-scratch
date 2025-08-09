# Transformer-from-scratch

The objective is to buid a transformer (same ai structure as chatGPT for example) from scratch. What we mean by "from scratch" in this project is that we will code everything using only classic modules such as numpy. The objective is also to try not to use torch. 

status of the project: just started

tokenizer:
We use the algorithm BPE with N=1000 on data.txt (some generated french text), resulting in vocab.pkl the vocabulary dictionary.

trandformer forward:
This code is meant to be the first step into building the transformer: build the whole architecture without the training part, test if it works and then implement the training. The test that is run on this shows that the dimensions are good, the logits dont explode, the attention map looks coherent.

tbc: add training
