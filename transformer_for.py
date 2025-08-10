import numpy as np

class transformer_forward:
    def __init__(self, vocab, d_model, context, n_head, n_layers, MLP_size=4):
        # d_model is d_embed
        assert d_model % n_head == 0, "d_model must be divisible by n_heads"
        self.vocab = list(vocab)
        self.vocab_size = len(vocab) # size of the vocabulary
        self.vocab = vocab 
        self.d_model = d_model #size of the vectors
        self.context = context
        self.token_embedding = np.random.randn(self.vocab_size, self.d_model).astype(np.float32)/np.sqrt(self.d_model)
        self.pos_embedding   = np.random.randn(self.context, self.d_model).astype(np.float32)/np.sqrt(self.d_model)
        self.n_head = n_head
        self.d_head = self.d_model//n_head  #we consider dmodel divisible by nhead
        self.n_layers = n_layers
        self.MLP_size = MLP_size   #size of the multilayer perceptron
        # it is recommanded to normalize by sqrt(modelsize)
        scale = 1.0 / np.sqrt(self.d_model)
        self.W_q = np.random.randn(self.d_model, self.d_model).astype(np.float32)*scale #queries weights  (d_model, d_head)
        self.W_k = np.random.randn(self.d_model, self.d_model).astype(np.float32)*scale #keys weights     (d_model, d_head)
        self.W_v = np.random.randn(self.d_model, self.d_model).astype(np.float32)*scale #values weights   (d_model, d_head)
        self.W_o = np.random.randn(self.d_model, self.d_model).astype(np.float32)*scale #out weights   (d_model, d_head)

        #multilayer perceptron
        self.W_MLP1 = [np.random.randn(self.d_model, MLP_size*self.d_model).astype(np.float32)*scale for _ in range(n_layers)]
        self.W_MLP2 = [np.random.randn(MLP_size*self.d_model, self.d_model).astype(np.float32)*scale for _ in range(n_layers)]

        #normalization params
        self.ln_eps = 1e-5
        self.ln1_gamma = [np.ones((self.d_model,), dtype=np.float32) for _ in range(n_layers)]
        self.ln1_beta  = [np.zeros((self.d_model,), dtype=np.float32) for _ in range(n_layers)]
        self.ln2_gamma = [np.ones((self.d_model,), dtype=np.float32) for _ in range(n_layers)]
        self.ln2_beta  = [np.zeros((self.d_model,), dtype=np.float32) for _ in range(n_layers)]
        
    
    @staticmethod
    def softmax(a, axis=-1):
        #softmax (can work column wise)
        a_max = np.max(a, axis=axis, keepdims=True)
        exp = np.exp(a - a_max)
        return(exp / np.sum(exp, axis=axis, keepdims=True))
    
    @staticmethod
    def gelu(x):
        # approximation of gelu (soften relu)
        return(0.5*x*(1.0+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))))

    def layer_norm(self, x, gamma, beta):
        # normalization of layer
        mu = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return(gamma * (x - mu) / (std + self.ln_eps) + beta)
    
    def embed(self , idx):
        #embeding vectors (tokens)
        # idx : array of token ids, shape (batch, seq_len)
        idx = np.array(idx)
        batch, seq_len = idx.shape
        token_emb = self.token_embedding[idx] # (batch, seq_len, d_model)
        pos_emb   = self.pos_embedding[:seq_len, :][None, :, :] # (1, seq_len, d_model)
        token_emb = token_emb*np.sqrt(self.d_model) 
        x = token_emb + pos_emb  # (batch, seq_len, d_model)
        return(x)
        
    def split_heads(self, x):
        # split attention heads
        # x : (batch, seq, d_model) -> (batch, n_head, seq, d_head)
        b, s, _ = x.shape
        x = x.reshape(b, s, self.n_head, self.d_head)
        return(x.transpose(0, 2, 1, 3))
    
    def combine_heads(self, x):
        # recombine attetion heads
        # x: (batch, n_head, seq, d_head) -> (batch, seq, d_model)
        b, h, s, dh = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(b, s, h * dh)
        return(x)
    
    def causal_mask(self, seq_len):
        # causal mask for the triangle
        return np.tril(np.ones((seq_len, seq_len), dtype=bool))

    def self_attention(self, x, mask):
        # self attention 
        b, s, d = x.shape # x: batch, seq, d_model
        #linear layers
        Q = x @ self.W_q # queries (batch, seq, dmodel)
        K = x @ self.W_k # keys    (batch, seq, dmodel)
        V = x @ self.W_v # values  (batch, seq, dmodel)

        #we split the heads:
        Qh = self.split_heads(Q)  # (b, h, s, dh)
        Kh = self.split_heads(K)
        Vh = self.split_heads(V)
        scores = np.matmul(Qh, Kh.transpose(0,1,3,2)) / np.sqrt(self.d_head)

        # mask shape should be (s, s) or (b, s, s) 
        if mask.ndim == 2:
            mask_b = mask[None, None, :, :]  # (1,1,s,s) 
        elif mask.ndim == 3:
            mask_b = mask[:, None, :, :]     # (b,1,s,s)
        else:
            raise ValueError("mask must be 2D or 3D")
        
        scores = np.where(mask_b, scores, -1e9) #apply mask

        attn = self.softmax(scores, axis=-1)
        out_heads = np.matmul(attn, Vh)
        #combine:
        out = self.combine_heads(out_heads)
        out = out @ self.W_o #final linear layer
        return(out, attn)
    
    def mlp(self, x, layer_idx):
        #multi layer perceptron
        b, s, d = x.shape
        w1 = self.W_MLP1[layer_idx] # (d, 4d)
        w2 = self.W_MLP2[layer_idx] # (4d, d)
        h = x @ w1 # (b, s, 4d)
        h = self.gelu(h)
        out = h @ w2 # (b, s, d)
        return(out)
    
    def forward(self, idx):
        # idx: (seq,) or (batch, seq)
        x = self.embed(idx)  # (batch, seq, d_model)
        b, seq_len, _ = x.shape

        # causal mask for LM
        mask = self.causal_mask(seq_len)  # (seq, seq)
        attn_maps = []  

        for layer in range(self.n_layers):
            ln1_g = self.ln1_gamma[layer]
            ln1_b = self.ln1_beta[layer]
            y = self.layer_norm(x, ln1_g, ln1_b)   # (b, seq, d)
            attn_out, attn = self.self_attention(y, mask=mask)
            x = x + attn_out  # residual
            attn_maps.append(attn)
        
            ln2_g = self.ln2_gamma[layer]
            ln2_b = self.ln2_beta[layer]
            y2 = self.layer_norm(x, ln2_g, ln2_b)
            mlp_out = self.mlp(y2, layer)
            x = x + mlp_out  # residual

        # final logits tied to token embeddings
        logits = x @ self.token_embedding.T   # (b, seq, vocab_size)
        return(logits, attn_maps)
    
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # test
    vocab = ["<pad>", "a", "b", "c", "d", "e"]
    model = transformer_forward(vocab, d_model=32, context=16, n_head=4, n_layers=2)
    
    # batch de of to sequences of IDs
    ids = np.array([[1,2,3,4,0,0],[2,3,4,1,0,0]])
    logits, attn = model.forward(ids) # shape (2, seq, vocab_size)
    print("logits:", logits.shape, "min/max/mean/std:", logits.min(), logits.max(), logits.mean(), logits.std())
    plt.imshow(attn[0][0,0], cmap='viridis') #show the heatmap of one attention map
    plt.show()