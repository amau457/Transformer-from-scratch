import numpy as np
# we cant just write a backward function inside our forward transformer because there are a lot of
# tasks in cascade and it makes it quite complicated, the method that we will do is devide
# the transformer into smaller blocks (tasks), compute backward propagation on these
# blocks and then and the end aglomerate all the blocks together


class Linear:
    # first, let's do the backpropagation for linear blocks
    # we didnt explicitly called them linear blocks in the forward transformer
    # but there were : 
    # Q = x @ W_q   
    # K = x @ W_k   
    # V = x @ W_v   #linear products to calculate attention
    # out = out @ W_o #linear projection (product) for the output of attention
    # the MLP is w1 one linear layer , gelu (!!!non linear), w2 an other linear layer 
    # logits = x @ token_embedding.T the final reprojection on the vocab is also linear

    def __init__(self, in_dim, out_dim, scale=1.0):
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros((out_dim,), dtype=np.float32)
        # gradients
        self.dW = np.zeros_like(self.W) #same dim as W
        self.db = np.zeros_like(self.b) #same dim as b
        self.cache = None #for backward we cache some data during forward

    def forward(self, x):
        # x shape: (b, seq, in_dim)  OR (N, in_dim) N=batch*seq
        self.x_shape = x.shape
        # we can, like in transformer forward, just write:
        # out = x @ self.W + self.b
        # it does the same thing as flattening and then doing the operation
        # but it is restrictive for dimensions
        # it will be simplier to do the backpropagation if we used the flattened (reduced to 2 dim) tensors
        x_flat = x.reshape(-1, x.shape[-1])  # (N, in_dim)
        out_flat = x_flat @ self.W + self.b  # (N, out_dim)
        out = out_flat.reshape(*x.shape[:-1], self.W.shape[1])
        # cache x_flat for backward
        self.cache = x_flat
        return(out)

    def backward(self, dout):
        # we reflatten the output
        dout_flat = dout.reshape(-1, dout.shape[-1])  # (N, out_dim)
        x_flat = self.cache  # (N, in_dim) #we use the flatten x that we cached in forward 
        # grads
        self.dW[:] = x_flat.T @ dout_flat # copy (in_dim, out_dim)
        self.db[:] = dout_flat.sum(axis=0) #copy (out_dim,)
        dx_flat = dout_flat @ self.W.T # (N, in_dim)
        dx = dx_flat.reshape(*dout.shape[:-1], self.W.shape[0])
        return(dx)

    def get_params(self):
        #get
        return([(self.W, self.dW, 'W'), (self.b, self.db, 'b')])
    
class LayerNorm:
    #in second, we implement the backpropagation to the normalization layers
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((d_model,), dtype=np.float32)
        self.beta  = np.zeros((d_model,), dtype=np.float32) #like in the forward model
        self.dgamma = np.zeros_like(self.gamma) #gradient
        self.dbeta  = np.zeros_like(self.beta) #gradient
        self.cache = None #we cache things during forward for backward

    def forward(self, x):
        # x: (b, seq, d)
        mu = x.mean(axis=-1, keepdims=True) # (b, seq, 1)
        std = x.std(axis=-1, keepdims=True)
        #x_hat is x normalized, we scale and shift with gamma and beta to have out
        x_hat = (x - mu) / (std+self.eps) # (b, seq, d) 
        out = self.gamma * x_hat + self.beta
        self.cache = (x, x_hat, std)
        return(out)

    def backward(self, dout):
        # dout: (b, seq, d)
        x, x_hat, std = self.cache
        b, seq, d = x.shape
        N = d
        # gradients / gamma/beta
        self.dgamma[:] = np.sum(dout*x_hat, axis=(0,1)) #copy
        self.dbeta[:]  = np.sum(dout, axis=(0,1)) #copy
        # grad / x
        #formula:
        # dx_hat = dL/dy*gamma
        dx_hat = dout * self.gamma[None, None, :]
        dx_hat_flat = dx_hat.reshape(-1, N)
        x_hat_flat = x_hat.reshape(-1, N)
        std_flat = std.reshape(-1, 1)

        # formula:
        # dx = (1/std) * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
        m = x_hat_flat.shape[0]
        mean_dxhat = np.mean(dx_hat_flat, axis=1, keepdims=True) # (m,1)
        mean_dxhat_xhat = np.mean(dx_hat_flat * x_hat_flat, axis=1, keepdims=True)
        dx_flat = (dx_hat_flat - mean_dxhat - x_hat_flat * mean_dxhat_xhat) / std_flat
        dx = dx_flat.reshape(b, seq, N)
        return(dx)

    def get_params(self):
        # parmeters
        return([(self.gamma, self.dgamma, 'gamma'), (self.beta, self.dbeta, 'beta')])
    
class Embedding:
    # we now do the backward for embedding 
    def __init__(self, vocab_size, d_model, context):
        self.token_W = np.random.randn(vocab_size, d_model).astype(np.float32) / np.sqrt(d_model) #for token embed
        self.pos_W   = np.random.randn(context, d_model).astype(np.float32) / np.sqrt(d_model) # for pos embed
        self.dtoken_W = np.zeros_like(self.token_W) #gradient
        self.dpos_W = np.zeros_like(self.pos_W) #gradient
        self.cache = None #we cache things during forward for backward
        self.d_model = d_model

    def forward(self, idx):
        # #embeding vectors (tokens)
        # idx : array of token ids, shape (batch, seq_len)
        # the same as embed from transformer_for
        seq = idx.shape[1]
        token_emb = self.token_W[idx] # (b, seq, d)
        pos_emb = self.pos_W[:seq][None, :, :] # (1, seq, d)
        token_emb = token_emb*np.sqrt(self.d_model)
        out = token_emb + pos_emb # (b, seq, d)
        self.cache = (idx, seq)
        return(out)

    def backward(self, dout):
        # dout: (b, seq, d)
        idx, seq= self.cache #we take the cache
        # token grads:
        self.dtoken_W[:] = 0
        idx_flat = idx.reshape(-1)
        dout_flat = dout.reshape(-1, dout.shape[-1])
        #formula:
        # dWtoken = sum(sqrt(d)*dout)
        np.add.at(self.dtoken_W, idx_flat, dout_flat*(np.sqrt(self.d_model)))

        # pos grads:
        # posgradient = sum over batch of dout
        self.dpos_W[:] = 0
        self.dpos_W[:seq] = np.sum(dout, axis=0)    # (seq, d)
        #no need to return anything (we dont need to propagate the gradient to idxs)
        #because gradient / idx does not make sens

    def get_params(self):
        #parameters
        return([(self.token_W, self.dtoken_W, 'token_W'),(self.pos_W,   self.dpos_W,   'pos_W')])
    
class MultiHeadAttention:
    #we calculate the backpropagation for the selfattention on multiple heads
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        scale = 1.0 / np.sqrt(d_model)
        # weight matrices (project to full d_model then split into heads)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

        # grads
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)

        self.cache = None # cache to store things from forward to use in backward
    
    @staticmethod
    def softmax(x, axis=-1):
        m = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - m)
        return(e / np.sum(e, axis=axis, keepdims=True))

    def split_heads(self, x):
        # split attention heads
        # x : (batch, seq, d_model) -> (batch, n_head, seq, d_head)
        b, s, d = x.shape
        x = x.reshape(b, s, self.n_heads, self.d_head)
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

    def forward(self, x, mask=None):
        # self attention 
        _, s, _ = x.shape
        #linear layers
        Q = x @ self.W_q # queries (batch, seq, dmodel)
        K = x @ self.W_k # keys    (batch, seq, dmodel)
        V = x @ self.W_v # values  (batch, seq, dmodel)

        #we split the heads:
        Qh = self.split_heads(Q)  # (b, h, s, dh)
        Kh = self.split_heads(K)
        Vh = self.split_heads(V)
        scores = np.matmul(Qh, Kh.transpose(0,1,3,2)) / np.sqrt(self.d_head)

        if mask is None:
            mask = self.causal_mask(s)       # shape (s, s)
        if mask.ndim == 2:
            mask_b = mask[None, None, :, :]  # (1,1,s,s)
        elif mask.ndim == 3:
            mask_b = mask[:, None, :, :]     # (b,1,s,s)
        else:
            raise ValueError("mask must be None, 2D or 3D")
        
        scores = np.where(mask_b, scores, -1e9) #apply mask

        attn = self.softmax(scores, axis=-1)
        out_heads = np.matmul(attn, Vh)
        #combine:
        out = self.combine_heads(out_heads)
        out_proj = out @ self.W_o #final linear layer 
        # cache everything needed for backward
        self.cache = (x, Qh, Kh, Vh, attn, out)
        return(out_proj, attn)

    def backward(self, dout):
        # dout: (b, s, d_model)
        x, Qh, Kh, Vh, attn, out = self.cache
        b, s, d = x.shape
        h = self.n_heads
        dh = self.d_head

        # output projection W_o
        out_flat = out.reshape(-1, d)        
        dout_flat = dout.reshape(-1, d)
        #formula:
        # y = out @ Wo at the end of forward
        # for A, B 2 matrices and C = A@B
        # then dL/dB = A.transpose@dL/dC
        # and dL/dA = dL/dC@B.transpose
        # see https://robotchinwag.com/posts/gradient-of-matrix-multiplicationin-deep-learning/
        # so dWo = out.transpose @ dy
        self.dW_o[:] = out_flat.T @ dout_flat #we do that
        # we have dout = dy@Wo.transpose
        dout_preproj_flat = dout_flat @ self.W_o.T 
        dout_preproj = dout_preproj_flat.reshape(b, s, d) # (b,s,d)

        # split heads
        # here we reverse combine head
        d_out_heads = dout_preproj.reshape(b, s, h, dh).transpose(0,2,1,3) # (b,h,s,dh)

        # out_heads = attn@Vh 
        # d_attn = d_out_heads@Vh^T (same formula as above)
        # dVh = attn^T @ d_out_heads
        dVh = np.matmul(attn.transpose(0,1,3,2), d_out_heads)  # (b,h,s,dh)
        d_attn = d_out_heads@Vh.transpose(0,1,3,2)

        # attn = softmax(scores)
        # efficient formula: ds = (d_attn - sum(d_attn * attn, axis=-1, keepdims=True)) * attn
        tmp = np.sum(d_attn * attn, axis=-1, keepdims=True)
        dscores = (d_attn - tmp) * attn                        

        # scores = Qh @ Kh^T / sqrt(dh)
        dQh = dscores@Kh/np.sqrt(dh)            
        dKh = dscores.transpose(0,1,3,2)@Qh/np.sqrt(dh)  

        # combine heads inverse:
        dQ = dQh.transpose(0,2,1,3).reshape(b, s, d) # (b,s,d)
        dK = dKh.transpose(0,2,1,3).reshape(b, s, d)
        dV = dVh.transpose(0,2,1,3).reshape(b, s, d)

        
        x_flat = x.reshape(-1, d)      # (N, d)
        dQ_flat = dQ.reshape(-1, d)
        dK_flat = dK.reshape(-1, d)
        dV_flat = dV.reshape(-1, d)

        # Q = x @ W_q (same for k and v)
        # so dW_q = x.T@dQ
        self.dW_q[:] = x_flat.T @ dQ_flat
        self.dW_k[:] = x_flat.T @ dK_flat
        self.dW_v[:] = x_flat.T @ dV_flat

        # gradient / x from three paths
        dx_from_q = dQ_flat @ self.W_q.T
        dx_from_k = dK_flat @ self.W_k.T
        dx_from_v = dV_flat @ self.W_v.T
        dx_flat = dx_from_q + dx_from_k + dx_from_v
        dx = dx_flat.reshape(b, s, d)
        return(dx)

    def get_params(self):
        return ([
            (self.W_q, self.dW_q, 'W_q'), (self.W_k, self.dW_k, 'W_k'),
            (self.W_v, self.dW_v, 'W_v'), (self.W_o, self.dW_o, 'W_o')
        ])
    
class MLP:
    # we have now to do the last block of our transformer, the multi layer perceptron
    # we already did the 2 linear parts of the perceptron
    # but we still have to do the gelu part and combine
    def __init__(self, d_model, ff_mult=4):
        self.d_model = d_model
        self.d_ff = ff_mult * d_model
        scale = 1.0 / np.sqrt(d_model)
        self.fc1 = Linear(d_model, self.d_ff, scale=scale) # we use linear
        self.fc2 = Linear(self.d_ff, d_model, scale=scale)
        self.cache = None
    
    @staticmethod
    def gelu(x):
        return(0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))))
    
    @staticmethod
    def gelu_prime(x):
        # approx of the derivative
        phi = np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
        tanh_phi = np.tanh(phi)
        dphi_dx = np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)
        return(0.5 * (1 + tanh_phi) + 0.5 * x * (1 - tanh_phi**2) * dphi_dx)

    def forward(self, x):
        # linar, gelu , linear
        h = self.fc1.forward(x)  #first linear
        h_act = self.gelu(h) #gelu
        out = self.fc2.forward(h_act) # second linear
        # cache h for backward
        self.cache = h
        return(out)

    def backward(self, dout):
        d_h_act = self.fc2.backward(dout) #gradient from linear
        h = self.cache
        d_h = d_h_act * self.gelu_prime(h) #gradient from gelu
        dx = self.fc1.backward(d_h) #grad from linear
        return(dx)

    def get_params(self):
        return(self.fc1.get_params() + self.fc2.get_params())

# now we fusion every blocks in one
class TransformerBlock:
    def __init__(self, d_model, n_heads, ff_mult=4):
        # linear, attention, linear, mlp => out
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, ff_mult)

    def forward(self, x, mask=None):
        # x: (b,s,d)
        y = self.ln1.forward(x)
        attn_out, attn_map = self.attn.forward(y, mask)
        x = x + attn_out
        y2 = self.ln2.forward(x)
        mlp_out = self.mlp.forward(y2)
        x = x + mlp_out
        # store maps for ploting
        self.attn_map = attn_map
        return(x)

    def backward(self, dout):
        # we do the backward
        #residual1 : dout
        d_mlp_in = self.mlp.backward(dout)
        d_ln2_in = dout + d_mlp_in
        d_after_attn = self.ln2.backward(d_ln2_in)
        d_attn_in = d_after_attn
        d_ln1_out = self.attn.backward(d_attn_in)
        d_x_before = self.ln1.backward(d_ln1_out)
        dx = d_x_before + d_after_attn
        return(dx)
    
class transformer:
    # we can stack some blocks from the block transformer and use
    # them as a whole (simple) transformer
    def __init__(self, vocab_size, d_model, context, n_heads, n_layers, ff_mult=4):
        self.embed = Embedding(vocab_size, d_model, context)
        self.blocks = [TransformerBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, idx, mask=None):
        x = self.embed.forward(idx)     
        for block in self.blocks:
            x = block.forward(x, mask) #we forward all blocks
        # logits = x @ token_W.T  
        logits = x @ self.embed.token_W.T   
        self._cache = x  # stores x
        return(logits)

    def backward_from_logits(self, dlogits):
        x = self._cache
        b, seq, d = x.shape
        x_flat = x.reshape(-1, d) # (N, d)
        dlogits_flat = dlogits.reshape(-1, dlogits.shape[-1]) # (N, V)
        dW_out = (x_flat.T @ dlogits_flat).T
        dx = dlogits @ self.embed.token_W 

        # backprop through transformer blocks
        for block in reversed(self.blocks):
            dx = block.backward(dx)  # each returns dx to previous layer
        # then embed
        self.embed.backward(dx) 
        self.embed.dtoken_W += dW_out

    def params(self):
        # collect parameters from all modules for optimizer
        ps = []
        ps += self.embed.get_params()
        for i, block in enumerate(self.blocks):
            ps += [ (p, g, f'block{i}.ln1.{n}') for (p, g, n) in block.ln1.get_params() ]
            ps += [ (p, g, f'block{i}.attn.{n}') for (p, g, n) in block.attn.get_params() ]
            ps += [ (p, g, f'block{i}.ln2.{n}') for (p, g, n) in block.ln2.get_params() ]
            ps += [ (p, g, f'block{i}.mlp.{n}') for (p, g, n) in block.mlp.get_params() ]
        return(ps)


def sgd_step(params, lr):
    # params: list of (param_array, grad_array, name)
    # optimizer (very simple)
    for p, g, _ in params:
        p -= lr * g

def cross_entropy_loss_and_grad(logits, targets):
    # loss
    # recives logits from the model and targets token ids
    # and computes the loss
    # gives also the grad / logits to give back to the backprop
    # logits: (b, seq, V), targets: (b, seq) int
    b, seq, V = logits.shape
    logits_flat = logits.reshape(-1, V)   # (N, V)
    targets_flat = targets.reshape(-1)    # (N,)

    #softmax
    m = np.max(logits_flat, axis=1, keepdims=True)
    exp = np.exp(logits_flat - m)
    probs = exp / np.sum(exp, axis=1, keepdims=True)  # (N, V)

    # loss: negative log-likelihood
    N = logits_flat.shape[0]
    logp = np.log(probs[np.arange(N), targets_flat] + 1e-20)
    loss = -np.mean(logp)

    # gradient dL / dlogits
    dlogits = probs
    dlogits[np.arange(N), targets_flat] -= 1.0
    dlogits /= N  # because we averaged loss
    dlogits = dlogits.reshape(b, seq, V)
    return(loss, dlogits)

def softmax(a, axis=-1):
    a_max = np.max(a, axis=axis, keepdims=True)
    e = np.exp(a - a_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def generate(model, prompt_ids, max_new_tokens=20, top_k=None, sample=False):
    idx = np.array(prompt_ids, dtype=np.int32)
    batch = idx.shape[0]

    # determine context length
    try:
        context = model.embed.pos_W.shape[0]  
    except Exception:
        context = getattr(model, 'context', None)

    for _ in range(max_new_tokens):
        # forward pass
        logits = model.forward(idx)         
        last_logits = logits[:, -1, :]

        # top-k: mask out everything except top_k largest logits
        V = last_logits.shape[-1]
        if top_k is not None and 0 < top_k < V:
            kth = np.partition(last_logits, -top_k, axis=-1)[:, -top_k][:, None]
            last_logits = np.where(last_logits < kth, -1e9, last_logits)

        probs = softmax(last_logits, axis=-1)

        if sample:
            # this one takes with respect to probs
            next_ids = np.array([np.random.choice(V, p=probs[i]) for i in range(batch)], dtype=np.int32)
        else:
            # greedy, just take the highest prob
            next_ids = probs.argmax(axis=-1).astype(np.int32)
      
        idx = np.concatenate([idx, next_ids[:, None]], axis=1)
        if context is not None and idx.shape[1] > context:
            idx = idx[:, -context:]
    return (idx)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tokenizer import BPE_tokenizer, tokenizer, detokenizer 
    corpus_sentences = [
    "salut à tous",
    "bonjour tout le monde",
    "comment ça va",
    "je suis en train de tester",
    "voici une autre phrase pour le corpus",
    "elle voulait le surprendre",
    "je suis ton père"
]
    words_for_bpe = [s for s in corpus_sentences]
    vocab = BPE_tokenizer(words_for_bpe, N=100)
    if "<pad>" not in vocab:
        # create a new ordered vocab with pad first
        new_vocab = {"<pad>": 0}
        for k in vocab.keys():
            new_vocab[k] = 0
        vocab = new_vocab
    
    seqs = []
    for s in corpus_sentences:
        ids = tokenizer(s, vocab) # token ids
        seqs.append(ids)
    
    # compute max length and pad
    max_len = max(len(x) for x in seqs)
    pad_idx = list(vocab.keys()).index("<pad>")

    def pad_batch(seqs, max_len, pad_idx):
        x = np.full((len(seqs), max_len), pad_idx, dtype=np.int32)
        for i, seq in enumerate(seqs):
            x[i, :len(seq)] = seq
        return x

    x = pad_batch(seqs, max_len, pad_idx)
    y = np.roll(x, -1, axis=1)

    V = len(vocab)
    d_model = 64
    context = max_len
    n_heads = 4
    n_layers = 2

    model = transformer(V, d_model, context, n_heads, n_layers, ff_mult=4)

    # training loop
    lr = 1e-2
    epochs = 1000
    batch_size = len(x)  # small dataset: use full-batch (overfit, good to test the code)

    for epoch in range(epochs):
        logits = model.forward(x)             
        loss, dlogits = cross_entropy_loss_and_grad(logits, y)
        # zero grads
        for p, g, _ in model.params():
            g[:] = 0
        # backward
        model.backward_from_logits(dlogits)
        # update
        sgd_step(model.params(), lr)

        if epoch % 20 == 0:
            print(f"epoch {epoch} loss {loss:.4f}")
        
    prompt_text = "voici une"
    prompt_ids_list = tokenizer(prompt_text, vocab)
    prompt_array = np.array([prompt_ids_list])
    gen_ids = generate(model, prompt_array, max_new_tokens=15, top_k=50, sample=True)
    generated_text = detokenizer(gen_ids[0].tolist(), vocab)
    print("result: ", generated_text)