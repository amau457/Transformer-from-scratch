import sys

class collection:
    # class of object collection, that represent the colection of tokens we have in the BPE method
    def __init__(self):
        self.col = {}
    
    def reinitialize(self):
        #reinitalize the count for every tokens
        for key in self.col:
            self.col[key] = 0 
        
    def count(self, token):
        #O(1)
        self.col[token] = self.col.get(token, 0) + 1

    def count_2(self, token):
        # depreciated because very slow: O(n) n size of vocab
        test = False
        for key in self.col:
            if key == token: # increase the count if already existing
                self.col[key] += 1
                test = True
                break
        if not test: #add the token to the collection if not already existing
            self.col.setdefault(token, 1)

def counter(word):
    # take a word and returns the word with letters separated with spaces
    res = ''
    for a in word:
        res += a+' '
    return(res)

def replace(word, vocab, max_tokens_len):
    # Tokenize word using a collection of vocab. Longest match first.
    # we return a list
    res_tokens = []
    k = 0
    L = len(word)
    while k < L:
        # we ignore spaces
        if word[k].isspace():
            k += 1
            continue

        matched = None
        # first the longest tokens
        max_try = min(max_tokens_len, L - k)
        for length in range(max_try, 0, -1):
            candidate = word[k:k+length]
            if candidate in vocab:
                matched = candidate
                break
        if matched is None:
            matched = word[k]
        res_tokens.append(matched)
        k += len(matched)
    return res_tokens


def read_file(path):
    # reads a txt file with text and split every caracter with a space 
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    words = text.split()
    return(words)

def BPE_tokenizer(words, N):
    # byte pair encoding (N execution of the loop)
    tokens = collection()
    # initialization of all unigrams in the vocab:
    for w in words:
        for c in w:
            tokens.col.setdefault(c, 0)
    max_tokens_len = 1 
    for step in range(N):
        freqs = {} #to store frequency of occurences
        for word in words:
            if step == 0:
                splited_word = counter(word)
            else:
                splited_word = replace(word, tokens.col, max_tokens_len)
            previous = None
            for caract in splited_word:
                if previous is not None:
                    pair = previous + caract     #create pair of tokens
                    freqs[pair] = freqs.get(pair, 0) + 1
                previous = caract
        if not freqs:
            # we already fusioned everything
            break
        best = max(freqs, key=freqs.get) #the pair with the highest frequency (the first one occured)
        if len(best) > max_tokens_len:
            max_tokens_len = len(best)
        # add the new token to the vocab
        tokens.col.setdefault(best, 0)
    tokens.col = dict(sorted(tokens.col.items())) #we sort the vocab in lexicographical order
    return(tokens.col)

def tokenizer(entry, vocab):
    # to use in practice, returns a list of integer (the order of the token in the vocab)
    keys = list(vocab.keys())
    max_token_size = max(len(k) for k in vocab)
    tokens_list = replace(entry, vocab, max_token_size)
    res = []
    for a in tokens_list:
        idx = keys.index(a)
        res.append(idx)
    return(res)

def detokenizer(entry, vocab):
    # to use in practice, returns a string from a list of intergers (tokens)
    keys = list(vocab.keys())
    res = ''
    for a in entry: #a is an integer
        res+= keys[a]
    return(res)

if __name__ == "__main__":
    import pickle
    words = read_file('data.txt')
    col = BPE_tokenizer(words, 1000)
    word = 'Elle l’était davantage, un de ses grands désirs, qu’elle n’avait jamais avoué à Julien de peur de le choquer, était de le voir quitter, ne fût-ce que pour un jour, son triste habit noir. Avec une adresse vraiment admirable chez une femme si naturelle, elle obtint d’abord de M. de Moirod, et ensuite de M. le sous-préfet de Maugiron, que Julien serait nommé garde d’honneur de préférence à cinq ou six jeunes gens, fils de fabricants fort aisés, et dont deux au moins étaient d’une exemplaire piété. M. Valenod, qui comptait prêter sa calèche aux plus jolies femmes de la ville et faire admirer ses beaux normands, consentit à donner un de ses chevaux à Julien, l’être qu’il haïssait le plus. Mais tous les gardes d’honneur avaient à eux ou d’emprunt quelqu’un de ces beaux habits bleu de ciel avec deux épaulettes de colonel en argent, qui avaient brillé sept ans auparavant. Mme de Rênal voulait un habit neuf, et il ne lui restait que quatre jours pour envoyer à Besançon, et en faire revenir l’habit d’uniforme, les armes, le chapeau, etc., tout ce qui fait un garde d’honneur. Ce qu’il y a de plaisant, c’est qu’elle trouvait imprudent de faire faire l’habit de Julien à Verrières. Elle voulait le surprendre, lui et la ville.'
    print(replace(word, col, max(len(k) for k in col)))
    print(tokenizer("salut à tous", col))
    print(detokenizer(tokenizer("salut à tous", col), col))
    with open("vocab.pkl", "wb") as f:
        pickle.dump(col, f, protocol=pickle.HIGHEST_PROTOCOL)
   