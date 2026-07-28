"""
TAPPA 5 (parte 1/2) — Il Transformer completo, in numpy, senza framework.

Qui assembliamo TUTTI i pezzi delle tappe precedenti in un LLM vero e proprio
(architettura decoder-only, la stessa di GPT/Claude). Pesi casuali: qui ci
interessa vedere l'ARCHITETTURA e ogni singola operazione, senza magia.

Il modello, per intero:

    x = embedding[token] + posizione            <- Tappa 3 + Tappa 4
    ripeti N volte:
        x = x + attention( layernorm(x) )       <- comunicazione TRA token
        x = x + feedforward( layernorm(x) )     <- elaborazione DENTRO ogni token
    logits = layernorm(x) @ W_out               <- un punteggio per ogni parola
    probs  = softmax(logits)                    <- ...trasformato in probabilità

È tutto qui. Davvero.
"""

import numpy as np

rng = np.random.default_rng(0)


# --- Mattoncini elementari ------------------------------------------------
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def gelu(x):
    # attivazione non-lineare (Tappa 1), variante "morbida" della ReLU
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def layernorm(x, gamma, beta, eps=1e-5):
    """Il 'termostato': media 0, deviazione 1, poi riscala con pesi imparati.
    Senza, i numeri esplodono attraversando decine di blocchi."""
    mu = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True)
    return gamma * (x - mu) / (sigma + eps) + beta


# --- Multi-head attention (Tappa 4), ora con più teste --------------------
def multi_head_attention(x, p, n_teste):
    T, D = x.shape
    d_testa = D // n_teste

    Q = (x @ p["Wq"]).reshape(T, n_teste, d_testa).transpose(1, 0, 2)
    K = (x @ p["Wk"]).reshape(T, n_teste, d_testa).transpose(1, 0, 2)
    V = (x @ p["Wv"]).reshape(T, n_teste, d_testa).transpose(1, 0, 2)

    punteggi = Q @ K.transpose(0, 2, 1) / np.sqrt(d_testa)      # rilevanza
    maschera = np.triu(np.ones((T, T)), k=1).astype(bool)       # niente futuro!
    punteggi[:, maschera] = -np.inf

    pesi = softmax(punteggi)                                     # somma a 1
    out = pesi @ V                                               # media pesata

    out = out.transpose(1, 0, 2).reshape(T, D)                   # riunisci le teste
    return out @ p["Wo"], pesi


# --- Feed-forward: il layer della Tappa 1. Qui abita la CONOSCENZA. -------
def feed_forward(x, p):
    return gelu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]      # D -> 4D -> D


# --- IL BLOCCO TRANSFORMER: quattro righe. È TUTTO. ----------------------
def blocco(x, p, n_teste):
    a, pesi = multi_head_attention(layernorm(x, p["g1"], p["b_ln1"]), p, n_teste)
    x = x + a                                    # <- CONNESSIONE RESIDUA
    f = feed_forward(layernorm(x, p["g2"], p["b_ln2"]), p)
    x = x + f                                    # <- CONNESSIONE RESIDUA
    return x, pesi


# --- Il modello completo --------------------------------------------------
def init_modello(V, D, T, n_blocchi, n_teste):
    def n(*shape):
        return rng.normal(scale=0.02, size=shape)

    m = {"E": n(V, D), "P": n(T, D), "W_out": n(D, V),
         "g_f": np.ones(D), "b_f": np.zeros(D), "blocchi": []}
    for _ in range(n_blocchi):
        m["blocchi"].append({
            "Wq": n(D, D), "Wk": n(D, D), "Wv": n(D, D), "Wo": n(D, D),
            "W1": n(D, 4 * D), "b1": np.zeros(4 * D),      # espansione 4x
            "W2": n(4 * D, D), "b2": np.zeros(D),
            "g1": np.ones(D), "b_ln1": np.zeros(D),
            "g2": np.ones(D), "b_ln2": np.zeros(D),
        })
    return m


def forward(m, tokens, n_teste):
    T = len(tokens)
    x = m["E"][tokens] + m["P"][:T]              # token + posizione (Tappe 3-4)

    tutti_i_pesi = []
    for p in m["blocchi"]:                        # <- il "residual stream"
        x, pesi = blocco(x, p, n_teste)
        tutti_i_pesi.append(pesi)

    x = layernorm(x, m["g_f"], m["b_f"])
    logits = x @ m["W_out"]                       # <- ECCO I LOGIT
    return logits, tutti_i_pesi


if __name__ == "__main__":
    V, D, T, N_BLOCCHI, N_TESTE = 1000, 64, 10, 4, 8
    m = init_modello(V, D, T, N_BLOCCHI, N_TESTE)
    tokens = rng.integers(0, V, size=T)

    logits, pesi = forward(m, tokens, N_TESTE)

    print("=== IL PERCORSO DI UN TOKEN DENTRO IL MODELLO ===")
    print(f"  input                : {T} token (interi)")
    print(f"  dopo embedding+pos   : ({T}, {D})   <- ogni token è un vettore")
    print(f"  dopo {N_BLOCCHI} blocchi     : ({T}, {D})   <- STESSA forma! il")
    print("                          residual stream non cambia mai dimensione:")
    print("                          ogni blocco ci SOMMA sopra il suo contributo")
    print(f"  dopo W_out           : {logits.shape}  <- un punteggio per OGNI parola")
    print()

    print("=== I LOGIT (per l'ULTIMA posizione: 'qual è la prossima parola?') ===")
    ultimi = logits[-1]
    print("  primi 8 logit grezzi :", np.round(ultimi[:8], 3))
    print("  -> numeri arbitrari: possono essere negativi, non sommano a nulla.")
    print("     Sono 'punteggi di plausibilità', uno per parola del vocabolario.\n")

    probs = softmax(ultimi)
    print("  dopo softmax         :", np.round(probs[:8], 5))
    print(f"  -> ora sono PROBABILITÀ. Somma di tutte: {probs.sum():.4f}")
    print(f"     Parola più probabile: token #{probs.argmax()} ({probs.max():.2%})")
    print("     (Il modello ha pesi casuali: le previsioni sono spazzatura.")
    print("      Ma il MECCANISMO è esattamente quello di un LLM vero.)\n")

    print("=== QUANTI PARAMETRI? (il 'numero di parametri' di cui si parla) ===")
    p0 = m["blocchi"][0]
    par_att = sum(p0[k].size for k in ("Wq", "Wk", "Wv", "Wo"))
    par_ffn = sum(p0[k].size for k in ("W1", "b1", "W2", "b2"))
    tot = m["E"].size + m["P"].size + m["W_out"].size + \
        sum(v.size for b in m["blocchi"] for v in b.values())

    print(f"  attention (per blocco): {par_att:>8,}   ({par_att/(par_att+par_ffn):.0%})")
    print(f"  feed-forward (per blocco): {par_ffn:>5,}   ({par_ffn/(par_att+par_ffn):.0%})")
    print(f"  TOTALE modello        : {tot:>8,}")
    print("\n  => I 2/3 dei parametri stanno nel FEED-FORWARD, non nell'attention!")
    print("     L'attention decide CHI GUARDA CHI (comunicazione).")
    print("     Il feed-forward è dove sono immagazzinati i FATTI (conoscenza).")
    print("     Questo rapporto è lo stesso in GPT-3 e in tutti i modelli reali.")
