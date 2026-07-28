"""
TAPPA 3 — Token, embedding e spazi latenti.

Una rete mangia NUMERI, ma noi abbiamo TESTO. Questa tappa costruisce il ponte.

  PARTE A — BPE: come si spezza il testo in token (algoritmo vero, ~20 righe)
  PARTE B — Embedding: perché la tabella degli embedding È un layer della Tappa 1
  PARTE C — Spazio latente: lo addestriamo davvero e guardiamo la semantica EMERGERE
"""

import numpy as np
from collections import Counter


# =========================================================================
# PARTE A — BPE (Byte-Pair Encoding): come nascono davvero i token
# =========================================================================
# L'algoritmo è stupido: trova la coppia di simboli adiacenti più frequente,
# fondila in un simbolo nuovo, ripeti. La FREQUENZA decide la granularità.

def bpe_impara(corpus, n_fusioni):
    parole = [list(p) + ["</w>"] for p in corpus.split()]   # parti dai caratteri
    fusioni = []

    for _ in range(n_fusioni):
        coppie = Counter()
        for p in parole:
            for i in range(len(p) - 1):
                coppie[(p[i], p[i + 1])] += 1
        if not coppie:
            break
        migliore = coppie.most_common(1)[0][0]              # la coppia più frequente
        fusioni.append(migliore)

        nuove = []                                          # fondila ovunque appaia
        for p in parole:
            i, out = 0, []
            while i < len(p):
                if i < len(p) - 1 and (p[i], p[i + 1]) == migliore:
                    out.append(p[i] + p[i + 1])
                    i += 2
                else:
                    out.append(p[i])
                    i += 1
            nuove.append(out)
        parole = nuove

    return fusioni, parole


# =========================================================================
# PARTE C — Il corpus giocattolo: frasi con una struttura regolare.
# =========================================================================
# Nota: NESSUNA etichetta, nessuna categoria. Solo frasi. Le categorie dovranno
# EMERGERE dai pesi da sole.
FRASI = [
    "il gatto mangia il pesce", "il cane mangia la carne",
    "il gatto beve il latte",   "il cane beve la acqua",
    "il topo mangia il pesce",  "il topo beve la acqua",
    "il gatto dorme",           "il cane dorme",  "il topo dorme",
    "il re governa il regno",   "la regina governa il regno",
    "il re comanda il popolo",  "la regina comanda il popolo",
    "il re dorme",              "la regina dorme",
    "il uomo mangia la carne",  "la donna mangia il pesce",
    "il uomo beve la acqua",    "la donna beve il latte",
    "il uomo dorme",            "la donna dorme",
]


def costruisci_vocabolario(frasi):
    parole = sorted({w for f in frasi for w in f.split()})
    stoi = {w: i for i, w in enumerate(parole)}     # string -> id  (la "tokenizzazione")
    itos = {i: w for w, i in stoi.items()}          # id -> string
    return stoi, itos


def coppie_contesto(frasi, stoi, finestra=2):
    """Dati di addestramento: (parola_centrale, parola_vicina). È lo skip-gram."""
    dati = []
    for f in frasi:
        ids = [stoi[w] for w in f.split()]
        for i, centro in enumerate(ids):
            for j in range(max(0, i - finestra), min(len(ids), i + finestra + 1)):
                if i != j:
                    dati.append((centro, ids[j]))
    return np.array(dati)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def coseno(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def vicini(parola, E, stoi, itos, k=3):
    v = E[stoi[parola]]
    sim = [(coseno(v, E[i]), itos[i]) for i in range(len(itos)) if itos[i] != parola]
    return sorted(sim, reverse=True)[:k]


if __name__ == "__main__":
    # ---------------- PARTE A: BPE ----------------
    print("=== PARTE A — BPE: come nascono i token ===")
    corpus_bpe = ("tokenizzazione tokenizzare token tokenizzato "
                  "realizzazione realizzare organizzazione organizzare")
    fusioni, parole = bpe_impara(corpus_bpe, n_fusioni=18)
    print("Prime fusioni imparate dal corpus:")
    print("  ", [a + b for a, b in fusioni[:10]])
    print("Risultato della tokenizzazione:")
    for p in parole[:4]:
        print("  ", p)
    print("=> I pezzi FREQUENTI diventano un token unico. Nessuno li ha scelti a mano:")
    print("   li ha decisi la statistica del corpus.\n")

    # ---------------- PARTE B + C: embedding e spazio latente ----------------
    stoi, itos = costruisci_vocabolario(FRASI)
    V = len(stoi)        # dimensione del vocabolario
    D = 8                # dimensione dello spazio latente (in un LLM vero: migliaia)

    print("=== PARTE B — La tabella degli embedding È un layer ===")
    print(f"Vocabolario: {V} token. Spazio latente: {D} dimensioni.")
    print(f"=> Matrice degli embedding E: {V}x{D} = {V*D} pesi, addestrati come")
    print("   qualunque altro peso della Tappa 1, con la discesa della Tappa 2.\n")

    rng = np.random.default_rng(0)
    E = rng.normal(scale=0.5, size=(V, D))    # embedding: DA IMPARARE
    W = rng.normal(scale=0.5, size=(D, V))    # layer di uscita: DA IMPARARE

    dati = coppie_contesto(FRASI, stoi)
    X, Y = dati[:, 0], dati[:, 1]
    n = len(dati)
    lr = 0.5

    print("=== PARTE C — Addestriamo lo spazio latente ===")
    print("Compito: data una parola, prevedi le parole che le stanno accanto.")
    print("(Nessuna etichetta, nessuna categoria: solo frasi grezze.)\n")

    for passo in range(3001):
        h = E[X]                              # LOOKUP = one_hot @ E  (Parte B!)
        logits = h @ W
        probs = softmax(logits)

        loss = -np.log(probs[np.arange(n), Y] + 1e-9).mean()

        # backpropagation (Tappa 2)
        dlogits = probs.copy()
        dlogits[np.arange(n), Y] -= 1
        dlogits /= n
        dW = h.T @ dlogits
        dh = dlogits @ W.T
        dE = np.zeros_like(E)
        np.add.at(dE, X, dh)                  # accumula il gradiente sulle righe usate

        W -= lr * dW                          # discesa del gradiente
        E -= lr * dE

        if passo % 1000 == 0:
            print(f"  passo {passo:>4}   loss = {loss:.4f}")

    # ---------------- Che cosa è EMERSO nello spazio? ----------------
    print("\n=== Chi si è avvicinato a chi? (similarità del coseno) ===")
    for parola in ["gatto", "re", "mangia", "il"]:
        vs = ", ".join(f"{w} ({s:.2f})" for s, w in vicini(parola, E, stoi, itos))
        print(f"  vicini di '{parola:<7}': {vs}")

    print("\nNessuno ha detto al modello che gatto/cane/topo sono animali,")
    print("o che re/regina sono sovrani. Ha solo dovuto PREVEDERE il contesto.")
    print("Le categorie sono emerse da sole come GEOMETRIA: questo è lo spazio latente.")
