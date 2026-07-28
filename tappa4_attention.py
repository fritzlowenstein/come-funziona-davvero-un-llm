"""
TAPPA 4 — L'Attention.

Problema (dalla Tappa 3): gli embedding sono STATICI. "riso" ha un solo vettore,
ma in "mangio il riso" e "un riso contagioso" significa due cose diverse.
Serve un'operazione che permetta a ogni token di GUARDARE gli altri e
aggiornare sé stesso in base al contesto.

  PARTE A — perché la media semplice non basta
  PARTE B — la formula:  softmax(Q @ K.T / sqrt(d)) @ V
  PARTE C — la maschera causale (perché un LLM non può sbirciare il futuro)
  PARTE D — ADDESTRIAMO un'attention e guardiamo DOVE ha imparato a guardare
  PARTE E — multi-head: più attention in parallelo
"""

import numpy as np


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# =========================================================================
# PARTE B — L'attention, nuda e cruda. Sono 4 righe.
# =========================================================================
def attention(X, Wq, Wk, Wv, causale=False):
    Q = X @ Wq                     # "che cosa sto cercando?"
    K = X @ Wk                     # "che cosa offro io?"
    V = X @ Wv                     # "che informazione porto con me"

    d = Q.shape[-1]
    punteggi = Q @ K.T / np.sqrt(d)          # rilevanza di ogni token per ogni altro

    if causale:                               # PARTE C: vieta di guardare avanti
        N = X.shape[0]
        maschera = np.triu(np.ones((N, N)), k=1).astype(bool)
        punteggi[maschera] = -np.inf          # -inf -> dopo la softmax diventa 0

    pesi = softmax(punteggi)                  # punteggi -> percentuali (somma = 1)
    return pesi @ V, pesi                     # media PESATA dei Value


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------------
    # PARTE A — perché la media semplice non basta
    # ---------------------------------------------------------------------
    print("=== PARTE A — La media semplice è cieca ===")
    print("Media: ogni token pesa 1/N. Ma per disambiguare 'riso' conta MOLTO")
    print("'contagioso' e NULLA 'un'. Serve una media PESATA, con pesi calcolati")
    print("dal contenuto. Ecco a cosa serve l'attention.\n")

    # ---------------------------------------------------------------------
    # PARTE B/C — il meccanismo, e la maschera causale
    # ---------------------------------------------------------------------
    print("=== PARTE C — La maschera causale ===")
    N, D = 5, 8
    X = rng.normal(size=(N, D))
    Wq, Wk, Wv = (rng.normal(size=(D, D)) * 0.3 for _ in range(3))

    _, pesi = attention(X, Wq, Wk, Wv, causale=True)
    print("Matrice dei pesi di attention (riga i = a chi guarda il token i):")
    print(np.round(pesi, 2))
    print("=> Triangolo superiore = 0: il token 2 NON può vedere i token 3,4.")
    print("   Un LLM prevede il token successivo: se sbirciasse il futuro,")
    print("   copierebbe la risposta invece di impararla.")
    print("   Ogni riga somma a 1 (è una distribuzione di attenzione).\n")

    # ---------------------------------------------------------------------
    # PARTE D — LA PROVA: addestriamo un'attention e vediamo dove guarda.
    # ---------------------------------------------------------------------
    # Compito: la sequenza è [A, B, SELETTORE].
    #   se SELETTORE == S0  -> l'output giusto è A
    #   se SELETTORE == S1  -> l'output giusto è B
    # A e B sono token casuali: NON si può risolvere a memoria.
    # L'unico modo è che l'ultimo token IMPARI A GUARDARE la posizione giusta
    # in base al proprio contenuto. È esattamente il "routing" dell'attention.
    print("=== PARTE D — Addestriamo l'attention su un compito che la richiede ===")
    print("Sequenza: [A, B, SELETTORE].  S0 -> rispondi A.  S1 -> rispondi B.")
    print("A e B sono casuali: impossibile risolverlo a memoria.\n")

    N_CONT = 6                      # token di contenuto: 0..5
    S0, S1 = 6, 7                   # i due selettori
    V_SIZE, D, T = 8, 16, 3

    E = rng.normal(scale=0.3, size=(V_SIZE, D))   # embedding (Tappa 3)
    P = rng.normal(scale=0.3, size=(T, D))        # embedding POSIZIONALI (vedi sotto)
    Wq = rng.normal(scale=0.3, size=(D, D))
    Wk = rng.normal(scale=0.3, size=(D, D))
    Wv = rng.normal(scale=0.3, size=(D, D))
    Wo = rng.normal(scale=0.3, size=(D, V_SIZE))  # proiezione finale sul vocabolario

    def batch(n):
        a = rng.integers(0, N_CONT, size=n)
        b = rng.integers(0, N_CONT, size=n)
        sel = rng.integers(0, 2, size=n)
        toks = np.stack([a, b, np.where(sel == 0, S0, S1)], axis=1)
        target = np.where(sel == 0, a, b)
        return toks, target

    lr, n = 0.05, 256
    for passo in range(4001):
        toks, target = batch(n)
        X = E[toks] + P                          # embedding + posizione
        Q, K, Vv = X @ Wq, X @ Wk, X @ Wv
        sc = Q @ K.transpose(0, 2, 1) / np.sqrt(D)
        m = np.triu(np.ones((T, T)), k=1).astype(bool)
        sc[:, m] = -np.inf
        att = softmax(sc)
        ctx = att @ Vv                           # media pesata dei Value
        h = ctx[:, -1, :]                        # ci interessa l'ultimo token
        logits = h @ Wo
        probs = softmax(logits)
        loss = -np.log(probs[np.arange(n), target] + 1e-9).mean()

        # backprop (Tappa 2) — scritta a mano
        dlogits = probs.copy(); dlogits[np.arange(n), target] -= 1; dlogits /= n
        dWo = h.T @ dlogits
        dh = dlogits @ Wo.T
        dctx = np.zeros_like(ctx); dctx[:, -1, :] = dh
        datt = dctx @ Vv.transpose(0, 2, 1)
        dVv = att.transpose(0, 2, 1) @ dctx
        dsc = att * (datt - (datt * att).sum(-1, keepdims=True))
        dsc = np.where(np.isfinite(sc), dsc, 0.0) / np.sqrt(D)
        dQ = dsc @ K
        dK = dsc.transpose(0, 2, 1) @ Q
        dWq = np.einsum('bti,btj->ij', X, dQ)
        dWk = np.einsum('bti,btj->ij', X, dK)
        dWv = np.einsum('bti,btj->ij', X, dVv)
        dX = dQ @ Wq.T + dK @ Wk.T + dVv @ Wv.T
        dE = np.zeros_like(E); np.add.at(dE, toks, dX)
        dP = dX.sum(axis=0)

        for W, dW in ((Wo, dWo), (Wq, dWq), (Wk, dWk), (Wv, dWv), (E, dE), (P, dP)):
            W -= lr * dW

        if passo % 1000 == 0:
            acc = (probs.argmax(1) == target).mean()
            print(f"  passo {passo:>4}   loss = {loss:.4f}   accuratezza = {acc:.1%}")

    # --- DOVE ha imparato a guardare l'ultimo token? ---
    print("\n=== DOVE guarda l'attention dopo l'addestramento? ===")
    for sel, nome in ((S0, "S0 (-> deve rispondere A, in posizione 0)"),
                      (S1, "S1 (-> deve rispondere B, in posizione 1)")):
        toks = np.array([[3, 5, sel]])
        X = E[toks] + P
        Q, K = X @ Wq, X @ Wk
        sc = Q @ K.transpose(0, 2, 1) / np.sqrt(D)
        att = softmax(sc)[0, -1]
        print(f"  selettore {nome}")
        print(f"    attenzione dell'ultimo token -> pos0: {att[0]:.2f}   "
              f"pos1: {att[1]:.2f}   pos2(sé): {att[2]:.2f}")

    print("\n=> Il modello ha imparato DA SOLO a puntare l'attenzione sulla posizione")
    print("   giusta a seconda del contenuto. Nessuno gliel'ha detto: è emerso dal")
    print("   gradiente. Questo è il 'routing' dinamico dell'informazione.")

    # ---------------------------------------------------------------------
    # PARTE E — Multi-head
    # ---------------------------------------------------------------------
    print("\n=== PARTE E — Multi-head attention ===")
    print("Una sola attention impara UNA relazione. Ma in una frase ce ne sono tante")
    print("simultanee: chi-è-il-soggetto, a-cosa-si-riferisce-'lo', qual-è-il-tempo-verbale...")
    print("Soluzione: N attention IN PARALLELO ('teste'), ognuna con i suoi Wq/Wk/Wv.")
    print("Ogni testa impara una relazione diversa; i risultati si concatenano.")
    print("=> È solo 'fai la stessa cosa 32 volte con pesi diversi'. Nient'altro.")
