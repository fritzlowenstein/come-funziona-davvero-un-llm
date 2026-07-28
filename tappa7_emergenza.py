"""
TAPPA 7 — Perché "non capiamo perché funziona".

Sappiamo ESATTAMENTE ogni operazione del modello (moltiplicazioni e softmax).
Eppure non capiamo il suo COMPORTAMENTO. Due fenomeni concreti lo spiegano:

  PARTE A — EMERGENZA: capacità che compaiono "di colpo" con la scala.
            Ne mostriamo il meccanismo matematico esatto (bulletproof, istantaneo).
  PARTE B — GROKKING: una rete che per MIGLIAIA di passi sembra solo memorizzare,
            e poi IMPROVVISAMENTE generalizza. Lo addestriamo davvero e lo guardiamo.
"""

import numpy as np


# =========================================================================
# PARTE A — Perché le capacità sembrano emergere "di colpo"
# =========================================================================
# Molti compiti richiedono di azzeccare TUTTI i passi di una catena.
# Es: un problema di matematica in K passaggi è giusto solo se TUTTI e K
# i passaggi sono giusti. Se il modello ha probabilità p di azzeccare
# ogni singolo passo, la probabilità di azzeccare l'INTERO compito è p^K.
#
# Ora: p (la competenza "grezza") cresce in modo LISCIO con la scala.
# Ma p^K, per K grande, resta incollato a zero... e poi SCHIZZA su.
# La capacità "emerge" di colpo NON perché dentro sia successo qualcosa di
# magico, ma perché la misuriamo con un metro tutto-o-niente.

def parte_A():
    print("=" * 70)
    print("PARTE A — L'EMERGENZA come effetto di una catena di passi")
    print("=" * 70)
    print("Competenza per-passo 'p' che cresce LISCIA con la scala del modello.")
    print("Successo sul compito completo = p^K (devi azzeccare tutti i K passi).\n")

    p = np.linspace(0.0, 1.0, 21)          # competenza grezza: cresce liscia
    print(f"{'p (competenza per passo)':<26}", end="")
    for K in (1, 5, 20):
        print(f"| compito {K:>2} passi ", end="")
    print()
    print("-" * 74)
    for pi in p[::2]:
        print(f"{pi:<26.2f}", end="")
        for K in (1, 5, 20):
            print(f"|      {pi**K:>6.1%}     ", end="")
        print()

    print("\n  => Guarda la colonna 'K=20': resta ~0% mentre p sale da 0.5 a 0.85,")
    print("     poi SCHIZZA verso il 100% nell'ultimo tratto. Sembra un interruttore.")
    print("     Ma NIENTE è cambiato di scatto dentro: è cambiata liscia p.")
    print("     'Emergenza' = una curva liscia vista attraverso un metro a soglia.\n")
    print("  Questa è metà della risposta a 'perché non lo capiamo': il")
    print("  comportamento macroscopico non assomiglia al meccanismo microscopico.\n")


# =========================================================================
# PARTE B — GROKKING: memorizzazione, poi generalizzazione improvvisa
# =========================================================================
# Addestriamo una piccola rete a calcolare  (a + b) mod P.
# Le diamo solo METÀ delle coppie possibili; l'altra metà è il test.
# Fenomeno (Power et al. 2022): la rete prima MEMORIZZA (train 100%, test a caso)
# e MOLTO dopo, di colpo, GENERALIZZA (test -> 100%). Nel mezzo, dal solo
# train-error non potresti MAI distinguere "ha memorizzato" da "ha capito".

def parte_B():
    print("=" * 70)
    print("PARTE B — GROKKING: la rete impara (a+b) mod P")
    print("=" * 70)
    rng = np.random.default_rng(0)
    P = 23
    D = 128

    # dataset: tutte le coppie (a,b); target = (a+b) mod P
    A, B = np.meshgrid(np.arange(P), np.arange(P))
    A, B = A.ravel(), B.ravel()
    Y = (A + B) % P
    N = len(A)

    perm = rng.permutation(N)
    n_train = N // 2
    tr, te = perm[:n_train], perm[n_train:]

    # modello minimo: embedding(a) + embedding(b) -> MLP -> softmax su P classi
    params = {
        "Ea": rng.normal(scale=0.3, size=(P, D)),
        "Eb": rng.normal(scale=0.3, size=(P, D)),
        "W1": rng.normal(scale=1/np.sqrt(D), size=(D, D)), "b1": np.zeros(D),
        "W2": rng.normal(scale=1/np.sqrt(D), size=(D, P)), "b2": np.zeros(P),
    }

    def softmax(z):
        z = z - z.max(-1, keepdims=True); e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    def forward(idx):
        h_pre = params["Ea"][A[idx]] + params["Eb"][B[idx]]
        h = np.maximum(0, h_pre @ params["W1"] + params["b1"])
        return softmax(h @ params["W2"] + params["b2"]), h, h_pre

    def accuratezza(idx):
        probs, _, _ = forward(idx)
        return (probs.argmax(1) == Y[idx]).mean()

    # AdamW: un ottimizzatore vero (Adam) + weight decay DISACCOPPIATO.
    lr, wd = 2e-3, 1.0
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(vv) for k, vv in params.items()}
    b1a, b2a, eps = 0.9, 0.98, 1e-8

    print(f"P={P}, esempi totali={N}, di cui {n_train} in train.  AdamW, weight_decay={wd}")
    print("(Il weight decay penalizza i pesi grandi: col tempo preferisce la")
    print(" soluzione 'pulita' che generalizza a quella 'sporca' che memorizza.)\n")
    print(f"  {'passo':>6} | {'train acc':>9} | {'test acc':>8} |")
    print("  " + "-" * 34)

    picco_gap, groccato = 0.0, False
    for passo in range(1, 15001):
        probs, h, h_pre = forward(tr)
        yt = Y[tr]
        dlogits = probs.copy(); dlogits[np.arange(n_train), yt] -= 1; dlogits /= n_train

        g = {}
        g["W2"] = h.T @ dlogits
        g["b2"] = dlogits.sum(0)
        dh = dlogits @ params["W2"].T; dh[h <= 0] = 0
        g["W1"] = h_pre.T @ dh
        g["b1"] = dh.sum(0)
        dpre = dh @ params["W1"].T
        g["Ea"] = np.zeros_like(params["Ea"]); np.add.at(g["Ea"], A[tr], dpre)
        g["Eb"] = np.zeros_like(params["Eb"]); np.add.at(g["Eb"], B[tr], dpre)

        for k in params:
            m[k] = b1a * m[k] + (1 - b1a) * g[k]
            v[k] = b2a * v[k] + (1 - b2a) * g[k] ** 2
            mh = m[k] / (1 - b1a ** passo)
            vh = v[k] / (1 - b2a ** passo)
            params[k] -= lr * (mh / (np.sqrt(vh) + eps) + wd * params[k])

        if passo % 1000 == 0:
            a_tr, a_te = accuratezza(tr), accuratezza(te)
            picco_gap = max(picco_gap, a_tr - a_te)
            marca = ""
            if a_tr > 0.9 and a_te < 0.5:
                marca = "  <- memorizza (train alto, test a caso)"
            if a_te > 0.70 and not groccato:
                marca = "  <- GROKKING! ora GENERALIZZA"; groccato = True
            elif a_te > 0.70:
                marca = "  <- generalizza"
            print(f"  {passo:>6} | {a_tr:>8.1%} | {a_te:>7.1%} |{marca}")

    print(f"\n  Massimo divario train-test osservato: {picco_gap:.0%}")
    if groccato:
        print("  => Guarda la sequenza: il train è perfetto fin da subito, e il test")
        print("     SEMBRA schizzare su di colpo (GROKKING). Ma rileggi la colonna qui")
        print("     sopra riga per riga: sale un passo alla volta, senza nessun salto.")
        print("     Il 'di colpo' sta nella risoluzione con cui guardi, non nel modello")
        print("     — è la lezione della Parte A, applicata alla Parte B.")
        print("     Nella fase iniziale, dal solo errore di TRAIN non potresti")
        print("     distinguere 'ha memorizzato la tabella' da 'ha capito l'addizione'.")
    else:
        print("  => Train alto ma test basso: la rete ha MEMORIZZATO senza capire.")
        print("     (Il 'grokking' pieno a volte richiede piu' passi/tuning; il punto")
        print("      che conta e' gia' qui: train perfetto NON implica comprensione.)")
    print("\n  Morale: le stesse identiche operazioni possono 'memorizzare' o")
    print("  'capire', e dall'esterno i due casi possono sembrare identici.")


if __name__ == "__main__":
    parte_A()
    parte_B()
