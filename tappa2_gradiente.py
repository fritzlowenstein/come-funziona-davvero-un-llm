"""
TAPPA 2 — L'apprendimento è la discesa dentro una funzione.

Alla fine della Tappa 1 avevamo una rete con pesi CASUALI: inutile.
Qui rispondiamo a: "da dove vengono i numeri giusti dentro le matrici W?"

Risposta: non li scriviamo, li CERCHIAMO. Con tre ingredienti:
  1. una LOSS  -> un numero che dice quanto sbagliamo
  2. il GRADIENTE -> in che direzione muovere ogni peso per far scendere la loss
  3. la DISCESA DEL GRADIENTE -> ripeti piccoli passi in discesa, tante volte

Il problema-giocattolo è lo XOR: NON è risolvibile senza un layer nascosto e una
non-linearità. È la prova pratica di ciò che dicevamo alla fine della Tappa 1.
"""

import numpy as np

# --- IL PROBLEMA: XOR ------------------------------------------------------
# input -> output atteso.  (0,0)->0   (0,1)->1   (1,0)->1   (1,1)->0
X = np.array([[0., 0.],
              [0., 1.],
              [1., 0.],
              [1., 1.]])
Y = np.array([[0.], [1.], [1.], [0.]])


def sigmoid(z):
    # attivazione non-lineare che schiaccia qualunque numero tra 0 e 1
    return 1.0 / (1.0 + np.exp(-z))


# --- LA RETE: 2 -> 4 -> 1.  Esattamente la struttura della Tappa 1. --------
def init_pesi(seed=1):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(size=(2, 4)), "b1": np.zeros(4),   # layer nascosto
        "W2": rng.normal(size=(4, 1)), "b2": np.zeros(1),   # layer di uscita
    }


def forward(p, X):
    """Il 'passaggio in avanti': input -> previsione. Come in Tappa 1."""
    z1 = X @ p["W1"] + p["b1"]
    h = sigmoid(z1)                 # <- la non-linearità, senza cui XOR è impossibile
    z2 = h @ p["W2"] + p["b2"]
    y_pred = sigmoid(z2)
    return y_pred, h                 # 'h' ci servirà nella backprop


# --- 1. LA LOSS: quanto sbagliamo, in un solo numero ----------------------
def loss(p, X, Y):
    y_pred, _ = forward(p, X)
    return np.mean((y_pred - Y) ** 2)     # errore quadratico medio


# --- 2a. IL GRADIENTE, metodo STUPIDO (numerico) --------------------------
# "spingo il peso di un pelo e guardo se la loss sale o scende".
# Funziona sempre, non richiede matematica... ma è LENTISSIMO.
def gradiente_numerico(p, X, Y, eps=1e-5):
    grads = {}
    for nome in p:
        g = np.zeros_like(p[nome])
        for idx in np.ndindex(p[nome].shape):    # UN peso alla volta!
            originale = p[nome][idx]
            p[nome][idx] = originale + eps
            loss_su = loss(p, X, Y)
            p[nome][idx] = originale - eps
            loss_giu = loss(p, X, Y)
            p[nome][idx] = originale             # rimetti a posto
            g[idx] = (loss_su - loss_giu) / (2 * eps)   # la pendenza
        grads[nome] = g
    return grads


# --- 2b. IL GRADIENTE, metodo INTELLIGENTE (backpropagation) --------------
# Stesso identico risultato, ma TUTTI i gradienti in UN solo passaggio all'indietro.
# Parte dall'errore in fondo e lo "spalma" indietro layer per layer.
def gradiente_backprop(p, X, Y):
    n = len(X)
    y_pred, h = forward(p, X)

    # si parte dall'errore in uscita e si risale la rete a ritroso:
    dz2 = (2.0 / n) * (y_pred - Y) * y_pred * (1 - y_pred)   # errore al layer 2
    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0)

    dh = dz2 @ p["W2"].T                                      # ...propagato indietro...
    dz1 = dh * h * (1 - h)                                    # errore al layer 1
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


if __name__ == "__main__":
    # === PROVA 1: la backprop calcola la STESSA cosa del metodo stupido? ===
    print("=== I due gradienti coincidono? ===")
    p = init_pesi()
    g_num = gradiente_numerico(p, X, Y)
    g_bp = gradiente_backprop(p, X, Y)
    for nome in p:
        uguali = np.allclose(g_num[nome], g_bp[nome], atol=1e-6)
        print(f"  {nome}: numerico == backprop ? {uguali}")
    print("=> La backprop NON è magia: è solo il modo VELOCE di calcolare")
    print("   la stessa identica pendenza.")
    print(f"   Numerico: {sum(v.size for v in p.values())} passaggi di rete. Backprop: 1.\n")

    # === PROVA 2: addestriamo davvero. Discesa del gradiente. ==============
    print("=== Addestramento (discesa del gradiente) ===")
    p = init_pesi()
    learning_rate = 3.0

    for passo in range(20001):
        grads = gradiente_backprop(p, X, Y)
        for nome in p:
            # IL CUORE DI TUTTO IL DEEP LEARNING, in una riga:
            p[nome] -= learning_rate * grads[nome]

        if passo % 4000 == 0:
            print(f"  passo {passo:>5}   loss = {loss(p, X, Y):.6f}")

    # === RISULTATO =========================================================
    print("\n=== La rete ha imparato lo XOR? ===")
    y_pred, _ = forward(p, X)
    for i in range(4):
        print(f"  input {X[i]} -> previsto {y_pred[i][0]:.3f}   (atteso {Y[i][0]:.0f})")

    print("\nNessuno ha PROGRAMMATO lo XOR. Abbiamo solo definito 'quanto sbagli'")
    print("e fatto rotolare i pesi in discesa. La soluzione è stata TROVATA.")
