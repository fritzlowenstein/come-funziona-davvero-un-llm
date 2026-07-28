"""
TAPPA 1 — Il neurone, e perché "pesi = layer".

Obiettivo: dimostrare che un "layer" di una rete neurale NON è un concetto
diverso dai "pesi". Un layer È una matrice di pesi. Punto.

Lo dimostriamo calcolando la stessa cosa in due modi:
  (A) a mano, con i cicli, un neurone alla volta;
  (B) con una singola moltiplicazione matrice-vettore.
I due risultati sono identici.
"""

import numpy as np


# --- (1) UN SINGOLO NEURONE, in puro Python, senza magia -----------------
def neurone(inputs, weights, bias, attivazione):
    z = 0.0
    for i in range(len(inputs)):
        z += inputs[i] * weights[i]   # combina input e PESI
    z += bias
    return attivazione(z)


def relu(x):
    # la funzione di attivazione più comune: "passa il positivo, azzera il negativo"
    return max(0.0, x) if np.isscalar(x) else np.maximum(0.0, x)


# --- (2) UN LAYER A MANO = un neurone per ogni riga di pesi ---------------
def layer_a_mano(inputs, W, bias):
    # W ha forma (n_neuroni, n_input): ogni RIGA sono i pesi di un neurone.
    n_neuroni = len(W)
    out = []
    for j in range(n_neuroni):
        out.append(neurone(inputs, W[j], bias[j], relu))
    return np.array(out)


# --- (3) LO STESSO LAYER, come UNA moltiplicazione matrice-vettore --------
def layer_matriciale(inputs, W, bias):
    return relu(W @ inputs + bias)     # <-- questo è "il layer". Tutto qui.


if __name__ == "__main__":
    np.random.seed(0)

    inputs = np.array([0.5, -1.2, 3.0])      # 3 numeri in ingresso
    W = np.random.randn(4, 3)                # 4 neuroni, ognuno con 3 pesi -> matrice 4x3
    bias = np.random.randn(4)                # un bias per neurone

    a = layer_a_mano(inputs, W, bias)
    b = layer_matriciale(inputs, W, bias)

    print("Output calcolato coi cicli   :", np.round(a, 4))
    print("Output calcolato con W @ x   :", np.round(b, 4))
    print("Sono identici?               :", np.allclose(a, b))
    print()
    print(f"Numero di pesi in QUESTO layer: {W.size} (la matrice) + {bias.size} (bias)")
    print("=> 'il layer' e 'i pesi' sono LO STESSO OGGETTO.")

    # --- (4) UNA RETE = più layer impilati. Output di uno = input del prossimo.
    print("\n--- Una mini-rete a 3 layer (3 -> 4 -> 4 -> 2) ---")
    W1, b1 = np.random.randn(4, 3), np.random.randn(4)
    W2, b2 = np.random.randn(4, 4), np.random.randn(4)
    W3, b3 = np.random.randn(2, 4), np.random.randn(2)

    h1 = layer_matriciale(inputs, W1, b1)   # primo layer
    h2 = layer_matriciale(h1, W2, b2)       # il suo output entra nel secondo
    out = W3 @ h2 + b3                       # ultimo layer (senza attivazione)

    totale_parametri = W1.size + b1.size + W2.size + b2.size + W3.size + b3.size
    print("Output finale della rete     :", np.round(out, 4))
    print(f"Parametri totali della rete  : {totale_parametri}")
    print("In un LLM questo stesso conteggio arriva a miliardi: nient'altro che")
    print("numeri dentro matrici come queste.")
