"""
TAPPA 9 — Chiudere il cerchio: è SOLO scala.

Non c'è teoria nuova qui. C'è una dimostrazione aritmetica di un'unica tesi:
il modello che hai costruito nella Tappa 5 e i più grandi LLM del mondo hanno
LA STESSA IDENTICA ARCHITETTURA. Cambiano solo tre numeri: quanti layer,
quanto sono larghi, su quanti dati sono addestrati.

Usiamo ESATTAMENTE la formula del conteggio parametri di `tappa5_transformer.py`
per contare i parametri dei modelli veri. Nessun addestramento: solo moltiplicazioni.
"""


def conta_parametri(V, D, T, n_layer, tied=True):
    """Parametri di un Transformer decoder-only. È la struttura della Tappa 5.

    V = vocabolario, D = larghezza (residual stream), T = contesto, n_layer = blocchi.
    """
    emb_token = V * D                     # tabella embedding (Tappa 3)
    emb_pos = T * D                       # embedding posizionali (Tappa 4)

    per_blocco = (
        4 * D * D                         # attention: Wq, Wk, Wv, Wo  (Tappa 4)
        + 8 * D * D + 5 * D               # feed-forward: D->4D->D + bias (Tappa 5)
        + 4 * D                           # due LayerNorm (gamma, beta)   (Tappa 5)
    )
    blocchi = n_layer * per_blocco

    ln_finale = 2 * D
    # unembedding (D->V): spesso CONDIVISO con la tabella embedding ("weight tying")
    unembed = 0 if tied else V * D

    totale = emb_token + emb_pos + blocchi + ln_finale + unembed
    return totale, per_blocco


# Configurazioni reali (dai paper). (nome, V, D, T, n_layer, tied)
# tied = l'unembedding riusa la tabella degli embedding (GPT-2 e GPT-3 si, il
# mini-GPT della Tappa 5 e Llama-3 no: hanno pesi propri in cima).
MODELLI = [
    ("Il TUO mini-GPT (Tappa 5)",     38,    96,   64,   3, False),
    ("GPT-2 small (2019)",         50257,   768, 1024,  12, True),
    ("GPT-2 XL (2019)",           50257,  1600, 1024,  48, True),
    ("GPT-3 175B (2020)",         50257, 12288, 2048,  96, True),
    ("Llama-3 70B (2024)",        128000,  8192, 8192,  80, False),
]


def umano(n):
    for soglia, suff in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= soglia:
            return f"{n/soglia:.1f}{suff}"
    return str(n)


if __name__ == "__main__":
    print("=" * 78)
    print("STESSA ARCHITETTURA, SOLO SCALA DIVERSA")
    print("=" * 78)
    print(f"{'modello':<28}{'layer':>6}{'largh. D':>9}{'contesto':>9}"
          f"{'parametri':>12}")
    print("-" * 78)
    for nome, V, D, T, L, tied in MODELLI:
        tot, _ = conta_parametri(V, D, T, L, tied)
        print(f"{nome:<28}{L:>6}{D:>9}{T:>9}{umano(tot):>12}")

    print("\n=> Ogni riga usa LA STESSA funzione conta_parametri(). Cambiano solo")
    print("   tre numeri: layer, larghezza, contesto. L'architettura è identica:")
    print("   embedding + [attention + feed-forward] x N + unembedding.")
    print("   (Ignora GQA e SwiGLU: sul Llama-3 sbaglia del 6% per difetto.)\n")

    # --- Dove stanno i parametri, in GPT-3? ---
    print("=" * 78)
    print("DOVE VIVONO I 175 MILIARDI (GPT-3)")
    print("=" * 78)
    V, D, T, L = 50257, 12288, 2048, 96
    tot, per_blocco = conta_parametri(V, D, T, L)
    attn = 4 * D * D
    ffn = 8 * D * D + 5 * D
    print(f"  per ogni blocco:  attention {umano(attn):>7}  ({attn/per_blocco:.0%})"
          f"   feed-forward {umano(ffn):>7}  ({ffn/per_blocco:.0%})")
    print(f"  x {L} blocchi:     {umano(per_blocco*L)} di parametri nei blocchi")
    print(f"  embedding:        {umano((V+T)*D)}")
    print("  => Come nella Tappa 5: ~2/3 della conoscenza è nel feed-forward.\n")

    # --- Il costo di addestramento, reso tangibile ---
    print("=" * 78)
    print("QUANTO CALCOLO SERVE PER ADDESTRARLO (la regola 6N)")
    print("=" * 78)
    # Regola empirica: addestrare costa ~6 * (n_parametri) * (n_token) FLOP.
    n_token = 300e9                       # GPT-3 fu addestrato su ~300 miliardi di token
    flop = 6 * tot * n_token
    print(f"  parametri N        : {umano(tot)}")
    print(f"  token di training  : {umano(n_token)}")
    print(f"  calcolo totale ~6·N·D_token = {flop:.2e} FLOP")
    # una GPU potente fa ~1e15 FLOP/s (utilizzati). Tempo su UNA GPU:
    sec = flop / 1e15
    print(f"  su UNA GPU (~1e15 FLOP/s) = {sec/86400/365:.0f} anni")
    print(f"  => per questo si usano MIGLIAIA di GPU in parallelo per settimane.")
    print("     Il TUO ciclo 'w -= lr * gradiente' (Tappa 2) è lo stesso.")
    print("     Cambia solo quante volte, e su quanto, lo esegui.\n")

    # --- Parametri E dati: la lezione di Chinchilla ---
    print("=" * 78)
    print("NON SOLO PARAMETRI: I DATI (la lezione di Chinchilla, 2022)")
    print("=" * 78)
    # Le scaling laws raffinate dicono: a parità di calcolo, parametri e token
    # vanno scalati INSIEME (~20 token per parametro). GPT-3 era sbilanciato.
    for nome, n_par, n_tok in [("GPT-3 175B", 175e9, 300e9),
                               ("Llama-3 70B", 70e9, 15e12)]:
        print(f"  {nome:<14} {umano(n_par):>7} parametri, {umano(n_tok):>6} token"
              f"  -> {n_tok/n_par:>5.1f} token/parametro"
              f"   ({6*n_par*n_tok:.1e} FLOP)")
    print("  => GPT-3 era SOTTO-addestrato: troppi parametri per i suoi dati.")
    print("     Llama-3 70B è 2.5 volte più piccolo ma molto più capace: ha visto")
    print("     50 volte più testo. La scala non è solo il modello: sono i dati.\n")

    print("=" * 78)
    print("IL PUNTO DI TUTTO IL CORSO")
    print("=" * 78)
    print("""  Hai costruito, riga per riga, un oggetto che differisce dai più grandi
  LLM del mondo SOLO per tre numeri e per la quantità di dati e calcolo.
  Non c'è un ingrediente segreto che non hai visto. Non c'è un modulo del
  'ragionamento' nascosto da qualche parte. C'è quello che hai scritto tu,
  moltiplicato per un miliardo.""")
