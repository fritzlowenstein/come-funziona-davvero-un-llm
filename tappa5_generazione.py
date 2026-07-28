"""
TAPPA 5 (parte 2/2) — Un mini-GPT ADDESTRATO: i logit e la generazione.

L'architettura è ESATTAMENTE quella di `tappa5_transformer.py`. L'unica differenza:
qui usiamo PyTorch, che calcola i gradienti AUTOMATICAMENTE invece di costringerci
a scriverli a mano come nelle Tappe 2 e 4; NON aggiunge nessun concetto nuovo,  
fa la backprop al posto nostro.
Lavoriamo a livello di CARATTERE (invece di token BPE) per tenere tutto minuscolo.
Il corpus è ridicolo (pochi KB), quindi il modello imparerà a malapena a fare
parole italiane plausibili. Va benissimo: il punto è vedere il MECCANISMO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
torch.set_num_threads(1)   # altrimenti su piu' core le somme cambiano ordine

# ---------------------------------------------------------------- il corpus
TESTO = """
La rete neurale non contiene regole scritte da un programmatore.
La rete neurale contiene soltanto numeri, e quei numeri sono i pesi.
I pesi vengono trovati dalla discesa del gradiente, non vengono scritti a mano.
Il modello impara a prevedere la parola successiva, e nient'altro.
Per prevedere bene la parola successiva il modello deve capire il contesto.
Il contesto viene raccolto dal meccanismo di attenzione.
L'attenzione confronta ogni parola con tutte le altre parole della frase.
Ogni parola diventa un vettore, e la posizione del vettore rappresenta il significato.
Lo spazio dei vettori si chiama spazio latente, e nessuno lo ha progettato.
Il significato emerge dalla statistica del testo, non da una definizione.
Un blocco del transformer somma il suo contributo al flusso residuo.
Il flusso residuo attraversa tutti i blocchi senza cambiare dimensione.
La conoscenza del modello abita nelle matrici del feed forward.
L'attenzione muove informazione tra i token, il feed forward la elabora.
Alla fine il modello produce i logit, un punteggio per ogni parola possibile.
La funzione softmax trasforma i logit in probabilita che sommano a uno.
Da quelle probabilita viene pescata una parola, e il ciclo ricomincia.
Il modello genera una parola alla volta, e non torna mai indietro.
La temperatura decide quanto il modello sia prudente oppure creativo.
Una temperatura bassa rende il testo ripetitivo ma sicuro.
Una temperatura alta rende il testo creativo ma incoerente.
L'addestramento minimizza una funzione di costo chiamata loss.
La loss misura quanto il modello sbaglia nel prevedere il testo.
Il gradiente indica la direzione in cui i pesi devono muoversi.
La backpropagation calcola tutti i gradienti in un solo passaggio.
Nessuno ha programmato la conoscenza, la conoscenza e stata trovata.
Il modello non ragiona come un uomo, il modello prevede il testo.
Eppure dalla previsione del testo emerge qualcosa che somiglia al ragionamento.
Non capiamo perche funzioni cosi bene, e questo e il vero mistero.
Le regole sono semplici, il comportamento che ne emerge non lo e affatto.
"""
TESTO = TESTO.strip() * 4          # ripetiamo: corpus giocattolo

chars = sorted(set(TESTO))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
dati = torch.tensor([stoi[c] for c in TESTO], dtype=torch.long)

# ------------------------------------------------------------ iperparametri
T = 64          # quanti caratteri di contesto
D = 96          # dimensione del residual stream
N_TESTE = 3
N_BLOCCHI = 3
BATCH = 32
PASSI = 800      # di proposito NON alleniamo fino alla memorizzazione totale:
                 # ci serve un modello ancora "incerto", per vedere le probabilità


class Attention(nn.Module):
    """Multi-head attention causale — la Tappa 4, identica."""
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)     # Wq, Wk, Wv in un colpo
        self.proj = nn.Linear(D, D, bias=False)        # Wo
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def forward(self, x):
        B, t, d = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        # separa le teste
        q = q.view(B, t, N_TESTE, d // N_TESTE).transpose(1, 2)
        k = k.view(B, t, N_TESTE, d // N_TESTE).transpose(1, 2)
        v = v.view(B, t, N_TESTE, d // N_TESTE).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)     # Q @ K.T / sqrt(d)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))  # no futuro
        att = F.softmax(att, dim=-1)                              # -> pesi
        y = att @ v                                               # media pesata
        y = y.transpose(1, 2).contiguous().view(B, t, d)
        return self.proj(y)


class Blocco(nn.Module):
    """Le quattro righe della Tappa 5."""
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.att = Attention()
        self.ffn = nn.Sequential(          # il layer della Tappa 1: D -> 4D -> D
            nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)
        )

    def forward(self, x):
        x = x + self.att(self.ln1(x))      # residuo: comunicazione TRA token
        x = x + self.ffn(self.ln2(x))      # residuo: elaborazione DENTRO il token
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_tok = nn.Embedding(V, D)          # Tappa 3
        self.emb_pos = nn.Embedding(T, D)          # Tappa 4: la posizione
        self.blocchi = nn.Sequential(*[Blocco() for _ in range(N_BLOCCHI)])
        self.ln_f = nn.LayerNorm(D)
        self.testa = nn.Linear(D, V, bias=False)   # -> LOGIT

    def forward(self, idx, target=None):
        B, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.emb_tok(idx) + self.emb_pos(pos)  # token + posizione
        x = self.blocchi(x)                        # il residual stream
        x = self.ln_f(x)
        logits = self.testa(x)                     # (B, t, V) <- UN PUNTEGGIO PER CARATTERE

        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, V), target.view(-1))
        return logits, loss

    @torch.no_grad()
    def genera(self, idx, n_nuovi, temperatura=1.0, top_k=None):
        """IL CICLO WHILE CHE FA PARLARE UN LLM."""
        for _ in range(n_nuovi):
            ctx = idx[:, -T:]                          # la 'finestra di contesto'
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperatura    # solo l'ULTIMA posizione
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)          # logit -> probabilità
            prossimo = torch.multinomial(probs, 1)     # PESCA una parola
            idx = torch.cat((idx, prossimo), dim=1)    # riaccodala e ricomincia
        return idx


def batch():
    i = torch.randint(len(dati) - T - 1, (BATCH,))
    x = torch.stack([dati[j:j + T] for j in i])
    y = torch.stack([dati[j + 1:j + T + 1] for j in i])   # il target è il testo
    return x, y                                           # SPOSTATO DI UNO!


def decodifica(t):
    return "".join(itos[i] for i in t.tolist())


if __name__ == "__main__":
    m = MiniGPT()
    n_par = sum(p.numel() for p in m.parameters())
    print(f"=== Mini-GPT: {n_par:,} parametri, {N_BLOCCHI} blocchi, vocabolario di {V} caratteri")
    print("(GPT-3 ne ha 175 miliardi. Stessa architettura, identica.)\n")

    print("=== IL COMPITO DI ADDESTRAMENTO ===")
    x, y = batch()
    print(f"  input : ...{decodifica(x[0][-40:])!r}")
    print(f"  target: ...{decodifica(y[0][-40:])!r}")
    print("  => Il target è l'input SPOSTATO DI UN CARATTERE.")
    print("     Tutto qui: 'indovina il prossimo'. Nessuna etichetta, nessun umano.\n")

    print("=== PRIMA dell'addestramento (pesi casuali) ===")
    start = torch.zeros((1, 1), dtype=torch.long)
    print(f"  {decodifica(m.genera(start, 80)[0])!r}\n")

    print("=== ADDESTRAMENTO (discesa del gradiente — Tappa 2) ===")
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for passo in range(PASSI):
        x, y = batch()
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()          # <- la backprop della Tappa 2, fatta da PyTorch
        opt.step()               # <- w -= lr * gradiente
        if passo % 500 == 0:
            print(f"  passo {passo:>4}   loss = {loss.item():.4f}")
    print(f"  passo {PASSI:>4}   loss = {loss.item():.4f}")
    print("  (Ci fermiamo qui di proposito: se addestrassimo di più su un corpus")
    print("   così piccolo il modello lo MEMORIZZEREBBE e sarebbe sicuro al 100%")
    print("   di ogni carattere. Si chiama overfitting.)\n")

    print("=== DOPO l'addestramento ===")
    testo = decodifica(m.genera(start, 300, temperatura=0.8)[0])
    print(f"  {testo!r}\n")

    # ---------------------------------------------------------------- I LOGIT
    print("=== GUARDIAMO I LOGIT DA VICINO ===")
    # Scegliamo un prompt GENUINAMENTE AMBIGUO: nel corpus esistono
    # "Il modello impara", "Il modello genera", "Il modello non ragiona",
    # "Il modello prevede"... quindi il modello DEVE essere incerto.
    prompt = "\nIl modello "
    idx = torch.tensor([[stoi[c] for c in prompt]])
    logits, _ = m(idx)
    ultimi = logits[0, -1]                    # i logit per il PROSSIMO carattere

    print(f"  prompt: {prompt!r}")
    print(f"  -> il modello produce {len(ultimi)} logit (uno per carattere del vocabolario)\n")

    probs = F.softmax(ultimi, dim=-1)
    top = torch.topk(probs, 5)
    print("  I 5 candidati più probabili:")
    for p, i in zip(top.values, top.indices):
        print(f"    {itos[i.item()]!r:>6}  logit={ultimi[i]:+6.2f}   probabilità={p:.1%}")
    print("  => I logit sono punteggi GREZZI (anche negativi). La softmax li rende")
    print("     probabilità. Poi se ne pesca uno. Questo è TUTTO ciò che fa un LLM.\n")

    # ---------------------------------------------------------- LA TEMPERATURA
    print("=== LA TEMPERATURA: dividere i logit prima della softmax ===")
    for temp in (0.1, 0.5, 1.0, 2.0):
        p = F.softmax(ultimi / temp, dim=-1)
        top3 = torch.topk(p, 3)
        s = "  ".join(f"{itos[i.item()]!r}:{v:.0%}" for v, i in zip(top3.values, top3.indices))
        print(f"  T={temp:<4} -> {s}")
    print("  => T bassa: le differenze si AMPLIFICANO -> quasi deterministico, ripetitivo.")
    print("     T alta : le differenze si APPIATTISCONO -> creativo, incoerente.")
    print("     Non c'è nessuna 'creatività' nel modello: c'è una divisione.\n")

    print("=== GENERAZIONE A TEMPERATURE DIVERSE ===")
    for temp in (0.3, 1.0, 1.6):
        out = decodifica(m.genera(idx.clone(), 90, temperatura=temp)[0])[len(prompt):]
        print(f"  T={temp}: {out!r}")
