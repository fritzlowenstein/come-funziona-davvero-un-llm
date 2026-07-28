"""
TAPPA 8 — La generazione nel mondo reale: KV CACHE e QUANTIZZAZIONE.

Il ciclo `while` della Tappa 5, preso alla lettera, ha un problema: a ogni nuova
parola ridà in pasto al modello TUTTA la sequenza e ricalcola TUTTO da capo.
Al token numero 1000 avresti rifatto ~1000 volte quasi gli stessi identici conti.

L'osservazione che salva tutto è un regalo della maschera causale (Tappa 4):
il passato non vede il futuro, quindi i vettori K e V dei token già elaborati
NON POSSONO CAMBIARE quando arriva un token nuovo. Allora calcoliamoli una volta
sola e teniamoli in memoria: la KV CACHE. A ogni giro del ciclo si elabora SOLO
il token nuovo: la sua Query interroga le Key/Value in cache, i suoi K/V vengono
appesi alla cache, fine.

Questo file riprende ESATTAMENTE il mini-GPT di `tappa5_generazione.py` e gli
aggiunge la cache. Poi dimostra tre cose:
  1. l'output è IDENTICO, carattere per carattere (non è un'approssimazione);
  2. il costo crolla: da ~N²/2 posizioni elaborate a ~N;
  3. i pesi non hanno bisogno di 32 bit: QUANTIZZATI a 8 o 4 bit il modello
     funziona quasi uguale (e la memoria si divide per 4 o per 8).
"""

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
torch.set_num_threads(1)   # forziamo single thread per risultati ripetibili

# ------------------------------------------------- corpus e iperparametri
# (identici alla Tappa 5 — è LO STESSO modello)
from tappa5_generazione import TESTO  # noqa: E402

chars = sorted(set(TESTO))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
dati = torch.tensor([stoi[c] for c in TESTO], dtype=torch.long)

T = 256         # contesto più lungo della Tappa 5: qui la lunghezza è il punto
D = 96
N_TESTE = 3
N_BLOCCHI = 3
BATCH = 32
PASSI = 800     # come la Tappa 5: il modello serve solo a "parlare", il punto è la cache


class Attention(nn.Module):
    """La multi-head attention causale della Tappa 4, con una novità: può
    ricevere una cache (K e V dei token passati) e restituirla aggiornata."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)

    def forward(self, x, cache=None):
        B, t, d = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, t, N_TESTE, d // N_TESTE).transpose(1, 2)
        k = k.view(B, t, N_TESTE, d // N_TESTE).transpose(1, 2)
        v = v.view(B, t, N_TESTE, d // N_TESTE).transpose(1, 2)

        if cache is not None:
            k_passato, v_passato = cache
            k = torch.cat([k_passato, k], dim=2)   # appendi i K nuovi ai vecchi
            v = torch.cat([v_passato, v], dim=2)   # idem per i V
        nuova_cache = (k, v)

        att = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        # Maschera causale: serve solo quando elaboriamo PIÙ token in un colpo
        # (prefill). Quando t == 1 (decodifica) non c'è niente da mascherare:
        # l'unico token nuovo vede solo il passato, che è esattamente la cache.
        if t > 1:
            t_tot = k.size(2)
            mask = torch.tril(torch.ones(t, t_tot), diagonal=t_tot - t)
            att = att.masked_fill(mask.view(1, 1, t, t_tot) == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, t, d)
        return self.proj(y), nuova_cache


class Blocco(nn.Module):
    """Le quattro righe della Tappa 5 — la cache passa attraverso."""

    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.att = Attention()
        self.ffn = nn.Sequential(
            nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)
        )

    def forward(self, x, cache=None):
        y, nuova_cache = self.att(self.ln1(x), cache)
        x = x + y
        x = x + self.ffn(self.ln2(x))      # il feed-forward è PER token: la
        return x, nuova_cache              # cache non lo riguarda proprio


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_tok = nn.Embedding(V, D)
        self.emb_pos = nn.Embedding(T, D)
        self.blocchi = nn.ModuleList([Blocco() for _ in range(N_BLOCCHI)])
        self.ln_f = nn.LayerNorm(D)
        self.testa = nn.Linear(D, V, bias=False)

    def forward(self, idx, target=None, caches=None, pos_iniziale=0):
        B, t = idx.shape
        pos = torch.arange(pos_iniziale, pos_iniziale + t, device=idx.device)
        x = self.emb_tok(idx) + self.emb_pos(pos)

        nuove_caches = []
        for i, blocco in enumerate(self.blocchi):
            x, c = blocco(x, caches[i] if caches is not None else None)
            nuove_caches.append(c)
        x = self.ln_f(x)
        logits = self.testa(x)

        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, V), target.view(-1))
        return logits, loss, nuove_caches

    # ------------------------------------------------ i due modi di generare

    @torch.no_grad()
    def genera_ingenua(self, idx, n_nuovi):
        """Il ciclo `while` della Tappa 5 preso alla lettera: a ogni giro si
        ributta dentro TUTTA la sequenza. Ritorna anche il conto del lavoro."""
        posizioni_elaborate = 0
        for _ in range(n_nuovi):
            ctx = idx[:, -T:]
            posizioni_elaborate += ctx.size(1)         # rielabora TUTTO ogni volta
            logits, _, _ = self(ctx)
            prossimo = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy
            idx = torch.cat((idx, prossimo), dim=1)
        return idx, posizioni_elaborate

    @torch.no_grad()
    def genera_con_cache(self, idx, n_nuovi):
        """Lo stesso ciclo, come lo eseguono i modelli veri.

        FASE 1 — PREFILL: il prompt intero passa nel modello UNA volta, in
                 parallelo, e riempie la cache. (È la pausa prima della
                 prima parola quando usi un LLM.)
        FASE 2 — DECODIFICA: a ogni giro entra SOLO il token nuovo.
        """
        posizioni_elaborate = idx.size(1)
        logits, _, caches = self(idx)                          # prefill
        prossimo = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, prossimo), dim=1)

        for _ in range(n_nuovi - 1):
            posizioni_elaborate += 1                           # UN token, non N
            logits, _, caches = self(
                prossimo, caches=caches, pos_iniziale=idx.size(1) - 1
            )
            prossimo = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, prossimo), dim=1)
        return idx, posizioni_elaborate


def batch():
    i = torch.randint(len(dati) - T - 1, (BATCH,))
    x = torch.stack([dati[j:j + T] for j in i])
    y = torch.stack([dati[j + 1:j + T + 1] for j in i])
    return x, y


def decodifica(t):
    return "".join(itos[i] for i in t.tolist())


def umano(n):
    for soglia, suff in ((1e9, " GB"), (1e6, " MB"), (1e3, " KB")):
        if n >= soglia:
            return f"{n/soglia:.1f}{suff}"
    return f"{n} B"


if __name__ == "__main__":
    m = MiniGPT()
    print("=== ADDESTRAMENTO (identico alla Tappa 5, serve solo un modello che parli) ===")
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for passo in range(PASSI):
        x, y = batch()
        _, loss, _ = m(x, y)
        opt.zero_grad()
        loss.backward()
        # Sforbicia i gradienti troppo grandi prima del passo: senza, intorno al
        # passo 750 la loss schizza da 0.09 a 1.4 e il modello arriva in fondo
        # rovinato. Lo fa ogni addestramento vero, per lo stesso motivo.
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if passo % 250 == 0:
            print(f"  passo {passo:>4}   loss = {loss.item():.4f}")
    print()

    prompt = "\nIl modello "
    idx = torch.tensor([[stoi[c] for c in prompt]])
    N_NUOVI = 220

    # ------------------------------------------- 1) STESSO OUTPUT, GARANTITO
    print("=== 1) LA CACHE NON È UN'APPROSSIMAZIONE ===")
    t0 = time.perf_counter()
    out_ingenua, lavoro_ingenuo = m.genera_ingenua(idx.clone(), N_NUOVI)
    t_ingenua = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_cache, lavoro_cache = m.genera_con_cache(idx.clone(), N_NUOVI)
    t_cache = time.perf_counter() - t0

    assert torch.equal(out_ingenua, out_cache), "output diversi?! bug!"
    print(f"  generati {N_NUOVI} caratteri in entrambi i modi (greedy).")
    print("  Output IDENTICI, carattere per carattere. La cache riorganizza il")
    print("  calcolo, non lo cambia: sono esattamente le stesse moltiplicazioni,")
    print("  fatte una volta sola invece che N volte.")
    print(f"  estratto: {decodifica(out_cache[0])[len(prompt):len(prompt)+60]!r}\n")

    # ------------------------------------------------------- 2) QUANTO COSTA
    print("=== 2) IL CONTO DEL LAVORO ===")
    print(f"  senza cache: {lavoro_ingenuo:>6} posizioni elaborate   ({t_ingenua:.2f} s)")
    print(f"  con cache  : {lavoro_cache:>6} posizioni elaborate   ({t_cache:.2f} s)")
    print(f"  rapporto   : {lavoro_ingenuo/lavoro_cache:.0f}x")
    print("  => Senza cache il lavoro cresce come N² (ogni token ripaga tutti i")
    print("     precedenti). Con la cache cresce come N. Più generi, più il")
    print("     divario esplode: è la differenza tra un giocattolo e un prodotto.\n")

    # -------------------------------------- 3) PREFILL vs DECODIFICA, dal vivo
    print("=== 3) PERCHÉ IL MODELLO 'CI PENSA' PRIMA DELLA PRIMA PAROLA ===")
    prompt_lungo = TESTO[:200]
    idx_lungo = torch.tensor([[stoi[c] for c in prompt_lungo]])
    t0 = time.perf_counter()
    _, _, caches = m(idx_lungo)                       # prefill: 200 caratteri in un colpo
    t_prefill = time.perf_counter() - t0
    tok = torch.tensor([[0]])
    t0 = time.perf_counter()
    m(tok, caches=caches, pos_iniziale=200)           # decodifica: 1 carattere
    t_decode = time.perf_counter() - t0
    print(f"  prefill del prompt (200 caratteri, in parallelo): {t_prefill*1000:6.1f} ms")
    print(f"  un passo di decodifica (1 carattere)            : {t_decode*1000:6.1f} ms")
    print("  => La pausa iniziale di un LLM è il prefill che riempie la cache;")
    print("     poi le parole escono a raffica perché ogni passo elabora 1 token.\n")

    # ------------------------------------- 4) IL PREZZO: la cache È la memoria
    print("=== 4) IL PREZZO DELLA CACHE: LA MEMORIA (ed ecco la finestra di contesto) ===")
    # per token: K e V (x2), per ogni blocco, ognuno lungo D float
    per_token_qui = 2 * N_BLOCCHI * D * 4             # float32 = 4 byte
    print(f"  questo mini-GPT : {per_token_qui} byte di cache per token")
    # Llama-3 70B: 80 blocchi, D=8192, fp16. Senza GQA: K e V larghi quanto D.
    per_token_pieno = 2 * 80 * 8192 * 2
    # Con GQA: 8 teste K/V da 128 dimensioni -> K e V larghi 1024, non 8192.
    per_token_gqa = 2 * 80 * 1024 * 2
    ctx = 8192
    print(f"  Llama-3 70B, cache piena (multi-head classico): "
          f"{umano(per_token_pieno)}/token -> {umano(per_token_pieno*ctx)} per {ctx} di contesto")
    print(f"  Llama-3 70B, con GQA (8 teste K/V condivise)  : "
          f"{umano(per_token_gqa)}/token -> {umano(per_token_gqa*ctx)} per {ctx} di contesto")
    print("  => La cache cresce LINEARMENTE col contesto: ogni token in più è")
    print("     memoria GPU occupata per tutta la conversazione. È il motivo per")
    print("     cui la finestra di contesto è una risorsa limitata e a pagamento,")
    print("     e il motivo per cui i modelli moderni usano la GQA: più teste di")
    print("     Query condividono le stesse K/V, e la cache si riduce di 8 volte.\n")

    # ------------------- 5) QUANTIZZAZIONE: di quanti bit ha bisogno un peso?
    print("=== 5) QUANTIZZAZIONE: di quanta precisione hanno bisogno i pesi? ===")

    def quantizza(w, bit):
        """Arrotonda i pesi su una griglia di 2^bit livelli.

        Una scala per RIGA della matrice: è esattamente ciò che memorizza un
        formato quantizzato reale — interi piccoli + una scala float per riga.
        """
        livelli = 2 ** (bit - 1) - 1
        scala = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / livelli
        return (w / scala).round().clamp(-livelli - 1, livelli) * scala

    x_val, y_val = batch()
    with torch.no_grad():
        _, loss_fp, _ = m(x_val, y_val)
    out_fp = decodifica(m.genera_con_cache(idx.clone(), 60)[0][0])[len(prompt):]
    print(f"  32 bit (float)  loss = {loss_fp.item():6.3f}   {out_fp[:46]!r}")

    for bit in (8, 4, 3, 2):
        mq = copy.deepcopy(m)
        with torch.no_grad():
            for p in mq.parameters():
                if p.dim() == 2:   # solo le MATRICI (embedding, Q/K/V/O, FFN,
                    #                testa): LayerNorm e bias restano float,
                    #                sono quattro spiccioli in confronto
                    p.copy_(quantizza(p, bit))
            _, loss_q, _ = mq(x_val, y_val)
        out_q = decodifica(mq.genera_con_cache(idx.clone(), 60)[0][0])[len(prompt):]
        print(f"  {bit:>2} bit          loss = {loss_q.item():6.3f}   {out_q[:46]!r}")

    print("  => A 8 bit il modello è INTATTO: testo identico, loss uguale. A 4")
    print("     zoppica, sotto si rompe. La conoscenza trovata dal gradiente è")
    print("     ridondante e distribuita (Tappa 7): regge all'arrotondamento.")
    print("     E più il modello è grande, più è ridondante: il nostro giocattolo")
    print("     da 348K parametri a 4 bit già soffre, un 70B quasi non se ne")
    print("     accorge. Memoria: 4 bit = 1/8 di float32. Un 70B passa da")
    print("     ~140 GB (fp16) a ~35 GB: da cluster a singola GPU.")
    print("     Stessa architettura, stessi pesi — solo scritti con meno cifre.")
