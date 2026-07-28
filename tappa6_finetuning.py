"""
TAPPA 6 — Da "indovina la parola" a un assistente.

Il mini-GPT della Tappa 5 è un COMPLETATORE DI TESTO: continua il testo, non
risponde. Qui mostriamo, con codice, le tre fasi che lo trasformano in assistente:

  FASE 1  PRETRAINING          -> impara la lingua e i fatti (Tappa 5, in grande)
  FASE 2  INSTRUCTION TUNING   -> impara il FORMATO domanda->risposta
  FASE 3  RLHF (concettuale)   -> impara QUALE risposta è preferibile

Idea dell'esperimento: prendiamo LA STESSA architettura e la addestriamo su due
"internet" con FORMA diversa. Il modello imita la forma dei dati, qualunque sia.
=> È la prova che il "comportamento da assistente" NON è nell'architettura:
   è nei DATI su cui lo alleni. Cambi i dati, cambi il comportamento.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)   # idem, per risultati ripetibili: vedi listato precedente

# =========================================================================
# Due "internet" con la STESSA conoscenza ma FORMA diversa.
# =========================================================================
# "Internet grezzo": testo che scorre, come troverebbe un base model.
INTERNET_GREZZO = """
la capitale della francia e parigi e si trova sulla senna. la capitale della
italia e roma. la capitale della spagna e madrid. il sole e una stella. la
luna orbita intorno alla terra. l acqua bolle a cento gradi. parigi e una
citta molto grande. roma e una citta antica. madrid e la capitale spagnola.
""".strip()

# "Dati da assistente": ogni esempio ha la forma DOMANDA -> RISPOSTA, con dei
# marcatori speciali che delimitano i turni (qui: | per Utente, > per Assistente).
INSTRUCT = """
|qual e la capitale della francia?>parigi.<
|qual e la capitale della italia?>roma.<
|qual e la capitale della spagna?>madrid.<
|dove si trova parigi?>sulla senna.<
|cosa e il sole?>una stella.<
|a quanti gradi bolle l acqua?>cento gradi.<
|cosa orbita intorno alla terra?>la luna.<
|qual e la capitale spagnola?>madrid.<
""".strip()


def prepara(testo):
    chars = sorted(set(testo))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    dati = torch.tensor([stoi[c] for c in testo], dtype=torch.long)
    return chars, stoi, itos, dati


# --- Modello: identico alla Tappa 5, in versione compatta ----------------
T, D, N_TESTE, N_BLOCCHI = 48, 64, 4, 3


class Blocco(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.register_buffer("mask", torch.tril(torch.ones(T, T)))

    def attn(self, x):
        B, t, d = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, t, N_TESTE, d // N_TESTE).transpose(1, 2)
        k = k.view(B, t, N_TESTE, d // N_TESTE).transpose(1, 2)
        v = v.view(B, t, N_TESTE, d // N_TESTE).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        a = a.masked_fill(self.mask[:t, :t] == 0, float("-inf"))
        y = (F.softmax(a, dim=-1) @ v).transpose(1, 2).contiguous().view(B, t, d)
        return self.proj(y)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.blocchi = nn.Sequential(*[Blocco() for _ in range(N_BLOCCHI)])
        self.ln_f = nn.LayerNorm(D)
        self.testa = nn.Linear(D, V, bias=False)

    def forward(self, idx, target=None):
        B, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t))
        x = self.ln_f(self.blocchi(x))
        logits = self.testa(x)
        loss = None if target is None else F.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss

    @torch.no_grad()
    def genera(self, idx, n, stoi_stop=None, temp=0.4):
        for _ in range(n):
            logits, _ = self(idx[:, -T:])
            probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat((idx, nxt), dim=1)
            if stoi_stop is not None and nxt.item() == stoi_stop:
                break
        return idx


def allena(dati, V, passi=1500):
    m = GPT(V)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(passi):
        i = torch.randint(len(dati) - T - 1, (32,))
        x = torch.stack([dati[j:j + T] for j in i])
        y = torch.stack([dati[j + 1:j + T + 1] for j in i])
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


if __name__ == "__main__":
    DOMANDA = "qual e la capitale della francia?"

    # ===== FASE 1: BASE MODEL (addestrato su internet grezzo) =============
    print("=" * 68)
    print("FASE 1 — BASE MODEL: addestrato a completare 'internet grezzo'")
    print("=" * 68)
    chars, stoi, itos, dati = prepara(INTERNET_GREZZO)
    base = allena(dati, len(chars))

    # Un base model si sonda con un PREFISSO DI TESTO, non con una domanda:
    # non conosce nemmeno il carattere "?", perché non c'è nel testo grezzo.
    prefisso = "la capitale della francia e"
    idx = torch.tensor([[stoi[c] for c in prefisso]])
    out = base.genera(idx, 55)[0].tolist()
    print(f"  Prefisso: {prefisso!r}")
    print(f"  Uscita  : {''.join(itos[i] for i in out)!r}")
    print("  => Completa correttamente 'parigi': la CONOSCENZA c'è.")
    print("     Ma prova a fargli una DOMANDA vera e non sa rispondere: continua")
    print("     e basta il testo. Non conosce nemmeno il carattere '?'.\n")

    # ===== FASE 2: INSTRUCT MODEL (stesso modello, dati con FORMA Q->A) ===
    print("=" * 68)
    print("FASE 2 — INSTRUCTION TUNING: stessa architettura, dati Q->A")
    print("=" * 68)
    chars2, stoi2, itos2, dati2 = prepara(INSTRUCT)
    instr = allena(dati2, len(chars2))

    prompt = f"|{DOMANDA}>"                      # nota i marcatori di turno
    idx = torch.tensor([[stoi2[c] for c in prompt]])
    stop = stoi2["<"]
    out = instr.genera(idx, 40, stoi_stop=stop)[0].tolist()
    completa = "".join(itos2[i] for i in out)
    risposta = completa.split(">")[-1].rstrip("<")
    print(f"  Prompt : {prompt!r}   (| = turno utente, > = turno assistente)")
    print(f"  Uscita : {completa!r}")
    print(f"  RISPOSTA ESTRATTA: {risposta!r}")
    print("  => Stessa conoscenza, ma ORA risponde e si FERMA. Non ha imparato")
    print("     fatti nuovi: ha imparato il FORMATO 'domanda -> risposta -> stop'.\n")

    # ===== FASE 3: RLHF (concettuale, non addestrato qui) ================
    print("=" * 68)
    print("FASE 3 — RLHF: scegliere QUALE risposta è preferibile")
    print("=" * 68)
    print("""  L'instruct model sa rispondere, ma non sa se è MEGLIO rispondere:
    A) "parigi."                              (secca)
    B) "La capitale della Francia è Parigi."  (completa, educata)
    C) "non lo so"                            (inutile)

  Le fasi 1-2 non hanno modo di preferire B. RLHF aggiunge questo:
    1. il modello genera più risposte candidate;
    2. degli umani le ORDINANO per preferenza (B > A > C);
    3. si allena un 'reward model' a predire quel giudizio umano;
    4. il modello viene ri-ottimizzato per MASSIMIZZARE quel reward
       (invece di minimizzare la loss di predizione).

  => Qui il segnale non è più 'quale parola viene dopo nel testo' ma
     'quanto è BUONA questa risposta secondo gli umani'. È il passo che
     rende un assistente utile, onesto e prudente: l'ALLINEAMENTO.""")

    print("\n" + "=" * 68)
    print("MORALE")
    print("=" * 68)
    print("""  Stessa identica architettura (Tappa 5) in tutte e tre le fasi.
  Ciò che cambia è SOLO su quali dati e con quale segnale la alleni:
    - testo grezzo      -> sa il mondo, ma blatera
    - coppie Q->A       -> risponde e si ferma
    - preferenze umane  -> risponde BENE
  L'assistente non è nell'algoritmo. È nei dati e nel segnale di addestramento.""")
