# Come funziona davvero un LLM — i sorgenti

Il codice del libro *Come funziona davvero un LLM. La guida per programmatori*.

Dieci programmi, circa 1.700 righe in tutto. Ognuno corrisponde a una tappa del libro e fa
davvero la cosa di cui quella tappa parla: non simulazioni, non pseudocodice. Il neurone somma
numeri veri, il gradiente scende davvero, il Transformer si addestra e genera testo.

## I file

| File | Tappa | Cosa dimostra |
|---|---|---|
| `tappa1_neurone.py` | 1 | che «layer» e «pesi» sono la stessa cosa: lo stesso calcolo fatto a mano con i cicli e come una moltiplicazione matrice-vettore |
| `tappa2_gradiente.py` | 2 | discesa del gradiente e backpropagation, con il gradiente verificato numericamente riga per riga |
| `tappa3_embedding.py` | 3 | tokenizzazione BPE costruita da zero, e embedding che imparano a disporsi nello spazio |
| `tappa4_attention.py` | 4 | l'attention: Q, K, V, la maschera causale, il multi-testa |
| `tappa5_transformer.py` | 5 | un Transformer completo in numpy puro, senza framework: forward pass e conteggio dei parametri |
| `tappa5_generazione.py` | 5 | un mini-GPT addestrato con PyTorch: logit, temperatura, top-p, generazione |
| `tappa6_finetuning.py` | 6 | dal pretraining al fine-tuning: lo stesso modello che cambia comportamento cambiando i dati |
| `tappa7_emergenza.py` | 7 | il grokking: la generalizzazione che arriva di colpo, molto dopo la memorizzazione |
| `tappa8_kvcache.py` | 8 | KV cache e quantizzazione, cronometrate: quanto costa davvero generare un token |
| `tappa9_scala.py` | 9 | il conteggio dei parametri di GPT-2, GPT-3 e Llama-3 con la stessa formula della Tappa 5 |

La Tappa 10 non ha codice: è una tappa di lettura.

## Come si eseguono

```
python3 tappa1_neurone.py
```

Uno alla volta, nell'ordine che preferisci. Stampano tutto su terminale, non aprono finestre,
non scaricano niente da internet, non serve una GPU.

**Tienili nella stessa cartella**: `tappa8_kvcache.py` importa `tappa5_generazione.py` per
riusarne il modello.

## Cosa serve installato

Python 3 (sono scritti e provati su 3.10, non usano sintassi recente) e:

```
pip install numpy torch
```

Nel dettaglio, se vuoi installare il minimo:

- **solo numpy** — tappe 1, 2, 3, 4, 5 (`transformer`), 7
- **PyTorch** — tappe 5 (`generazione`), 6, 8
- **niente** — la tappa 9 è aritmetica in Python puro

PyTorch va bene nella versione CPU, che è molto più leggera da scaricare.

## Perché i tuoi numeri sono uguali a quelli stampati nel libro

I quattro programmi che addestrano davvero (5, 6, 7, 8) partono da un seme fisso, e quelli in
PyTorch aggiungono anche questo:

```python
torch.set_num_threads(1)
```

Non è un dettaglio di prestazioni. I numeri in virgola mobile non sono associativi: `(a+b)+c` e
`a+(b+c)` possono dare risultati diversi nell'ultima cifra. Su più core PyTorch spezza le somme
fra i thread, quindi l'ordine cambia col numero di core, e quella differenza minuscola, moltiplicata
per centinaia di passi di addestramento, diventa visibile. Con un thread solo l'ordine è sempre lo
stesso, e la loss che leggi sul tuo terminale è quella che leggi nel libro.

Toglilo e i programmi funzionano lo stesso, solo più in fretta e con numeri leggermente diversi
dai nostri.

## Se vuoi giocarci

Il modo migliore di usarli è rompere qualcosa e vedere cosa succede. Qualche punto d'attacco:

- `PASSI` nelle tappe 5 e 8: cosa cambia se addestri di più, o molto di meno?
- la temperatura e il `top_p` nella generazione della tappa 5: dove finisce il testo quando alzi
  la temperatura oltre 1?
- il numero di teste e di blocchi nel mini-GPT: quanti parametri costano, e servono davvero?
- i bit della quantizzazione nella tappa 8: a che punto il modello smette di parlare?

## Un avvertimento onesto

Sono programmi didattici, non una libreria. Sono scritti per essere **letti**, quindi preferiscono
sempre la chiarezza all'efficienza: cicli espliciti dove una vettorizzazione sarebbe più veloce,
nomi lunghi, nessuna astrazione che nasconda il calcolo. Non usarli in produzione. Usali per
capire cosa c'è dentro quelli veri, e poi leggi il codice di chi la produzione la fa.

## Licenza

MIT. Puoi usarli, modificarli e includerli nei tuoi progetti, anche commerciali, senza altre
condizioni. Il testo del libro non è coperto da questa licenza: è solo il codice.
