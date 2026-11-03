###############################################################
# MARS TURIZM CHATBOT  |  RAG (PARCALAMA, VEKTOR, ESIKLI ERISIM)
###############################################################
# Bu dosya, soruya en yakin sirket bilgisi satirlarini bulur. Cevabi uretmek
# bot.py'nin isidir; burasi yalnizca "hangi satirlar ilgili?" sorusunu yanitlar.
#
# NEDEN SATIR BAZLI PARCA
#   data/mars_turizm.txt dosyasinda her satir TEK bir olgudur (fiyat, sure,
#   kural...). Satir zaten dogal bir parcadir; sabit uzunluklu bolme bir
#   olguyu ortadan kesebilir. Satir numarasi da kaynak gostermek icin yeter.
#
# NEDEN ANAHTAR KELIME DEGIL, EMBEDDING
#   Turkce ekler kelimeyi degistirir: ders ornegi "cay" kelimesini buluyor ama
#   "cayin" kelimesini bulamiyordu. bge-m3 anlami vektore cevirdigi icin
#   "cayin fiyati" ile "cay 500 TL'dir" yakin cikar. Vektorler L2-normalize
#   edilir; boylece nokta carpim = kosinus benzerligi olur.
#
# NEDEN ONCE MEVCUT SORU, SONRA BIRLESIK (KARARLAR K-15)
#   "Cay ne kadar?" sorusundan sonra "CEO kim?" gelirse, onceki soruyu
#   her zaman eklemek cay satirini getirirdi. Bu yuzden once YALNIZ mevcut
#   soru aranir; hic sonuc yoksa (ornegin "Peki ya VIP?") onceki soruyla
#   birlestirilip tekrar aranir.
#
# GIRDI    ayarlar.DATA_PATH, Ollama (bge-m3)
# CIKTI    load_chunks, embed, load_index, search, rank, retrieve
###############################################################

import functools

import numpy as np
import ollama

import ayarlar


###############################################################
# STEP 1: PARCALAMA
###############################################################
# Baslik (1. satir) ve bos satirlar atlanir. "line" 1 tabanlidir. UTF-8 BOM ve
# Windows satir sonu (\r) temizlenir; git autocrlf dosyayi CRLF yapabilir.

def load_chunks(path: str = ayarlar.DATA_PATH) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        satirlar = f.read().replace("\r", "").split("\n")
    chunks = []
    for no, satir in enumerate(satirlar, start=1):
        if no == 1 or not satir.strip():
            continue
        chunks.append({"line": no, "text": satir.strip()})
    return chunks


###############################################################
# STEP 2: EMBEDDING
###############################################################
# Istemci tembel olusturulur: `import rag` Ollama'nin acik olmasini gerektirmez
# (birim testler Ollama kapaliyken calisir).

@functools.lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client(host=ayarlar.OLLAMA_HOST)


def embed(texts: list[str]) -> np.ndarray:
    yanit = _client().embed(model=ayarlar.EMBED_MODEL, input=texts)
    matris = np.array(yanit.embeddings, dtype=float)
    norm = np.linalg.norm(matris, axis=1, keepdims=True)
    norm[norm == 0] = 1  # sifir vektorde 0'a bolmeyi onle
    return matris / norm


# Arama kelimeleri (K-41): bazi satirlar sorulma bicimindeki kelimeleri hic
# icermez. "Ekonomi paket 250.000 TL" satirinda fiyat/ucret/tutar gecmedigi icin
# "Mars'a gitmek ne kadar tutar?" bu satiri ilk 5'e bile getirmiyordu. Kelimeler
# YALNIZCA embedding'e eklenir; modele ve ekrana belgedeki satir aynen gider.
# Bicim: "5: fiyat, ucret" (satir no: kelimeler); "#" ile baslayan satir yorum.

def load_search_words(path: str = ayarlar.KELIME_PATH) -> dict[int, str]:
    kelimeler = {}
    with open(path, encoding="utf-8-sig") as f:
        for satir in f.read().replace("\r", "").split("\n"):
            if satir.strip() and not satir.startswith("#"):
                no, metin = satir.split(":", 1)
                kelimeler[int(no)] = metin.strip()
    return kelimeler


def load_index(path: str = ayarlar.DATA_PATH, embed_fn=embed,
               kelime_path: str | None = ayarlar.KELIME_PATH) -> dict:
    chunks = load_chunks(path)
    kelimeler = load_search_words(kelime_path) if kelime_path else {}
    metinler = [f"{c['text']} ({kelimeler[c['line']]})" if c["line"] in kelimeler
                else c["text"] for c in chunks]
    matrix = embed_fn(metinler)
    return {"chunks": chunks, "matrix": matrix}


###############################################################
# STEP 3: ARAMA
###############################################################
# rank: tum parcalar, esiksiz (eval_rag.py esigi buradan olcer).
# search: rank + esik + ilk k. k/threshold None ise ayarlar'dan CAGRI ANINDA
# okunur; boylece ayarlar yeniden yuklenince (reload) yeni deger gecerli olur.

def rank(query: str, index: dict, embed_fn=embed) -> list[dict]:
    q = np.asarray(embed_fn([query]))[0]
    skorlar = index["matrix"] @ q
    sonuc = [
        {"line": c["line"], "text": c["text"], "score": float(s)}
        for c, s in zip(index["chunks"], skorlar)
    ]
    return sorted(sonuc, key=lambda h: h["score"], reverse=True)


def search(query: str, index: dict, embed_fn=embed,
           k: int | None = None, threshold: float | None = None) -> list[dict]:
    k = ayarlar.TOP_K if k is None else k
    threshold = ayarlar.RAG_ESIK if threshold is None else threshold
    return [h for h in rank(query, index, embed_fn) if h["score"] >= threshold][:k]


###############################################################
# STEP 4: ONCE MEVCUT SORU, SONRA BIRLESIK (K-15)
###############################################################
# mode: "current" (mevcut soru yetti), "combined" (onceki soruyla birlesince
# bulundu), "none" (hicbiri).

def retrieve(question: str, previous: str | None, index: dict, embed_fn=embed,
             k: int | None = None, threshold: float | None = None) -> tuple[list[dict], str]:
    hits = search(question, index, embed_fn, k, threshold)
    if hits:
        return hits, "current"
    if previous:
        hits = search(previous + " " + question, index, embed_fn, k, threshold)
        if hits:
            return hits, "combined"
    return [], "none"
