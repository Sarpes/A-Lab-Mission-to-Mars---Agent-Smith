import re

import numpy as np

import eval_rag
import rag

VOCAB = ["cay", "kahve", "fiyat", "iptal", "vip", "paket"]


def sahte_embed(texts):
    matris = np.zeros((len(texts), len(VOCAB)))
    for i, metin in enumerate(texts):
        for kelime in re.findall(r"\w+", metin.lower()):
            if kelime in VOCAB:
                matris[i, VOCAB.index(kelime)] += 1
    norm = np.linalg.norm(matris, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return matris / norm


def kur(tmp_path, items):
    dosya = tmp_path / "mini.txt"
    dosya.write_text("BASLIK\nCay fiyat 500 TL.\nIptal 30 gun once.\n"
                     "VIP paket fiyat 750 TL.\n", encoding="utf-8")
    index = rag.load_index(str(dosya), embed_fn=sahte_embed)
    cached = eval_rag.make_cached_embed(sahte_embed)
    return index, cached, eval_rag.build_rankings(items, index, cached)


ITEMS = [
    # tam eslesme: puan 1.0 (satir 2)
    {"id": "A1", "soru": "cay fiyat", "onceki": None, "beklenen": [2]},
    # yalniz birlesik yedekle bulunur: "peki" sozlukte yok, mevcut puan 0
    {"id": "A2", "soru": "peki", "onceki": "vip paket fiyat", "beklenen": [4]},
    # kapsam disi, hicbir sozcuk eslesmez: puan 0
    {"id": "Y1", "soru": "ceo kim", "onceki": None, "beklenen": []},
    # kapsam disi ama kismi eslesme: "fiyat" satir 2 icin 0.707, satir 4 icin 0.577 verir
    {"id": "Y2", "soru": "fiyat", "onceki": None, "beklenen": []},
]


def test_sweep_counts_and_combined_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_rag, "ESIKLER", [0.30, 0.80])
    index, cached, items = kur(tmp_path, ITEMS)
    dusuk, yuksek = eval_rag.sweep(items, index, cached)
    # t=0.30: A1 ve A2 (yalniz birlesik yedekle) tutulur; Y2 "fiyat" 0.707 ile
    # satir getirir -> ret 1/2
    assert (dusuk["tutuldu"], dusuk["n_cevap"], dusuk["ret"], dusuk["n_yok"]) == (2, 2, 1, 2)
    # t=0.80: Y2 (0.707) artik reddedilir -> ret 2/2; A1 ve A2 hala tutulur (1.0)
    assert (yuksek["tutuldu"], yuksek["ret"]) == (2, 2)
    # A2'nin gercekten birlesik yoldan bulundugunu dogrula
    assert rag.retrieve("peki", "vip paket fiyat", index, cached, threshold=0.30)[1] == "combined"


def test_hits_at_uses_retrieve(tmp_path):
    index, cached, items = kur(tmp_path, ITEMS)
    a2 = items[1]
    beklenen, _ = rag.retrieve(a2["soru"], a2["onceki"], index, cached, threshold=0.5)
    assert eval_rag.hits_at(a2, 0.5, index, cached) == beklenen
    assert [h["line"] for h in beklenen][0] == 4


def s(t, tutuldu, ret, n_cevap=10, n_yok=10):
    return {"t": t, "tutuldu": tutuldu, "n_cevap": n_cevap, "ret": ret,
            "n_yok": n_yok, "denge": (tutuldu / n_cevap + ret / n_yok) / 2}


def test_pick_threshold_equal_denge_prefers_higher_ret():
    # ikisi de denge 0.7: (8/10 + 6/10)/2 ve (6/10 + 8/10)/2
    secilen = eval_rag.pick_threshold([s(0.40, 8, 6), s(0.60, 6, 8)])
    assert secilen["t"] == 0.60  # ters yazilsa 0.40 secilirdi


def test_pick_threshold_equal_denge_and_ret_prefers_lower_t():
    secilen = eval_rag.pick_threshold([s(0.60, 8, 6), s(0.40, 8, 6)])
    assert secilen["t"] == 0.40  # ters yazilsa 0.60 secilirdi
