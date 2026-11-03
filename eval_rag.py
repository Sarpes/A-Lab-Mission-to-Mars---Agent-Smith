###############################################################
# MARS TURIZM CHATBOT  |  ERISIM TESTI VE ESIK TARAMASI
###############################################################
# Bu dosya, rag.py'nin dogru satiri getirip getirmedigini ve cevapsiz (kapsam
# disi) sorularda bos dondugunu OLCER; RAG_ESIK degerini bu olcumden secer.
#
# NEDEN DONDURULMUS SORU SETI (eval/rag_sorular.json)
#   Farkli esikler ayni sorularla olculmezse sayilar kiyaslanamaz. Set bir kez
#   onaylanip dondurulur; degisiklik KARARLAR'a yeni bir kayit gerektirir.
#   Dondurulmadan once "_durum" taslak notunu tasir.
#
# NEDEN DENGELI DOGRULUK (denge = (tutuldu_orani + ret_orani) / 2)
#   Sette ~28 cevaplanabilir, ~12 kapsam disi soru var. Duz dogruluk, esigi
#   dusuk tutup hic reddetmeyen bir botu 28/40 ile iyi gosterirdi. Iki sinifin
#   orani esit agirlikli ortalanir; boylece iki hata da ayni agirlikta sayilir.
#
# NEDEN BERABERLIKTE YUKSEK RET
#   Esit denge'de ret'i yuksek olan esik secilir. Cevaplanabilir bir soruyu
#   reddetmek bir KACIRMADIR (kullanici tekrar sorar); ama kapsam disi bir soruya
#   satir getirmek modelin cevap UYDURMASINA davetiyedir. Hala esitse dusuk esik.
#
# METRIK TANIMLARI
#   recall@1 / recall@3  esikten bagimsiz, ham siralama. "onceki" bos olan soruda
#                        mevcut soru siralamasi; "onceki" dolu olan takip
#                        sorusunda BIRLESIK siralama (onceki + " " + soru),
#                        cunku takip sorusu tek basina cevaplanamaz.
#   tutuldu@t            retrieve ile ayni mantik: mevcut siralamanin en iyi
#                        puani >= t ise onunla, degilse (onceki varsa) birlesikle;
#                        beklenen satir ilk 3 ve puani >= t ise tutuldu.
#   ret@t                kapsam disi soruda hic satir donmedi.
#
# GIRDI    eval/rag_sorular.json, data/mars_turizm.txt, Ollama (bge-m3)
# CIKTI    ekrana tablo, outputs/rag_sweep.csv (eklemeli)
###############################################################

import csv
import datetime
import json
import os

import ollama

import ayarlar
import rag

SORU_YOLU = os.path.join(ayarlar.KOK, "eval", "rag_sorular.json")
CSV_YOLU = os.path.join(ayarlar.KOK, "outputs", "rag_sweep.csv")
ESIKLER = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30 ... 0.70


###############################################################
# STEP 1: SORULARI OKU, SIRALAMALARI BIR KEZ HESAPLA
###############################################################
# Her farkli metin yalnizca bir kez embed edilir (onbellek); tum esikler bu
# siralamalardan hesaplanir, Ollama'ya esik basina tekrar gidilmez.

def load_items(path: str = SORU_YOLU) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["sorular"]


def build_rankings(items: list[dict], index: dict) -> list[dict]:
    onbellek = {}

    def cached_embed(texts):
        if texts[0] not in onbellek:
            onbellek[texts[0]] = rag.embed(texts)
        return onbellek[texts[0]]

    for it in items:
        it["mevcut"] = rag.rank(it["soru"], index, cached_embed)
        it["birlesik"] = (
            rag.rank(it["onceki"] + " " + it["soru"], index, cached_embed)
            if it["onceki"] else None
        )
    return items


###############################################################
# STEP 2: ESIK BAZLI METRIKLER
###############################################################

def hits_at(item: dict, t: float) -> list[dict]:
    # rag.retrieve ile ayni sira: mevcut, yetmezse birlesik, yoksa bos.
    for siralama in (item["mevcut"], item["birlesik"]):
        if siralama and siralama[0]["score"] >= t:
            return [h for h in siralama if h["score"] >= t][:ayarlar.TOP_K]
    return []


def raw_ranking(item: dict) -> list[dict]:
    return item["birlesik"] if item["onceki"] else item["mevcut"]


def found(item: dict, hits: list[dict]) -> bool:
    return any(h["line"] in item["beklenen"] for h in hits)


def recall_counts(items: list[dict]) -> tuple[int, int, int]:
    cevap = [it for it in items if it["beklenen"]]
    r1 = sum(found(it, raw_ranking(it)[:1]) for it in cevap)
    r3 = sum(found(it, raw_ranking(it)[:3]) for it in cevap)
    return r1, r3, len(cevap)


def sweep(items: list[dict]) -> list[dict]:
    cevap = [it for it in items if it["beklenen"]]
    yok = [it for it in items if not it["beklenen"]]
    satirlar = []
    for t in ESIKLER:
        tutuldu = sum(found(it, hits_at(it, t)) for it in cevap)
        ret = sum(not hits_at(it, t) for it in yok)
        denge = (tutuldu / len(cevap) + ret / len(yok)) / 2
        satirlar.append({"t": t, "tutuldu": tutuldu, "n_cevap": len(cevap),
                         "ret": ret, "n_yok": len(yok), "denge": denge})
    return satirlar


def pick_threshold(satirlar: list[dict]) -> dict:
    # En yuksek denge; esitlikte yuksek ret; yine esitse dusuk t.
    return max(satirlar, key=lambda s: (round(s["denge"], 9), s["ret"], -s["t"]))


###############################################################
# STEP 3: RAPOR
###############################################################

def pct(a: int, b: int) -> str:
    return f"{a}/{b} = %{100 * a / b:.0f}"


def fmt_hits(siralama: list[dict], n: int = 3) -> str:
    return ", ".join("{}@{:.3f}".format(h["line"], h["score"]) for h in siralama[:n])


def print_report(items: list[dict], satirlar: list[dict], secilen: dict) -> None:
    r1, r3, n = recall_counts(items)
    print(f"recall@1 {pct(r1, n)} | recall@3 {pct(r3, n)}\n")
    print(f"{'t':>5} | {'tutuldu':>16} | {'ret':>16} | denge")
    for s in satirlar:
        isaret = "  <- t*" if s["t"] == secilen["t"] else ""
        print(f"{s['t']:>5.2f} | {pct(s['tutuldu'], s['n_cevap']):>16} | "
              f"{pct(s['ret'], s['n_yok']):>16} | {s['denge']:.3f}{isaret}")
    t = secilen["t"]
    print(f"\nt* = {t:.2f}  (denge {secilen['denge']:.3f})\n")
    print("--- Kacirmalar (ham siralama ilk 3'te yok, veya t*'da tutulmadi) ---")
    for it in items:
        if it["beklenen"]:
            r3_ok = found(it, raw_ranking(it)[:3])
            if not (r3_ok and found(it, hits_at(it, t))):
                etiket = "recall@3" if not r3_ok else f"esik@{t:.2f}"
                print(f"{it['id']} [{etiket}] {it['soru']!r} beklenen={it['beklenen']} "
                      f"ilk3: {fmt_hits(raw_ranking(it))}")
        elif hits_at(it, t):
            print(f"{it['id']} [ret degil@{t:.2f}] {it['soru']!r} getirilen: "
                  f"{fmt_hits(hits_at(it, t))}")


def embed_digest() -> str:
    for m in ollama.Client(host=ayarlar.OLLAMA_HOST).list().models:
        if m.model.split(":")[0] == ayarlar.EMBED_MODEL.split(":")[0]:
            return m.digest
    return "bilinmiyor"


def append_csv(satirlar: list[dict], secilen: dict, digest: str) -> None:
    os.makedirs(os.path.dirname(CSV_YOLU), exist_ok=True)
    yeni = not os.path.exists(CSV_YOLU)
    zaman = datetime.datetime.now().isoformat(timespec="seconds")
    with open(CSV_YOLU, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if yeni:
            w.writerow(["zaman", "model", "digest", "t", "tutuldu", "n_cevap",
                        "ret", "n_yok", "denge", "secilen"])
        for s in satirlar:
            w.writerow([zaman, ayarlar.EMBED_MODEL, digest, s["t"], s["tutuldu"],
                        s["n_cevap"], s["ret"], s["n_yok"], f"{s['denge']:.4f}",
                        int(s["t"] == secilen["t"])])


###############################################################
# STEP 4: CALISTIR
###############################################################

def main() -> None:
    index = rag.load_index()
    items = build_rankings(load_items(), index)
    satirlar = sweep(items)
    secilen = pick_threshold(satirlar)
    print_report(items, satirlar, secilen)
    append_csv(satirlar, secilen, embed_digest())
    print(f"\nCSV: {CSV_YOLU}")


if __name__ == "__main__":
    main()
