###############################################################
# MARS TURIZM CHATBOT  |  AYARLAR
###############################################################
# Bu dosya projenin TEK ayar kaynagidir: yollar, model adlari, esikler,
# limitler. rag.py, bot.py ve app.py sabitleri buradan okur; hicbiri kendi
# icinde sayi veya yol tanimlamaz.
#
# NEDEN AYRI BIR DOSYA
#   Esik ve model adi gibi degerler olcumle degisir (RAG_ESIK eval_rag.py ile
#   olculdu, bkz. docs/OLCUMLER.md). Degerler tek yerde durursa, bir degisiklik
#   her modulde ayni anda gecerli olur ve iki yerde farkli sayi kalmaz.
#
# NASIL DEGISTIRILIR
#   Her sabit, ayni adli ortam degiskeniyle (veya .env dosyasiyla) ezilebilir.
#   Kodu degistirmeden deneme yapmak icin: .env icine TOP_K=5 yazmak yeter.
#   Ortamda zaten tanimli bir degisken .env tarafindan EZILMEZ.
#
# GIRDI    .env (istege bagli), ortam degiskenleri
# CIKTI    modul duzeyinde sabitler (importlib.reload ile yeniden okunabilir)
###############################################################

import os

from dotenv import load_dotenv

###############################################################
# STEP 1: YOLLAR VE .env YUKLEME
###############################################################
# KOK, bu dosyanin klasorudur; calisma dizini nereden baslatilirsa baslatilsin
# yollar sasmaz. override=False: terminalde set edilmis deger .env'i yener.

KOK = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(KOK, ".env"), override=False)

DATA_PATH = os.path.join(KOK, "data", "mars_turizm.txt")
PERSONA_PATH = os.path.join(KOK, "persona.txt")
LOG_PATH = os.path.join(KOK, "logs", "sohbet.jsonl")


###############################################################
# STEP 2: TIP DONUSTURUCULER
###############################################################
# Ortam degiskenleri her zaman metindir. "0.4" metnini 0.4'e, "true" metnini
# True'ya acikca ceviriyoruz; aksi halde bool("false") True olurdu.

def _metin(anahtar, varsayilan):
    return os.environ.get(anahtar, varsayilan)


def _ondalik(anahtar, varsayilan):
    return float(os.environ.get(anahtar, varsayilan))


def _tamsayi(anahtar, varsayilan):
    return int(os.environ.get(anahtar, varsayilan))


def _mantik(anahtar, varsayilan):
    return os.environ.get(anahtar, str(varsayilan)).strip().lower() in ("1", "true")


###############################################################
# STEP 3: AYARLAR
###############################################################

# --- Ollama ve modeller ------------------------------------------------------
OLLAMA_HOST = _metin("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = _metin("LLM_MODEL", "qwen3:14b")
EMBED_MODEL = _metin("EMBED_MODEL", "bge-m3")
TEMPERATURE = _ondalik("TEMPERATURE", 0.6)
THINK = _mantik("THINK", False)
NUM_CTX = _tamsayi("NUM_CTX", 8192)
KEEP_ALIVE = _metin("KEEP_ALIVE", "30m")

# --- Erisim (retrieval) ------------------------------------------------------
TOP_K = _tamsayi("TOP_K", 3)
RAG_ESIK = _ondalik("RAG_ESIK", 0.55)        # eval_rag.py taramasi, bkz. docs/OLCUMLER.md
TEKRAR_ESIK = _ondalik("TEKRAR_ESIK", 0.92)  # GECICI

# --- Sohbet limitleri --------------------------------------------------------
GECMIS_TUR = _tamsayi("GECMIS_TUR", 10)
PARALEL_YUVA = _tamsayi("PARALEL_YUVA", 2)
SORU_MAX = _tamsayi("SORU_MAX", 500)
ERISIM_SIFRESI = _metin("ERISIM_SIFRESI", "")
