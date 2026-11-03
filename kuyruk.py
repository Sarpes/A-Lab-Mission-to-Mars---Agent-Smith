###############################################################
# MARS TURIZM CHATBOT  |  KUYRUK (SIRA SAYACI)
###############################################################
# Ollama ayni anda PARALEL_YUVA kadar cevap uretir. Bu dosya, uygulamanin
# kendi sira sayacidir: yuva doluysa bekleyen kisiye sirasi soylenir
# ("Onunde 1 insan var, Mr. Anderson.") ve yuva bosalinca cevap baslar.
#
# NEDEN AYRI DOSYA
#   Streamlit olmadan test edilebilsin diye. Sira mantigi saf Python
#   (threading.Semaphore + sayac); app.py yalnizca make_queue ile kuyrugu
#   bir kez kurar ve her cevabi slot() icinde akitir.
#
# NEDEN finally (BaseException DAHIL)
#   Tarayici sekmesi akis sirasinda kapanirsa Streamlit betigi
#   StopException / RerunException ile keser; bunlar bazi surumlerde
#   Exception degil BaseException alt sinifidir. "except Exception" yuvayi
#   kacirirdi ve iki kapanan sekme uygulamayi kalici olarak kilitlerdi.
#   finally her durumda calisir (tests/test_kuyruk.py bunu kanitlar).
#
# GIRDI    yuva sayisi (ayarlar.PARALEL_YUVA)
# CIKTI    make_queue, slot
###############################################################

import contextlib
import threading
from typing import Callable, Iterator


###############################################################
# STEP 1: KUYRUK
###############################################################
# sem: bos yuvalar. lock: waiting sayacini korur. waiting: su an sirada
# bekleyen (yuva almamis) kisi sayisi.

def make_queue(slots: int) -> dict:
    return {"sem": threading.Semaphore(slots), "lock": threading.Lock(), "waiting": 0}


###############################################################
# STEP 2: YUVA ALMA
###############################################################
# Bos yuva varsa hemen alinir, on_wait cagrilmaz. Yoksa kisi siraya girer,
# sira numarasi (kendisi dahil bekleyen sayisi) on_wait ile bildirilir ve
# yuva bosalana kadar beklenir. Beklerken kesilirse sayac geri alinir ve
# hic alinmamis yuva serbest birakilmaz.

@contextlib.contextmanager
def slot(q: dict, on_wait: Callable[[int], None]) -> Iterator[None]:
    if not q["sem"].acquire(blocking=False):
        with q["lock"]:
            q["waiting"] += 1
            sira = q["waiting"]
        try:
            on_wait(sira)
            q["sem"].acquire()
        finally:
            with q["lock"]:
                q["waiting"] -= 1
    try:
        yield
    finally:
        q["sem"].release()
