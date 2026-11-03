###############################################################
# MARS TURIZM CHATBOT  |  KILIT (SIFRE DENEME SINIRI)
###############################################################
# TEHDIT MODELI
#   Uygulama demo sirasinda herkese acik bir tunel adresinden (Funnel)
#   erisilir ve tek bir ORTAK sifreyle korunur. Linki bilen biri sifreyi kaba
#   kuvvetle (brute force) deneyebilir: tek sekmeden art arda, yeni sekmeler
#   acarak (oturum sayaci sifirlanir) veya bircok oturumdan PARALEL olarak.
#   IP basina sinir islemez: tunel arkasinda tum istekler ayni adresten gelir.
#
# IKI KATMAN
#   1. Oturum basina: her yanlistan sonra 1, 2, 4, 8, 8... sn bekleme;
#      8 yanlistan sonra o oturum kapanir (app.py).
#   2. Surec geneli SIRALAMA: tum oturumlarda son GENEL_PENCERE_SN icinde
#      GENEL_DENEME_MAX'tan fazla basarisizlik varsa yavaslatma devrededir.
#      Bu durumda TUM oturumlar icin GENEL_YAVASLATMA_SN'de en fazla BIR
#      deneme degerlendirilir: her deneme sira zamani alir
#      (sira = max(simdi, next_slot); next_slot = sira + gecikme) ve kilit
#      DISINDA o zamana kadar uyur. Paralel oturum acmak tahmin hizini
#      artirmaz. Sirada GENEL_BEKLEYEN_MAX kisi varsa yeni deneme hemen
#      reddedilir ("sira dolu"): basarisizlik sayilmaz, uyumaz; boylece
#      bekleyen Streamlit is parcacigi sayisi da sinirli kalir.
#
# NEDEN KILITLEMEK DEGIL SIRALAMAK (R14, R15)
#   Genel bir kilit, linki bilen tek kisinin canli demoyu herkese kapatmasina
#   izin verirdi. Siralama kaba kuvveti saniyede bir tahmine indirir ama
#   dogru sifre sirasi gelince her zaman iceri girer: demo ayakta kalir.
#
# KAYIT KURALI
#   Her deneme, kilit altinda GECICI basarisizlik olarak kaydedilir; sifre
#   dogruysa cancel_attempt ile geri alinir (arka arkaya giris yapan seyirci
#   sayaci doldurmaz). Yarida kesilen deneme basarisiz sayilir (temkinli
#   taraf). Saat (clock) kilit ICINDE okunur ki kayitlar sirali kalsin ve
#   eskiler bastan budanabilsin. Kilit hicbir zaman uyurken tutulmaz.
#
# NEDEN AYRI DOSYA
#   Kurallar saf Python; saat ve uyku disaridan verilir ki testler beklemeden
#   zamani ileri saralabilsin (tests/test_kilit.py). Sinirlar ayarlar.py'de.
#
# GIRDI    saat (time.monotonic), sinirlar
# CIKTI    new_guard, begin_attempt, wait_turn, cancel_attempt, backoff,
#          session_locked
###############################################################

import collections
import threading
import time
from typing import Callable

import ayarlar


###############################################################
# STEP 1: SUREC GENELI DURUM
###############################################################
# failures: basarisiz (veya sonucu henuz belli olmayan) deneme anlari, artan
# sirada. next_slot: siradaki denemenin en erken degerlendirilme ani.
# pending: sira zamanini bekleyen deneme sayisi.

def new_guard() -> dict:
    return {"lock": threading.Lock(), "failures": collections.deque(),
            "next_slot": 0.0, "pending": 0}


###############################################################
# STEP 2: DENEMEYI BASLATMA (TEK KILIT ALTINDA)
###############################################################
# Donus: {"rejected", "slowed", "token", "turn"}.
#   rejected: sira dolu; hicbir sey kaydedilmedi, pending degismedi.
#   slowed:   yavaslatma devrede; pending artirildi, wait_turn cagrilmali.
#   token:    gecici kaydin ani (cancel_attempt icin).
#   turn:     bu denemenin degerlendirilecegi an.

def begin_attempt(guard: dict, clock: Callable[[], float] = time.monotonic,
                  limit: int = ayarlar.GENEL_DENEME_MAX,
                  window: float = ayarlar.GENEL_PENCERE_SN,
                  delay: float = ayarlar.GENEL_YAVASLATMA_SN,
                  max_pending: int = ayarlar.GENEL_BEKLEYEN_MAX) -> dict:
    with guard["lock"]:
        now = clock()
        while guard["failures"] and guard["failures"][0] <= now - window:
            guard["failures"].popleft()
        slowed = len(guard["failures"]) > limit
        turn = now
        if slowed:
            if guard["pending"] >= max_pending:
                return {"rejected": True, "slowed": True, "token": None, "turn": None}
            turn = max(now, guard["next_slot"])
            guard["next_slot"] = turn + delay
            guard["pending"] += 1
        guard["failures"].append(now)
    return {"rejected": False, "slowed": slowed, "token": now, "turn": turn}


###############################################################
# STEP 3: SIRAYI BEKLEME (KILIT DISINDA)
###############################################################
# on_wait: arayuzun bekleme mesajini gostermesi icin (yalnizca yavaslatmada).
# pending her durumda finally icinde azaltilir: uyku Streamlit'in
# StopException'iyla (BaseException) kesilse bile sayac sisip kalmaz.

def wait_turn(guard: dict, attempt: dict, on_wait: Callable[[], None] | None = None,
              clock: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep) -> None:
    if not attempt["slowed"] or attempt["rejected"]:
        return
    try:
        if on_wait:
            on_wait()
        kalan = attempt["turn"] - clock()
        if kalan > 0:
            sleep(kalan)
    finally:
        with guard["lock"]:
            guard["pending"] -= 1


def cancel_attempt(guard: dict, token: float) -> None:
    # Basarili giris: gecici kaydi geri al. Kayit pencereden dusmusse sorun yok.
    with guard["lock"]:
        try:
            guard["failures"].remove(token)
        except ValueError:
            pass


###############################################################
# STEP 4: OTURUM BASINA KURAL
###############################################################
# n: bu oturumdaki basarisizlik sayisi (1'den baslar).

def backoff(n: int) -> float:
    return min(2 ** (n - 1), 8)


def session_locked(n: int, limit: int = ayarlar.SIFRE_DENEME_MAX) -> bool:
    return n >= limit
