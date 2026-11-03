###############################################################
# MARS TURIZM CHATBOT  |  KILIT (SIFRE DENEME SINIRI)
###############################################################
# Uygulama demo sirasinda herkese acik bir tunel adresinden erisilir. Tek
# ortak sifre kaba kuvvetle (brute force) denenmesin diye iki katman var:
#   1. Oturum basina: her basarisizliktan sonra bekleme 1, 2, 4, 8, 8... sn;
#      8 basarisizliktan sonra o oturum kapanir (app.py).
#   2. Surec geneli: tum oturumlarda son 10 dakikada 30'dan fazla
#      basarisizlik varsa, pencere gecene kadar hicbir deneme kabul edilmez.
#      Yeni sekme acarak oturum sayacini sifirlayan saldirgani bu yakalar.
#      IP basina sinir kullanilmadi: tunel arkasinda tum istekler ayni
#      adresten gelir, IP guvenilir degil.
#
# NEDEN AYRI DOSYA
#   Kurallar saf Python; saat (now) disaridan verilir ki testler beklemeden
#   zamani ileri saralabilsin (tests/test_kilit.py). Sinirlar ayarlar.py'de.
#
# GIRDI    basarisizlik anlari (time.monotonic), sinirlar
# CIKTI    new_guard, record_failure, is_locked, backoff, session_locked
###############################################################

import collections
import threading

import ayarlar


###############################################################
# STEP 1: SUREC GENELI SAYAC
###############################################################
# failures: basarisizlik anlari (artan sirada). Kilit, Streamlit'in her
# oturumu ayri is parcaciginda (thread) calistirdigi icin gereklidir.

def new_guard() -> dict:
    return {"lock": threading.Lock(), "failures": collections.deque()}


def record_failure(guard: dict, now: float) -> None:
    with guard["lock"]:
        guard["failures"].append(now)


def is_locked(guard: dict, now: float, limit: int = ayarlar.GENEL_DENEME_MAX,
              window: float = ayarlar.GENEL_PENCERE_SN) -> bool:
    # Pencereden dusen kayitlar silinir; kalanlar limitten fazlaysa kilitli.
    with guard["lock"]:
        while guard["failures"] and guard["failures"][0] <= now - window:
            guard["failures"].popleft()
        return len(guard["failures"]) > limit


###############################################################
# STEP 2: OTURUM BASINA KURAL
###############################################################
# n: bu oturumdaki basarisizlik sayisi (1'den baslar).

def backoff(n: int) -> float:
    return min(2 ** (n - 1), 8)


def session_locked(n: int, limit: int = ayarlar.SIFRE_DENEME_MAX) -> bool:
    return n >= limit
