###############################################################
# MARS TURIZM CHATBOT  |  KILIT (SIFRE DENEME SINIRI)
###############################################################
# Uygulama demo sirasinda herkese acik bir tunel adresinden erisilir. Tek
# ortak sifre kaba kuvvetle (brute force) denenmesin diye iki katman var:
#   1. Oturum basina: her basarisizliktan sonra bekleme 1, 2, 4, 8, 8... sn;
#      8 basarisizliktan sonra o oturum kapanir (app.py).
#   2. Surec geneli YAVASLATMA: tum oturumlarda son 10 dakikada 30'dan fazla
#      basarisizlik varsa, her deneme (dogru ya da yanlis) once
#      GENEL_YAVASLATMA_SN bekler, sonra normal degerlendirilir. Yeni sekme
#      acarak oturum sayacini sifirlayan saldirgani yavaslatir.
#
# NEDEN KILITLEME DEGIL YAVASLATMA (R14)
#   Genel bir kilit, linki bilen tek kisinin 31 yanlis denemeyle canli demoyu
#   10 dakika herkese kapatmasina izin verirdi. Yavaslatma kaba kuvveti
#   pahali kilar ama dogru sifre her zaman iceri girer.
#
# NEDEN TEK FONKSIYON (begin_attempt)
#   "Say ve kaydet" ayri iki cagri olsaydi, ayni anda gelen denemeler arada
#   kalip sayimi kacirabilirdi. begin_attempt tek kilit altinda eskileri
#   siler, bu denemeyi GECICI basarisizlik olarak kaydeder ve yavaslatma
#   bayragini dondurur. Sifre dogru cikarsa app.py cancel_attempt ile kaydi
#   geri alir; boylece demoda arka arkaya giris yapan seyirciler sayaci
#   doldurmaz. Deneme yarida kesilirse (sekme kapandi) kayit basarisizlik
#   olarak kalir; temkinli taraf budur. Kilit hicbir zaman uyurken tutulmaz:
#   bekleme app.py'de, kilit birakildiktan sonra yapilir.
#
# NEDEN AYRI DOSYA
#   Kurallar saf Python; saat (now) disaridan verilir ki testler beklemeden
#   zamani ileri saralabilsin (tests/test_kilit.py). Sinirlar ayarlar.py'de.
#
# GIRDI    deneme anlari (time.monotonic), sinirlar
# CIKTI    new_guard, begin_attempt, cancel_attempt, backoff, session_locked
###############################################################

import collections
import threading

import ayarlar


###############################################################
# STEP 1: SUREC GENELI SAYAC
###############################################################
# failures: basarisiz (veya sonucu henuz belli olmayan) deneme anlari, artan
# sirada. Kilit gereklidir: Streamlit her oturumu ayri is parcaciginda calistirir.

def new_guard() -> dict:
    return {"lock": threading.Lock(), "failures": collections.deque()}


def begin_attempt(guard: dict, now: float, limit: int = ayarlar.GENEL_DENEME_MAX,
                  window: float = ayarlar.GENEL_PENCERE_SN) -> tuple[float, bool]:
    # Donus: (jeton, yavaslat). Yavaslatma, bu denemeden ONCEKI kayitlara bakar:
    # pencerede limitten fazla basarisizlik varsa True.
    with guard["lock"]:
        while guard["failures"] and guard["failures"][0] <= now - window:
            guard["failures"].popleft()
        yavaslat = len(guard["failures"]) > limit
        guard["failures"].append(now)
    return now, yavaslat


def cancel_attempt(guard: dict, token: float) -> None:
    # Basarili giris: gecici kaydi geri al. Kayit pencereden dusmusse sorun yok.
    with guard["lock"]:
        try:
            guard["failures"].remove(token)
        except ValueError:
            pass


###############################################################
# STEP 2: OTURUM BASINA KURAL
###############################################################
# n: bu oturumdaki basarisizlik sayisi (1'den baslar).

def backoff(n: int) -> float:
    return min(2 ** (n - 1), 8)


def session_locked(n: int, limit: int = ayarlar.SIFRE_DENEME_MAX) -> bool:
    return n >= limit
