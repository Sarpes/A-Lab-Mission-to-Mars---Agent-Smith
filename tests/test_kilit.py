import threading

import pytest

import kilit

AYAR = {"limit": 30, "window": 600, "delay": 5, "max_pending": 5}


def saat(t):
    return lambda: t


def _doldur(g, adet, t0):
    # adet kadar basarisiz deneme kaydet (hicbiri geri alinmaz, pending sifirlanir)
    for i in range(adet):
        d = kilit.begin_attempt(g, saat(t0 + i), **AYAR)
        kilit.wait_turn(g, d, clock=saat(t0 + i), sleep=lambda s: None)


def test_backoff_doubles_and_caps_at_8():
    assert [kilit.backoff(n) for n in range(1, 7)] == [1, 2, 4, 8, 8, 8]


def test_session_locked_after_max_failures():
    assert not kilit.session_locked(7, limit=8)
    assert kilit.session_locked(8, limit=8)


def test_slowdown_boundary_30_vs_31():
    g = kilit.new_guard()
    _doldur(g, 30, 1000.0)
    # Pencerede 30 basarisizlik var: "30'dan fazla" degil, yavaslatma yok.
    d = kilit.begin_attempt(g, saat(1030.0), **AYAR)
    assert d["slowed"] is False and d["turn"] == 1030.0
    # Bu deneme de basarisiz sayildi -> 31; sonraki deneme yavaslar.
    d = kilit.begin_attempt(g, saat(1031.0), **AYAR)
    assert d["slowed"] is True and d["rejected"] is False
    assert g["pending"] == 1


def test_slowdown_expires_with_window():
    g = kilit.new_guard()
    _doldur(g, 31, 1000.0)  # anlar 1000..1030 (31 kayit)
    d = kilit.begin_attempt(g, saat(1100.0), **AYAR)
    assert d["slowed"] is True
    kilit.wait_turn(g, d, clock=saat(1100.0), sleep=lambda s: None)
    # 1631'de 1000..1030 arasindaki kayitlar (<= 1631 - 600 = 1031) pencereden
    # duser; geriye 1100'deki kayit kalir.
    d = kilit.begin_attempt(g, saat(1631.0), **AYAR)
    assert d["slowed"] is False
    assert list(g["failures"]) == [1100.0, 1631.0]


def test_inactive_slowdown_no_wait_no_spacing():
    g = kilit.new_guard()
    uykular = []
    for _ in range(3):
        d = kilit.begin_attempt(g, saat(50.0), **AYAR)
        assert d["slowed"] is False and d["turn"] == 50.0
        kilit.wait_turn(g, d, clock=saat(50.0), sleep=uykular.append)
    assert uykular == []
    assert g["next_slot"] == 0.0 and g["pending"] == 0


def test_concurrent_attempts_get_turns_spaced_by_delay():
    g = kilit.new_guard()
    _doldur(g, 31, 0.0)
    ayar = dict(AYAR, max_pending=100)
    k = 20
    baslat = threading.Barrier(k)
    siralar, kilit_sirasi = [], threading.Lock()

    def deneme():
        baslat.wait()
        d = kilit.begin_attempt(g, saat(100.0), **ayar)
        with kilit_sirasi:
            siralar.append(d["turn"])

    ipler = [threading.Thread(target=deneme) for _ in range(k)]
    for t in ipler:
        t.start()
    for t in ipler:
        t.join()
    # 100'deki K deneme: 100, 105, 110, ... tam 5 sn aralikla, cakisma yok
    assert sorted(siralar) == [100.0 + 5 * i for i in range(k)]
    assert g["pending"] == k
    assert len(g["failures"]) == 31 + k


def test_wait_turn_sleeps_until_turn():
    g = kilit.new_guard()
    _doldur(g, 31, 0.0)
    d1 = kilit.begin_attempt(g, saat(100.0), **AYAR)
    d2 = kilit.begin_attempt(g, saat(101.0), **AYAR)
    assert (d1["turn"], d2["turn"]) == (100.0, 105.0)
    uykular = []
    kilit.wait_turn(g, d2, clock=saat(101.0), sleep=uykular.append)
    assert uykular == [4.0]


def test_pending_cap_rejects_without_recording():
    g = kilit.new_guard()
    _doldur(g, 31, 0.0)
    for _ in range(5):
        assert kilit.begin_attempt(g, saat(100.0), **AYAR)["rejected"] is False
    kayit, sonraki = len(g["failures"]), g["next_slot"]
    d = kilit.begin_attempt(g, saat(100.0), **AYAR)  # 6. bekleyen
    assert d["rejected"] is True
    assert len(g["failures"]) == kayit
    assert g["pending"] == 5 and g["next_slot"] == sonraki
    uykular = []
    kilit.wait_turn(g, d, sleep=uykular.append)  # reddedilen beklemez
    assert uykular == [] and g["pending"] == 5


class SahteDurdurma(BaseException):
    pass


def test_pending_decremented_when_wait_raises():
    g = kilit.new_guard()
    _doldur(g, 31, 0.0)
    d = kilit.begin_attempt(g, saat(100.0), **AYAR)
    assert g["pending"] == 1

    def kes(_):
        raise SahteDurdurma()

    d2 = kilit.begin_attempt(g, saat(100.0), **AYAR)
    with pytest.raises(SahteDurdurma):
        kilit.wait_turn(g, d2, clock=saat(100.0), sleep=kes)
    assert g["pending"] == 1
    kilit.wait_turn(g, d, clock=saat(100.0), sleep=lambda s: None)
    assert g["pending"] == 0


def test_success_removes_provisional_record():
    g = kilit.new_guard()
    _doldur(g, 30, 1000.0)
    d = kilit.begin_attempt(g, saat(1030.0), **AYAR)
    kilit.cancel_attempt(g, d["token"])  # dogru sifre: basarisizlik sayilmaz
    assert len(g["failures"]) == 30
    assert kilit.begin_attempt(g, saat(1031.0), **AYAR)["slowed"] is False
    # zaten pencereden dusmus bir jetonu geri almak hata vermez
    kilit.cancel_attempt(g, 1.0)


def test_concurrent_attempts_record_exactly_n():
    g = kilit.new_guard()
    baslat = threading.Barrier(8)

    def is_parcacigi():
        baslat.wait()
        for _ in range(100):
            kilit.begin_attempt(g, saat(5.0), **dict(AYAR, max_pending=10**6))

    ipler = [threading.Thread(target=is_parcacigi) for _ in range(8)]
    for t in ipler:
        t.start()
    for t in ipler:
        t.join()
    assert len(g["failures"]) == 800
