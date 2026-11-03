import threading

import kilit


def test_backoff_doubles_and_caps_at_8():
    assert [kilit.backoff(n) for n in range(1, 7)] == [1, 2, 4, 8, 8, 8]


def test_session_locked_after_max_failures():
    assert not kilit.session_locked(7, limit=8)
    assert kilit.session_locked(8, limit=8)


def _doldur(g, adet, t0):
    # adet kadar basarisiz deneme kaydet (hicbiri geri alinmaz)
    for i in range(adet):
        kilit.begin_attempt(g, t0 + i, limit=30, window=600)


def test_slowdown_boundary_30_vs_31():
    g = kilit.new_guard()
    _doldur(g, 30, 1000.0)
    # Pencerede 30 basarisizlik var: "30'dan fazla" degil, yavaslatma yok.
    _, yavas = kilit.begin_attempt(g, 1030.0, limit=30, window=600)
    assert yavas is False
    # Bu deneme de basarisiz sayildi -> 31; sonraki deneme yavaslar.
    _, yavas = kilit.begin_attempt(g, 1031.0, limit=30, window=600)
    assert yavas is True


def test_slowdown_expires_with_window():
    g = kilit.new_guard()
    _doldur(g, 31, 1000.0)  # anlar 1000..1030
    _, yavas = kilit.begin_attempt(g, 1100.0, limit=30, window=600)
    assert yavas is True
    # 1631'de 1000..1031 arasindaki kayitlar (<= now - window) pencereden duser;
    # geriye 1100'deki kayit kalir.
    _, yavas = kilit.begin_attempt(g, 1631.0, limit=30, window=600)
    assert yavas is False
    assert list(g["failures"]) == [1100.0, 1631.0]


def test_success_removes_provisional_record():
    g = kilit.new_guard()
    _doldur(g, 30, 1000.0)
    jeton, _ = kilit.begin_attempt(g, 1030.0, limit=30, window=600)
    kilit.cancel_attempt(g, jeton)  # dogru sifre: basarisizlik sayilmaz
    assert len(g["failures"]) == 30
    _, yavas = kilit.begin_attempt(g, 1031.0, limit=30, window=600)
    assert yavas is False
    # zaten pencereden dusmus bir jetonu geri almak hata vermez
    kilit.cancel_attempt(g, 1.0)


def test_concurrent_attempts_record_exactly_n():
    g = kilit.new_guard()
    baslat = threading.Barrier(8)

    def is_parcacigi():
        baslat.wait()
        for _ in range(100):
            kilit.begin_attempt(g, 5.0, limit=30, window=600)

    ipler = [threading.Thread(target=is_parcacigi) for _ in range(8)]
    for t in ipler:
        t.start()
    for t in ipler:
        t.join()
    assert len(g["failures"]) == 800
