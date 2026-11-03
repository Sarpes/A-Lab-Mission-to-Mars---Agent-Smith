import threading

import kilit


def test_backoff_doubles_and_caps_at_8():
    assert [kilit.backoff(n) for n in range(1, 7)] == [1, 2, 4, 8, 8, 8]


def test_session_locked_after_max_failures():
    assert not kilit.session_locked(7, limit=8)
    assert kilit.session_locked(8, limit=8)


def test_global_lock_after_limit_within_window_then_expires():
    g = kilit.new_guard()
    t = 1000.0
    for i in range(30):
        kilit.record_failure(g, t + i)
    # 30 basarisizlik: sinir "30'dan fazla", kilit yok
    assert not kilit.is_locked(g, t + 30, limit=30, window=600)
    kilit.record_failure(g, t + 30)
    assert kilit.is_locked(g, t + 31, limit=30, window=600)
    # ilk kayit 1000'de; 1600'den sonra pencereden duser, sayi 30'a iner
    assert kilit.is_locked(g, t + 599, limit=30, window=600)
    assert not kilit.is_locked(g, t + 600.5, limit=30, window=600)


def test_old_failures_do_not_count():
    g = kilit.new_guard()
    for i in range(40):
        kilit.record_failure(g, float(i))
    # hepsi 10 dakikadan eski
    assert not kilit.is_locked(g, 700.0, limit=30, window=600)
    assert len(g["failures"]) == 0


def test_record_failure_is_thread_safe():
    g = kilit.new_guard()
    ipler = [threading.Thread(target=lambda: [kilit.record_failure(g, 5.0) for _ in range(100)])
             for _ in range(8)]
    for t in ipler:
        t.start()
    for t in ipler:
        t.join()
    assert len(g["failures"]) == 800
