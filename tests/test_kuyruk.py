import threading

import pytest

import kuyruk


def _bekle_ki(kosul, sure=2.0):
    olay = threading.Event()
    for _ in range(int(sure / 0.01)):
        if kosul():
            return True
        olay.wait(0.01)
    return kosul()


def test_two_slots_third_waits():
    q = kuyruk.make_queue(2)
    birak = [threading.Event(), threading.Event()]
    girdi = [threading.Event(), threading.Event()]

    def tutan(i):
        with kuyruk.slot(q, on_wait=lambda n: None):
            girdi[i].set()
            birak[i].wait(5)

    tutanlar = [threading.Thread(target=tutan, args=(i,)) for i in range(2)]
    for t in tutanlar:
        t.start()
    assert all(e.wait(2) for e in girdi)

    sira, ucuncu_girdi = [], threading.Event()

    def ucuncu():
        with kuyruk.slot(q, on_wait=sira.append):
            ucuncu_girdi.set()

    t3 = threading.Thread(target=ucuncu)
    t3.start()
    assert _bekle_ki(lambda: sira == [1])
    assert not ucuncu_girdi.is_set()
    assert q["waiting"] == 1

    birak[0].set()
    assert ucuncu_girdi.wait(2)
    birak[1].set()
    for t in (*tutanlar, t3):
        t.join(2)
    assert q["waiting"] == 0
    assert q["sem"]._value == 2


def test_free_slot_does_not_call_on_wait():
    q = kuyruk.make_queue(2)
    cagri = []
    with kuyruk.slot(q, on_wait=cagri.append):
        assert q["waiting"] == 0
    assert cagri == []


class SahteDurdurma(BaseException):
    """Streamlit'in StopException / RerunException benzeri: Exception degil."""


@pytest.mark.parametrize("hata", [RuntimeError, SahteDurdurma])
def test_slot_released_when_stream_raises(hata):
    q = kuyruk.make_queue(2)
    with pytest.raises(hata):
        with kuyruk.slot(q, on_wait=lambda n: None):
            raise hata("akis koptu")
    assert q["sem"]._value == 2
    assert q["waiting"] == 0


def test_waiting_restored_when_interrupted_while_queued():
    # Sirada beklerken on_wait icinde kesilirse (sekme kapandi) sayac geri alinir
    # ve hic alinmamis yuva serbest birakilmaz.
    q = kuyruk.make_queue(1)
    q["sem"].acquire()

    def kes(n):
        raise SahteDurdurma()

    with pytest.raises(SahteDurdurma):
        with kuyruk.slot(q, on_wait=kes):
            pass
    assert q["waiting"] == 0
    assert q["sem"]._value == 0
