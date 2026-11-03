import importlib
import os

import ayarlar

ANAHTARLAR = ["TOP_K", "GECMIS_TUR", "SORU_MAX", "PARALEL_YUVA", "TEMPERATURE", "THINK"]


def test_defaults(monkeypatch):
    # .env veya ortamda kalan degerler varsayilani gizlemesin
    for anahtar in ANAHTARLAR:
        monkeypatch.delenv(anahtar, raising=False)
    importlib.reload(ayarlar)
    assert ayarlar.TOP_K == 3
    assert ayarlar.GECMIS_TUR == 10
    assert ayarlar.SORU_MAX == 500
    assert ayarlar.PARALEL_YUVA == 2
    assert ayarlar.DATA_PATH.endswith(os.path.join("data", "mars_turizm.txt"))


def test_env_override(monkeypatch):
    monkeypatch.setenv("TEMPERATURE", "0.4")
    monkeypatch.setenv("THINK", "true")
    importlib.reload(ayarlar)
    assert ayarlar.TEMPERATURE == 0.4
    assert isinstance(ayarlar.TEMPERATURE, float)
    assert ayarlar.THINK is True
    monkeypatch.undo()
    importlib.reload(ayarlar)
