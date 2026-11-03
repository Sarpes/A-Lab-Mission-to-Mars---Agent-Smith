import os

import pytest
from streamlit.testing.v1 import AppTest

import ayarlar

APP = os.path.join(ayarlar.KOK, "app.py")


# Sifre kontrolu persona ve Ollama'dan once gelir; bu testler Ollama istemez.
@pytest.mark.parametrize("sifre, beklenen", [
    ("", "tanımlı değil"),
    ("kisa123", "en az 8 karakter"),
])
def test_app_refuses_missing_or_short_password(monkeypatch, sifre, beklenen):
    monkeypatch.setattr(ayarlar, "ERISIM_SIFRESI", sifre)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert len(at.error) == 1
    assert beklenen in at.error[0].value
    assert "K-18" in at.error[0].value
