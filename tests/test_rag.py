import re

import numpy as np
import pytest

import rag

VOCAB = ["cay", "kahve", "fiyat", "iptal", "gun", "evcil", "hayvan"]


def sahte_embed(texts):
    """Sabit kelime dagarcigi uzerinde torba-kelime vektoru, L2-normalize."""
    matris = np.zeros((len(texts), len(VOCAB)))
    for i, metin in enumerate(texts):
        for kelime in re.findall(r"\w+", metin.lower()):
            if kelime in VOCAB:
                matris[i, VOCAB.index(kelime)] += 1
    norm = np.linalg.norm(matris, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return matris / norm


@pytest.fixture
def index(tmp_path):
    dosya = tmp_path / "mini.txt"
    dosya.write_text(
        "BASLIK\n\n"
        "Cay fiyat 500 TL.\n"
        "Kahve ucretsiz.\n"
        "\n"
        "Iptal 30 gun once.\n"
        "Evcil hayvan kabul edilmez.\n",
        encoding="utf-8",
    )
    return rag.load_index(str(dosya), embed_fn=sahte_embed)


def test_load_chunks_real_file():
    chunks = rag.load_chunks()
    assert len(chunks) == 11
    assert chunks[0]["line"] == 3
    assert chunks[0]["text"].startswith("Şirket 2042")
    assert "Mars'a bekleriz" in chunks[-1]["text"]
    assert all(c["text"].strip() for c in chunks)


def test_load_chunks_line_numbers_skip_blanks(index):
    assert [c["line"] for c in index["chunks"]] == [3, 4, 6, 7]


def test_search_sorted_topk_and_threshold(index):
    hits = rag.search("cay kahve fiyat", index, embed_fn=sahte_embed, k=2, threshold=0.1)
    assert 0 < len(hits) <= 2
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s >= 0.1 for s in scores)
    assert rag.search("uzay roket", index, embed_fn=sahte_embed, threshold=0.1) == []


def test_rank_returns_all_without_threshold(index):
    tum = rag.rank("uzay roket", index, embed_fn=sahte_embed)
    assert len(tum) == len(index["chunks"])


def test_retrieve_current_mode(index):
    hits, mode = rag.retrieve("kahve", "iptal gun", index, embed_fn=sahte_embed, threshold=0.1)
    assert mode == "current"
    assert hits[0]["line"] == 4


def test_retrieve_combined_fallback(index):
    hits, mode = rag.retrieve("uzay roket", "evcil hayvan", index, embed_fn=sahte_embed, threshold=0.1)
    assert mode == "combined"
    assert hits and hits[0]["line"] == 7


def test_retrieve_none(index):
    assert rag.retrieve("uzay roket", "gezegen", index, embed_fn=sahte_embed, threshold=0.1) == ([], "none")
    assert rag.retrieve("uzay roket", None, index, embed_fn=sahte_embed, threshold=0.1) == ([], "none")
