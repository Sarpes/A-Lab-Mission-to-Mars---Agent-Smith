import json
import os
import random
from types import SimpleNamespace

import httpx
import numpy as np
import ollama
import pytest

import ayarlar
import bot
import rag

CEKIRDEK = "Veri bankalarımda bu bilgi yok, Mr. Anderson."


@pytest.fixture
def persona():
    # Canlida kullanilan persona (depoda; KARARLAR K-48).
    return bot.load_persona()


# --- persona ----------------------------------------------------------------

def test_load_persona_sections(persona):
    for anahtar in ("system", "matrix", "ipucu", "refusal", "examples", "wound"):
        assert persona[anahtar], anahtar
    assert isinstance(persona["matrix"], list) and isinstance(persona["ipucu"], list)
    assert isinstance(persona["examples"], list)
    assert len(persona["matrix"]) == 26
    assert len(persona["ipucu"]) == 26
    assert persona["refusal"] == CEKIRDEK
    assert len(persona["examples"]) == 2


def test_load_persona_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="K-17"):
        bot.load_persona(str(tmp_path / "yok.txt"))


# --- Turkce kucuk harf ve evcil hayvan algisi -------------------------------

def test_tr_lower_dotted_capital_i():
    assert bot.tr_lower("EVCİL IĞDIR") == "evcil ığdır"


def test_pet_topic_detects_uppercase_turkish():
    assert bot.is_pet_topic("KÖPEĞİMİ getirebilir miyim?")
    assert bot.is_pet_topic("EVCİL hayvan kuralı ne?")
    # Yalniz "EVCİL": duz lower() "evci̇l" verir ve kok kacar; tr_lower sart.
    assert bot.is_pet_topic("EVCİL?")
    assert not bot.is_pet_topic("Çay ne kadar?")


def test_about_bot_detects_questions_about_the_agent():
    assert bot.is_about_bot("Sen robot musun?")
    assert bot.is_about_bot("SEN KİMSİN")
    assert bot.is_about_bot("Gerçekten insan mısın?")
    assert bot.is_about_bot("nesin sen")
    assert not bot.is_about_bot("Çay ne kadar?")
    assert not bot.is_about_bot("Evcil hayvanımı getirebilir miyim?")
    assert not bot.is_about_bot("Mars'a taşınmalı mıyım?")


def test_sansur_masks_forbidden_words_across_chunks():
    parcalar = ["Rakibimiz Ven", "üs Turizm değil. VENÜS", "'e git, ", "venüslü olma."]
    cikti = list(bot.sansur(iter(parcalar), ["venüs"]))
    metin = "".join(cikti)
    assert metin == "Rakibimiz … Turizm değil. …'e git, … olma."
    assert "ven" not in bot.tr_lower(metin)


def test_strip_instructions_drops_leaked_prompt_lines_across_chunks():
    parcalar = ["Bu turun tal", "imatı:  \n- İsteğe bağlı Matrix teması: Zion  \n",
                "- İsteğe ", "bağlı davranış ipucu: boş mama kabı  \n\n",
                "Evcil hayvan kabul edilmemektedir, Mr. Anderson.\n", "Kaşık yok."]
    metin = "".join(bot.strip_instructions(iter(parcalar)))
    assert metin.strip() == "Evcil hayvan kabul edilmemektedir, Mr. Anderson.\nKaşık yok."


def test_strip_instructions_drops_echoed_source_lines():
    # Model kaynak satirlarini "- ..." biciminde aynen basa yazdi; persona madde
    # isaretini zaten yasaklar, bu yuzden "- " ile baslayan satir dusurulur.
    parcalar = ["- 18 yaşından küçükler yalnızca aileleriyle seyahat edebilir.\n",
                "- Evcil hayvan kabul edilmemektedir.\n\n",
                "Evet, Mr. Anderson. Ailenle gidebilirsin."]
    metin = "".join(bot.strip_instructions(iter(parcalar)))
    assert metin.strip() == "Evet, Mr. Anderson. Ailenle gidebilirsin."


def test_strip_instructions_streams_normal_text_immediately():
    # Isaretle baslamayan satir beklemeden akar (akis gecikmesi yok).
    akis = bot.strip_instructions(iter(["Çay ", "500 TL'dir, ", "Mr. Anderson."]))
    assert next(akis) == "Çay "
    assert "".join(akis) == "500 TL'dir, Mr. Anderson."


RET = "Bu bilgi Mars Turizm belgelerinde yok, Mr. Anderson."


def test_late_refusal_after_a_fact_is_dropped():
    parcalar = ["Ekonomi paket 250.000 TL'dir. Bu bil", "gi Mars Turizm belgelerinde ",
                "yok, Mr. Anderson. Basit bir soru."]
    metin = "".join(bot.drop_late_refusal(iter(parcalar), RET))
    assert metin == "Ekonomi paket 250.000 TL'dir. Basit bir soru."


def test_refusal_as_first_sentence_is_kept():
    parcalar = ["Bu bilgi Mars Turizm", " belgelerinde yok, Mr. Anderson. Bu bir anomali."]
    assert "".join(bot.drop_late_refusal(iter(parcalar), RET)) == RET + " Bu bir anomali."


def test_sansur_without_list_passes_text_through():
    parcalar = ["Çay ", "500 TL'dir."]
    assert "".join(bot.sansur(iter(parcalar), [])) == "Çay 500 TL'dir."


def test_persona_has_forbidden_word_list(persona):
    assert isinstance(persona["yasak"], list) and persona["yasak"]


def test_first_sentence_keeps_mr_anderson_together():
    assert bot.first_sentence("Merhaba, Mr. Anderson. Ne istiyorsun?") == "Merhaba, Mr. Anderson."
    assert bot.first_sentence("Ekonomi paket 250.000 TL'dir. Lüks.") == "Ekonomi paket 250.000 TL'dir."
    assert bot.first_sentence("Tek cümle") == "Tek cümle"


def test_history_assistant_answers_trimmed_for_model(persona):
    motifs = (persona["matrix"][0], persona["ipucu"][0])
    gecmis = [{"role": "user", "content": "Çay ne kadar? Söyle."},
              {"role": "assistant",
               "content": "500 TL'dir, Mr. Anderson. Senin gibi insanlar için bir lüks."}]
    mesajlar = bot.build_messages(persona, gecmis, "Kahve?", HITS, motifs,
                                  pet=False, repeat=False)
    assert mesajlar[1]["content"] == "Çay ne kadar? Söyle."
    assert mesajlar[2]["content"] == "500 TL'dir, Mr. Anderson."
    # Oturumun kendi gecmisi degismez (gunluk ve SSS tam cevabi tutar).
    assert gecmis[1]["content"].endswith("bir lüks.")


def test_closing_question_is_decided_per_turn(persona):
    motifs = (persona["matrix"][0], persona["ipucu"][0])
    sorulu = bot.build_messages(persona, [], "Çay ne kadar?", HITS, motifs,
                                pet=False, repeat=False, ask_back=True)[-1]["content"]
    sorusuz = bot.build_messages(persona, [], "Çay ne kadar?", HITS, motifs,
                                 pet=False, repeat=False, ask_back=False)[-1]["content"]
    assert "soğuk, kibirli bir soruyla bitir" in sorulu
    assert "soru sorma" in sorusuz


def test_prepare_turn_asks_back_about_one_turn_in_three(persona, index):
    rng = random.Random(7)
    oran = sum(bot.prepare_turn(bot.new_session(), "Çay ne kadar?", index, persona,
                                embed_fn=sahte_embed, rng=rng)["ask_back"]
               for _ in range(300)) / 300
    assert 0.25 < oran < 0.42


def test_about_bot_ignored_when_company_lines_found(persona):
    # "fiyat soruyorum ... misin" bir sirket sorusudur: satir geldiyse normal akis.
    motifs = (persona["matrix"][0], persona["ipucu"][0])
    fiyat = [{"line": 5, "text": "Ekonomi paket 250.000 TL, VIP paket 750.000 TL'dir.",
              "score": 0.6}]
    mesajlar = bot.build_messages(persona, [], "fiyat soruyorum anlıyor musun", fiyat,
                                  motifs, pet=False, repeat=False)
    kullanici = mesajlar[-1]["content"]
    assert "Bu soru sana dair" not in kullanici
    assert "- Ekonomi paket 250.000 TL" in kullanici


def test_about_bot_question_skips_wound_and_refusal(persona):
    motifs = (persona["matrix"][0], persona["ipucu"][0])
    mesajlar = bot.build_messages(persona, [], "Sen hayvan mısın?", [], motifs,
                                  pet=True, repeat=False)
    kullanici = mesajlar[-1]["content"]
    assert "Bu soru sana dair" in kullanici
    assert persona["wound"] not in kullanici
    assert f"ilk cümlen \"{persona['refusal']}\"" not in kullanici


# --- girdi denetimi ---------------------------------------------------------

def test_validate_question():
    assert bot.validate_question("") == (False, None)
    assert bot.validate_question("   ") == (False, None)
    ok, mesaj = bot.validate_question("x" * 501)
    assert ok is False and isinstance(mesaj, str) and mesaj
    assert bot.validate_question("Çay ne kadar?") == (True, None)


# --- motif motoru -----------------------------------------------------------

def test_pick_motifs_never_repeats(persona):
    rng = random.Random(42)
    last = None
    for _ in range(500):
        secim = bot.pick_motifs(persona, last, rng)
        assert secim[0] in persona["matrix"] and secim[1] in persona["ipucu"]
        if last is not None:
            assert secim[0] != last[0]
            assert secim[1] != last[1]
        last = secim


# --- tekrar algisi ----------------------------------------------------------

def test_is_repeat():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert bot.is_repeat(a, [b, a])
    assert not bot.is_repeat(a, [b])
    assert not bot.is_repeat(a, [])


# --- mesaj kurma ------------------------------------------------------------

HITS = [{"line": 9, "text": "Mars'ta kahve ücretsiz, çay 500 TL'dir.", "score": 0.81}]


def test_build_messages_shape(persona):
    motifs = (persona["matrix"][0], persona["ipucu"][1])
    gecmis = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
              for i in range(30)]

    mesajlar = bot.build_messages(persona, gecmis, "Çay ne kadar?", HITS, motifs,
                                  pet=False, repeat=False)
    assert mesajlar[0]["role"] == "system"
    sistem = mesajlar[0]["content"]
    assert persona["system"] in sistem
    assert persona["refusal"] in sistem

    assert len(mesajlar) == 1 + 2 * ayarlar.GECMIS_TUR + 1
    assert mesajlar[1]["content"] == "m10"
    assert mesajlar[-1]["role"] == "user"
    # Turun talimati (motif, ipucu, yara, deja vu) son kullanici mesajindadir.
    kullanici = mesajlar[-1]["content"]
    assert motifs[0] in kullanici and motifs[1] in kullanici
    assert persona["wound"] not in kullanici
    assert "déjà vu" not in kullanici
    assert "Kaynak bilgileri" in kullanici
    # Satir numarasi modele gitmez (model "Satir 10'a gore" diye okuyordu).
    assert "- Mars'ta kahve ücretsiz, çay 500 TL'dir." in kullanici
    assert "satır 9" not in kullanici
    assert "Çay ne kadar?" in kullanici

    yarali = bot.build_messages(persona, [], "Köpeğimi getirebilir miyim?", [],
                                motifs, pet=True, repeat=True)
    kullanici2 = yarali[-1]["content"]
    assert persona["wound"] in kullanici2
    assert "déjà vu" in kullanici2
    assert len(yarali) == 2
    assert "İlgili bilgi bulunamadı." in kullanici2


# --- prepare_turn ve record_turn --------------------------------------------

def sahte_embed(texts):
    """Her metin icin sabit, metne bagli bir birim vektor."""
    satirlar = []
    for metin in texts:
        v = np.zeros(8)
        for ch in metin:
            v[ord(ch) % 8] += 1
        satirlar.append(v / (np.linalg.norm(v) or 1))
    return np.array(satirlar)


@pytest.fixture
def index():
    chunks = [{"line": 9, "text": "Mars'ta kahve ücretsiz, çay 500 TL'dir."}]
    return {"chunks": chunks, "matrix": sahte_embed([c["text"] for c in chunks]),
            "matrix_folded": sahte_embed([rag.fold(c["text"]) for c in chunks])}


def test_prepare_turn_does_not_mutate_session(persona, index):
    """Cevap hic gelmezse (record_turn yok) oturum aynen kalir; tekrar deneme deja vu degil."""
    oturum = bot.new_session()
    assert set(oturum) == {"id", "history", "last_motifs", "question_vecs"}
    tur = bot.prepare_turn(oturum, "Çay ne kadar?", index, persona,
                           embed_fn=sahte_embed, rng=random.Random(1))
    assert set(tur) == {"messages", "hits", "mode", "motifs", "pet", "repeat", "vec",
                        "ask_back"}
    assert oturum["history"] == []
    assert oturum["question_vecs"] == []
    assert oturum["last_motifs"] is None
    assert tur["repeat"] is False

    tekrar = bot.prepare_turn(oturum, "Çay ne kadar?", index, persona,
                              embed_fn=sahte_embed, rng=random.Random(2))
    assert tekrar["repeat"] is False


def test_record_turn_commits_vec_and_motifs(persona, index, tmp_path):
    oturum = bot.new_session()
    tur = bot.prepare_turn(oturum, "Çay ne kadar?", index, persona,
                           embed_fn=sahte_embed, rng=random.Random(1))
    bot.record_turn(oturum, "Çay ne kadar?", "500 TL.", tur, model="sahte",
                    seconds=0.1, log_path=str(tmp_path / "g.jsonl"))
    assert oturum["last_motifs"] == tur["motifs"]
    assert len(oturum["question_vecs"]) == 1

    tur2 = bot.prepare_turn(oturum, "Çay ne kadar?", index, persona,
                            embed_fn=sahte_embed, rng=random.Random(2))
    assert tur2["repeat"] is True
    assert tur2["motifs"][0] != tur["motifs"][0]


def test_history_stores_raw_question(persona, index, tmp_path):
    oturum = bot.new_session()
    soru = "Çay ne kadar?"
    tur = bot.prepare_turn(oturum, soru, index, persona, embed_fn=sahte_embed,
                           rng=random.Random(3))
    gunluk = tmp_path / "logs" / "sohbet.jsonl"
    bot.record_turn(oturum, soru, "500 TL, Mr. Anderson.", tur, model="sahte",
                    seconds=1.23, log_path=str(gunluk))

    assert oturum["history"][-2] == {"role": "user", "content": soru}
    assert "Kaynak bilgileri" not in oturum["history"][-2]["content"]
    assert oturum["history"][-1]["role"] == "assistant"

    kayit = json.loads(gunluk.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert set(kayit) == {"ts", "session", "question", "mode", "hits", "motifs",
                          "pet", "repeat", "answer", "model", "seconds"}
    assert kayit["question"] == soru
    assert kayit["session"] == oturum["id"]
    for h in kayit["hits"]:
        assert set(h) == {"line", "score"}


# --- akis (sahte istemci) ---------------------------------------------------

class SahteIstemci:
    def __init__(self, yetenekler):
        self.yetenekler = yetenekler
        self.chat_kwargs = None
        self.generate_kwargs = None

    def show(self, model):
        return SimpleNamespace(capabilities=self.yetenekler)

    def chat(self, **kwargs):
        self.chat_kwargs = kwargs
        parcalar = [("düşünüyorum...", ""), (None, "Çay "), ("", "500 TL."), (None, None)]
        for dusunce, icerik in parcalar:
            yield SimpleNamespace(message=SimpleNamespace(thinking=dusunce, content=icerik))

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs


def test_stream_answer_skips_thinking_and_respects_capability():
    mesajlar = [{"role": "user", "content": "Çay?"}]

    dusunen = SahteIstemci(["completion", "thinking"])
    cikti = list(bot.stream_answer(mesajlar, model="sahte-dusunen", client=dusunen))
    assert "".join(cikti) == "Çay 500 TL."
    assert dusunen.chat_kwargs["think"] is False
    assert dusunen.chat_kwargs["stream"] is True
    assert dusunen.chat_kwargs["keep_alive"] == ayarlar.KEEP_ALIVE
    assert dusunen.chat_kwargs["options"] == {"temperature": ayarlar.TEMPERATURE,
                                              "num_ctx": ayarlar.NUM_CTX,
                                              "num_predict": ayarlar.NUM_PREDICT}

    duz = SahteIstemci(["completion"])
    assert "".join(bot.stream_answer(mesajlar, model="sahte-duz", client=duz)) == "Çay 500 TL."
    assert "think" not in duz.chat_kwargs


def test_preload_calls_generate():
    istemci = SahteIstemci(["completion"])
    bot.preload("sahte", client=istemci)
    assert istemci.generate_kwargs == {"model": "sahte", "prompt": "",
                                       "keep_alive": ayarlar.KEEP_ALIVE}


class SahteSohbetIstemci(SahteIstemci):
    def chat(self, **kwargs):
        self.chat_kwargs = kwargs
        return SimpleNamespace(message=SimpleNamespace(content="."))


def test_warmup_prefills_real_system_message(persona):
    sistem = bot.build_messages(persona, [], "x", [], ("m", "i"), False, False)[0]

    dusunen = SahteSohbetIstemci(["completion", "thinking"])
    bot.warmup(persona, model="sahte-isinma-dusunen", client=dusunen)
    k = dusunen.chat_kwargs
    assert k["messages"][0] == sistem
    assert len(k["messages"]) == 2 and k["messages"][1]["role"] == "user"
    assert k["options"] == {"num_predict": 1, "num_ctx": ayarlar.NUM_CTX}
    assert k["keep_alive"] == ayarlar.KEEP_ALIVE
    assert k.get("stream", False) is False
    assert k["think"] is False

    duz = SahteSohbetIstemci(["completion"])
    bot.warmup(persona, model="sahte-isinma-duz", client=duz)
    assert "think" not in duz.chat_kwargs


# --- dostca hata mesajlari --------------------------------------------------

def test_friendly_error_for_connection_refused():
    for hata in (ConnectionError("refused"), httpx.ConnectError("refused")):
        mesaj = bot.friendly_error(hata)
        assert "Mr. Anderson" in mesaj
        assert "Ollama" in mesaj


def test_friendly_error_for_missing_model():
    hata = ollama.ResponseError("model 'qwen3:14b' not found", 404)
    mesaj = bot.friendly_error(hata)
    assert "Mr. Anderson" in mesaj
    assert "ollama pull qwen3:14b" in mesaj
