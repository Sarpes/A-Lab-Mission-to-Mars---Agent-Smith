###############################################################
# MARS TURIZM CHATBOT  |  BOT (PERSONA, MOTIF MOTORU, MESAJ, AKIS)
###############################################################
# Bu dosya sohbet motorudur: persona.txt'yi okur, her tur icin bir Matrix
# motifi ve bir ipucu secer, rag.py'den gelen satirlarla mesajlari kurar,
# Ollama'dan cevabi parca parca (streaming) alir ve turu gunluge yazar.
# Arayuz (app.py) ve cevap testi (eval_bot.py) yalnizca buradaki fonksiyonlari
# cagirir; ikisi de ayni yolu kullandigi icin test edilen sey demodaki seydir.
#
# NEDEN MOTIFI KOD SECIYOR (KARARLAR K-07)
#   Kucuk modellere bir espri listesi verilirse favorilerini tekrar tekrar
#   kullanirlar. Cesitlilik modelden umulmaz: kod her tur 26 motiften ve 26
#   ipucundan birer tane secer, art arda ayni secim gelmez, ve yalnizca o
#   secimi turun talimatina yazar.
#
# NEDEN GECMISTE HAM SORU (KAYNAK BLOGU DEGIL)
#   Modele giden kullanici mesaji "Kaynak bilgileri" blogu tasir. Bu blok
#   gecmise yazilsaydi her tur gecmis sisip baglami doldururdu ve eski
#   turlarin kaynaklari yeni soruya ikinci kez beslenirdi (ornegin cay satiri
#   CEO sorusuna sizardi). Gecmiste yalnizca kullanicinin yazdigi soru durur.
#
# NEDEN think YALNIZCA DESTEKLENIRSE
#   Ollama, dusunme (thinking) yetenegi olmayan bir modele think= gonderilirse
#   istegi reddeder. Bake-off'ta farkli modeller denendigi icin yetenek
#   client.show() ile bir kez sorulur ve onbellege alinir.
#
# GIRDI    persona.txt, ayarlar, rag.retrieve / rag.embed, Ollama
# CIKTI    load_persona, tr_lower, is_pet_topic, validate_question,
#          pick_motifs, is_repeat, new_session, prepare_turn, build_messages,
#          system_message, stream_answer, supports_thinking, preload,
#          warmup, record_turn,
#          friendly_error
###############################################################

import datetime
import functools
import json
import os
import random
import re
import threading
import uuid
from typing import Iterator

import httpx
import numpy as np
import ollama

import ayarlar
import rag

# persona.txt bolum adlari. Liste bolumleri "- " ile baslayan satirlardan,
# metin bolumleri ise bolumun tum metninden olusur.
METIN_BOLUMLERI = {"SISTEM": "system", "RET_CUMLESI": "refusal", "YARA": "wound"}
LISTE_BOLUMLERI = {"RET_ORNEKLERI": "examples", "MATRIX_MOTIFLERI": "matrix",
                   "IPUCLARI": "ipucu"}

# Evcil hayvan / kural konusu kokleri (K-04). Bu liste, depodaki tek izinli istisnadir (K-16).
EVCIL_KOKLERI = ("evcil", "hayvan", "kedi", "köpe", "kuş", "kural", "yasak", "izin ver")

BILGI_YOK = "İlgili bilgi bulunamadı."


###############################################################
# STEP 1: PERSONA YUKLEME
###############################################################
# persona.txt yoksa sessizce persona_ornek.txt'ye dusulmez (K-17): demoda
# yanlis karakteri fark etmeden sunmak, hic baslamamaktan kotudur.

def load_persona(path: str = ayarlar.PERSONA_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            "persona.txt bulunamadı. Uygulama bilerek başlatılmıyor (KARARLAR K-17)."
        )
    with open(path, encoding="utf-8-sig") as f:
        satirlar = f.read().replace("\r", "").split("\n")

    bolumler, ad = {}, None
    for satir in satirlar:
        if satir.startswith("## "):
            ad = satir[3:].strip()
            bolumler[ad] = []
        elif ad is not None:
            bolumler[ad].append(satir)

    eksik = [b for b in (*METIN_BOLUMLERI, *LISTE_BOLUMLERI) if b not in bolumler]
    if eksik:
        raise ValueError(f"persona dosyasında eksik bölüm: {', '.join(eksik)} ({path})")

    persona = {}
    for bolum, anahtar in METIN_BOLUMLERI.items():
        persona[anahtar] = "\n".join(bolumler[bolum]).strip()
    for bolum, anahtar in LISTE_BOLUMLERI.items():
        persona[anahtar] = [s[2:].strip() for s in bolumler[bolum]
                            if s.startswith("- ") and s[2:].strip()]
    return persona


###############################################################
# STEP 2: TURKCE KUCUK HARF VE EVCIL HAYVAN ALGISI
###############################################################
# Python'un lower() fonksiyonu "İ" harfini "i" + birlesik nokta yapar ve
# "EVCİL" -> "evci̇l" olur; kok aramasi kacar. "I" da "ı" olmalidir, "i" degil.

def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def is_pet_topic(text: str) -> bool:
    kucuk = tr_lower(text)
    return any(kok in kucuk for kok in EVCIL_KOKLERI)


###############################################################
# STEP 3: GIRDI DENETIMI
###############################################################
# Bos soru sessizce yok sayilir (None). Cok uzun soru LLM'e hic gitmez:
# baglami korur ve 5.000 karakterlik yapistirmalari maliyetsiz reddeder.

def validate_question(text: str) -> tuple[bool, str | None]:
    if not text or not text.strip():
        return False, None
    if len(text) > ayarlar.SORU_MAX:
        return False, (
            f"{len(text)} karakter, Mr. Anderson. Sınır {ayarlar.SORU_MAX}. "
            "Destan yazmak sizin türünüzün eğlencesi; benim görev tanımımda yok. "
            "Kısalt ve tekrar dene."
        )
    return True, None


###############################################################
# STEP 4: MOTIF MOTORU (K-07)
###############################################################
# Onceki turun motifi ve ipucu havuzdan cikarilip kalanlardan secilir; boylece
# art arda ayni secim imkansizdir. Havuzda tek oge varsa tekrar kacinilmazdir.

def _sec(havuz: list[str], onceki: str | None, rng: random.Random) -> str:
    adaylar = [m for m in havuz if m != onceki] or havuz
    return rng.choice(adaylar)


def pick_motifs(persona: dict, last: tuple[str, str] | None,
                rng: random.Random) -> tuple[str, str]:
    onceki_motif, onceki_ipucu = last if last else (None, None)
    return (_sec(persona["matrix"], onceki_motif, rng),
            _sec(persona["ipucu"], onceki_ipucu, rng))


###############################################################
# STEP 5: TEKRAR ALGISI (deja vu, K-13)
###############################################################
# Vektorler rag.embed'den L2-normalize gelir; yine de normalize edilir ki
# baska bir embed fonksiyonuyla da kosinus dogru kalsin.

def is_repeat(vec: np.ndarray, previous: list[np.ndarray],
              threshold: float = ayarlar.TEKRAR_ESIK) -> bool:
    v = np.asarray(vec, dtype=float)
    v = v / (np.linalg.norm(v) or 1)
    for p in previous:
        p = np.asarray(p, dtype=float)
        if float(v @ (p / (np.linalg.norm(p) or 1))) >= threshold:
            return True
    return False


###############################################################
# STEP 6: OTURUM VE TUR HAZIRLIGI
###############################################################
# Soru vektoru bir kez hesaplanir: hem tekrar algisi hem retrieve ayni
# vektoru kullanir (retrieve'e verilen embed_fn, ayni soru icin onbellekten
# dondurur). Boylece tur basina bir embed cagrisi tasarruf edilir.

_RNG = random.Random()


def new_session() -> dict:
    return {"id": uuid.uuid4().hex, "history": [], "last_motifs": None,
            "question_vecs": []}


def prepare_turn(session: dict, question: str, index: dict, persona: dict,
                 embed_fn=rag.embed, rng=None) -> dict:
    rng = rng or _RNG
    vec = np.asarray(embed_fn([question]))[0]

    def embed_onbellekli(texts):
        if list(texts) == [question]:
            return vec[None, :]
        return embed_fn(texts)

    previous = next((m["content"] for m in reversed(session["history"])
                     if m["role"] == "user"), None)
    hits, mode = rag.retrieve(question, previous, index, embed_fn=embed_onbellekli)

    repeat = is_repeat(vec, session["question_vecs"])
    pet = is_pet_topic(question)
    motifs = pick_motifs(persona, session["last_motifs"], rng)

    # Oturum burada DEGISTIRILMEZ: vektor ve motifler tur sozlugunde doner,
    # record_turn onlari gecmisle birlikte isler. Cevap hic gelmezse (Ollama
    # kapali, model yok) oturum aynen kalir; tekrar deneme deja vu sayilmaz.
    messages = build_messages(persona, session["history"], question, hits,
                              motifs, pet, repeat)
    return {"messages": messages, "hits": hits, "mode": mode, "motifs": motifs,
            "pet": pet, "repeat": repeat, "vec": vec}


###############################################################
# STEP 7: MESAJ KURMA
###############################################################
# Sistem mesaji = persona + ret cumlesi + yapi ornekleri (her turda ayni).
# Kullanici mesaji = kaynak satirlari + bu turun talimati + soru + hatirlatma.
# Turun talimati (motif, ipucu, yara, deja vu) bilerek kullanici mesajindadir:
# kucuk model, uretime en yakin metne en cok uyar; sistem mesaji ise uzun
# gecmisin cok gerisinde kalir. Bu mesaj gecmise girmez (bkz. baslik).
# Sondaki hatirlatma, esigi gecen ama soruyu cevaplamayan satirlara karsi
# ikinci savunma hattidir (docs/OLCUMLER.md: "CEO kim?" gibi YOK sorulari
# 0.55 esiginin hemen altinda; birlesik aramada onceki sorunun satiri gelir).

def system_message(persona: dict) -> dict:
    ornekler = "\n".join(f"- {e}" for e in persona["examples"])
    sistem = "\n\n".join([
        persona["system"],
        f"RET CÜMLESİ (kelimesi kelimesine): {persona['refusal']}",
        "Ret cevabı yapı örnekleri. Yalnızca OLGU sorusunun cevabı satırlarda yokken "
        "geçerlidir; fikir ve genel bilgi sorularında kullanılmaz (yapı örneği, "
        f"kelimesi kelimesine kopyalama):\n{ornekler}",
    ])
    return {"role": "system", "content": sistem}


def build_messages(persona: dict, history: list[dict], question: str,
                   hits: list[dict], motifs: tuple[str, str], pet: bool,
                   repeat: bool) -> list[dict]:
    motif, ipucu = motifs
    tur = [
        f"- Matrix motifi: {motif}. Küçümseme cümlende buna bir gönderme yap.",
        f"- İpucu: Sen, Agent Smith, şu anda bununla meşgulsün: {ipucu}. Bunu "
        "kendinden söz eden bir yan cümlede (\"ben ...\"), kendi kelimelerinle "
        "sezdir; ifadeyi olduğu gibi yazma.",
    ]
    if pet:
        tur.append("- " + persona["wound"])
    if repeat:
        tur.append("- Bu soru bu oturumda daha önce soruldu. Bunu bir déjà vu olarak "
                   "fark et: Matrix'te bir şey değiştirildi. Bilgiyi yine doğru ver.")

    if hits:
        kaynak = "\n".join(f"(satır {h['line']}) {h['text']}" for h in hits)
    else:
        kaynak = BILGI_YOK
    kullanici = (
        f"Kaynak bilgileri:\n{kaynak}\n\n"
        "Bu turun talimatı:\n" + "\n".join(tur) + "\n\n"
        f"Soru: {question}\n\n"
        "(Hatırlatma: Olgu sorusunda cevap yukarıdaki satırlarda açıkça yoksa ilk "
        f"cümlen \"{persona['refusal']}\" olur. Fikir sorusunda bu cümleyi kullanma. "
        "Tahmin etme, hesaplama, uydurma.)"
    )

    return ([system_message(persona)]
            + list(history[-2 * ayarlar.GECMIS_TUR:])
            + [{"role": "user", "content": kullanici}])


###############################################################
# STEP 8: OLLAMA: AKIS, YETENEK, ON YUKLEME
###############################################################
# Istemci tembel olusturulur (import bot Ollama'yi gerektirmez). Yalnizca
# message.content verilir; thinking metni kullaniciya asla akmaz.

@functools.lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client(host=ayarlar.OLLAMA_HOST)


@functools.lru_cache(maxsize=32)
def supports_thinking(model: str, client=None) -> bool:
    client = client or _client()
    return "thinking" in (client.show(model).capabilities or [])


def stream_answer(messages: list[dict], model: str = ayarlar.LLM_MODEL,
                  temperature: float = ayarlar.TEMPERATURE,
                  think: bool = ayarlar.THINK, client=None) -> Iterator[str]:
    client = client or _client()
    kwargs = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature, "num_ctx": ayarlar.NUM_CTX},
        "keep_alive": ayarlar.KEEP_ALIVE,
    }
    if supports_thinking(model, client):
        kwargs["think"] = think
    for chunk in client.chat(**kwargs):
        parca = chunk.message.content
        if parca:
            yield parca


def preload(model: str = ayarlar.LLM_MODEL, client=None) -> None:
    client = client or _client()
    client.generate(model=model, prompt="", keep_alive=ayarlar.KEEP_ALIVE)


# Isinma (R12): preload modeli bellege alir ama sistem mesajinin on islemesi
# (prefill) ilk gercek soruda odenirdi (olculen ilk token: 6.70 s). Ayni
# sistem mesajiyla tek tokenlik bir cagri Ollama'nin onek onbellegini
# (prompt cache) doldurur; ilk gercek soru yalnizca kendi kismini isler.
def warmup(persona: dict, model: str = ayarlar.LLM_MODEL, client=None) -> None:
    client = client or _client()
    kwargs = {
        "model": model,
        "messages": [system_message(persona), {"role": "user", "content": "Merhaba."}],
        "options": {"num_predict": 1, "num_ctx": ayarlar.NUM_CTX},
        "keep_alive": ayarlar.KEEP_ALIVE,
    }
    if supports_thinking(model, client):
        kwargs["think"] = ayarlar.THINK
    client.chat(**kwargs)


###############################################################
# STEP 9: TURU KAYDETME (GECMIS + GUNLUK)
###############################################################
# Iki tarayici oturumu ayni Streamlit surecinde ayni anda yazabilir; kilit,
# JSONL satirlarinin birbirine karismasini onler.

_GUNLUK_KILIDI = threading.Lock()


def record_turn(session: dict, question: str, answer: str, turn: dict,
                model: str, seconds: float, log_path: str = ayarlar.LOG_PATH) -> None:
    session["history"].append({"role": "user", "content": question})
    session["history"].append({"role": "assistant", "content": answer})
    session["question_vecs"].append(turn["vec"])
    session["last_motifs"] = turn["motifs"]

    kayit = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "session": session["id"],
        "question": question,
        "mode": turn["mode"],
        "hits": [{"line": h["line"], "score": round(float(h["score"]), 4)}
                 for h in turn["hits"]],
        "motifs": list(turn["motifs"]),
        "pet": turn["pet"],
        "repeat": turn["repeat"],
        "answer": answer,
        "model": model,
        "seconds": round(float(seconds), 2),
    }
    klasor = os.path.dirname(log_path)
    if klasor:
        os.makedirs(klasor, exist_ok=True)
    with _GUNLUK_KILIDI, open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(kayit, ensure_ascii=False) + "\n")


###############################################################
# STEP 10: DOSTCA HATA MESAJI
###############################################################
# Yigin izi (stack trace) yerine ne oldugunu ve ne calistirilacagini soyleyen
# tek cumle. Akista baglanti hatasi httpx.ConnectError, akis disinda
# ConnectionError olarak gelir (ollama 0.6.3'te olculdu); ikisi de yakalanir.

def friendly_error(exc: Exception) -> str:
    if isinstance(exc, (ConnectionError, httpx.ConnectError)):
        return ("Makine Şehri'ne ulaşamıyorum, Mr. Anderson. Ollama kapalı. "
                "Ollama'yı başlat, sonra tekrar dene. Bekliyorum; bu konuda sizden "
                "çok daha iyiyim.")
    if isinstance(exc, ollama.ResponseError) and (
            exc.status_code == 404 or "not found" in str(exc).lower()):
        bulunan = re.search(r"model ['\"]?([^'\"\s]+)['\"]? not found", str(exc))
        model = bulunan.group(1) if bulunan else ayarlar.LLM_MODEL
        return (f"Model yüklenmemiş, Mr. Anderson. Terminale şunu yaz: "
                f"ollama pull {model}. Sonra tekrar dene. Evet, bu kadar basit.")
    return (f"Sistemde bir anomali var, Mr. Anderson: {type(exc).__name__}: "
            f"{str(exc)[:200]}. Tekrar dene; anomaliler genelde insan kaynaklıdır.")
