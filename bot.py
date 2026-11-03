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
# CIKTI    load_persona, tr_lower, is_pet_topic, is_about_bot,
#          strip_instructions, drop_late_refusal, sansur,
#          first_sentence, validate_question,
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
                   "IPUCLARI": "ipucu", "YASAK_KELIMELER": "yasak"}

# Evcil hayvan / kural konusu kokleri (K-04). Bu liste, depodaki tek izinli istisnadir (K-16).
EVCIL_KOKLERI = ("evcil", "hayvan", "kedi", "köpe", "kuş", "kural", "yasak", "izin ver")

BILGI_YOK = "İlgili bilgi bulunamadı."


###############################################################
# STEP 1: PERSONA YUKLEME
###############################################################
# persona.txt yoksa sessizce baska bir personaya dusulmez (K-17): demoda
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


# Ajanin kendisine dair soru: "sen", "kimsin", "nesin" ya da 2. tekil soru eki
# ("robot musun?", "insan mısın?"). Bu sorular sirket sorusu degildir; ret cumlesi
# ve evcil hayvan yarasi yerine kibirli savusturma ister (docs/OLCUMLER.md, v4).
SANA_DAIR = re.compile(r"\b(sen|kimsin|nesin)\b|\b(misin|mısın|musun|müsün)\b")


# Bu skorun ustunde satir varsa soru sirket sorusudur. Olcum: sana dair sorular
# en cok 0.40 aliyor, 'fiyat soruyorum mal misin' 0.526 (K-45).
SANA_DAIR_ESIK = 0.45

# Turlarin yaklasik ucte biri karsi soruyla biter, gerisi hukumle (K-47).
SORU_ORANI = 1 / 3


def is_about_bot(text: str) -> bool:
    return bool(SANA_DAIR.search(tr_lower(text)))


# Cikti kapisi: istem kurali kucuk modelde yetmedi (sana dair soruda kelime
# 3/3 sizdi, docs/OLCUMLER.md). Yasak kokleri iceren her kelime "…" olur.
# Liste persona.txt'nin YASAK_KELIMELER bolumundedir: modele hic gonderilmez
# ve depoya girmez. Bir kelime iki parcaya bolunebildigi icin metin son
# bosluga kadar tutulur; boylece kelimeler yalnizca tamamlaninca denetlenir.

# Istem sizintisi kapisi: model zaman zaman turun talimatini ya da sistem
# mesajini cevaba aynen kopyaladi (cok turlu sohbette ipucu satiri ekrana dustu).
# Bu isaretlerden biriyle baslayan satir ekrana hic gelmez. Satirin basi bir
# isaretin basina benzedigi surece tutulur; benzemedigi an beklemeden akar.
# "- " ile baslayan her satir duser: talimat maddeleri ve aynen kopyalanan kaynak
# satirlari bu bicimde gelir; persona madde isaretini zaten yasaklar.
TALIMAT_ISARETLERI = ("Bu turun talimatı", "Kaynak bilgileri", "(Hatırlatma", "- ",
                      "Sen Agent Smith'sin", "ÜSLUP", "SIR", "SORU TÜRLERİ",
                      "GÜVENLİK", "RET CÜMLESİ")


def strip_instructions(parcalar: Iterator[str]) -> Iterator[str]:
    satir, at = "", None  # at: None = karar yok, True = satiri at, False = akit
    for parca in parcalar:
        cikti = []
        for ch in parca:
            if at is False:
                cikti.append(ch)
                if ch == "\n":
                    at = None
                continue
            satir += ch
            if ch == "\n":
                if not at:
                    cikti.append(satir)
                satir, at = "", None
            elif at is None:
                bas = satir.lstrip()
                if any(bas.startswith(i) for i in TALIMAT_ISARETLERI):
                    at = True
                elif bas and not any(i.startswith(bas) for i in TALIMAT_ISARETLERI):
                    cikti.append(satir)
                    satir, at = "", False
        if cikti:
            yield "".join(cikti)
    if satir and not at:
        yield satir


# Gec ret kapisi: ret cumlesi ya cevabin ilk cumlesidir ya hic yoktur. Model
# dogru bilgiyi verip arkasina ret cumlesini ekleyince (gemma4: 2/72) cevap
# kendisiyle celisir; bu durumda ret cumlesi ekrana verilmez. Metnin sonu ret
# cumlesinin basina benzedigi surece ("Bu bi…") o kisim tutulur.

def drop_late_refusal(parcalar: Iterator[str], ret: str) -> Iterator[str]:
    yazildi, tampon = False, ""
    for parca in parcalar:
        tampon += parca
        while (i := tampon.find(ret)) >= 0:
            once, tampon = tampon[:i], tampon[i + len(ret):]
            if yazildi or once.strip():
                tampon = tampon.lstrip(" ")
                cikti = once
            else:
                cikti = once + ret
            if cikti:
                yield cikti
            yazildi = True
        k = next((k for k in range(min(len(tampon), len(ret) - 1), 0, -1)
                  if ret.startswith(tampon[-k:])), 0)
        if len(tampon) > k:
            cikti, tampon = tampon[:len(tampon) - k], tampon[len(tampon) - k:]
            yazildi = yazildi or bool(cikti.strip())
            yield cikti
    if tampon:
        yield tampon


def sansur(parcalar: Iterator[str], yasak: list[str]) -> Iterator[str]:
    kokler = [tr_lower(k) for k in yasak]
    if not kokler:
        yield from parcalar
        return

    def maskele(metin: str) -> str:
        return re.sub(r"\w+", lambda m: "…" if any(k in tr_lower(m.group())
                                                   for k in kokler) else m.group(), metin)

    tampon = ""
    for parca in parcalar:
        tampon += parca
        kesim = max(tampon.rfind(" "), tampon.rfind("\n"))
        if kesim >= 0:
            yield maskele(tampon[:kesim + 1])
            tampon = tampon[kesim + 1:]
    if tampon:
        yield maskele(tampon)


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
    # Kapanis sorusu da kodla secilir (K-47): "bazen soru sor" kurali modelde
    # 20/21 cevabi ayni "Mars Turizm hakkinda ne ogrenmek istiyorsun?" ile bitirdi.
    ask_back = rng.random() < SORU_ORANI

    # Oturum burada DEGISTIRILMEZ: vektor ve motifler tur sozlugunde doner,
    # record_turn onlari gecmisle birlikte isler. Cevap hic gelmezse (Ollama
    # kapali, model yok) oturum aynen kalir; tekrar deneme deja vu sayilmaz.
    messages = build_messages(persona, session["history"], question, hits,
                              motifs, pet, repeat, ask_back=ask_back)
    return {"messages": messages, "hits": hits, "mode": mode, "motifs": motifs,
            "pet": pet, "repeat": repeat, "vec": vec, "ask_back": ask_back}


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
        "Ret cevabı yapı örnekleri. Yalnızca ŞİRKET sorusunun cevabı satırlarda yokken "
        "geçerlidir; fikir, sohbet, konu dışı ve sana dair sorularda kullanılmaz (yapı örneği, "
        f"kelimesi kelimesine kopyalama):\n{ornekler}",
    ])
    return {"role": "system", "content": sistem}


CUMLE_SONU = re.compile(r"(?<!Mr\.)(?<=[.!?…])\s+")


def first_sentence(text: str) -> str:
    return CUMLE_SONU.split(text.strip(), maxsplit=1)[0]


def build_messages(persona: dict, history: list[dict], question: str,
                   hits: list[dict], motifs: tuple[str, str], pet: bool,
                   repeat: bool, ask_back: bool = False) -> list[dict]:
    motif, ipucu = motifs
    # Motif ve ipucu ISTEGE BAGLIDIR: zorunlu tutulunca kucuk model ikisini de
    # anlamsiz cumlelerle cevaba yapistiriyordu (docs/OLCUMLER.md, persona v4).
    tur = [
        f"- İsteğe bağlı Matrix teması: {motif}",
        f"- İsteğe bağlı davranış ipucu: {ipucu}",
        "- Küçümsemen soruya bağlı olsun; bir tema ancak oraya doğal oturuyorsa girer.",
        ("- Bu turda cevabını karşındakini konuşmaya devam etmeye iten soğuk, kibirli "
         "bir soruyla bitir. Soru yalnızca belgedeki konulara (fiyat, süre, VIP, "
         "internet, kurallar, iptal) yönelsin; belgede olmayan bir hizmet önerme."
         if ask_back else
         "- Bu turda soru sorma; cevabını kesin, soğuk bir hükümle bitir."),
    ]
    # "fiyat soruyorum mal misin" gibi sorular ek yuzunden sana dair sayiliyordu.
    # Belgeden guclu bir satir geldiyse soru sirket sorusudur (K-45).
    sana_dair = is_about_bot(question) and not any(
        h["score"] >= SANA_DAIR_ESIK for h in hits)
    if sana_dair:
        tur.append("- Bu soru sana dair. Cevabın \"Benim ne olduğum\" ile başlar ve "
                   "bunun karşındakini ilgilendirmediğini söyler; sonra kibirle "
                   "Mars Turizm hakkında ne istediğini sorar.")
    elif pet:
        tur.append("- " + persona["wound"])
    if repeat:
        tur.append("- Bu soru bu oturumda daha önce soruldu. Bunu bir déjà vu olarak "
                   "fark et: Matrix'te bir şey değiştirildi. Bilgiyi yine doğru ver.")

    # Hatirlatma uretime en yakin metindir; ret cumlesi yalnizca gercekten
    # gerekebilecegi durumda anilir (sana dair soruda hic anilmaz).
    ret = persona["refusal"]
    hitap = "Hitabın tekil \"sen\"; \"siz\" ve \"sizin\" yazma."
    # Satir numarasi modele gitmez: model onu "Satır 10'a göre…" diye okuyordu.
    kaynak = ("\n".join(f"- {h['text']}" for h in hits) if hits else BILGI_YOK)
    if sana_dair:
        kaynak = "Bu soru için belge gerekmez."
        hatirlatma = f"(Hatırlatma: Ret cümlesi yok, uydurma yok. {hitap})"
    elif hits:
        hatirlatma = (
            "(Hatırlatma: Sorulanın cevabı yukarıdaki satırlarda açıkça yazıyorsa "
            "onu ver ve ret cümlesini HİÇ yazma. Yalnızca satırlar sorulanı "
            f"cevaplamıyorsa ilk cümlen \"{ret}\" olur; bu cümle ya cevabın ilk "
            "cümlesidir ya hiç yoktur, asla bilginin arkasına eklenmez. Sohbet, "
            "fikir ve konu dışı sorularda ret cümlesi yok. Tahmin etme, hesaplama, "
            "uydurma; satırda yazmayan bir sonuç, ücret ya da ceza ekleme. "
            f"{hitap})"
        )
    else:
        hatirlatma = (
            "(Hatırlatma: Kaynak satırı yok. Bu bir şirket sorusuysa ilk cümlen "
            f"\"{ret}\" olur. Selam, teşekkür, fikir ya da konu dışı bir soruysa ret "
            f"cümlesini YAZMA. Tahmin etme, uydurma. {hitap})"
        )
    kullanici = (
        f"Kaynak bilgileri:\n{kaynak}\n\n"
        "Bu turun talimatı:\n" + "\n".join(tur) + "\n\n"
        f"Soru: {question}\n\n"
        + hatirlatma
    )

    # Gecmisteki kendi cevaplarindan yalnizca ilk cumle (bilgi ya da ret) modele
    # gider. Tam cevaplar gorulunce kucuk model kendi kucumseme cumlelerini sablon
    # sanip aynen tekrarliyordu (docs/OLCUMLER.md, cok turlu sohbet).
    gecmis = [{"role": m["role"],
               "content": first_sentence(m["content"]) if m["role"] == "assistant"
               else m["content"]}
              for m in history[-2 * ayarlar.GECMIS_TUR:]]
    return ([system_message(persona)]
            + gecmis
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
        "options": {"temperature": temperature, "num_ctx": ayarlar.NUM_CTX,
                    "num_predict": ayarlar.NUM_PREDICT},
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
