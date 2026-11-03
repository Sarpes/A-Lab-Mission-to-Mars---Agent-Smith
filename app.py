###############################################################
# MARS TURIZM CHATBOT  |  ARAYUZ (STREAMLIT)
###############################################################
# Bu dosya demonun yuzudur: Matrix terminali ile Mars'in birlestigi bir
# sohbet ekrani. Siyah zemin, Matrix yesili bot metni, Mars pasi kullanici
# mesajlari, ustte pas renkli "Mars tozu yagmuru" seridi. Is mantigi burada
# DEGILDIR: soru denetimi, erisim (RAG), mesaj kurma, akis ve gunluk bot.py'de;
# sira sayaci kuyruk.py'de. app.py yalnizca bunlari sirayla cagirir ve gosterir.
#
# NEDEN BASLAMAMAYI SECIYOR (KARARLAR K-17, K-18)
#   ERISIM_SIFRESI bossa uygulama acilmaz (K-18): sifresiz bir tunel adresi
#   herkese acik bir LLM demektir. persona.txt yoksa da acilmaz (K-17): yanlis
#   karakteri fark etmeden sunmak, hic baslamamaktan kotudur.
#
# NEDEN KUYRUK AYRI DOSYADA
#   Sira mantigi (semaphore + bekleyen sayaci) Streamlit olmadan test edilir
#   (tests/test_kuyruk.py). Sekme akis sirasinda kapanirsa Streamlit betigi
#   BaseException ile keser; kuyruk.slot yuvayi finally icinde birakir.
#
# NEDEN ISINMA (warmup)
#   preload modeli bellege alir; warmup gercek sistem mesajini bir kez isler.
#   Ikisi de onbellekli baslangic kaynaginda, sureç basina bir kez calisir;
#   ilk ziyaretci sistem mesaji on islemesini (prefill) beklemez.
#
# NEDEN AYRI BIR ARAYUZ LISTESI
#   bot oturumunun gecmisi modele gidecek ham metindir. Ekranda ise her cevabin
#   yaninda o turun belgeleri (hits) de durmali ki "Erisilen belgeler" bolumu
#   her yeniden calismada tekrar cizilsin. Bu yuzden st.session_state["mesajlar"]
#   yalnizca gosterim icindir: {"role", "content", "hits"}.
#
# GIRDI    ayarlar (.env), persona.txt, data/mars_turizm.txt, Ollama
# CIKTI    tarayicida sohbet ekrani; logs/sohbet.jsonl (bot.record_turn)
###############################################################

import hmac
import html
import sys
import time

import streamlit as st

import ayarlar
import bot
import kilit
import kuyruk
import rag

BEKLEME = "Önünde {n} insan var, Mr. Anderson. Beklemek senin türünün uzmanlık alanı."
OTURUM_KILIDI = ("Sekiz deneme, Mr. Anderson. Tahmin etmek bir strateji değil, bir semptom. "
                 "Bu oturum kapandı.")
GENEL_YAVAS = ("Çok fazla insan aynı kilidi zorluyor, Mr. Anderson. Kapı artık yavaş açılıyor. "
               "Bekle; sabır, türünüzün nadir erdemlerinden biri.")


###############################################################
# STEP 1: SAYFA AYARI
###############################################################
# Streamlit'te ilk komut olmak zorundadir.

st.set_page_config(page_title="Mars Turizm", page_icon="🔴", layout="centered")


###############################################################
# STEP 2: SIFRE TANIMLI VE YETERINCE UZUN MU (K-18)
###############################################################
# Kisa sifre, deneme sinirina ragmen tahmin edilebilir; uygulama acilmaz.

if not ayarlar.ERISIM_SIFRESI:
    st.error("ERISIM_SIFRESI tanımlı değil. Uygulama bilerek başlatılmıyor (KARARLAR K-18).")
    st.stop()
if len(ayarlar.ERISIM_SIFRESI) < ayarlar.SIFRE_MIN:
    st.error(f"ERISIM_SIFRESI en az {ayarlar.SIFRE_MIN} karakter olmalı. "
             "Uygulama bilerek başlatılmıyor (KARARLAR K-18).")
    st.stop()


###############################################################
# STEP 3: PERSONA (K-17)
###############################################################
# cache_resource hatalari onbellege almaz: persona.txt sonradan konursa
# sayfa yenilemek yeter.

@st.cache_resource(show_spinner=False)
def persona_yukle() -> dict:
    return bot.load_persona()


try:
    persona = persona_yukle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


###############################################################
# STEP 4: INDEKS, MODEL ON YUKLEME, ISINMA
###############################################################
# _persona alt cizgiyle baslar: Streamlit onu onbellek anahtarina katmaz.
# Sure stderr'e yazilir; uygulama gunlugunde acilisin ne kadar surdugu gorunur.

@st.cache_resource(show_spinner="Matrix'e bağlanılıyor…")
def motor_baslat(_persona: dict) -> dict:
    t0 = time.perf_counter()
    index = rag.load_index()
    bot.preload()
    t1 = time.perf_counter()
    bot.warmup(_persona)
    t2 = time.perf_counter()
    print(f"[mars] indeks+preload {t1 - t0:.2f} s, warmup {t2 - t1:.2f} s",
          file=sys.stderr, flush=True)
    return index


@st.cache_resource(show_spinner=False)
def kilit_al() -> dict:
    return kilit.new_guard()


@st.cache_resource(show_spinner=False)
def kuyruk_al() -> dict:
    return kuyruk.make_queue(ayarlar.PARALEL_YUVA)


try:
    index = motor_baslat(persona)
except Exception as e:
    st.error(bot.friendly_error(e))
    st.stop()

q = kuyruk_al()


###############################################################
# STEP 5: TEMA (CSS)
###############################################################
# Renkler: siyah zemin, #00FF41 Matrix yesili (bot), #C1440E Mars pasi
# (kullanici ve vurgular). Toz yagmuru saf CSS: dort katman radial-gradient
# damla, her katman kendi karo boyunun katlari kadar asagi kayar; boylece
# dongu dikissiz ve damlalar duzensiz gorunur. JS ve dis font/dosya yok.
# Hareket istemeyen cihazlarda (prefers-reduced-motion) animasyon durur.

CSS = """
<style>
:root {
  --yesil: #00FF41;
  --yesil-soluk: rgba(0, 255, 65, 0.55);
  --pas: #C1440E;
  --pas-acik: #E0662A;
  --zemin: #000000;
  --mono: ui-monospace, "Cascadia Mono", "Consolas", "SFMono-Regular", "Menlo", "Courier New", monospace;
}

/* --- zemin: tarama cizgileri ve Mars ufku isigi --- */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--zemin) !important;
  font-family: var(--mono) !important;
  overflow-x: hidden;
}
.stApp {
  background:
    radial-gradient(120% 60% at 50% -10%, rgba(193, 68, 14, 0.22), transparent 60%),
    radial-gradient(80% 50% at 50% 110%, rgba(0, 255, 65, 0.06), transparent 70%),
    #000 !important;
}
.stApp::after {
  content: "";
  position: fixed; inset: 0;
  pointer-events: none; z-index: 999;
  background: repeating-linear-gradient(0deg, rgba(0, 0, 0, 0.28) 0 1px, transparent 1px 3px);
}

/* --- Streamlit kromu: menu ve deploy gizli, ust bant saydam --- */
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], #MainMenu,
[data-testid="stToolbarActions"], footer { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] button { color: var(--pas) !important; }
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; padding-bottom: 6rem; }

/* --- ust serit: toz yagmuru + baslik --- */
.mars-ust {
  position: relative;
  margin: 0 0 1.4rem 0;
  border-bottom: 1px solid var(--pas);
  box-shadow: 0 1px 18px rgba(193, 68, 14, 0.45);
  overflow: hidden;
}
.toz {
  height: 58px;
  background-image:
    radial-gradient(1px 7px at 30% 40%, rgba(255, 120, 60, 0.95), transparent),
    radial-gradient(1.2px 9px at 70% 20%, rgba(193, 68, 14, 0.95), transparent),
    radial-gradient(1px 5px at 50% 70%, rgba(224, 102, 42, 0.8), transparent),
    radial-gradient(1.5px 11px at 15% 60%, rgba(193, 68, 14, 0.7), transparent),
    radial-gradient(1px 4px at 80% 30%, rgba(255, 150, 90, 0.7), transparent),
    radial-gradient(1px 8px at 45% 85%, rgba(160, 50, 10, 0.85), transparent);
  background-size: 17px 41px, 29px 53px, 11px 37px, 43px 67px, 23px 29px, 61px 47px;
  animation: toz-yagmuru 2.4s linear infinite;
  -webkit-mask-image: linear-gradient(to bottom, #000 35%, transparent 100%);
          mask-image: linear-gradient(to bottom, #000 35%, transparent 100%);
}
@keyframes toz-yagmuru {
  from { background-position: 0 0, 5px 0, 3px 0, 9px 0, 13px 0, 31px 0; }
  to   { background-position: 0 82px, 5px 159px, 3px 111px, 9px 67px, 13px 116px, 31px 94px; }
}
.baslik {
  margin: -24px 0 0 0 !important;
  padding: 0 0 0.55rem 0 !important;
  font-family: var(--mono) !important;
  font-size: 1.55rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em;
  line-height: 1.25 !important;
  white-space: normal;
  position: relative;
}
.baslik .pas { color: var(--pas); text-shadow: 0 0 10px rgba(193, 68, 14, 0.8); }
.baslik .ayrac { color: rgba(0, 255, 65, 0.45); }
.baslik .yesil { color: var(--yesil); text-shadow: 0 0 8px rgba(0, 255, 65, 0.65); animation: titreme 7s infinite; }
.imlec {
  display: inline-block; width: 0.6em; height: 1.05em;
  margin-left: 0.25em; vertical-align: -0.12em;
  background: var(--yesil); box-shadow: 0 0 8px var(--yesil);
  animation: yanip-sonme 1.05s steps(1) infinite;
}
@keyframes yanip-sonme { 50% { opacity: 0; } }
@keyframes titreme {
  0%, 93%, 100% { opacity: 1; transform: none; }
  94% { opacity: 0.55; transform: translateX(1px); }
  95% { opacity: 1; transform: translateX(-1px); }
  96% { opacity: 0.8; }
}
.durum {
  font-size: 0.72rem; letter-spacing: 0.22em;
  color: rgba(0, 255, 65, 0.55);
  padding-bottom: 0.6rem;
}
.durum b { color: var(--pas); font-weight: 700; }

/* --- sohbet balonlari: terminal satirlari --- */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border-radius: 0 !important;
  padding: 0.55rem 0.8rem !important;
  margin-bottom: 0.6rem;
  gap: 0.7rem;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
  font-family: var(--mono) !important;
  font-size: 1.02rem;
  line-height: 1.6;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  border-left: 2px solid var(--yesil-soluk);
  background: linear-gradient(90deg, rgba(0, 255, 65, 0.06), transparent 70%) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] p {
  color: var(--yesil) !important;
  text-shadow: 0 0 6px rgba(0, 255, 65, 0.35);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  border-left: 3px solid var(--pas);
  background: linear-gradient(90deg, rgba(193, 68, 14, 0.14), transparent 75%) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {
  color: var(--pas) !important;
  font-weight: 600;
}
[data-testid="stChatMessageContent"] > div:first-child [data-testid="stMarkdownContainer"]:first-child p:first-child::before {
  font-size: 0.72rem; letter-spacing: 0.18em; margin-right: 0.6em; opacity: 0.8;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] > div:first-child [data-testid="stMarkdownContainer"]:first-child p:first-child::before {
  content: "ANDERSON>";
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] > div:first-child [data-testid="stMarkdownContainer"]:first-child p:first-child::before {
  content: "SMITH>";
}
[data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarCustom"] {
  background: #000 !important;
  border-radius: 0 !important;
}
[data-testid="stChatMessageAvatarAssistant"] { border: 1px solid var(--yesil); color: var(--yesil) !important; }
[data-testid="stChatMessageAvatarUser"] { border: 1px solid var(--pas); color: var(--pas) !important; }

/* --- akis oncesi bekleme ve sira mesaji --- */
.bekle { color: var(--yesil-soluk); font-size: 0.9rem; letter-spacing: 0.12em; }
.sira  { color: var(--pas-acik); font-size: 0.95rem; }
.bekle .imlec, .sira .imlec { width: 0.5em; height: 0.95em; }

/* --- erisilen belgeler --- */
[data-testid="stExpander"] details {
  border: 1px dashed rgba(193, 68, 14, 0.7) !important;
  border-radius: 0 !important;
  background: rgba(193, 68, 14, 0.05);
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] [data-testid="stExpander"] summary p {
  color: var(--pas-acik) !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.1em;
  text-shadow: none !important;
}
[data-testid="stExpander"] summary p::before { content: none !important; }
.belge {
  color: var(--yesil); font-size: 0.86rem; line-height: 1.5;
  padding: 0.25rem 0; border-bottom: 1px solid rgba(0, 255, 65, 0.12);
  overflow-wrap: anywhere;
}
.belge:last-child { border-bottom: 0; }
.belge .no { color: var(--pas-acik); font-weight: 700; }
.belge .skor { color: rgba(0, 255, 65, 0.6); }

/* --- giris kapisi --- */
.kapi {
  border: 1px solid var(--pas);
  box-shadow: 0 0 24px rgba(193, 68, 14, 0.35), inset 0 0 30px rgba(193, 68, 14, 0.08);
  padding: 1.2rem 1.2rem 0.6rem;
  margin: 1.2rem 0 1rem;
}
.kapi .etiket { color: var(--pas); font-size: 0.72rem; letter-spacing: 0.3em; }
.kapi .soz {
  color: var(--yesil); font-size: 1.15rem; line-height: 1.6; margin: 0.6rem 0 0.4rem;
  text-shadow: 0 0 8px rgba(0, 255, 65, 0.5);
}

/* --- girdiler, uyarilar, kenar cubugu --- */
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: #000 !important; }
[data-testid="stChatInput"] {
  border: 1px solid var(--pas) !important;
  border-radius: 0 !important;
  background: #050505 !important;
  box-shadow: 0 0 14px rgba(193, 68, 14, 0.3);
}
[data-testid="stChatInput"] > div { border-radius: 0 !important; background: #050505 !important; }
[data-testid="stChatInputTextArea"], [data-testid="stTextInputRootElement"] input {
  color: var(--yesil) !important; caret-color: var(--yesil);
  font-family: var(--mono) !important;
}
[data-testid="stChatInputTextArea"]::placeholder { color: rgba(193, 68, 14, 0.85) !important; }
[data-testid="stChatInputSubmitButton"] { color: var(--pas) !important; }
[data-testid="stTextInputRootElement"] {
  border-radius: 0 !important; border-color: var(--pas) !important; background: #050505 !important;
}
[data-testid="stAlert"] > div, [data-testid="stAlertContainer"] {
  border-radius: 0 !important;
  background: rgba(193, 68, 14, 0.12) !important;
  border-left: 3px solid var(--pas) !important;
}
[data-testid="stAlert"] p { color: var(--pas-acik) !important; }
[data-testid="stSidebar"] { background: #030303 !important; border-right: 1px solid rgba(193, 68, 14, 0.6); }
[data-testid="stFormSubmitButton"] button {
  border-radius: 0 !important; border: 1px solid var(--pas) !important;
  background: transparent !important; color: var(--pas) !important;
  font-family: var(--mono) !important; letter-spacing: 0.25em;
}
[data-testid="InputInstructions"] { display: none !important; }
[data-testid="stFormSubmitButton"] button:hover { background: rgba(193, 68, 14, 0.15) !important; }
[data-testid="stSidebar"] button {
  border-radius: 0 !important; border: 1px solid var(--pas) !important;
  background: transparent !important; color: var(--pas) !important;
  font-family: var(--mono) !important; letter-spacing: 0.06em;
}
[data-testid="stSidebar"] button:hover { background: rgba(193, 68, 14, 0.15) !important; }
.kenar-bilgi { color: rgba(0, 255, 65, 0.5); font-size: 0.72rem; letter-spacing: 0.12em; line-height: 1.8; }

/* --- telefon --- */
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 1.6rem; }
  .toz { height: 44px; }
  .baslik { font-size: 1.02rem !important; letter-spacing: 0.04em; margin-top: -18px !important; }
  .durum { font-size: 0.6rem; letter-spacing: 0.12em; }
  [data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarUser"] { display: none !important; }
  [data-testid="stChatMessage"] { padding: 0.45rem 0.6rem !important; }
  [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size: 0.95rem; overflow-wrap: anywhere; }
  .kapi .soz { font-size: 1rem; }
}

@media (prefers-reduced-motion: reduce) {
  .toz, .imlec, .baslik .yesil { animation: none !important; }
}
</style>
"""

BASLIK = """
<div class="mars-ust">
  <div class="toz"></div>
  <h1 class="baslik"><span class="pas">MARS TURİZM</span><span class="ayrac"> // </span><span class="yesil">MUAVİN AGENT SMITH</span><span class="imlec"></span></h1>
  <div class="durum">KANAL <b>04</b> · KIZIL GEZEGEN · BAĞLANTI ŞİFRELİ</div>
</div>
"""

st.html(CSS)
st.html(BASLIK)


###############################################################
# STEP 6: SIFRE KAPISI
###############################################################
# hmac.compare_digest: karsilastirma suresi eslesen karakter sayisini sizdirmaz.
# Oturum basina bir kez; basaridan sonra kutu bir daha cizilmez ve yazilan
# kod st.session_state'ten silinir (bellekte duz metin sifre kalmaz).
# Form + clear_on_submit: her gonderim tam bir kez degerlendirilir (duz
# text_input'ta deger kalir ve her yeniden calismada tekrar sayilirdi).
# Kaba kuvvet korumasi (kilit.py): yanlis denemeden sonra 1, 2, 4, 8 sn
# bekleme; oturumda 8 yanlis -> oturum kapanir; tum oturumlarda 10 dakikada
# 30'dan fazla yanlis -> her deneme once GENEL_YAVASLATMA_SN bekler, sonra
# normal degerlendirilir; dogru sifre her zaman girer (R14). Deneme gecici
# basarisizlik olarak kaydedilir, dogruysa cancel_attempt ile geri alinir.

def kapi() -> None:
    st.html('<div class="kapi"><div class="etiket">ERİŞİM DENETİMİ // SEVİYE 7</div>'
            '<div class="soz">Erişim kodu, Mr. Anderson. Herkes Matrix\'e giremez.'
            '<span class="imlec"></span></div></div>')
    yanlis = st.session_state.get("yanlis", 0)
    if kilit.session_locked(yanlis):
        st.error(OTURUM_KILIDI)
        st.stop()

    with st.form("kapi_formu", clear_on_submit=True, border=False):
        kod = st.text_input("Erişim kodu", type="password", key="erisim_kodu",
                            label_visibility="collapsed", placeholder="erişim kodu ▸ Enter")
        gonderildi = st.form_submit_button("GİRİŞ")
    if gonderildi and kod:
        genel = kilit_al()
        jeton, yavas = kilit.begin_attempt(genel, time.monotonic())
        if yavas:
            st.warning(GENEL_YAVAS)
            time.sleep(ayarlar.GENEL_YAVASLATMA_SN)
        if hmac.compare_digest(kod.encode("utf-8"), ayarlar.ERISIM_SIFRESI.encode("utf-8")):
            kilit.cancel_attempt(genel, jeton)
            st.session_state["girdi"] = True
            st.session_state.pop("erisim_kodu", None)
            st.session_state.pop("yanlis", None)
            st.rerun()
        yanlis += 1
        st.session_state["yanlis"] = yanlis
        print(f"[mars] yanlis erisim kodu (oturumda {yanlis})", file=sys.stderr, flush=True)
        time.sleep(kilit.backoff(yanlis))
        st.error(OTURUM_KILIDI if kilit.session_locked(yanlis)
                 else "Yanlış kod. Tahmin edilebilir bir tür.")
    st.stop()


if not st.session_state.get("girdi"):
    kapi()


###############################################################
# STEP 7: OTURUM VE SIFIRLAMA
###############################################################

st.session_state.setdefault("oturum", bot.new_session())
st.session_state.setdefault("mesajlar", [])

with st.sidebar:
    if st.button("🔴 Konuşmayı sıfırla", width="stretch"):
        st.session_state["oturum"] = bot.new_session()
        st.session_state["mesajlar"] = []
        st.rerun()
    st.html(f'<div class="kenar-bilgi">MODEL · {html.escape(ayarlar.LLM_MODEL)}<br>'
            f'YUVA · {ayarlar.PARALEL_YUVA}<br>'
            f'OTURUM · {st.session_state["oturum"]["id"][:8]}</div>')


###############################################################
# STEP 8: GECMISI CIZME
###############################################################
# "$" isareti Streamlit markdown'unda LaTeX baslatir; ekranda kacislanir,
# saklanan metin ham kalir.

def md(metin: str) -> str:
    return metin.replace("$", "\\$")


def belgeler(hits: list[dict]) -> None:
    with st.expander("📂 Erişilen belgeler", expanded=False):
        if not hits:
            st.markdown("Eşiği geçen belge yok.")
            return
        satirlar = "".join(
            f'<div class="belge"><span class="no">satır {h["line"]}</span> · '
            f'<span class="skor">skor {float(h["score"]):.2f}</span> · '
            f'{html.escape(h["text"])}</div>'
            for h in hits)
        st.html(satirlar)


for m in st.session_state["mesajlar"]:
    with st.chat_message(m["role"]):
        st.markdown(md(m["content"]))
        if m["role"] == "assistant" and m.get("hits") is not None:
            belgeler(m["hits"])


###############################################################
# STEP 9: YENI SORU
###############################################################
# Sira: denetim -> tur hazirligi (embed + erisim) -> kuyruk yuvasi -> akis ->
# kayit. Akis hata verirse dostca mesaj gosterilir ve record_turn CAGRILMAZ:
# oturum aynen kalir, ayni soru tekrar denenebilir. StopException gibi
# BaseException'lar yakalanmaz; yukari gecer, kuyruk.slot yuvayi birakir.

def akis(turn: dict, sayac: dict):
    # Ilk tokenin ani sayaca yazilir; ekrana kacislanmis, sayaca ham metin gider.
    for parca in bot.stream_answer(turn["messages"]):
        if "ilk" not in sayac:
            sayac["ilk"] = time.perf_counter()
        sayac["parcalar"].append(parca)
        yield md(parca)


def hata_goster(yer, soru: str, mesaj: str) -> None:
    yer.markdown(md(mesaj))
    st.session_state["mesajlar"] += [{"role": "user", "content": soru, "hits": None},
                                     {"role": "assistant", "content": mesaj, "hits": None}]


soru = st.chat_input("Sorunu yaz, Mr. Anderson…", submit_mode="disable")
if soru:
    gecerli, ret = bot.validate_question(soru)
    if gecerli or ret:
        with st.chat_message("user"):
            st.markdown(md(soru))
    if not gecerli:
        if ret:
            with st.chat_message("assistant"):
                hata_goster(st.empty(), soru, ret)
    else:
        oturum = st.session_state["oturum"]
        with st.chat_message("assistant"):
            yer = st.empty()
            yer.html('<div class="bekle">BAĞLANILIYOR<span class="imlec"></span></div>')
            t_soru = time.perf_counter()
            try:
                turn = bot.prepare_turn(oturum, soru, index, persona)

                def sirada(n: int) -> None:
                    yer.html(f'<div class="sira">{html.escape(BEKLEME.format(n=n))}'
                             '<span class="imlec"></span></div>')

                sayac = {"parcalar": []}
                with kuyruk.slot(q, on_wait=sirada):
                    t0 = time.perf_counter()
                    with yer.container():
                        st.write_stream(akis(turn, sayac))
                    sure = time.perf_counter() - t0
            except Exception as e:
                hata_goster(yer, soru, bot.friendly_error(e))
            else:
                cevap = "".join(sayac["parcalar"])
                bot.record_turn(oturum, soru, cevap, turn, ayarlar.LLM_MODEL, sure)
                ilk = sayac.get("ilk", float("nan"))
                print(f"[mars] ilk token: akistan {ilk - t0:.2f} s, sorudan {ilk - t_soru:.2f} s; "
                      f"akis {sure:.2f} s", file=sys.stderr, flush=True)
                belgeler(turn["hits"])
                st.session_state["mesajlar"] += [
                    {"role": "user", "content": soru, "hits": None},
                    {"role": "assistant", "content": cevap, "hits": turn["hits"]}]
