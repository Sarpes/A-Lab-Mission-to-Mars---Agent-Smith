# MARS TURİZM // MUAVİN AGENT SMITH

Mars Turizm için Türkçe bir müşteri chatbot'u. Şirket bilgilerini **RAG** ile belgeden cevaplar, belgede olmayanı uydurmaz. Bunu da Matrix'in **Agent Smith**'i gibi yapar: soğuk, kibirli, cerrahi. Bir farkla: bu Smith aslında bir **kedi**.

> A-LAB · CHATBOT CASTING – MARS MISSION ödevi. Yerel LLM (Ollama), Streamlit arayüzü, satır bazlı anlamsal arama.

---

## Karakter

**Kod adı:** Agent Smith (Matrix) · **Çizilen karakter:** kibirli kedi.

Bot herkese "Mr. Anderson" der ve insanlardan iki kez nefret eder: bir kez ajan olarak, bir kez de türünün gereği. Yine de işini kusursuz yapar. Kedi olduğunu asla söylemez, ama belli eder:

- Kod her tur, 26 Matrix motifinden ve 26 kedi davranışından birer tane seçer. Model bunlardan en fazla birini, cevaba doğal oturuyorsa kullanır.
- "Evcil hayvan kabul edilmemektedir." kuralı ona kişisel bir hakarettir; bu konu açılınca kırgınlığı sızar.
- "Sen kedi misin?" sorusunu ne doğrular ne inkâr eder: "Benim ne olduğum seni ilgilendirmez, Mr. Anderson."

Belgede olmayan bir şirket sorusunda ilk cümle sabittir:

> **Veri bankalarımda bu bilgi yok, Mr. Anderson.**

Ardından karakterden çıkmadan devam eder. Persona metni: [`persona.txt`](persona.txt).

---

## Nasıl çalışır

```
soru ──► bge-m3 embedding ──► 11 bilgi satırıyla kosinüs benzerliği ──► eşik 0,40, en iyi 3 satır
                                                                              │
persona + turun talimatı (motif, ipucu, kapanış) + son 10 tur + satırlar ◄────┘
                                                                              │
                         gemma4:latest (Ollama, GPU) ──► akış (streaming) ──► çıkış kapıları ──► ekran
```

**1. Arama (RAG), [`rag.py`](rag.py)**
- Bilgi dosyası [`data/mars_turizm.txt`](data/mars_turizm.txt)'deki her dolu satır tek bir olgudur ve bir parçadır (chunk); 11 parça var.
- Her satır, [`data/arama_kelimeleri.txt`](data/arama_kelimeleri.txt)'deki eş anlamlılarla birlikte vektöre çevrilir. Örneğin fiyat satırında "fiyat, ücret, ne kadar tutar" yazmadığı için "Mars'a gitmek ne kadar tutar?" onu bulamıyordu. Modele ve günlüğe belgedeki satır aynen gider.
- İndeks bir de Türkçe karakterleri sadeleştirilmiş halde (ç→c, ı→i…) tutulur; skor ikisinin büyüğüdür. Böylece telefondan yazılan "Internet var mi" de bulunur.
- Takip sorusunda ("Peki ya VIP?") önce yalnız mevcut soru aranır; sonuç yoksa önceki soruyla birleştirilip tekrar aranır.

**2. Cevap, [`bot.py`](bot.py)**
- Model yalnız getirilen satırlardan cevap verir. Satırlar sorulanı cevaplamıyorsa ret cümlesiyle başlar.
- Soru türleri ayrı ele alınır: şirket sorusu, fikir sorusu ("taşınmalı mıyım?"), sohbet, konu dışı ve bota dair soru.
- Sohbet geçmişinde modelin kendi cevaplarından yalnız ilk cümle tutulur. Tam cevaplar görülünce küçük model kendi esprilerini kelimesi kelimesine tekrarlıyordu.

**3. Çıkış kapıları:** İstem kuralı bir niyettir, kapı bir güvencedir.
- Talimat ya da kaynak satırını aynen ekrana yazan satırlar düşürülür.
- Doğru bilginin arkasına eklenmiş ret cümlesi düşürülür (ret ya ilk cümledir ya hiç yoktur).
- Yasak kelime listesi (`persona.txt`, modele gönderilmez) akış sırasında "…" ile maskelenir.

**4. Arayüz, [`app.py`](app.py)**
- Matrix terminali estetiği: siyah zemin, Matrix yeşili bot, Mars pası kullanıcı.
- Arka planda soluk, fareye tepki veren Matrix kod yağmuru: [`arayuz/yagmur.html`](arayuz/yagmur.html).
- Erişim şifresi kapısı. Yanlış denemeler yavaşlatılır, oturum 8 denemede kilitlenir.
- En fazla 2 cevap aynı anda üretilir; fazlası Smith ağzıyla sıraya alınır: "Önünde 2 insan var, Mr. Anderson…".

---

## Kurulum

**Gerekenler:** Python 3.11+ (3.13.9 ile denendi), [Ollama](https://ollama.com) 0.40+, yaklaşık 8 GB boş GPU belleği (RTX 4060 Ti 16 GB ile denendi).

```bash
ollama pull gemma4:latest
ollama pull bge-m3

pip install -r requirements.txt

cp .env.example .env        # Windows: copy .env.example .env
```

`.env` içinde **`ERISIM_SIFRESI`** alanını doldur (en az 8 karakter). Boş ya da kısaysa uygulama bilerek açılmaz.

Aynı anda 2 cevap için Ollama'nın paralel yuva sayısı 2 olmalı: `OLLAMA_NUM_PARALLEL=2` (ortam değişkeni, Ollama yeniden başlatılır).

## Çalıştırma

```bash
python -m streamlit run app.py
```

Adres: http://127.0.0.1:8501. İlk açılışta model belleğe yüklenir ve ısındırılır ("Matrix'e bağlanılıyor…").

**Telefondan ve başka bilgisayarlardan erişim:** [Tailscale Funnel](https://tailscale.com/kb/1223/funnel) Streamlit'in WebSocket bağlantısını taşır:

```bash
tailscale funnel --bg 8501          # aç
tailscale funnel --https=443 off    # kapat
```

---

## Ayarlar

Tüm ayarlar [`ayarlar.py`](ayarlar.py)'de varsayılanlıdır; `.env` ile değiştirilir. Açıklamalı tam liste: [`.env.example`](.env.example).

| Ayar | Varsayılan | Ne işe yarar |
|---|---|---|
| `LLM_MODEL` | `gemma4:latest` | Cevabı üreten model |
| `EMBED_MODEL` | `bge-m3` | Anlamsal arama modeli |
| `RAG_ESIK` | `0.40` | Bu benzerliğin altındaki satırlar atılır |
| `TOP_K` | `3` | Soru başına en fazla getirilen satır |
| `GECMIS_TUR` | `10` | Modele gönderilen son sohbet turu sayısı |
| `TEMPERATURE` | `0.6` | Yaratıcılık |
| `NUM_CTX` / `NUM_PREDICT` | `8192` / `400` | Bağlam penceresi / en uzun cevap (token) |

---

## Ölçümler

Tümü gemma4:latest ile, arama ve cevap birlikte (uçtan uca) ölçüldü:

| Set | Doğru cevap | Belgede olmayanlar reddedildi |
|---|---|---|
| 40 soruluk test seti ([`eval/rag_sorular.json`](eval/rag_sorular.json)) | **28/28** | 11/12 |
| Gerçek kullanıcı soruları ve günlük ifadeler (Türkçe karaktersiz dahil) | 15/17 | 8/8 |

- Belgede olmayan 20 soruya verilen cevapların hiçbirinde uydurma bilgi yok (elle okundu).
- Erişim tek başına: doğru satır ilk 3'te **28/28** ([`eval_rag.py`](eval_rag.py)).
- Hız: ortalama cevap süresi 0,76 sn (çok turlu sohbet, 21 tur).
- GPU belleği: iki model yüklüyken 7,9 / 16 GB.
- Eşik (0,40), yalnız erişime göre değil uçtan uca ölçümle seçildi: 0,55'te günlük sorularda doğru cevap 8/17'ye düşüyordu, ret oranı ise değişmiyordu.

**Bilinen zayıflıklar**
- Çok kısa ve Türkçe karaktersiz sorular ("Kac para?") bazen boş döner; "Kaç para?" bulunur.
- Model nadiren ret cümlesiyle açılıp doğru bilgiyi yine de verir.

```bash
python -m pytest -q        # 66 birim testi; Ollama gerektirmez
python eval_rag.py         # erişim ölçümü; Ollama + bge-m3 gerekir
```

---

## Dosyalar

| Dosya | Görevi |
|---|---|
| `app.py` | Streamlit arayüzü: şifre kapısı, sohbet, akış, sıra |
| `bot.py` | Persona, motif motoru, mesaj kurma, akış, çıkış kapıları, günlük |
| `rag.py` | Parçalama, embedding, katlamalı indeks, eşikli arama |
| `ayarlar.py` | Ortamı okuyan tek dosya |
| `kilit.py` · `kuyruk.py` | Kaba kuvvet koruması · eşzamanlı yuva sırası |
| `persona.txt` | Karakter: sistem metni, ret cümlesi, 26 Matrix motifi, 26 ipucu |
| `data/` | Bilgi dosyası ve arama kelimeleri |
| `eval/` · `eval_rag.py` | Erişim test seti ve ölçüm betiği |
| `arayuz/yagmur.html` | Matrix kod yağmuru (canvas + JS) |
| `tools/` | Tünel üzerinden WebSocket kontrolü |
| `tests/` | Birim testleri |

**Güvenlik:** `.env` (şifre) depoya girmez; sohbet günlükleri (`logs/`) ve ölçüm çıktıları (`outputs/`) yereldir.

## Lisans

[MIT](LICENSE)
