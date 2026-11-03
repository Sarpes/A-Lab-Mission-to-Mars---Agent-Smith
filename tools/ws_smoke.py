#####################################################################
# ws_smoke.py - Funnel uzerinden Streamlit WebSocket duman testi
#
# NE: Basligi "WS TEST", bir "Tikla" dugmesi ve session_state sayaci olan sayfa.
# NEDEN: Sayac yalnizca tarayici <-> sunucu WebSocket baglantisi calisirsa
#        artar. Funnel arkasinda WebSocket engelleniyorsa sayfa acilir ama
#        sayac kipirdamaz (karar K-11'in en riskli varsayimi).
#####################################################################
import streamlit as st

st.title("WS TEST")

# Sayac oturum durumunda tutulur; her tiklama sunucuya WebSocket ile gider
if "sayac" not in st.session_state:
    st.session_state.sayac = 0

if st.button("Tıkla"):
    st.session_state.sayac += 1

st.write(f"Sayaç: {st.session_state.sayac}")
