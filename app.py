import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime
import os

st.set_page_config(page_title="BİST100 Yapay Zeka Radarı", layout="wide")

st.title("📈 BİST100 Derin Öğrenme Fiyat Projeksiyonu")
st.markdown("""
**Proje Felsefesi:** Bu sistemdeki her hisse, kendi geçmiş verileriyle eğitilmiş bağımsız LSTM yapay sinir ağları tarafından analiz edilir.
**Yasal Sınır:** Tüm tahminler Borsa İstanbul'un günlük **±%10 devre kesici** limitlerine göre kısıtlanmıştır. *(Yatırım tavsiyesi değildir.)*
""")

hazir_modeller = [f.replace("_model.h5", "") for f in os.listdir("src/models") if f.endswith(".h5")]
hazir_modeller.sort()

if not hazir_modeller:
    st.warning("Lütfen önce arka planda eğitim kodunu çalıştırın.")
else:
    sekme1, sekme2 = st.tabs(["📊 Tüm Piyasa Radarı (Gün Gün Analiz)", "🎯 Bireysel Hisse Analizi"])

    # ---------------- SEKME 1: TOPLU TARAMA ----------------
    with sekme1:
        st.subheader(f"Toplu Hisse Tarayıcı ({len(hazir_modeller)} Yapay Zeka Modeli)")
        st.markdown("Arka planda eğitilen modeller kullanılarak ilk **3 günün** fiyat projeksiyonu ve **bir önceki güne göre** zincirleme yüzdelik değişimleri hesaplanır.")

        if st.button("Piyasayı Tarat (Tahminleri Hesapla)"):
            progress_bar = st.progress(0)
            durum_metni = st.empty()

            sonuclar = []

            for i, hisse in enumerate(hazir_modeller):
                durum_metni.text(f"Yapay Zeka Analiz Ediyor: {hisse} ({i+1}/{len(hazir_modeller)})")
                try:
                    df = yf.download(hisse, period="1y", interval="1d", progress=False)
                    kapanis_ham = df['Close']
                    if isinstance(kapanis_ham, pd.DataFrame):
                        kapanis_ham = kapanis_ham.squeeze()

                    kapanis = pd.to_numeric(kapanis_ham, errors='coerce').dropna()

                    if len(kapanis) >= 60:
                        fiyatlar = kapanis.values.reshape(-1, 1)
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        olcekli_veri = scaler.fit_transform(fiyatlar)
                        son_60 = olcekli_veri[-60:].reshape(1, 60, 1)

                        model_yolu = f'src/models/{hisse}_model.h5'
                        model = load_model(model_yolu)

                        tahmin_olcekli = model.predict(son_60, verbose=0)
                        ham_tahmin_tl = scaler.inverse_transform(tahmin_olcekli.reshape(-1, 1)).flatten()

                        # Devre kesici hesabı
                        limitli_tahmin_tl = []
                        referans_fiyat = fiyatlar[-1][0]
                        for ham_tahmin in ham_tahmin_tl:
                            tavan = referans_fiyat * 1.10
                            taban = referans_fiyat * 0.90
                            kesilmis = float(np.clip(ham_tahmin, taban, tavan))
                            limitli_tahmin_tl.append(kesilmis)
                            referans_fiyat = kesilmis

                        # --- YENİ ZİNCİRLEME MATEMATİK ---
                        suanki_fiyat = fiyatlar[-1][0]
                        gun1_fiyat = limitli_tahmin_tl[0] # Yarın
                        gun2_fiyat = limitli_tahmin_tl[1] # Sonraki Gün
                        gun3_fiyat = limitli_tahmin_tl[2] # 3. Gün

                        # Her günün değişimi BİR ÖNCEKİ GÜNE göre hesaplanıyor
                        gun1_degisim = ((gun1_fiyat - suanki_fiyat) / suanki_fiyat) * 100
                        gun2_degisim = ((gun2_fiyat - gun1_fiyat) / gun1_fiyat) * 100
                        gun3_degisim = ((gun3_fiyat - gun2_fiyat) / gun2_fiyat) * 100

                        # Toplam sıralama için 3 günlük bileşik getiri/götürü (Sadece listeyi sıralamak için kullanacağız)
                        total_degisim = ((gun3_fiyat - suanki_fiyat) / suanki_fiyat) * 100

                        sonuclar.append({
                            "Hisse": hisse,
                            "Mevcut": suanki_fiyat,
                            "1. Gün": gun1_fiyat,
                            "1. Gün %": gun1_degisim,
                            "2. Gün": gun2_fiyat,
                            "2. Gün %": gun2_degisim,
                            "3. Gün": gun3_fiyat,
                            "3. Gün %": gun3_degisim,
                            "Siralama_Skoru": total_degisim
                        })

                        tf.keras.backend.clear_session()

                except Exception as e:
                    pass

                progress_bar.progress((i + 1) / len(hazir_modeller))

            durum_metni.success("✅ Tüm piyasa taraması tamamlandı!")

            if sonuclar:
                df_sonuc = pd.DataFrame(sonuclar)

                # Tabloyu en kârlı olandan en zararlı olana doğru mantıklı bir şekilde sıralıyoruz
                df_sonuc = df_sonuc.sort_values(by="Siralama_Skoru", ascending=False).reset_index(drop=True)

                # Devasa 3 Günlük Analiz Tablosu (Markdown)
                md_tablo = "| Hisse | Mevcut | 1. Gün (Yarın) | 1. Gün İvmesi | 2. Gün | 2. Gün İvmesi | 3. Gün | 3. Gün İvmesi |\n"
                md_tablo += "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"

                for index, row in df_sonuc.iterrows():
                    # Her gün için dinamik ok işaretleri oluşturuyoruz
                    ok1 = "🟢" if row['1. Gün %'] > 0 else "🔴" if row['1. Gün %'] < 0 else "⚪"
                    ok2 = "🟢" if row['2. Gün %'] > 0 else "🔴" if row['2. Gün %'] < 0 else "⚪"
                    ok3 = "🟢" if row['3. Gün %'] > 0 else "🔴" if row['3. Gün %'] < 0 else "⚪"

                    md_tablo += (
                        f"| **{row['Hisse']}** "
                        f"| {row['Mevcut']:.2f} ₺ "
                        f"| {row['1. Gün']:.2f} ₺ | {ok1} {row['1. Gün %']:.2f} % "
                        f"| {row['2. Gün']:.2f} ₺ | {ok2} {row['2. Gün %']:.2f} % "
                        f"| {row['3. Gün']:.2f} ₺ | {ok3} {row['3. Gün %']:.2f} % |\n"
                    )

                st.markdown(md_tablo)

    # ---------------- SEKME 2: BİREYSEL ANALİZ (Değişmedi) ----------------
    with sekme2:
        st.subheader("Tekil Hisse Projeksiyonu")
        hisse_secim = st.selectbox("Grafik Analizi İçin Hisse Seçin:", hazir_modeller)

        if st.button("Hisseye Özel Analizi Çalıştır"):
            with st.spinner(f"{hisse_secim} verileri çekiliyor..."):
                try:
                    df = yf.download(hisse_secim, period="1y", interval="1d", progress=False)
                    kapanis_ham = df['Close']
                    if isinstance(kapanis_ham, pd.DataFrame):
                        kapanis_ham = kapanis_ham.squeeze()

                    kapanis = pd.to_numeric(kapanis_ham, errors='coerce').dropna()
                    fiyatlar = kapanis.values.reshape(-1, 1)

                    scaler = MinMaxScaler(feature_range=(0, 1))
                    olcekli_veri = scaler.fit_transform(fiyatlar)
                    son_60 = olcekli_veri[-60:].reshape(1, 60, 1)

                    model = load_model(f'src/models/{hisse_secim}_model.h5')
                    tahmin_olcekli = model.predict(son_60, verbose=0)
                    ham_tahmin_tl = scaler.inverse_transform(tahmin_olcekli.reshape(-1, 1)).flatten()

                    limitli_tahmin_tl = []
                    taban_listesi = []
                    tavan_listesi = []
                    referans_fiyat = fiyatlar[-1][0]

                    for ham_tahmin in ham_tahmin_tl:
                        tavan_fiyat = referans_fiyat * 1.10
                        taban_fiyat = referans_fiyat * 0.90
                        kesilmis_fiyat = float(np.clip(ham_tahmin, taban_fiyat, tavan_fiyat))

                        limitli_tahmin_tl.append(kesilmis_fiyat)
                        taban_listesi.append(taban_fiyat)
                        tavan_listesi.append(tavan_fiyat)
                        referans_fiyat = kesilmis_fiyat

                    son_tarih = kapanis.index[-1]
                    tahmin_tarihleri = [son_tarih + datetime.timedelta(days=i) for i in range(1, 8)]

                    st.markdown("---")
                    sol_sutun, sag_sutun = st.columns([2, 3])

                    with sol_sutun:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.plot(kapanis.index[-30:], fiyatlar[-30:], label='Son 30 Gün', marker='o', linewidth=2)
                        ax.plot(tahmin_tarihleri, limitli_tahmin_tl, label='Yasal Tahmin', color='green', marker='x', linewidth=2)
                        ax.set_title(f"{hisse_secim} - Fiyat Projeksiyonu")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig, use_container_width=True)

                    with sag_sutun:
                        st.markdown(f"**Son İşlem Günü Fiyatı (Referans):** {fiyatlar[-1][0]:.2f} ₺")

                        md_tablo = "| Tarih | Yasal Taban (-10%) | Yasal Tavan (+10%) | 🤖 Modelin Tahmini |\n|:---|:---:|:---:|:---:|\n"
                        for tarih, taban, tavan, fiyat in zip(tahmin_tarihleri, taban_listesi, tavan_listesi, limitli_tahmin_tl):
                            md_tablo += f"| {tarih.strftime('%d.%m.%Y')} | {taban:.2f} ₺ | {tavan:.2f} ₺ | **{fiyat:.2f} ₺** |\n"

                        st.markdown(md_tablo)

                except Exception as e:
                    st.error(f"Sistem Hatası: {e}")
