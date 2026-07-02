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
    sekme1, sekme2 = st.tabs(["📊 Tüm Piyasa Radarı (1 Haftalık Analiz)", "🎯 Bireysel Hisse Analizi"])

    # ---------------- SEKME 1: TOPLU TARAMA (BUTON EKLENDİ) ----------------
    with sekme1:
        st.subheader(f"Toplu Hisse Tarayıcı ({len(hazir_modeller)} Yapay Zeka Modeli)")
        st.markdown("Arka planda eğitilen modeller kullanılarak **5 işlem gününün (1 Hafta)** fiyat projeksiyonu ve zincirleme yüzdelik değişimleri hesaplanır.")
        
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

                        limitli_tahmin_tl = []
                        referans_fiyat = fiyatlar[-1][0] 
                        for ham_tahmin in ham_tahmin_tl:
                            tavan = referans_fiyat * 1.10
                            taban = referans_fiyat * 0.90
                            kesilmis = float(np.clip(ham_tahmin, taban, tavan))
                            limitli_tahmin_tl.append(kesilmis)
                            referans_fiyat = kesilmis 
                        
                        suanki_fiyat = fiyatlar[-1][0]
                        gun1_fiyat = limitli_tahmin_tl[0] 
                        gun2_fiyat = limitli_tahmin_tl[1] 
                        gun3_fiyat = limitli_tahmin_tl[2] 
                        gun4_fiyat = limitli_tahmin_tl[3] 
                        gun5_fiyat = limitli_tahmin_tl[4] 
                        
                        gun1_degisim = ((gun1_fiyat - suanki_fiyat) / suanki_fiyat) * 100
                        gun2_degisim = ((gun2_fiyat - gun1_fiyat) / gun1_fiyat) * 100
                        gun3_degisim = ((gun3_fiyat - gun2_fiyat) / gun2_fiyat) * 100
                        gun4_degisim = ((gun4_fiyat - gun3_fiyat) / gun3_fiyat) * 100
                        gun5_degisim = ((gun5_fiyat - gun4_fiyat) / gun4_fiyat) * 100
                        
                        total_degisim = ((gun5_fiyat - suanki_fiyat) / suanki_fiyat) * 100
                        
                        sonuclar.append({
                            "Hisse Kodu": hisse, "Mevcut Fiyat": round(suanki_fiyat, 2),
                            "1. Gün Fiyat": round(gun1_fiyat, 2), "1. Gün %": round(gun1_degisim, 2),
                            "2. Gün Fiyat": round(gun2_fiyat, 2), "2. Gün %": round(gun2_degisim, 2),
                            "3. Gün Fiyat": round(gun3_fiyat, 2), "3. Gün %": round(gun3_degisim, 2),
                            "4. Gün Fiyat": round(gun4_fiyat, 2), "4. Gün %": round(gun4_degisim, 2),
                            "5. Gün Fiyat": round(gun5_fiyat, 2), "5. Gün %": round(gun5_degisim, 2),
                            "Siralama_Skoru": total_degisim
                        })
                        
                        tf.keras.backend.clear_session()
                
                except Exception as e:
                    pass 
                
                progress_bar.progress((i + 1) / len(hazir_modeller))
            
            durum_metni.success("✅ Tüm piyasa taraması tamamlandı!")
            
            if sonuclar:
                df_sonuc = pd.DataFrame(sonuclar)
                df_sonuc = df_sonuc.sort_values(by="Siralama_Skoru", ascending=False).reset_index(drop=True)
                
                # --- İNDİRME BUTONU BURADA ---
                # Türkçe karakterlerin Excel'de bozulmaması için 'utf-8-sig' kullanıyoruz
                csv_data = df_sonuc.drop(columns=['Siralama_Skoru']).to_csv(index=False).encode('utf-8-sig')
                
                st.markdown("<br>", unsafe_allow_html=True) # Araya biraz boşluk
                st.download_button(
                    label="📥 Tüm Sonuçları Excel / CSV Olarak İndir",
                    data=csv_data,
                    file_name=f"BIST100_Haftalik_Radar_{datetime.date.today().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv",
                    help="Bu dosyayı indirip doğrudan Excel'de açabilirsiniz."
                )
                st.markdown("---")
                # ------------------------------

                md_tablo = "| Hisse | Mevcut Fiyat | 1. Gün | 2. Gün | 3. Gün | 4. Gün | 5. Gün |\n"
                md_tablo += "|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n"
                
                for index, row in df_sonuc.iterrows():
                    ok1 = "🟢" if row['1. Gün %'] > 0 else "🔴" if row['1. Gün %'] < 0 else "⚪"
                    ok2 = "🟢" if row['2. Gün %'] > 0 else "🔴" if row['2. Gün %'] < 0 else "⚪"
                    ok3 = "🟢" if row['3. Gün %'] > 0 else "🔴" if row['3. Gün %'] < 0 else "⚪"
                    ok4 = "🟢" if row['4. Gün %'] > 0 else "🔴" if row['4. Gün %'] < 0 else "⚪"
                    ok5 = "🟢" if row['5. Gün %'] > 0 else "🔴" if row['5. Gün %'] < 0 else "⚪"
                    
                    md_tablo += (
                        f"| **{row['Hisse Kodu']}** "
                        f"| {row['Mevcut Fiyat']:.2f} ₺ "
                        f"| {ok1} %{row['1. Gün %']:.2f} ➔ {row['1. Gün Fiyat']:.2f} ₺ "
                        f"| {ok2} %{row['2. Gün %']:.2f} ➔ {row['2. Gün Fiyat']:.2f} ₺ "
                        f"| {ok3} %{row['3. Gün %']:.2f} ➔ {row['3. Gün Fiyat']:.2f} ₺ "
                        f"| {ok4} %{row['4. Gün %']:.2f} ➔ {row['4. Gün Fiyat']:.2f} ₺ "
                        f"| {ok5} %{row['5. Gün %']:.2f} ➔ {row['5. Gün Fiyat']:.2f} ₺ |\n"
                    )
                
                st.markdown(md_tablo)

    # ---------------- SEKME 2: BİREYSEL ANALİZ (Aynı Kaldı) ----------------
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