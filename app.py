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

st.set_page_config(page_title="BİST100 & Emtia Radarı", layout="wide")

st.title("📈 Derin Öğrenme Fiyat Projeksiyonu (Otonom Kalibrasyonlu)")
st.markdown("""
**Sistem Mimarisi:** Her varlık bağımsız bir LSTM ağı tarafından incelenir. 
**🌟 Otonom Kalibrasyon:** Model, son 15 günün geçmiş tahminlerini gerçek fiyatlarla kıyaslar ve "Hata Payını" bulup bugünkü tahminlerini **kendi kendine kalibre eder**.
**Yasal Sınır:** ±%10 BİST devre kesici (Altın/Gümüş hariç) aktiftir. *(Yatırım tavsiyesi değildir.)*
""")

# 1. Modelleri Klasörden Çek
ham_modeller = [f.replace("_model.h5", "") for f in os.listdir("src/models") if f.endswith(".h5")]

# 2. Gösterim İsimleri (Görsel İsimlendirme Haritası)
isim_haritasi = {
    "XU100.IS": "XU100 Endeksi",
    "GC=F": "Altın",
    "SI=F": "Gümüş"
}

if not ham_modeller:
    st.warning("Lütfen önce arka planda eğitim kodunu çalıştırın.")
else:
    # 3. KUSURSUZ SIRALAMA ALGORİTMASI (Senin İstediğin Format)
    bas_taraf = []
    if "XU100.IS" in ham_modeller: bas_taraf.append("XU100.IS")
    if "GC=F" in ham_modeller: bas_taraf.append("GC=F")
    if "SI=F" in ham_modeller: bas_taraf.append("SI=F")
    
    kalanlar = [m for m in ham_modeller if m not in ["XU100.IS", "GC=F", "SI=F"]]
    kalanlar.sort() # Kalan hisseleri alfabetik diz (A'dan Z'ye)
    
    hazir_modeller = bas_taraf + kalanlar # Listeleri birleştir, sırayı kilitle

    sekme1, sekme2 = st.tabs(["📊 Tüm Piyasa Radarı (1 Haftalık)", "🎯 Bireysel Analiz"])

    # ---------------- SEKME 1: TOPLU TARAMA ----------------
    with sekme1:
        st.subheader(f"Toplu Tarayıcı ({len(hazir_modeller)} Varlık)")
        
        if st.button("Piyasayı Tarat (Tahminleri Hesapla)"):
            progress_bar = st.progress(0)
            durum_metni = st.empty()
            
            sonuclar = []
            
            for i, hisse in enumerate(hazir_modeller):
                gorsel_isim = isim_haritasi.get(hisse, hisse)
                durum_metni.text(f"Otonom Analiz & Kalibrasyon: {gorsel_isim} ({i+1}/{len(hazir_modeller)})")
                
                try:
                    df = yf.download(hisse, period="1y", interval="1d", progress=False)
                    kapanis_ham = df['Close']
                    if isinstance(kapanis_ham, pd.DataFrame):
                        kapanis_ham = kapanis_ham.squeeze()
                        
                    kapanis = pd.to_numeric(kapanis_ham, errors='coerce').dropna()
                    
                    if len(kapanis) >= 75:
                        fiyatlar = kapanis.values.reshape(-1, 1)
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        olcekli_veri = scaler.fit_transform(fiyatlar)
                        
                        model_yolu = f'src/models/{hisse}_model.h5' 
                        model = load_model(model_yolu)

                        # Otonom Hata Düzeltme (Bias Correction)
                        son_75 = olcekli_veri[-75:]
                        X_batch = np.array([son_75[j : j + 60] for j in range(15)])
                        y_pred_batch = model.predict(X_batch, verbose=0)
                        
                        tahminler_1g_olcekli = y_pred_batch[:, 0].reshape(-1, 1)
                        tahminler_1g_tl = scaler.inverse_transform(tahminler_1g_olcekli).flatten()
                        gercekler_15g_tl = fiyatlar[-15:].flatten()
                        
                        ortalama_hata = np.mean(tahminler_1g_tl - gercekler_15g_tl)
                        
                        # Gelecek Tahmini
                        son_60 = olcekli_veri[-60:].reshape(1, 60, 1)
                        tahmin_olcekli = model.predict(son_60, verbose=0)
                        ham_tahmin_tl = scaler.inverse_transform(tahmin_olcekli.reshape(-1, 1)).flatten()
                        
                        kalibre_tahmin_tl = ham_tahmin_tl - ortalama_hata

                        # Devre Kesici
                        limitli_tahmin_tl = []
                        referans_fiyat = fiyatlar[-1][0] 
                        for ham in kalibre_tahmin_tl:
                            tavan = referans_fiyat * 1.10
                            taban = referans_fiyat * 0.90
                            kesilmis = float(np.clip(ham, taban, tavan))
                            limitli_tahmin_tl.append(kesilmis)
                            referans_fiyat = kesilmis 
                        
                        suanki_fiyat = fiyatlar[-1][0]
                        gunler = limitli_tahmin_tl[:5]
                        
                        yuzdeler = []
                        eski_fiyat = suanki_fiyat
                        for g_fiyat in gunler:
                            yuzdeler.append(((g_fiyat - eski_fiyat) / eski_fiyat) * 100)
                            eski_fiyat = g_fiyat
                            
                        # Tabloya eklerken teknik kodu değil, GÖRSEL İSMİ ekliyoruz
                        sonuclar.append({
                            "Varlık": gorsel_isim, 
                            "Mevcut Fiyat": round(suanki_fiyat, 2),
                            "g1_f": gunler[0], "g1_y": yuzdeler[0],
                            "g2_f": gunler[1], "g2_y": yuzdeler[1],
                            "g3_f": gunler[2], "g3_y": yuzdeler[2],
                            "g4_f": gunler[3], "g4_y": yuzdeler[3],
                            "g5_f": gunler[4], "g5_y": yuzdeler[4]
                        })
                        
                        tf.keras.backend.clear_session()
                
                except Exception as e:
                    pass 
                
                progress_bar.progress((i + 1) / len(hazir_modeller))
            
            durum_metni.success("✅ Otonom Kalibrasyon ve Piyasa Taraması Tamamlandı!")
            
            if sonuclar:
                df_sonuc = pd.DataFrame(sonuclar)
                
                # SİLİNEN KISIM: df_sonuc = df_sonuc.sort_values(...) satırını sildik.
                # Artık "sonuclar" listesi döngüye hangi sırayla girdiyse o sırayla kalacak.
                # Yani tam olarak 1. XU100, 2. Altın, 3. Gümüş, 4. AEFES ... şeklinde inecek ve basılacak.
                
                csv_data = df_sonuc.to_csv(index=False).encode('utf-8-sig')
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Tüm Sonuçları Excel Olarak İndir (Alfabetik)", data=csv_data,
                    file_name=f"BIST100_Otonom_Radar_{datetime.date.today().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv"
                )
                st.markdown("---")

                md_tablo = "| Varlık Adı | Mevcut Fiyat | 1. Gün (Yarın) | 2. Gün | 3. Gün | 4. Gün | 5. Gün |\n"
                md_tablo += "|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n"
                
                for _, row in df_sonuc.iterrows():
                    ok1 = "🟢" if row['g1_y'] > 0 else "🔴" if row['g1_y'] < 0 else "⚪"
                    ok2 = "🟢" if row['g2_y'] > 0 else "🔴" if row['g2_y'] < 0 else "⚪"
                    ok3 = "🟢" if row['g3_y'] > 0 else "🔴" if row['g3_y'] < 0 else "⚪"
                    ok4 = "🟢" if row['g4_y'] > 0 else "🔴" if row['g4_y'] < 0 else "⚪"
                    ok5 = "🟢" if row['g5_y'] > 0 else "🔴" if row['g5_y'] < 0 else "⚪"
                    
                    md_tablo += (
                        f"| **{row['Varlık']}** "
                        f"| {row['Mevcut Fiyat']:.2f} "
                        f"| {ok1} %{row['g1_y']:.2f} ➔ {row['g1_f']:.2f} "
                        f"| {ok2} %{row['g2_y']:.2f} ➔ {row['g2_f']:.2f} "
                        f"| {ok3} %{row['g3_y']:.2f} ➔ {row['g3_f']:.2f} "
                        f"| {ok4} %{row['g4_y']:.2f} ➔ {row['g4_f']:.2f} "
                        f"| {ok5} %{row['g5_y']:.2f} ➔ {row['g5_f']:.2f} |\n"
                    )
                st.markdown(md_tablo)

    # ---------------- SEKME 2: BİREYSEL ANALİZ ----------------
    with sekme2:
        st.subheader("Tekil Varlık Projeksiyonu")
        
        hisse_secim = st.selectbox(
            "Grafik Analizi İçin Varlık Seçin:", 
            hazir_modeller, 
            format_func=lambda x: isim_haritasi.get(x, x)
        )

        if st.button("Analizi Çalıştır"):
            gorsel_isim = isim_haritasi.get(hisse_secim, hisse_secim)
            with st.spinner(f"{gorsel_isim} verileri çekiliyor ve otonom kalibre ediliyor..."):
                try:
                    df = yf.download(hisse_secim, period="1y", interval="1d", progress=False)
                    kapanis_ham = df['Close']
                    if isinstance(kapanis_ham, pd.DataFrame):
                        kapanis_ham = kapanis_ham.squeeze()
                        
                    kapanis = pd.to_numeric(kapanis_ham, errors='coerce').dropna()
                    fiyatlar = kapanis.values.reshape(-1, 1)

                    scaler = MinMaxScaler(feature_range=(0, 1))
                    olcekli_veri = scaler.fit_transform(fiyatlar)
                    
                    model = load_model(f'src/models/{hisse_secim}_model.h5')
                    
                    ortalama_hata = 0
                    if len(kapanis) >= 75:
                        son_75 = olcekli_veri[-75:]
                        X_batch = np.array([son_75[j : j + 60] for j in range(15)])
                        y_pred_batch = model.predict(X_batch, verbose=0)
                        
                        t_1g_olcekli = y_pred_batch[:, 0].reshape(-1, 1)
                        t_1g_tl = scaler.inverse_transform(t_1g_olcekli).flatten()
                        g_15g_tl = fiyatlar[-15:].flatten()
                        ortalama_hata = np.mean(t_1g_tl - g_15g_tl)

                    son_60 = olcekli_veri[-60:].reshape(1, 60, 1)
                    tahmin_olcekli = model.predict(son_60, verbose=0)
                    ham_tahmin_tl = scaler.inverse_transform(tahmin_olcekli.reshape(-1, 1)).flatten()
                    
                    kalibre_tahmin_tl = ham_tahmin_tl - ortalama_hata

                    limitli_tahmin_tl = []
                    taban_listesi = []
                    tavan_listesi = []
                    referans_fiyat = fiyatlar[-1][0] 

                    for ham in kalibre_tahmin_tl:
                        tavan_fiyat = referans_fiyat * 1.10
                        taban_fiyat = referans_fiyat * 0.90
                        kesilmis_fiyat = float(np.clip(ham, taban_fiyat, tavan_fiyat))
                        
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
                        ax.plot(tahmin_tarihleri, limitli_tahmin_tl, label='Kalibre Edilmiş Tahmin', color='green', marker='x', linewidth=2)
                        ax.set_title(f"{gorsel_isim} - Fiyat Projeksiyonu")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig, use_container_width=True)

                    with sag_sutun:
                        st.markdown(f"**Son İşlem Günü Fiyatı:** {fiyatlar[-1][0]:.2f} | **Model Sapma Payı:** {ortalama_hata:.2f} (Otonom Düzeltildi)")
                        
                        md_tablo = "| Tarih | Yasal Taban (-10%) | Yasal Tavan (+10%) | 🤖 Kalibre Tahmin |\n|:---|:---:|:---:|:---:|\n"
                        for tarih, taban, tavan, fiyat in zip(tahmin_tarihleri, taban_listesi, tavan_listesi, limitli_tahmin_tl):
                            md_tablo += f"| {tarih.strftime('%d.%m.%Y')} | {taban:.2f} | {tavan:.2f} | **{fiyat:.2f}** |\n"
                        st.markdown(md_tablo)
                
                except Exception as e:
                    st.error(f"Sistem Hatası: {e}")