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

st.set_page_config(page_title="BİST100 & Emtia Radarı V3", layout="wide")

st.error("""
**!PROJE TEST VE GELİŞTİRME AŞAMASINDADIR!** **!KESİNLİKLE GERÇEK PARAYLA KULLANILMASI ÖNERİLMEZ!** **!HİÇBİR YATIRIM TAVSİYESİ VE SORUMLULUK KABUL EDİLMEMEKTEDİR!**
""")

st.title("📈 Derin Öğrenme Fiyat Projeksiyonu V3 (BiLSTM + Huber)")
st.markdown("""
**Sistem Mimarisi (V3):** Çift Yönlü LSTM (Bidirectional) hücreleri ve Huber Şok Emici kayıp fonksiyonu ile donatılmış 4 Boyutlu (OHLCV, RSI, MACD) Yapay Zeka.
**🌟 Otonom Kalibrasyon:** Model, son 15 günün geçmiş tahminlerini gerçek fiyatlarla kıyaslar ve bugünkü tahminlerini kendi kendine kalibre eder.
""")

# 1. Canlı Dolar Kurunu Çekme (Dönüşüm İçin)
try:
    df_kur = yf.download("TRY=X", period="5d", interval="1d", progress=False)
    kapanis_kur = df_kur['Close']
    if isinstance(kapanis_kur, pd.DataFrame):
        kapanis_kur = kapanis_kur.squeeze()
    dolar_kuru = float(pd.to_numeric(kapanis_kur, errors='coerce').dropna().iloc[-1])
except Exception:
    dolar_kuru = 33.0

st.info(f"💱 **Sistemde Kullanılan Anlık Dolar Kuru:** {dolar_kuru:.2f} ₺")

# 2. Modelleri Klasörden Çek
ham_modeller = [f.replace("_model.h5", "") for f in os.listdir("src/models") if f.endswith(".h5")]

# 3. Gösterim İsimleri
isim_haritasi = {
    "XU100.IS": "XU100 Endeksi",
    "GC=F": "Gram Altın",
    "SI=F": "Gram Gümüş"
}

if not ham_modeller:
    st.warning("Lütfen önce arka planda eğitim kodunu çalıştırıp modelleri yükleyin.")
else:
    # KUSURSUZ SIRALAMA ALGORİTMASI
    bas_taraf = []
    if "XU100.IS" in ham_modeller: bas_taraf.append("XU100.IS")
    if "GC=F" in ham_modeller: bas_taraf.append("GC=F")
    if "SI=F" in ham_modeller: bas_taraf.append("SI=F")
    
    kalanlar = [m for m in ham_modeller if m not in ["XU100.IS", "GC=F", "SI=F"]]
    kalanlar.sort() 
    hazir_modeller = bas_taraf + kalanlar 

    # --- 3 SEKMELİ YENİ YAPI ---
    sekme1, sekme2, sekme3 = st.tabs(["📊 Tüm Piyasa Radarı", "🎯 Bireysel Analiz", "💼 Model Portföy & Simülatör"])

    # ---------------- SEKME 1: TOPLU TARAMA ----------------
    with sekme1:
        st.subheader(f"Toplu Tarayıcı ({len(hazir_modeller)} Varlık)")
        
        if st.button("Piyasayı Tarat (Tahminleri Hesapla)"):
            progress_bar = st.progress(0)
            durum_metni = st.empty()
            sonuclar = []
            
            for i, hisse in enumerate(hazir_modeller):
                gorsel_isim = isim_haritasi.get(hisse, hisse)
                durum_metni.text(f"V3 Analiz & Kalibrasyon: {gorsel_isim} ({i+1}/{len(hazir_modeller)})")
                
                try:
                    df = yf.download(hisse, period="1y", interval="1d", progress=False)
                    
                    df['Volume'] = df['Volume'].replace(0, np.nan).ffill().bfill()
                    
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = exp1 - exp2
                    
                    df.dropna(inplace=True)
                    
                    if len(df) >= 75:
                        features = df[['Close', 'Volume', 'RSI', 'MACD']].values
                        target = df[['Close']].values
                        
                        scaler_X = MinMaxScaler(feature_range=(0, 1))
                        scaler_y = MinMaxScaler(feature_range=(0, 1))
                        
                        scaled_X = scaler_X.fit_transform(features)
                        scaler_y.fit(target)
                        
                        model_yolu = f'src/models/{hisse}_model.h5' 
                        model = load_model(model_yolu)

                        son_75_X = scaled_X[-75:]
                        X_batch = np.array([son_75_X[j : j + 60] for j in range(15)])
                        y_pred_batch = model.predict(X_batch, verbose=0)
                        
                        tahminler_1g_olcekli = y_pred_batch[:, 0].reshape(-1, 1)
                        tahminler_1g_tl = scaler_y.inverse_transform(tahminler_1g_olcekli).flatten()
                        gercekler_15g_tl = target[-15:].flatten()
                        
                        ortalama_hata = np.mean(tahminler_1g_tl - gercekler_15g_tl)
                        
                        son_60_X = scaled_X[-60:].reshape(1, 60, 4)
                        tahmin_olcekli = model.predict(son_60_X, verbose=0)
                        ham_tahmin_tl = scaler_y.inverse_transform(tahmin_olcekli.reshape(-1, 1)).flatten()
                        
                        kalibre_tahmin_tl = ham_tahmin_tl - ortalama_hata

                        suanki_fiyat = target[-1][0]
                        if hisse in ["GC=F", "SI=F"]:
                            carpan = dolar_kuru / 31.1034768
                            kalibre_tahmin_tl = kalibre_tahmin_tl * carpan
                            referans_fiyat = suanki_fiyat * carpan
                            suanki_fiyat_gosterim = suanki_fiyat * carpan
                        else:
                            referans_fiyat = suanki_fiyat
                            suanki_fiyat_gosterim = suanki_fiyat

                        limitli_tahmin_tl = []
                        for ham in kalibre_tahmin_tl:
                            tavan = referans_fiyat * 1.10
                            taban = referans_fiyat * 0.90
                            kesilmis = float(np.clip(ham, taban, tavan))
                            limitli_tahmin_tl.append(kesilmis)
                            referans_fiyat = kesilmis 
                        
                        gunler = limitli_tahmin_tl[:5]
                        yuzdeler = []
                        eski_fiyat = suanki_fiyat_gosterim
                        for g_fiyat in gunler:
                            yuzdeler.append(((g_fiyat - eski_fiyat) / eski_fiyat) * 100)
                            eski_fiyat = g_fiyat
                            
                        # Portföy sıralaması için Total Getiri yüzdesini de sözlüğe ekliyoruz
                        total_getiri_yuzde = ((gunler[4] - suanki_fiyat_gosterim) / suanki_fiyat_gosterim) * 100

                        sonuclar.append({
                            "Varlık": gorsel_isim, 
                            "Mevcut Fiyat": round(suanki_fiyat_gosterim, 2),
                            "g1_f": gunler[0], "g1_y": yuzdeler[0],
                            "g2_f": gunler[1], "g2_y": yuzdeler[1],
                            "g3_f": gunler[2], "g3_y": yuzdeler[2],
                            "g4_f": gunler[3], "g4_y": yuzdeler[3],
                            "g5_f": gunler[4], "g5_y": yuzdeler[4],
                            "Total_Getiri": total_getiri_yuzde
                        })
                        
                        tf.keras.backend.clear_session()
                
                except Exception as e:
                    pass 
                
                progress_bar.progress((i + 1) / len(hazir_modeller))
            
            durum_metni.success("✅ V3 Piyasa Taraması Tamamlandı! Portföy sekmesini kontrol edebilirsiniz.")
            
            if sonuclar:
                df_sonuc = pd.DataFrame(sonuclar)
                
                # 🌟 SİSTEM HAFIZASI: Sonuçları 3. sekmede kullanmak üzere RAM'e kaydediyoruz
                st.session_state['df_hafiza'] = df_sonuc
                
                csv_data = df_sonuc.drop(columns=['Total_Getiri']).to_csv(index=False).encode('utf-8-sig')
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Tüm Sonuçları Excel Olarak İndir", data=csv_data,
                    file_name=f"BIST100_V3_Radar_{datetime.date.today().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv"
                )
                st.markdown("---")

                yari_nokta = len(df_sonuc) // 2 + (len(df_sonuc) % 2)
                df_sol = df_sonuc.iloc[:yari_nokta]
                df_sag = df_sonuc.iloc[yari_nokta:]
                
                sol_sutun, sag_sutun = st.columns(2)
                
                def tablo_olustur(df):
                    md_tablo = "| Varlık | Mevcut (₺) | 1. Gün | 2. Gün | 3. Gün | 4. Gün | 5. Gün |\n"
                    md_tablo += "|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n"
                    
                    for _, row in df.iterrows():
                        ok1 = "🟢" if row['g1_y'] > 0 else "🔴" if row['g1_y'] < 0 else "⚪"
                        ok2 = "🟢" if row['g2_y'] > 0 else "🔴" if row['g2_y'] < 0 else "⚪"
                        ok3 = "🟢" if row['g3_y'] > 0 else "🔴" if row['g3_y'] < 0 else "⚪"
                        ok4 = "🟢" if row['g4_y'] > 0 else "🔴" if row['g4_y'] < 0 else "⚪"
                        ok5 = "🟢" if row['g5_y'] > 0 else "🔴" if row['g5_y'] < 0 else "⚪"
                        
                        md_tablo += (
                            f"| **{row['Varlık'][:10]}** "
                            f"| {row['Mevcut Fiyat']:.2f} "
                            f"| {ok1} %{row['g1_y']:.1f} ➔ {row['g1_f']:.1f} "
                            f"| {ok2} %{row['g2_y']:.1f} ➔ {row['g2_f']:.1f} "
                            f"| {ok3} %{row['g3_y']:.1f} ➔ {row['g3_f']:.1f} "
                            f"| {ok4} %{row['g4_y']:.1f} ➔ {row['g4_f']:.1f} "
                            f"| {ok5} %{row['g5_y']:.1f} ➔ {row['g5_f']:.1f} |\n"
                        )
                    return md_tablo

                with sol_sutun: st.markdown(tablo_olustur(df_sol))
                with sag_sutun: st.markdown(tablo_olustur(df_sag))

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
            with st.spinner(f"{gorsel_isim} verileri çekiliyor ve V3 model ile analiz ediliyor..."):
                try:
                    df = yf.download(hisse_secim, period="1y", interval="1d", progress=False)
                    
                    df['Volume'] = df['Volume'].replace(0, np.nan).ffill().bfill()
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = exp1 - exp2
                    
                    df.dropna(inplace=True)

                    features = df[['Close', 'Volume', 'RSI', 'MACD']].values
                    target = df[['Close']].values

                    scaler_X = MinMaxScaler(feature_range=(0, 1))
                    scaler_y = MinMaxScaler(feature_range=(0, 1))
                    
                    scaled_X = scaler_X.fit_transform(features)
                    scaler_y.fit(target)
                    
                    model = load_model(f'src/models/{hisse_secim}_model.h5')
                    
                    ortalama_hata = 0
                    if len(df) >= 75:
                        son_75_X = scaled_X[-75:]
                        X_batch = np.array([son_75_X[j : j + 60] for j in range(15)])
                        y_pred_batch = model.predict(X_batch, verbose=0)
                        
                        t_1g_olcekli = y_pred_batch[:, 0].reshape(-1, 1)
                        t_1g_tl = scaler_y.inverse_transform(t_1g_olcekli).flatten()
                        g_15g_tl = target[-15:].flatten()
                        ortalama_hata = np.mean(t_1g_tl - g_15g_tl)

                    son_60_X = scaled_X[-60:].reshape(1, 60, 4)
                    tahmin_olcekli = model.predict(son_60_X, verbose=0)
                    ham_tahmin_tl = scaler_y.inverse_transform(tahmin_olcekli.reshape(-1, 1)).flatten()
                    
                    kalibre_tahmin_tl = ham_tahmin_tl - ortalama_hata

                    if hisse_secim in ["GC=F", "SI=F"]:
                        carpan = dolar_kuru / 31.1034768
                        kalibre_tahmin_tl = kalibre_tahmin_tl * carpan
                        fiyatlar_gosterim = target * carpan
                        ortalama_hata_gosterim = ortalama_hata * carpan
                    else:
                        fiyatlar_gosterim = target
                        ortalama_hata_gosterim = ortalama_hata

                    limitli_tahmin_tl = []
                    taban_listesi = []
                    tavan_listesi = []
                    referans_fiyat = fiyatlar_gosterim[-1][0] 

                    for ham in kalibre_tahmin_tl:
                        tavan_fiyat = referans_fiyat * 1.10
                        taban_fiyat = referans_fiyat * 0.90
                        kesilmis_fiyat = float(np.clip(ham, taban_fiyat, tavan_fiyat))
                        
                        limitli_tahmin_tl.append(kesilmis_fiyat)
                        taban_listesi.append(taban_fiyat)
                        tavan_listesi.append(tavan_fiyat)
                        referans_fiyat = kesilmis_fiyat 

                    son_tarih = df.index[-1]
                    tahmin_tarihleri = [son_tarih + datetime.timedelta(days=i) for i in range(1, 8)]

                    st.markdown("---")
                    sol_sutun_grafik, sag_sutun_grafik = st.columns([2, 3])

                    with sol_sutun_grafik:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.plot(df.index[-30:], fiyatlar_gosterim[-30:], label='Son 30 Gün', marker='o', linewidth=2)
                        ax.plot(tahmin_tarihleri, limitli_tahmin_tl, label='V3 Kalibre Tahmin', color='green', marker='x', linewidth=2)
                        ax.set_title(f"{gorsel_isim} - Fiyat Projeksiyonu (₺)")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig, use_container_width=True)

                    with sag_sutun_grafik:
                        st.markdown(f"**Son İşlem Günü Fiyatı:** {fiyatlar_gosterim[-1][0]:.2f} ₺ | **Model Sapma Payı:** {ortalama_hata_gosterim:.2f} ₺")
                        
                        md_tablo = "| Tarih | Yasal Taban (-10%) | Yasal Tavan (+10%) | 🤖 V3 Kalibre Tahmin |\n|:---|:---:|:---:|:---:|\n"
                        for tarih, taban, tavan, fiyat in zip(tahmin_tarihleri, taban_listesi, tavan_listesi, limitli_tahmin_tl):
                            md_tablo += f"| {tarih.strftime('%d.%m.%Y')} | {taban:.2f} ₺ | {tavan:.2f} ₺ | **{fiyat:.2f} ₺** |\n"
                        st.markdown(md_tablo)
                
                except Exception as e:
                    st.error(f"Sistem Hatası: {e}")

    # ---------------- SEKME 3: SİMÜLATÖR VE PORTFÖY ----------------
    with sekme3:
        st.subheader("💼 V3 Model Portföy & Yatırım Simülatörü")
        
        # Hafızada tarama verisi yoksa kullanıcıyı uyar
        if 'df_hafiza' not in st.session_state:
            st.warning("👉 Lütfen önce 'Tüm Piyasa Radarı' sekmesine gidip piyasayı taratın. Simülatör o verilere ihtiyaç duyar.")
        else:
            df_hafiza = st.session_state['df_hafiza']
            
            # --- MODEL PORTFÖY (EN İYİ 10 HİSSE) ---
            st.markdown("### 🏆 V3 Yapay Zeka Model Portföyü (En Yüksek Beklenti)")
            st.markdown("Modelin 5. günün sonunda **en yüksek net getiri yüzdesini** beklediği ilk 10 varlık.")
            
            # Total Getiriye göre büyükten küçüğe sıralayıp ilk 10'u alıyoruz
            df_top10 = df_hafiza.sort_values(by="Total_Getiri", ascending=False).head(10)
            
            # Şık bir bar grafiği
            st.bar_chart(data=df_top10.set_index('Varlık')['Total_Getiri'], use_container_width=True)
            
            md_portfoy = "| Sıra | Varlık Adı | Mevcut Fiyat | 5 Gün Sonra (Tahmin) | Beklenen Getiri |\n|:---:|:---|:---:|:---:|:---:|\n"
            for index, (sira, row) in enumerate(df_top10.iterrows(), 1):
                md_portfoy += f"| **{index}.** | **{row['Varlık']}** | {row['Mevcut Fiyat']:.2f} ₺ | {row['g5_f']:.2f} ₺ | 🟢 **%{row['Total_Getiri']:.2f}** |\n"
            st.markdown(md_portfoy)
            
            st.markdown("---")
            
            # --- YATIRIM SİMÜLATÖRÜ ---
            st.markdown("### 🧮 Yatırım Simülatörü (Ne Yatırsam Ne Olur?)")
            
            col_sim1, col_sim2 = st.columns(2)
            
            with col_sim1:
                # Kullanıcı sadece taranan hisseler arasından seçim yapabilir
                secilen_varlik = st.selectbox("Simüle Etmek İstediğiniz Varlık:", df_hafiza['Varlık'].tolist())
                yatirim_miktari = st.number_input("Yatırım Miktarı (₺):", min_value=100.0, value=10000.0, step=500.0)
            
            # Arkada Matematiği Çalıştır
            satir = df_hafiza[df_hafiza['Varlık'] == secilen_varlik].iloc[0]
            alinabilen_adet = yatirim_miktari / satir['Mevcut Fiyat']
            
            gun5_para = alinabilen_adet * satir['g5_f']
            net_kar_zarar = gun5_para - yatirim_miktari
            
            with col_sim2:
                st.info(f"📦 **Alınabilen Miktar (Lot/Gram):** {alinabilen_adet:.4f} Birim")
                
                if net_kar_zarar > 0:
                    st.success(f"💰 **5. Gün Sonundaki Kasanız:** {gun5_para:.2f} ₺\n\n🟢 **Net Kâr:** +{net_kar_zarar:.2f} ₺")
                elif net_kar_zarar < 0:
                    st.error(f"📉 **5. Gün Sonundaki Kasanız:** {gun5_para:.2f} ₺\n\n🔴 **Net Zarar:** {net_kar_zarar:.2f} ₺")
                else:
                    st.warning(f"⚖️ **5. Gün Sonundaki Kasanız:** {gun5_para:.2f} ₺\n\n⚪ **Kâr/Zarar Yok**")