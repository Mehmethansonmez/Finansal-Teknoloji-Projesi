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
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="BİST100 Fon Terminali V4", layout="wide")

# 📊 [YFINANCE GÜNCELLEMELERİNE KARŞI %100 ÇELİK ZIRH VERİ MOTORU]
def guvenli_veri_cek(ticker, period="1y"):
    try:
        df_raw = yf.download(ticker, period=period, interval="1d", progress=False)
        if df_raw.empty:
            return pd.DataFrame()
            
        # Eğer yfinance veriyi iç içe geçmiş (MultiIndex) sütunla getirdiyse en üst seviyeyi al
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
            
        # 🌟 KRİTİK ÇÖZÜM: Zaman dilimi (Timezone) farklarını tamamen silip indeksi eşitliyoruz
        df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
        
        res = pd.DataFrame(index=df_raw.index)
        
        if 'Close' in df_raw.columns:
            c_data = df_raw['Close']
            res['Close'] = c_data.iloc[:, 0] if isinstance(c_data, pd.DataFrame) else c_data
            
        if 'Volume' in df_raw.columns:
            v_data = df_raw['Volume']
            res['Volume'] = v_data.iloc[:, 0] if isinstance(v_data, pd.DataFrame) else v_data
            
        return res
    except Exception:
        return pd.DataFrame()

st.title("📈 Kuantum Fon Terminali V4 (Transformer + Sentiment)")
st.markdown("""
**Sistem Mimarisi (V4):** Çift Yönlü LSTM, Huber Şok Emici ve **Multi-Head Attention** (Transformer) destekli 5 Boyutlu Yapay Zeka (XU100 Makro Entegrasyonu).
**📰 Alternatif Veri (Sentiment):** Kendi kurduğunuz portföyü anlık haber akışıyla stres testine sokabilirsiniz.
""")

# 1. Canlı Dolar Kurunu Çekme
try:
    df_kur = guvenli_veri_cek("TRY=X", period="5d")
    dolar_kuru = float(df_kur['Close'].dropna().iloc[-1])
except Exception:
    dolar_kuru = 33.0

st.info(f"💱 **Sistemde Kullanılan Anlık Dolar Kuru:** {dolar_kuru:.2f} ₺")

# Modelleri Klasörden Çek
ham_modeller = [f.replace("_model.h5", "") for f in os.listdir("src/models") if f.endswith(".h5")]

isim_haritasi = {
    "XU100.IS": "XU100 Endeksi",
    "GC=F": "Gram Altın",
    "SI=F": "Gram Gümüş"
}

# --- HABER ÇEKME VE DUYGU ANALİZİ BOTU ---
def haber_duygu_analizi(sirket_kodu):
    sirket_ismi = sirket_kodu.replace(".IS", "")
    url = f"https://news.google.com/search?q={sirket_ismi}+hisse+borsa&hl=tr&gl=TR&ceid=TR:tr"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        mansetler = soup.find_all('h3')
        if not mansetler:
            return 0.0 
            
        toplam_skor = 0
        sayac = 0
        
        pozitifler = ['uçtu', 'arttı', 'yükseldi', 'kâr', 'anlaşma', 'ihale', 'büyüme', 'rekor', 'onay']
        negatifler = ['düştü', 'çöktü', 'zarar', 'iptal', 'ceza', 'dava', 'satış', 'uyarı', 'kriz']
        
        for man in mansetler[:5]: 
            metin = man.text.lower()
            haber_skoru = 0
            for p in pozitifler:
                if p in metin: haber_skoru += 0.2
            for n in negatifler:
                if n in metin: haber_skoru -= 0.3 
                
            toplam_skor += haber_skoru
            sayac += 1
            
        ortalama_skor = toplam_skor / sayac if sayac > 0 else 0
        return float(np.clip(ortalama_skor, -1.0, 1.0))
    except Exception:
        return 0.0 

if not ham_modeller:
    st.warning("Lütfen önce arka planda eğitim kodunu çalıştırıp modelleri yükleyin.")
else:
    bas_taraf = []
    if "XU100.IS" in ham_modeller: bas_taraf.append("XU100.IS")
    if "GC=F" in ham_modeller: bas_taraf.append("GC=F")
    if "SI=F" in ham_modeller: bas_taraf.append("SI=F")
    
    kalanlar = [m for m in ham_modeller if m not in ["XU100.IS", "GC=F", "SI=F"]]
    kalanlar.sort() 
    hazir_modeller = bas_taraf + kalanlar 

    sekme1, sekme2, sekme3, sekme4 = st.tabs([
        "📊 Tüm Piyasa Radarı (V4)", 
        "🎯 Bireysel Analiz", 
        "🏆 V4 Model Portföy", 
        "💼 Kendi Portföyüm (Risk Radarı)"
    ])

    # ---------------- SEKME 1: TOPLU TARAMA ----------------
    with sekme1:
        st.subheader(f"Toplu Tarayıcı ({len(hazir_modeller)} Varlık)")
        
        if st.button("Piyasayı Tarat (V4 Motorunu Çalıştır)"):
            progress_bar = st.progress(0)
            durum_metni = st.empty()
            sonuclar = []
            
            # Makro XU100 verisini çekiyoruz ve indeksini düzeltiyoruz
            df_makro = guvenli_veri_cek("XU100.IS", period="1y")
            makro_serisi = df_makro['Close'] if not df_makro.empty else None
            
            for i, hisse in enumerate(hazir_modeller):
                gorsel_isim = isim_haritasi.get(hisse, hisse)
                durum_metni.text(f"V4 Analizi (Attention): {gorsel_isim} ({i+1}/{len(hazir_modeller)})")
                
                try:
                    df = guvenli_veri_cek(hisse, period="1y")
                    if df.empty or makro_serisi is None:
                        continue
                        
                    df['Volume'] = df['Volume'].replace(0, np.nan).ffill().bfill()
                    
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = exp1 - exp2
                    
                    # 🌟 ASLA PATLAMAYAN DIRET ATAMA (Timezone uyuşmazlığı giderildi)
                    df['XU100_Close'] = makro_serisi
                    df['XU100_Close'] = df['XU100_Close'].ffill().bfill()
                    
                    df.dropna(inplace=True)
                    
                    if len(df) >= 75:
                        features = df[['Close', 'Volume', 'RSI', 'MACD', 'XU100_Close']].values
                        target = df[['Close']].values
                        
                        scaler_X = MinMaxScaler(feature_range=(0, 1))
                        scaler_y = MinMaxScaler(feature_range=(0, 1))
                        
                        scaled_X = scaler_X.fit_transform(features)
                        scaler_y.fit(target)
                        
                        model_yolu = f'src/models/{hisse}_model.h5' 
                        model = load_model(model_yolu, compile=False) 

                        son_75_X = scaled_X[-75:]
                        X_batch = np.array([son_75_X[j : j + 60] for j in range(15)])
                        y_pred_batch = model.predict(X_batch, verbose=0)
                        
                        tahminler_1g_olcekli = y_pred_batch[:, 0].reshape(-1, 1)
                        tahminler_1g_tl = scaler_y.inverse_transform(tahminler_1g_olcekli).flatten()
                        gercekler_15g_tl = target[-15:].flatten()
                        
                        ortalama_hata = np.mean(tahminler_1g_tl - gercekler_15g_tl)
                        
                        son_60_X = scaled_X[-60:].reshape(1, 60, 5) 
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
                            
                        total_getiri_yuzde = ((gunler[4] - suanki_fiyat_gosterim) / suanki_fiyat_gosterim) * 100

                        sonuclar.append({
                            "Hisse_Kodu": hisse, 
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
                except Exception:
                    pass 
                
                progress_bar.progress((i + 1) / len(hazir_modeller))
            
            if sonuclar:
                df_sonuc = pd.DataFrame(sonuclar)
                st.session_state['df_hafiza'] = df_sonuc
                durum_metni.success("✅ V4 Piyasa Taraması Tamamlandı! Tablolar aktif hale getirildi.")
                
                csv_data = df_sonuc.drop(columns=['Hisse_Kodu', 'Total_Getiri']).to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Tüm Sonuçları Excel Olarak İndir", data=csv_data,
                    file_name=f"BIST100_V4_Radar_{datetime.date.today().strftime('%Y-%m-%d')}.csv", mime="text/csv"
                )

                yari_nokta = len(df_sonuc) // 2 + (len(df_sonuc) % 2)
                df_sol = df_sonuc.iloc[:yari_nokta]
                df_sag = df_sonuc.iloc[yari_nokta:]
                
                def tablo_olustur(df_target):
                    md_tablo = "| Varlık | Mevcut (₺) | 1. Gün | 2. Gün | 3. Gün | 4. Gün | 5. Gün |\n"
                    md_tablo += "|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n"
                    for _, row in df_target.iterrows():
                        ok1 = "🟢" if row['g1_y'] > 0 else "🔴" if row['g1_y'] < 0 else "⚪"
                        ok2 = "🟢" if row['g2_y'] > 0 else "🔴" if row['g2_y'] < 0 else "⚪"
                        ok3 = "🟢" if row['g3_y'] > 0 else "🔴" if row['g3_y'] < 0 else "⚪"
                        ok4 = "🟢" if row['g4_y'] > 0 else "🔴" if row['g4_y'] < 0 else "⚪"
                        ok5 = "🟢" if row['g5_y'] > 0 else "🔴" if row['g5_y'] < 0 else "⚪"
                        md_tablo += (
                            f"| **{row['Varlık'][:10]}** | {row['Mevcut Fiyat']:.2f} "
                            f"| {ok1} %{row['g1_y']:.1f} ➔ {row['g1_f']:.1f} | {ok2} %{row['g2_y']:.1f} ➔ {row['g2_f']:.1f} "
                            f"| {ok3} %{row['g3_y']:.1f} ➔ {row['g3_f']:.1f} | {ok4} %{row['g4_y']:.1f} ➔ {row['g4_f']:.1f} "
                            f"| {ok5} %{row['g5_y']:.1f} ➔ {row['g5_f']:.1f} |\n"
                        )
                    return md_tablo

                sol_sutun, sag_sutun = st.columns(2)
                with sol_sutun: st.markdown(tablo_olustur(df_sol))
                with sag_sutun: st.markdown(tablo_olustur(df_sag))
            else:
                durum_metni.error("❌ Tarama yapıldı fakat eşleşen veri üretilemedi. Model isimlerini kontrol edin.")

    # ---------------- SEKME 2: BİREYSEL ANALİZ ----------------
    with sekme2:
        st.subheader("Tekil Varlık Projeksiyonu")
        hisse_secim = st.selectbox("Grafik Analizi İçin Varlık Seçin:", hazir_modeller, format_func=lambda x: isim_haritasi.get(x, x))

        if st.button("Analizi Çalıştır"):
            gorsel_isim = isim_haritasi.get(hisse_secim, hisse_secim)
            with st.spinner(f"{gorsel_isim} verileri V4 model ile analiz ediliyor..."):
                try:
                    df_makro = guvenli_veri_cek("XU100.IS", period="1y")
                    makro_serisi = df_makro['Close'] if not df_makro.empty else None

                    df = guvenli_veri_cek(hisse_secim, period="1y")
                    if df.empty or makro_serisi is None:
                        st.error("Veri kaynaklarına şu an erişilemiyor.")
                    else:
                        df['Volume'] = df['Volume'].replace(0, np.nan).ffill().bfill()
                        delta = df['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        df['RSI'] = 100 - (100 / (1 + rs))
                        
                        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                        df['MACD'] = exp1 - exp2
                        
                        df['XU100_Close'] = makro_serisi
                        df['XU100_Close'] = df['XU100_Close'].ffill().bfill()
                        df.dropna(inplace=True)

                        features = df[['Close', 'Volume', 'RSI', 'MACD', 'XU100_Close']].values
                        target = df[['Close']].values

                        scaler_X = MinMaxScaler(feature_range=(0, 1))
                        scaler_y = MinMaxScaler(feature_range=(0, 1))
                        
                        scaled_X = scaler_X.fit_transform(features)
                        scaler_y.fit(target)
                        
                        model = load_model(f'src/models/{hisse_secim}_model.h5', compile=False)
                        
                        ortalama_hata = 0
                        if len(df) >= 75:
                            son_75_X = scaled_X[-75:]
                            X_batch = np.array([son_75_X[j : j + 60] for j in range(15)])
                            y_pred_batch = model.predict(X_batch, verbose=0)
                            t_1g_tl = scaler_y.inverse_transform(y_pred_batch[:, 0].reshape(-1, 1)).flatten()
                            g_15g_tl = target[-15:].flatten()
                            ortalama_hata = np.mean(t_1g_tl - g_15g_tl)

                        son_60_X = scaled_X[-60:].reshape(1, 60, 5)
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
                        taban_listesi, tavan_listesi = [], []
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
                            ax.plot(tahmin_tarihleri, limitli_tahmin_tl, label='V4 Kalibre Tahmin', color='green', marker='x', linewidth=2)
                            ax.set_title(f"{gorsel_isim} - Fiyat Projeksiyonu (₺)")
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                            st.pyplot(fig, use_container_width=True)

                        with sag_sutun_grafik:
                            st.markdown(f"**Son İşlem Günü Fiyatı:** {fiyatlar_gosterim[-1][0]:.2f} ₺ | **Model Sapma Payı:** {ortalama_hata_gosterim:.2f} ₺")
                            md_tablo = "| Tarih | Yasal Taban (-10%) | Yasal Tavan (+10%) | 🤖 V4 Kalibre Tahmin |\n|:---|:---:|:---:|:---:|\n"
                            for tarih, taban, tavan, fiyat in zip(tahmin_tarihleri, taban_listesi, tavan_listesi, limitli_tahmin_tl):
                                md_tablo += f"| {tarih.strftime('%d.%m.%Y')} | {taban:.2f} ₺ | {tavan:.2f} ₺ | **{fiyat:.2f} ₺** |\n"
                            st.markdown(md_tablo)
                except Exception as e:
                    st.error(f"Sistem Hatası: {e}")

    # ---------------- SEKME 3: MODEL PORTFÖY ----------------
    with sekme3:
        st.subheader("🏆 V4 Yapay Zeka Model Portföyü (Top 10)")
        
        if 'df_hafiza' not in st.session_state:
            st.warning("👉 Lütfen önce 'Tüm Piyasa Radarı (V4)' sekmesine gidip piyasayı taratın.")
        else:
            df_hafiza = st.session_state['df_hafiza']
            df_top10 = df_hafiza.sort_values(by="Total_Getiri", ascending=False).head(10)
            
            st.bar_chart(data=df_top10.set_index('Varlık')['Total_Getiri'], use_container_width=True)
            
            md_portfoy = "| Sıra | Varlık Adı | Mevcut Fiyat | 5 Gün Sonra (Tahmin) | Getiri Oranı |\n|:---:|:---|:---:|:---:|:---:|\n"
            for index, (sira, row) in enumerate(df_top10.iterrows(), 1):
                md_portfoy += f"| **{index}.** | **{row['Varlık']}** | {row['Mevcut Fiyat']:.2f} ₺ | {row['g5_f']:.2f} ₺ | 🟢 **%{row['Total_Getiri']:.2f}** |\n"
            st.markdown(md_portfoy)

    # ---------------- SEKME 4: KENDİ PORTFÖYÜM & HABER ANALİZİ ----------------
    with sekme4:
        st.subheader("💼 Portföy Risk Radarı (Haber + Yapay Zeka)")
        
        if 'df_hafiza' not in st.session_state:
            st.warning("👉 Lütfen önce 'Tüm Piyasa Radarı (V4)' sekmesine gidip piyasayı taratın.")
        else:
            df_hafiza = st.session_state['df_hafiza']
            secilen_varliklar = st.multiselect("Portföyünüze Varlık Ekleyin:", df_hafiza['Varlık'].tolist())
            
            if secilen_varliklar:
                toplam_yatirim = 0
                portfoy_detay = []
                
                col1, col2 = st.columns(2)
                for i, varlik in enumerate(secilen_varliklar):
                    with col1 if i % 2 == 0 else col2:
                        miktar = st.number_input(f"{varlik} Yatırım Tutarı (₺):", min_value=0.0, value=1000.0, step=500.0, key=f"inp_{varlik}")
                        if miktar > 0:
                            toplam_yatirim += miktar
                            satir = df_hafiza[df_hafiza['Varlık'] == varlik].iloc[0]
                            portfoy_detay.append({
                                "Varlık": varlik, "Hisse_Kodu": satir['Hisse_Kodu'],
                                "Yatırım": miktar, "Mevcut_Fiyat": satir['Mevcut Fiyat'],
                                "V4_Saf_Hedef": satir['g5_f'], "V4_Saf_Getiri_Yuzde": satir['Total_Getiri']
                            })
                
                if portfoy_detay:
                    st.markdown("---")
                    if st.button("🚨 Portföyü Haber Stres Testine Sok"):
                        st.info("🌐 Haberler NLP motoruyla taranıp duygu skorlaması yapılıyor...")
                        
                        portfoy_sonuclari = []
                        toplam_yeni_kasa = 0
                        
                        for kalem in portfoy_detay:
                            if kalem['Hisse_Kodu'] in ["GC=F", "SI=F", "XU100.IS"]:
                                duygu_skoru = 0.0 
                            else:
                                duygu_skoru = haber_duygu_analizi(kalem['Hisse_Kodu'])
                                
                            risk_carpanı = 1 + (duygu_skoru * 0.10) 
                            v4_fark = kalem['V4_Saf_Hedef'] - kalem['Mevcut_Fiyat']
                            yeni_fark = v4_fark * risk_carpanı
                            yeni_hedef_fiyat = kalem['Mevcut_Fiyat'] + yeni_fark
                            
                            adet = kalem['Yatırım'] / kalem['Mevcut_Fiyat']
                            gelecek_para = float(adet * yeni_hedef_fiyat)
                            toplam_yeni_kasa += gelecek_para
                            
                            portfoy_sonuclari.append({
                                "Varlık": kalem['Varlık'], "Yatırılan": f"{kalem['Yatırım']:,.2f} ₺",
                                "Saf V4 Hedefi": f"%{kalem['V4_Saf_Getiri_Yuzde']:.2f}", "Medya Skoru": f"{duygu_skoru:.2f}",
                                "Yeni Haberli Hedef": f"%{((yeni_hedef_fiyat - kalem['Mevcut_Fiyat'])/kalem['Mevcut_Fiyat'])*100:.2f}",
                                "5. Gün Sonucu": f"{gelecek_para:,.2f} ₺"
                            })
                            
                        df_gosterim = pd.DataFrame(portfoy_sonuclari)
                        st.markdown("### 📊 Haber Duyarlılık Analizi Sonuçları")
                        st.success(f"**Toplam Portföy Değeri:** {toplam_yatirim:,.2f} ₺  ➔  **Beklenen Gelecek Kasa:** {toplam_yeni_kasa:,.2f} ₺")
                        st.dataframe(df_gosterim)