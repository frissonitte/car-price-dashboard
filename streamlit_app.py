import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

import shap


st.set_page_config(page_title="5. Grup Araç Fiyat Tahmin", layout="wide")
st.title("İkinci El Araç Fiyat Tahmin Sistemi")
st.markdown("Seçtiğiniz makine öğrenmesi modeliyle araç değer tahmini ve piyasa kıyaslaması")


@st.cache_resource
def load_assets():
    with open("outputs/ml_modeller.pkl", "rb") as f:
        ml = pickle.load(f)
    with open("outputs/islenmis_veri.pkl", "rb") as f:
        veri = pickle.load(f)
    return ml, veri


@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


ml, veri = load_assets()
modeller = ml["modeller"]
scaler = veri["scaler_std"]
le_dict = veri["le_dict"]
df_enc = veri["df_encoded"]
df_model = veri["df_model"]

feat_names_all = veri["feature_names"]
nan_mask_bool = np.array([df_enc[c].isna().all() for c in feat_names_all])
gecerli_idx = [i for i, skip in enumerate(nan_mask_bool) if not skip]
feat_names_model = [feat_names_all[i] for i in gecerli_idx]

df_model_sonuc = load_csv("outputs/model_sonuclari.csv")
df_piyasa = load_csv("outputs/piyasa_analizi.csv")
df_segment = load_csv("outputs/segmentli_araclar.csv")
df_istat = load_csv("outputs/ozet_istatistikler.csv")


def secenekler(col_name, fallback):
    if col_name in le_dict:
        return list(le_dict[col_name].classes_)
    return fallback


def model_gorunur_ad(model_adi):
    parcalar = model_adi.split(". ", 1)
    if len(parcalar) == 2 and parcalar[0].isdigit():
        return parcalar[1]
    return model_adi


def safe_encode(col_name, value):
    if col_name not in le_dict:
        return 0
    encoder = le_dict[col_name]
    siniflar = set(encoder.classes_)
    if value in siniflar:
        return int(encoder.transform([value])[0])
    return 0


def get_stat_value(name, default):
    if df_istat.empty or name not in df_istat.columns:
        return default
    value = pd.to_numeric(df_istat.loc[df_istat.index == "50%", name], errors="coerce")
    if len(value) == 0 or pd.isna(value.iloc[0]):
        return default
    return float(value.iloc[0])


def prepare_model_input(ozellikler):
    x_full = np.array([[ozellikler.get(f, 0.0) for f in feat_names_all]], dtype=float)
    x_scaled = scaler.transform(x_full)
    return x_scaled[:, gecerli_idx]


def predict_price(x_model, model_obj):
    return float(model_obj.predict(x_model)[0])


def durum_etiketi(ilan_fiyati, tahmin, esik=10.0):
    fark_pct = ((ilan_fiyati - tahmin) / max(tahmin, 1.0)) * 100
    if fark_pct <= -esik:
        return "FIRSAT", fark_pct
    if fark_pct >= esik:
        return "PAHALI", fark_pct
    return "MAKUL", fark_pct


def nadir_kombinasyon_uyarisi(df, yil, km):
    if yil >= 2022 and km > 150000:
        return "Bu yıl/km kombinasyonu veri setinde nadir olabilir. Tahmini dikkatli yorumlayın."

    if "Kilometre" in df and len(df) > 0:
        ust_sinir = float(df["Kilometre"].quantile(0.99))
        if km > ust_sinir:
            return f"Girilen kilometre, veri setinin %99 üst diliminin üzerinde ({ust_sinir:,.0f} km)."

    return None


st.sidebar.header("Araç Özelliklerini Girin")

default_model_name = "10. Extra Trees"
if not df_model_sonuc.empty and "Model" in df_model_sonuc.columns:
    sirali_modeller = [
        m for m in df_model_sonuc.sort_values("R2", ascending=False)["Model"].tolist() if m in modeller
    ]
    model_opsiyonlari = sirali_modeller + [m for m in modeller.keys() if m not in sirali_modeller]
else:
    model_opsiyonlari = list(modeller.keys())

default_idx = model_opsiyonlari.index(default_model_name) if default_model_name in model_opsiyonlari else 0
aktif_model_adi = st.sidebar.selectbox(
    "Tahmin Modeli",
    model_opsiyonlari,
    index=default_idx,
    format_func=model_gorunur_ad,
)
aktif_model = modeller[aktif_model_adi]

yil_min = int(df_model["Yil"].min()) if "Yil" in df_model else 2000
yil_max = int(df_model["Yil"].max()) if "Yil" in df_model else 2025
km_max = int(df_model["Kilometre"].max()) if "Kilometre" in df_model else 500000

marka_list = secenekler("Marka", ["Nissan", "Hyundai"])
vites_list = secenekler("Vites_Tipi", ["Otomatik", "Düz", "Yarı Otomatik"])
yakit_list = secenekler("Yakit_Tipi", ["Benzin", "Dizel", "LPG", "Hibrit", "Elektrik"])
kasa_list = secenekler("Kasa_Tipi", ["SUV", "Sedan", "Hatchback"])
cekis_list = secenekler("Cekis", ["Önden Çekiş"])
kimden_list = secenekler("Kimden", ["Sahibinden", "Galeriden", "Yetkili Bayiden"])

yil = st.sidebar.slider("Araç Yılı", yil_min, yil_max, min(2018, yil_max))
km = st.sidebar.slider("Kilometre", 0, km_max, 100000)
marka = st.sidebar.selectbox("Marka", marka_list)
vites = st.sidebar.selectbox("Vites Tipi", vites_list)
yakit = st.sidebar.selectbox("Yakıt Tipi", yakit_list)
kasa = st.sidebar.selectbox("Kasa Tipi", kasa_list)
cekis = st.sidebar.selectbox("Çekiş Tipi", cekis_list)
kimden = st.sidebar.selectbox("Kimden", kimden_list)

motor_gucu = st.sidebar.number_input("Motor Gücü (HP)", min_value=60, max_value=600, value=130)
motor_hacmi = st.sidebar.number_input("Motor Hacmi (cc)", min_value=800, max_value=6000, value=1500)
ilan_fiyati = st.sidebar.number_input(
    "İlan Fiyatı (Opsiyonel, TL)",
    min_value=0,
    value=0,
    step=10000,
    help="Girerseniz araç FIRSAT / MAKUL / PAHALI olarak sınıflandırılır.",
)

if "detayli_analiz" not in st.session_state:
    st.session_state.detayli_analiz = False

if st.sidebar.button("Fiyatı Tahmin Et", type="primary"):
    st.session_state.detayli_analiz = True

if st.session_state.detayli_analiz and st.sidebar.button("Detaylı Analizi Gizle"):
    st.session_state.detayli_analiz = False

med_hizlanma = get_stat_value("Hizlanma_0_100", 11.5)
med_uzunluk = get_stat_value("Uzunluk", 4394)
med_genislik = get_stat_value("Genislik", 1806)
med_yukseklik = get_stat_value("Yukseklik", 1615)
med_koltuk = get_stat_value("Koltuk_Sayisi", 5)
med_silindir = get_stat_value("Silindir_Sayisi", 4)
med_kasko = get_stat_value("Ort_Kasko", 15946)

ozellikler = {
    "Yil": yil,
    "Kilometre": km,
    "Motor_Hacmi": float(motor_hacmi),
    "Motor_Gucu": float(motor_gucu),
    "Hizlanma_0_100": med_hizlanma,
    "Uzunluk": med_uzunluk,
    "Genislik": med_genislik,
    "Yukseklik": med_yukseklik,
    "Koltuk_Sayisi": med_koltuk,
    "Silindir_Sayisi": med_silindir,
    "Ort_Kasko": med_kasko,
    "Marka_encoded": safe_encode("Marka", marka),
    "Vites_Tipi_encoded": safe_encode("Vites_Tipi", vites),
    "Yakit_Tipi_encoded": safe_encode("Yakit_Tipi", yakit),
    "Kasa_Tipi_encoded": safe_encode("Kasa_Tipi", kasa),
    "Cekis_encoded": safe_encode("Cekis", cekis),
    "Kimden_encoded": safe_encode("Kimden", kimden),
}

x_model = prepare_model_input(ozellikler)
prediction = predict_price(x_model, aktif_model)

mape_val = 5.15
r2_val = None
if not df_model_sonuc.empty:
    model_row = df_model_sonuc[df_model_sonuc["Model"] == aktif_model_adi]
    if not model_row.empty:
        mape_val = float(model_row.iloc[0]["MAPE"])
        r2_val = float(model_row.iloc[0]["R2"])
band = prediction * (mape_val / 100.0)

st.subheader("Tahmin Sonuçları")

nadir_uyari = nadir_kombinasyon_uyarisi(df_model, yil, km)
if nadir_uyari:
    st.warning(nadir_uyari)

c1, c2, c3 = st.columns(3)

delta_text = "İlan fiyatı girilmedi"
delta_color = "off"
delta_aciklama = "Karşılaştırma için İlan Fiyatı alanını doldurun"
if ilan_fiyati > 0:
    durum, fark_pct = durum_etiketi(float(ilan_fiyati), prediction, esik=10.0)
    abs_fark_pct = abs(fark_pct)
    if durum == "FIRSAT":
        delta_text = f"-{abs_fark_pct:.1f}%"
        delta_color = "inverse"
        delta_aciklama = "Piyasanın Altında"
    elif durum == "PAHALI":
        delta_text = f"+{abs_fark_pct:.1f}%"
        delta_color = "inverse"
        delta_aciklama = "Piyasanın Üstünde"
    else:
        delta_text = "Piyasa Değerinde"
        delta_color = "off"
        delta_aciklama = "Makul Aralık"

c1.metric(
    "Tahmini Piyasa Değeri",
    f"{prediction:,.0f} TL",
    delta=delta_text,
    delta_color=delta_color,
)
c2.metric("Beklenen Hata Payı (MAPE)", f"%{mape_val:.2f}")
c3.metric("Tahmini Aralık", f"{prediction - band:,.0f} - {prediction + band:,.0f} TL")

fiyat_min = float(df_model["Fiyat"].min()) if "Fiyat" in df_model and len(df_model) > 0 else max(0.0, prediction - band)
fiyat_max = float(df_model["Fiyat"].max()) if "Fiyat" in df_model and len(df_model) > 0 else prediction + band
if fiyat_max > fiyat_min:
    progress_orani = int(np.clip((prediction - fiyat_min) / (fiyat_max - fiyat_min) * 100, 0, 100))
    st.progress(progress_orani, text=f"Tahmin konumu: %{progress_orani} (Min: {fiyat_min:,.0f} TL | Max: {fiyat_max:,.0f} TL)")

st.caption(f"Fiyat karşılaştırması: {delta_aciklama}")
if ilan_fiyati > 0:
    if durum == "FIRSAT":
        st.success(f"İlan fiyatı FIRSAT görünüyor. Sapma: %{fark_pct:.1f}")
    elif durum == "PAHALI":
        st.error(f"İlan fiyatı PAHALI görünüyor. Sapma: %{fark_pct:.1f}")
    else:
        st.info(f"İlan fiyatı MAKUL aralıkta. Sapma: %{fark_pct:.1f}")

if st.session_state.detayli_analiz:
    shap_top = None
    shap_hata = None
    try:
        explainer = shap.TreeExplainer(aktif_model)
        shap_values = explainer.shap_values(x_model)
        shap_array = np.array(shap_values)
        if shap_array.ndim == 3:
            shap_row = shap_array[0, 0, :]
        elif shap_array.ndim == 2:
            shap_row = shap_array[0, :]
        else:
            shap_row = shap_array.reshape(-1)

        shap_df = pd.DataFrame({
            "Ozellik": feat_names_model,
            "SHAP": shap_row,
        })
        shap_df["Mutlak"] = shap_df["SHAP"].abs()
        shap_top = shap_df.sort_values("Mutlak", ascending=False).head(10)
    except Exception as e:
        shap_hata = str(e)

    st.divider()

    sol, sag = st.columns([1.15, 1.0])
    with sol:
        st.markdown("### Benzer İlan Özeti")
        if df_piyasa.empty:
            st.warning("piyasa_analizi.csv bulunamadı. Benzer ilan analizi gösterilemiyor.")
        else:
            filt = df_piyasa[
                (df_piyasa["Marka"] == marka)
                & (df_piyasa["Yil"].between(yil - 1, yil + 1))
                & (df_piyasa["Kilometre"].between(max(0, km - 30000), km + 30000))
            ]

            if filt.empty:
                st.info("Bu filtrelerde benzer ilan bulunamadı.")
            else:
                med_ilan = float(filt["Fiyat"].median())
                med_tahmin = float(filt["Tahmin_Fiyat"].median())
                ort_sapma = float(filt["Fark_Pct"].mean())
                s1, s2, s3 = st.columns(3)
                s1.metric("Benzer İlan Sayısı", f"{len(filt):,}")
                s2.metric("Medyan İlan Fiyatı", f"{med_ilan:,.0f} TL")
                s3.metric("Ort. Sapma", f"%{ort_sapma:.2f}")

                durum_sayim = filt["Durum"].value_counts().reindex(["FIRSAT", "MAKUL", "PAHALI"], fill_value=0)
                st.bar_chart(durum_sayim)

                cols = [
                    "Yil",
                    "Kilometre",
                    "Yakit_Tipi",
                    "Vites_Tipi",
                    "Fiyat",
                    "Tahmin_Fiyat",
                    "Fark_Pct",
                    "Durum",
                ]
                goster = filt[cols].sort_values("Fark_Pct").head(10).reset_index(drop=True)
                st.dataframe(goster, use_container_width=True)

    with sag:
        st.markdown("### Segment Dağılımı")
        if df_segment.empty:
            st.info("segmentli_araclar.csv bulunamadı.")
        else:
            seg = (
                df_segment[df_segment["Marka"] == marka]["Segment"]
                .value_counts(normalize=True)
                .mul(100)
                .round(1)
            )
            if seg.empty:
                st.info("Seçilen marka için segment bilgisi bulunamadı.")
            else:
                st.bar_chart(seg)
                st.caption("Yüzdesel dağılım, seçilen markadaki ilanlara göre hesaplanır.")

        st.markdown("### Model Bilgisi")
        if r2_val is not None:
            st.caption(f"Model: {model_gorunur_ad(aktif_model_adi)} | R²: {r2_val:.3f} | MAPE: %{mape_val:.2f}")
        else:
            st.caption(f"Model: {model_gorunur_ad(aktif_model_adi)} | MAPE: %{mape_val:.2f}")

        if hasattr(aktif_model, "feature_importances_"):
            fi = pd.Series(aktif_model.feature_importances_, index=feat_names_model).sort_values(ascending=False).head(8)
            st.dataframe(fi.rename("Önem Skoru").to_frame(), use_container_width=True)

        st.markdown("### SHAP Açıklaması")
        if shap_top is not None:
            st.caption("Pozitif SHAP değeri fiyatı artırır, negatif SHAP değeri düşürür.")
            st.dataframe(shap_top[["Ozellik", "SHAP"]], use_container_width=True)
        elif shap_hata:
            st.info("Bu model için SHAP hesaplanamadı. Ağaç tabanlı modellerde daha stabil çalışır.")

    with st.expander("Tahminde kullanılan özellikler"):
        st.dataframe(pd.DataFrame([ozellikler]).T.rename(columns={0: "Değer"}), use_container_width=True)
else:
    st.info("Canlı tahmin gösteriliyor. Detaylı analiz için sol panelden Fiyatı Tahmin Et butonuna basın.")

st.divider()
tab_ozet, tab_karsilastirma = st.tabs(["Özet", "Model Karşılaştırması"])

with tab_ozet:
    st.caption("Model seçmeden önce diğer sekmeden model özelliklerini görüntüleyebilirsiniz.")

with tab_karsilastirma:
    if df_model_sonuc.empty:
        st.warning("model_sonuclari.csv bulunamadı.")
    else:
        cols = ["Model", "R2", "RMSE", "MAE", "MAPE", "Sure_sn"]
        tablo = df_model_sonuc[cols].sort_values("R2", ascending=False).copy()
        tablo.insert(0, "Aktif", np.where(tablo["Model"] == aktif_model_adi, "       ✅", ""))
        st.dataframe(tablo, use_container_width=True)