Streamlit harici dosyalar Atalay Durmaz tarafindan hazirlanmistir.

# Ikinci El Arac Fiyat Tahmini Dashboard Paketi

Bu dashboard, ikinci el araclar icin model tabanli fiyat tahmini yapmanizi ve tahmini piyasa verileriyle hizlica karsilastirmanizi saglar. Tek ekran uzerinden model secimi, fiyat analizi ve aciklanabilirlik goruntulemesi sunar.

## Kapsam

Bu yuklemede sadece su dosyalar bulunur:

- `streamlit_app.py`
- `.streamlit/config.toml`
- `requirements.txt`
- `outputs/` klasoru altindaki dashboard girdileri (model ve veri dosyalari)

Not: Veri on isleme, model egitimi ve optimizasyon scriptleri bu pakette yer almaz.

## Gerekli Dosyalar

Dashboard'in calismasi icin asagidaki dosyalarin var olmasi gerekir:

- `outputs/islenmis_veri.pkl`
- `outputs/ml_modeller.pkl`
- `outputs/model_sonuclari.csv`
- `outputs/piyasa_analizi.csv`
- `outputs/segmentli_araclar.csv`
- `outputs/ozet_istatistikler.csv`

## Kurulum

```bash
pip install -r requirements.txt
```

## Calistirma

```bash
streamlit run streamlit_app.py
```

## Dashboard Ozeti

- Tahmin modeli secimi
- Canli fiyat tahmini
- Ilan fiyati ile FIRSAT / MAKUL / PAHALI karsilastirmasi
- Tahmin araligi ve piyasa ozeti
- Detayli analiz modunda SHAP aciklamasi

## Kisa Not

Bu paket yalnizca sunum ve demo amaclidir. Modeli yeniden egitmek veya tum pipeline'i calistirmak icin orijinal proje dosyalarina ihtiyac vardir.
