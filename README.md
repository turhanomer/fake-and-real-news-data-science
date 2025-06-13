# Sahte ve Gerçek Haber Sınıflandırma Projesi

Bu proje, makine öğrenmesi teknikleri kullanarak haber metinlerinin gerçek veya sahte olarak sınıflandırılmasını amaçlamaktadır. Proje, doğal dil işleme ve çeşitli sınıflandırma algoritmaları kullanarak yüksek doğrulukta tahminler yapmaktadır.

## Veri Seti Hakkında

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Proje iki ana veri seti kullanmaktadır:
- `True.csv`: Güvenilir kaynaklardan alınmış gerçek haber makaleleri
- `Fake.csv`: Tespit edilmiş sahte haber makaleleri

Her haber makalesi şu bilgileri içermektedir:
- Başlık
- Metin içeriği
- Konu
- Yayın tarihi

## Teknik Gereksinimler

- Gerekli Python paketleri (requirements.txt dosyasında belirtilmiştir):
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - nltk

## Kurulum ve Çalıştırma

1. Bu depoyu bilgisayarınıza klonlayın
2. Sanal ortam oluşturun (önerilir):
```bash
python -m venv venv
venv\Scripts\activate  # Windows için
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Analizi çalıştırın:
```bash
python analysis.py
```

## Proje Bileşenleri

### Ana Script
- `analysis.py`: Gelişmiş analiz scripti şunları içerir:
  - Veri setinin yüklenmesi ve ön işlenmesi
  - Metin vektörizasyonu
  - Model eğitimi ve değerlendirmesi
  - Performans metriklerinin hesaplanması
  - Görselleştirmelerin oluşturulması

### Çıktı Dosyaları
Analiz sonucunda oluşturulan dosyalar:
1. `analysis_results.txt`: Detaylı analiz sonuçları
2. `confusion_matrices`: Model performans görselleştirmesi
3. `roc_curves.png`: ROC eğrileri
4. `feature_distributions.png`: Özellik önemliliği dağılımları

## Kullanılan Makine Öğrenmesi Modelleri

Proje üç farklı sınıflandırma algoritması kullanmaktadır:

1. Naive Bayes Sınıflandırıcı
   - Metin sınıflandırma için hızlı ve verimli
   - Yüksek boyutlu verilerle iyi çalışır
   - Haber sınıflandırma için özellikle etkili

2. Karar Ağacı Sınıflandırıcı
   - Yorumlanabilir sonuçlar sağlar
   - Karmaşık veri desenlerini yakalayabilir
   - Özellik önemliliğini anlamak için kullanışlı

3. Random Forest Sınıflandırıcı
   - Daha yüksek doğruluk oranı
   - Aşırı öğrenmeye karşı dirençli
   - Özellik önemliliği analizi için ideal

## Performans Metrikleri

Modeller şu metriklerle değerlendirilmektedir:
- Doğruluk (Accuracy): Genel tahmin doğruluğu
- Kesinlik (Precision): Pozitif tahminlerin doğruluğu
- Duyarlılık (Recall): Tüm pozitif durumları bulma yeteneği
- F1-Skoru: Kesinlik ve duyarlılığın harmonik ortalaması

## Metin Ön İşleme

Proje, metin verilerini işlemek için gelişmiş teknikler kullanır:
- Noktalama işaretlerinin kaldırılması
- Küçük harfe dönüştürme
- Stop words (etkisiz kelimeler) temizleme
- Tokenization (kelime ayırma)
- Bigram özellik çıkarımı
