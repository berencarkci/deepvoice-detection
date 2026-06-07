# ASVspoof 2019 LA — Sonuç Özeti

> `asvspoof_results.py` tarafından `asvspoof_results.csv`'den üretilir.

Metrik = **EER** (düşük = daha iyi). Sınıf dengesizliği (~1:9) nedeniyle EER, dengeli doğruluk (BalACC) ve AUC esas alınır.

- **Havuzlanmış k-fold:** train+dev+eval birleştirilir, konuşmacıya göre GroupKFold(5).
- **Standart protokol:** train ile eğit, dev ile doğrula, eval ile test.

## Ana tablo — EER

| Yöntem | Havuzlanmış k-fold | Standart protokol (eval) | Oran |
|--------|--------------------|--------------------------|------|
| RF | 0.0902 ± 0.0154 | 0.1437 | 1.6× |
| SVM | 0.0454 ± 0.0060 | 0.1306 | 2.9× |
| Mel-CNN | 0.0063 ± 0.0029 | 0.0453 | 7.2× |
| Spec-CNN | 0.0041 ± 0.0029 | 0.1146 | 28.0× |

### Havuzlanmış 5-fold çapraz doğrulama

| Yöntem | EER | Accuracy | BalACC | Precision | Recall | F1 | AUC |
|--------|-----|----------|--------|-----------|--------|----|-----|
| RF | 0.0902 | 0.9311 | 0.6767 | 0.9468 | 0.3556 | 0.5114 | 0.9700 |
| SVM | 0.0454 | 0.9731 | 0.9265 | 0.8720 | 0.8673 | 0.8681 | 0.9906 |
| Mel-CNN | 0.0063 | 0.9968 | 0.9924 | 0.9801 | 0.9871 | 0.9836 | 0.9996 |
| Spec-CNN | 0.0041 | 0.9971 | 0.9956 | 0.9780 | 0.9937 | 0.9857 | 0.9998 |

### Standart protokol (dev / doğrulama)

| Yöntem | EER | Accuracy | BalACC | Precision | Recall | F1 | AUC |
|--------|-----|----------|--------|-----------|--------|----|-----|
| RF | 0.1170 | 0.9142 | 0.5868 | 0.9370 | 0.1750 | 0.2950 | 0.9504 |
| SVM | 0.0754 | 0.9539 | 0.7873 | 0.9546 | 0.5777 | 0.7198 | 0.9740 |

### Standart protokol (eval)

| Yöntem | EER | Accuracy | BalACC | Precision | Recall | F1 | AUC |
|--------|-----|----------|--------|-----------|--------|----|-----|
| RF | 0.1437 | 0.9035 | 0.6197 | 0.5718 | 0.2620 | 0.3593 | 0.9238 |
| SVM | 0.1306 | 0.9070 | 0.8271 | 0.5366 | 0.7264 | 0.6173 | 0.9392 |
| Mel-CNN | 0.0453 | 0.9493 | 0.9535 | 0.6807 | 0.9587 | 0.7961 | 0.9920 |
| Spec-CNN | 0.1146 | 0.8411 | 0.9063 | 0.3929 | 0.9886 | 0.5623 | 0.9532 |

"Oran" sütunu = standart protokol EER / havuzlanmış k-fold EER.
