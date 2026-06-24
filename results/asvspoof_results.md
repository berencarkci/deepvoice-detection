# ASVspoof 2019 LA — Sonuç Özeti

> `asvspoof_results.py` tarafından `asvspoof_results.csv`'den üretilir.

Değerlendirme: **saldırı-gruplu 5 katlı çapraz doğrulama** — her katta modeller görülmemiş saldırı türleriyle test edilir. Ana metrik **EER** (düşük = daha iyi); sınıf dengesizliği (~1:9) nedeniyle dengeli doğruluk (BalACC) ve AUC de raporlanır.

| Yöntem | EER | Accuracy | BalACC | Precision | Recall | F1 | AUC |
|--------|-----|----------|--------|-----------|--------|----|-----|
| RF | 0.1708 ± 0.0671 | 0.9023 | 0.6641 | 0.6579 | 0.3636 | 0.4300 | 0.9066 |
| SVM | 0.1374 ± 0.0549 | 0.8754 | 0.8500 | 0.5443 | 0.8172 | 0.6204 | 0.9355 |
| Mel-CNN | 0.0376 ± 0.0414 | 0.9190 | 0.9450 | 0.6758 | 0.9788 | 0.7644 | 0.9898 |
| Spec-CNN | 0.1244 ± 0.1515 | 0.8147 | 0.8916 | 0.6391 | 0.9838 | 0.6897 | 0.9127 |

EER değerleri 5 katın ortalaması ± standart sapma olarak verilir.
