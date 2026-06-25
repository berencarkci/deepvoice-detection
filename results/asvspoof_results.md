# ASVspoof 2019 LA — Sonuç Özeti

> `asvspoof_results.py` tarafından `asvspoof_results.csv`'den üretilir.

Değerlendirme: **saldırı-gruplu 5 katlı çapraz doğrulama** — her katta modeller görülmemiş saldırı türleriyle test edilir. Ana metrik **EER** (düşük = daha iyi); sınıf dengesizliği (~1:9) nedeniyle dengeli doğruluk (BalACC) ve AUC de raporlanır.

| Yöntem | EER | Accuracy | BalACC | Precision | Recall | F1 | AUC |
|--------|-----|----------|--------|-----------|--------|----|-----|
| RF | 0.1708 ± 0.0671 | 0.9023 | 0.6641 | 0.6579 | 0.3636 | 0.4300 | 0.9066 |
| SVM | 0.1374 ± 0.0549 | 0.8754 | 0.8500 | 0.5443 | 0.8172 | 0.6204 | 0.9355 |
| Mel-CNN | 0.0370 ± 0.0407 | 0.9678 | 0.9566 | 0.8030 | 0.9434 | 0.8570 | 0.9904 |
| Spec-CNN | 0.0954 ± 0.1022 | 0.8553 | 0.9002 | 0.6031 | 0.9539 | 0.6659 | 0.9424 |

EER değerleri 5 katın ortalaması ± standart sapma olarak verilir.
