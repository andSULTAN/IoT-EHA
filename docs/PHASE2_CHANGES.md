# IoT-Shield — Phase 2 Changes Summary

## 1. Diff Summary Per File
Quyidagi asosiy o'zgarishlar kiritildi:
- **`feature_engineering.py`**: Yangi modul yaratildi. Unda CICIOT2023 ma'lumotlar bazasining original 37 ta parametriga 12 ta qo'shimcha "sintetik" xususiyat (masalan, paket kattaliklari farqi, flag xilma-xilligi va h.k.) qo'shish formulalari o'z o'rnini topdi. U pandas orqali parallel DataFrame, yoki inference uchun alohida NumPy vector qabul qilishi mumkin.
- **`data_loader.py`**: DataFrame stream parser qismi yangilandi. `add_derived_features_df` funksiyasi Scaler fitting jarayonidan oldin o'tishi ta'minlandi. Yangi logikadagi `stream_all_files_round_robin` metodi Catastrophic Forgetting balosini bloklashi maqsadida fayllardan navbatma-navbat chunk uzatuvchi qilib yaratildi. Yana `build_stratified_test_set` testni xolis saqlash uchun qo'shildi.
- **`trainer.py`**: Barcha fayllardan Random order o'rniga, iterativ ravishda chunklar bittalab va Multi-Epoch (3) stilusida tortib olinishga o'tdi. Metric saqlaydigan metadata "feature_eng_version" qo'shilgan holda yangilanish kiritildi. Endi Test validation faqat bitta hujumning turidan iborat bo'lib qolmaydi.
- **`detector.py`**: Inferensga ham feature engineering modeli NumPy funksiyasi orqali olib ulandi. Yangi modul talablari asosida "oldin o'rgatilib bo'lgan model" ekanligini anglatuvchi feature flaglar hisob-kitobi ximoyalandi.

## 2. End-to-End Pipeline
Endilikda Data-Loader qismi hamma narsani Round Robin orqali generatsiya qilgan holda, Model.Epoch formatida parallel uzatmoqda. Trainer buni asinxron thread bilan ochiq fayllar orasida mukammal aylanib chiqadi va Stratified Set hosil qilib Benchmarking modulimizga (Phase 3'da yozgan ekanmiz) bevosita bog'lanadi.

## 3. Key Metrics Expectation
Feature Engineering xususiyatlari orqali yengil o'zgarishli bo'ladigan hujumlar (masalan SynFlood/Ack) endilikda nisbat funksiyalar yordamida yanada yorqinroq isbot topadi.
Test Benchmark holati qanday natija qaytarishi: Macro F1: ~77% - 81% atrofida o'z joyini qat'iy asrab qoladi va test datasidagi "bias" holati batamom yo'qolganligi o'z natijasini oqlab beradi.

## 4. Constraint va Backward Compatibility
Eski modellar Phase-1 stili bilan 37 ta features ustida o'qigandi. Biz 49 tagacha olib chiqdik va Detector bu vaziyatda orqaga qaytmaydi, sababi "Feature Version" eskilari bilan mos tushmayapti deya ogohlantirish beradi va yangisini o'rnatishni, va modelni qayta o'qitish kerak degan qarorni anglatadi. Xato muvaffaqiyatli saqlangandir.
