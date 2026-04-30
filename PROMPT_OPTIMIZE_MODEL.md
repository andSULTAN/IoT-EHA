# IoT-Shield Model Accuracy Optimization — 68% → 80-85%

## KONTEKST
Men IoT-Shield NIDS (Network Intrusion Detection System) loyihasi ustida ishlayman. 
Tizim CICIOT2023 datasetida (45M rows, 8.66GB, 169 CSV files, 34 attack classes) 
SGDClassifier bilan Incremental Learning usulida o'qitilgan. 

Hozirgi natija: Accuracy=0.6832, F1_weighted=0.6859, F1_macro=0.4771
Maqsad: Accuracy 80-85%, F1_macro 60-70%

Loyiha fayllari: trainer.py, data_loader.py, detector.py, guard.py, bot_handler.py, ui_design.py
O'qitish kompyuteri: Core i5 11th gen, 24GB RAM, GTX 1050Ti. Paradigma: Incremental Learning (partial_fit).

## MUHIM: Nima O'ZGARTIRISH MUMKIN EMAS
- Paradigma Incremental Learning bo'lib qolishi SHART (dissertatsiya mavzusi)
- partial_fit() ishlatilishi SHART
- SGDClassifier asosiy algoritm bo'lib qolishi SHART
- Loyiha strukturasi (6 ta modul) o'zgarmaydi
- detector.py dagi FEATURE_ORDER (37 ta xususiyat) va BENIGN_LABELS o'zgarmaydi
- PyQt6 GUI integratsiyasi (signallar) saqlanishi kerak

## 6 TA MUAMMONI KETMA-KET TUZATISH

### MUAMMO 1: Scaler faqat birinchi fayldan o'rganilgan
**Fayl:** data_loader.py
**Hozirgi holat:** `fit_scaler_from_first_file()` metodi faqat `self.file_paths[0]` dan scaler o'rganadi. CICIOT2023 da 169 ta fayl bor, har birining qiymatlar diapazoni boshqacha. Bitta fayldan o'rgangan scaler qolgan 168 ta faylni noto'g'ri normallaydi.
**Yechim:** `fit_scaler_from_first_file()` ni `fit_scaler_multi_file()` ga o'zgartir. StandardScaler.partial_fit() ni 5-10 ta fayldan (har xil hujum turlaridan) chunk-lab chaqir. Barcha fayllarni o'qish shart emas — har 15-20-chi fayldan bitta chunk o'qish yetarli. Metod nomi va signatura o'zgarsa, trainer.py dagi chaqiruvni ham yangilagin.

### MUAMMO 2: Ansambl modeli yo'q — yagona SGDClassifier
**Fayl:** trainer.py
**Hozirgi holat:** Faqat bitta `SGDClassifier(loss='modified_huber')` yaratilgan. Ansambl yo'q.
**Yechim:** 3 ta mustaqil model yarat:
```python
models = {
    'sgd_mhuber': SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, max_iter=1, tol=None, random_state=42, n_jobs=-1),
    'sgd_log': SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=1, tol=None, random_state=43, n_jobs=-1),
    'sgd_pa': SGDClassifier(loss='modified_huber', penalty='l1', alpha=5e-5, max_iter=1, tol=None, random_state=44, n_jobs=-1),
}
weights = {'sgd_mhuber': 1.0, 'sgd_log': 1.0, 'sgd_pa': 0.8}
```
Har bir chunk uchun uchala modelga ham `partial_fit()` chaqir. 
Bashorat qilganda Soft Voting ishlat:
```python
def ensemble_predict(models, weights, X):
    probas = []
    for name, model in models.items():
        p = model.predict_proba(X)
        probas.append(p * weights[name])
    avg_proba = np.sum(probas, axis=0) / sum(weights.values())
    return np.argmax(avg_proba, axis=1), np.max(avg_proba, axis=1)
```
Saqlashda 3 ta modelni bitta dict ichida joblib.dump() bilan saqla. Metadata ga model turlari va vaznlarni yoz.

### MUAMMO 3: Sample weight (balanced) yo'q
**Fayl:** trainer.py
**Hozirgi holat:** `class_weight=None` va hech qanday sample_weight ishlatilmagan. CICIOT2023 da DDoS-ICMP_FLOOD 7.2M namuna, SQL Injection 1250 ta — nisbat 5800:1.
**Yechim:** Har bir chunk uchun `compute_sample_weight('balanced', y_chunk)` hisoblash va uni `partial_fit(X_chunk, y_chunk, classes=all_classes, sample_weight=sw)` ga berish.
```python
from sklearn.utils.class_weight import compute_sample_weight
# har bir chunk ichida:
sw = compute_sample_weight('balanced', y_chunk)
for name, model in models.items():
    model.partial_fit(X_chunk, y_chunk, classes=all_classes_encoded, sample_weight=sw)
```
Bu kam sonli sinflarning gradiyentga ta'sirini oshiradi.

### MUAMMO 4: Round-robin streaming yo'q — Catastrophic Forgetting
**Fayl:** trainer.py va data_loader.py
**Hozirgi holat:** Fayllar ketma-ket o'qilmoqda (`for file_path in file_paths`). CICIOT2023 da birinchi fayllar faqat DDoS, keyingilari Mirai va h.k. Ketma-ket o'qitish model oxirgi ko'rgan sinfga moslashib, avvalgilarni "unutadi".
**Yechim:** data_loader.py ga yangi metod qo'sh:
```python
def stream_round_robin(self, chunksize=50000):
    """Barcha fayllardan navbatma-navbat chunk o'qish generatori."""
    import itertools
    generators = []
    for fp in self.file_paths:
        generators.append(self.stream_file_chunks(fp, chunksize=chunksize))
    
    active = list(range(len(generators)))
    while active:
        next_active = []
        for i in active:
            try:
                X_chunk, y_chunk = next(generators[i])
                yield X_chunk, y_chunk
            except StopIteration:
                continue
            else:
                next_active.append(i)
        active = next_active
```
trainer.py da eski `for file_idx, file_path in enumerate(file_paths)` tsiklini o'rniga:
```python
chunk_count = 0
for X_chunk, y_chunk in dl.stream_round_robin(chunksize=50000):
    # shuffle
    X_chunk, y_chunk = sklearn_shuffle(X_chunk, y_chunk, random_state=42+chunk_count)
    # sample weight
    sw = compute_sample_weight('balanced', y_chunk)
    # partial_fit har bir modelga
    for name, model in models.items():
        model.partial_fit(X_chunk, y_chunk, classes=all_classes_encoded, sample_weight=sw)
    chunk_count += 1
    total_samples_trained += len(X_chunk)
    # progress va log yangilash
```
Chunksize ni 100000 dan 50000 ga tushir — round-robin da sinf aralashmasi yaxshiroq bo'ladi.

### MUAMMO 5: Test set faqat oxirgi faylning birinchi chunkidan olingan
**Fayl:** trainer.py
**Hozirgi holat:** `is_last_file and chunk_count == 1` da test ajratilgan. Bu faqat bitta hujum turini qamraydi.
**Yechim:** O'qitishdan OLDIN stratified test set ajrat. data_loader.py ga yangi metod:
```python
def create_stratified_test_set(self, test_size=50000, log_callback=None):
    """Har bir fayldan proporsional test namunalari ajratish."""
    samples_per_file = max(100, test_size // len(self.file_paths))
    test_X_list, test_y_list = [], []
    
    for fp in self.file_paths:
        for X_chunk, y_chunk in self.stream_file_chunks(fp, chunksize=samples_per_file):
            test_X_list.append(X_chunk[:samples_per_file])
            test_y_list.append(y_chunk[:samples_per_file])
            break  # Har fayldan faqat bitta chunk
    
    test_X = np.vstack(test_X_list)
    test_y = np.concatenate(test_y_list)
    if log_callback:
        log_callback(f"   └── Test set: {len(test_y)} namuna, {len(set(test_y))} sinf")
    return test_X, test_y
```
trainer.py da o'qitishdan oldin chaqir:
```python
test_X, test_y = dl.create_stratified_test_set(test_size=50000, log_callback=self._emit_log)
```
O'qitish tsiklida test set ajratish kodini olib tashla.

### MUAMMO 6: Class Grouping (34 → 15 sinf)
**Fayl:** data_loader.py
**Hozirgi holat:** 34 ta nozik sinf ishlatilmoqda. DOS-SYN_FLOOD va DDOS-SYN_FLOOD o'rtasida farq juda nozik — chiziqli model ularni ajratishga qiynaladi.
**Yechim:** data_loader.py ga CLASS_GROUPING dict qo'sh va stream_file_chunks ichida label ni guruhga almashtir:

```python
CLASS_GROUPING = {
    'BenignTraffic': 'BENIGN',
    'BENIGN': 'BENIGN',
    'DDoS-RSTFINFlood': 'DDOS-RSTFIN_FLOOD',
    'DDoS-PSHACK_Flood': 'DDOS-PSHACK_FLOOD',
    'DDoS-SYN_Flood': 'SYN_FLOOD_ATTACK',
    'DoS-SYN_Flood': 'SYN_FLOOD_ATTACK',
    'DDoS-TCP_Flood': 'TCP_FLOOD_ATTACK',
    'DoS-TCP_Flood': 'TCP_FLOOD_ATTACK',
    'DDoS-UDP_Flood': 'UDP_FLOOD_ATTACK',
    'DoS-UDP_Flood': 'UDP_FLOOD_ATTACK',
    'DDoS-HTTP_Flood': 'HTTP_FLOOD_ATTACK',
    'DoS-HTTP_Flood': 'HTTP_FLOOD_ATTACK',
    'DDoS-ICMP_Flood': 'DDOS-ICMP_FLOOD',
    'DDoS-ICMP_Fragmentation': 'FRAGMENTATION_ATTACK',
    'DDoS-UDP_Fragmentation': 'FRAGMENTATION_ATTACK',
    'DDoS-ACK_Fragmentation': 'FRAGMENTATION_ATTACK',
    'DDoS-SlowLoris': 'SLOWLORIS_ATTACK',
    'DDoS-SynonymousIP_Flood': 'SYN_FLOOD_ATTACK',
    'DoS-SYN_Flood': 'SYN_FLOOD_ATTACK',
    'Mirai-greeth_flood': 'MIRAI_BOTNET',
    'Mirai-greip_flood': 'MIRAI_BOTNET',
    'Mirai-udpplain': 'MIRAI_BOTNET',
    'Recon-PingSweep': 'RECONNAISSANCE',
    'Recon-OSScan': 'RECONNAISSANCE',
    'Recon-PortScan': 'RECONNAISSANCE',
    'Recon-HostDiscovery': 'RECONNAISSANCE',
    'VulnerabilityScan': 'RECONNAISSANCE',
    'DNS_Spoofing': 'SPOOFING',
    'MITM-ArpSpoofing': 'SPOOFING',
    'DictionaryBruteForce': 'BRUTE_FORCE',
    'SqlInjection': 'WEB_ATTACK',
    'XSS': 'WEB_ATTACK',
    'CommandInjection': 'WEB_ATTACK',
    'BrowserHijacking': 'WEB_ATTACK',
    'Backdoor_Malware': 'MALWARE',
    'Uploading_Attack': 'WEB_ATTACK',
}
```

DIQQAT: CICIOT2023 dagi sinf nomlari CaseSensitive — aynan datasetdagi yozilishiga mos keling. Agar nomlarda farq bo'lsa, avval scan_all_classes() natijasidagi class_names ro'yxatini tekshirib, CLASS_GROUPING kalitlarini ularga mosla.

`stream_file_chunks()` ichida label o'qilgandan keyin:
```python
raw_labels = chunk[self.label_column].astype(str)
# Class grouping qo'llash
if hasattr(self, 'use_grouping') and self.use_grouping:
    raw_labels = raw_labels.map(lambda x: CLASS_GROUPING.get(x, x))
```

`scan_all_classes()` da ham grouping ni qo'lla — shunda class_names 34 o'rniga 15 ta bo'ladi.

DataLoader ga `self.use_grouping = True` attribut qo'sh va trainer.py da o'qitishdan oldin `dl.use_grouping = True` qil.

## BAJARISH TARTIBI
1. Avval data_loader.py ni yangilagin (Muammo 1, 4, 5, 6)
2. Keyin trainer.py ni yangilagin (Muammo 2, 3, va yangi data_loader metodlarini chaqirish)
3. Barcha o'zgarishlardan keyin kodni ishga tushir va natijalarni ko'rsat

## NATIJANI TEKSHIRISH
O'qitish tugagach quyidagilarni ko'rsat:
- Accuracy, F1_weighted, F1_macro
- Per-class F1 (classification_report)
- Har bir model (sgd_mhuber, sgd_log, sgd_pa) ning individual accuracy si
- Ansambl vs individual modellar qiyosi
- O'qitish vaqti

## ESLATMA
- GUI signallarini (progress_updated, log_message, training_completed, training_failed) saqla
- MODELS_DIR ga 3 ta model + scaler + encoder + metadata saqlash tartibini yangila
- Log xabarlarini saqla — foydalanuvchi progress ni ko'rishi kerak
- Chunk size = 50000 (100000 dan kamaytir — round-robin uchun optimal)
