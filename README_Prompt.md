# IoT-Shield Claude Code Optimization Prompts

Bu papkada IoT-Shield loyihasini bosqichma-bosqich yaxshilash uchun **3 ta professional texnik prompt** mavjud. Har biri Claude Code uchun to'liq spec sifatida yozilgan.

## 📋 Promptlar ro'yxati

| # | Fayl | Maqsad | Vaqt | Kutilgan natija |
|---|------|--------|------|-----------------|
| 1 | `PROMPT_1_Critical_Fixes.md` | Kritik bug'lar + xavfsizlik | 1-2 soat | 74.86% → **78-82%** |
| 2 | `PROMPT_2_Feature_Engineering.md` | Feature engineering + training loop | 2-3 soat | 78-82% → **82-86%** |
| 3 | `PROMPT_3_Ensemble_Benchmarking.md` | Ensemble + tuning + benchmarking | 3-4 soat | 82-86% → **85-89%** |

**Jami potentsial yaxshilanish**: Accuracy **74.86% → 85-89%+**

## 🔥 Muhim eslatma — TARTIB MUHIM!

Promptlarni **aynan shu tartibda** ishlating:

1. **Avval Prompt 1** — keyingi fazalar unga tayanadi
2. **Keyin Prompt 2** — Prompt 3 undan olingan feature engineering'ga tayanadi
3. **Oxirida Prompt 3** — ensemble va benchmarking

Agar tartibni o'zgartirsangiz, Claude Code mos kelmaslik xatolariga duchor bo'ladi.

## 📝 Har bir promptdan foydalanish usuli

### 1-qadam: Claude Code ni loyihangizda ishga tushiring

```bash
cd "AI Project"  # sizning loyihangiz papkasi
claude
```

### 2-qadam: Promptni to'liq nusxa ko'chiring

Masalan, `PROMPT_1_Critical_Fixes.md` ning TO'LIQ mazmunini Claude Code ga yuboring (birinchi qatordan oxirigacha).

### 3-qadam: Claude Code ishni bajarsin

U:
- Fayllarni o'qib chiqadi
- Spec bo'yicha o'zgartirishlarni kiritadi
- Testlarni bajaradi
- `PHASE1_CHANGES.md` (yoki 2, 3) hujjatini yaratadi
- Sizga hisobot beradi

### 4-qadam: Keyingi promptga o'tishdan oldin

Har bir promptdan keyin:

```bash
# O'zgarishlarni tekshirish
git diff  # agar git ishlatsangiz

# Kichik test qilish — model o'qitiladimi?
python main.py

# Agar hammasi yaxshi bo'lsa — commit qiling
git add . && git commit -m "Phase N complete"
```

Keyin faqat undan so'ng keyingi promptga o'ting.

## 🧪 Test qilish bo'yicha maslahat

Har bir promptdan keyin **to'liq dataset ustida** (45M qator) qayta o'qitish zarur emas — bu soatlab ketadi. Kichik subset (masalan 3-5 fayl, har birida 500K qator) ni ishlatib, o'zgarishlar ishlayotganini tezda tekshiring.

Faqat **oxirgi** bosqichdan keyin (Prompt 3 ham yakunlangach) **to'liq datasetda** yakuniy o'qitish qiling va dissertatsiya uchun natijalarni yozing.

## ⚠️ Agar nimadir noto'g'ri ketsa

Har bir prompt **mustaqil** — agar Prompt 2 da xatolik bo'lsa, Prompt 1 holatiga qaytish oson. Git ishlating:

```bash
# Har promptdan oldin
git checkout -b phase-1
# Ishlashdan keyin, agar yaxshi bo'lsa
git commit -am "Phase 1 done"
git checkout main && git merge phase-1

# Agar yomon bo'lsa
git checkout main  # o'zgarishlar rad etiladi
```

## 📊 Himoyaga tayyorgarlik uchun

Oxirgi promptdan keyin sizda shular bo'ladi:

1. ✅ To'liq optimallashtirilgan kod
2. ✅ 3 ta `PHASEn_CHANGES.md` hujjati (har birida nima qilinganini batafsil)
3. ✅ Benchmark report (per-class F1, latency, throughput)
4. ✅ Asl vs yangi natijalar qiyoslamasi

Bularning hammasi dissertatsiyangizga to'g'ridan-to'g'ri kiritilishi mumkin. Shuningdek, himoyada "qanday qilib accuracy ni oshirdingiz?" degan savolga aniq javob bera olasiz.

## 💡 Qo'shimcha maslahat

Har bir promptning oxirida Claude Code dan **summary report** so'raladi. Uni **saqlab qo'ying** — dissertatsiyaning "Tajriba natijalari" bobi uchun ideal material.

Agar Claude Code biror qadamda muammoga duch kelsa (masalan, `dotenv` o'rnatilmagan), u sizdan yordam so'raydi. Oddiygina "davom et" deb aytib yoki paketni o'rnatib, davom ettirishingiz mumkin.

---

**Omad tilaymiz!** 🛡️ Bu optimallashlar sizning dissertatsiyangizni sezilarli mustahkamlaydi.
