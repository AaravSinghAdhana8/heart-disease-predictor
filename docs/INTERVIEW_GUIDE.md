# Interview Guide — Early Heart Disease Prediction

Ye guide isliye hai taaki aap project ka har hissa apne shabdon me samjha sako.
Ise ek baar padho, phir app khol ke har cheez khud click karke dekho. Interview me
ratta nahi, samajh dikhni chahiye.

---

## 1. 60-second pitch (isko practice karo)

> "Maine ek heart disease risk prediction app pe kaam kiya hai jo 13 clinical
> values, jaise age, cholesterol, max heart rate aur ECG results, lekar batata
> hai ki patient ko heart disease hone ki probability kitni hai, aur **kyun**.
>
> Project mujhe ek pehle se bane version ke roop me mila tha jo har patient ko
> 'at risk' bata raha tha. Maine AI tool (Claude) ki madad se ise debug kiya.
> Pata chala ki Kaggle wale dataset me target label ulta hai: 1 ka matlab
> healthy hai, disease nahi. Original UCI dataset se saare 303 rows match karke
> ye confirm hua.
>
> Uske baad pipeline dobara banayi gayi: missing values handle ki, categorical
> features one-hot encode kiye, teen models cross-validation se compare kiye,
> aur Random Forest choose hua jo test set pe 0.89 ROC-AUC deta hai. Maine khud
> threshold ke saath experiment kiya aur recall vs false alarms ka trade-off
> dekha. App Streamlit me hai aur GitHub pe hosted hai."

## 2. Bug ki kahani (sabse strong point — STAR format)

**Situation:** Project already bana hua tha, par koi bhi data daalo, "at risk" aata tha.

**Task:** Pata lagana ki model galat kyun hai, aur use reliable banana.

**Action (step by step)** — ye kaam AI tool (Claude) ki madad se hua; aapko har step ka *kyun* samajhna hai:
1. **Reproduce kiya.** Ek healthy 30 saal ka patient (max heart rate 190, koi
   ECG problem nahi) daala → model bola **75.8% risk**. Ek bimaar 70 saal ka
   patient (3 blocked vessels, exercise pe chest pain) → **9% risk**. Matlab
   model ulta chal raha tha.
2. **Data ko sawaal kiya (EDA / sanity check).** Groupby karke dekha:
   - Exercise pe chest pain wale logon me target=1 sirf **23%**, bina pain wale me **70%**.
   - Jitne zyada blocked vessels, utna **kam** target=1.
   - Medical sense ke hisaab se ye ulta hai. Isse shak hua ki `target=1` asal me healthy hai.
3. **Proof liya.** Original UCI Cleveland dataset (jisme label "Yes/No" likha
   hai) se saare 303 rows match kiye. **Har ek row ka label ulta tha.**
4. **Fix kiya:** `has_disease = 1 - target`.
5. **Aur hidden problems bhi pakdi:** `ca = 4` aur `thal = 0` asal me missing
   values hain (original me "?" tha), unhe NaN banaya. Ek duplicate row hatai.
6. **Dobara na ho, iska intezaam:** `train.py` me sanity check hai, aur
   `tests/` me tests hain. Agar model healthy patient ko high risk bole to model
   save hi nahi hota.

**Result:** Healthy patient ab ~11% risk, bimaar ~93-96%. Test accuracy 80%, ROC-AUC 0.89.

**Lesson (ye zaroor bolna):** "Accuracy achhi ho tab bhi model galat ho sakta
hai. Purana model bhi accuracy ke hisaab se achha dikhta tha, kyunki woh ulte
label ko consistently seekh raha tha. Isliye main hamesha predictions ka domain
sense se sanity check karta hoon."

> Ye point samajhna zaroori hai: label ulta hone par bhi accuracy 80%+ aati hai,
> kyunki model ulte pattern ko perfectly seekh leta hai. Numbers se bug nahi
> dikhta, sirf common sense check se dikhta hai.

---

## 3. Pipeline — har step kya aur kyun

Code: `src/data.py`, `src/model.py`, `train.py`

### Features ke 3 type
| Type | Columns | Kya kiya | Kyun |
|---|---|---|---|
| Numeric | age, trestbps, chol, thalach, oldpeak, ca | Median se missing fill + StandardScaler | Scaling se Logistic Regression ko fair comparison milta hai (sab ek scale pe) |
| Binary | sex, fbs, exang | Waise hi (0/1) | Already number hain |
| Categorical | cp, restecg, slope, thal | Most-frequent se fill + One-Hot Encoding | Ye codes hain, quantity nahi |

**One-hot kyun?** `cp` me 0,1,2,3 codes hain. Agar number maan lo to model
sochega "3 is bigger than 1", jo bekaar hai, kyunki ye bas types hain. One-hot
har type ka alag 0/1 column bana deta hai.

**Median kyun, mean kyun nahi?** Median outliers se kam affect hota hai
(cholesterol me 564 jaisi values hain).

### `Pipeline` kyun use kiya?
Preprocessing + model ek hi object me. Fayde:
- Training aur prediction me **same** transformation guaranteed.
- Cross-validation me **data leakage** nahi hota: scaler har fold me sirf
  training part pe fit hota hai.
- Ek hi file save/load karni padti hai.

### Train/test split (80/20, stratified)
- 20% data ko pehle hi alag rakh diya; model selection me use nahi hua. Ye final "exam" hai.
- `stratify=y` → dono parts me disease ka ratio same (~46%).

### Cross-validation (5-fold × 5 repeats)
Training data ko 5 hisson me baanto, 4 pe train, 1 pe test, 5 baar ghumao. Poora
process 5 baar alag shuffle se repeat kiya. Dataset chhota hai (302 rows), to ek
split pe result luck pe depend karta hai; CV se stable estimate milta hai.

### Model comparison
| Model | CV ROC-AUC |
|---|---|
| Logistic Regression | 0.909 ± 0.043 |
| Random Forest | 0.910 ± 0.042 |
| Gradient Boosting | 0.889 ± 0.045 |

Random Forest select hua (highest mean ROC-AUC). **Honest baat:** RF aur LR ka
fark bahut chhota hai, ± std ke andar. Agar interviewer puche to bolo: "Dono
practically barabar hain. Agar explainability top priority hoti to main
Logistic Regression le leta."

### Final model ko poore data pe retrain kyun?
Test score nikalne ke baad, deploy wala model saare 302 rows pe train kiya.
Chhote dataset me har row kaam ki hai. Reported test score thoda conservative estimate hai.

---

## 4. Metrics — simple bhasha me

Confusion matrix (test set, 61 patients):

|  | Predicted healthy | Predicted disease |
|---|---|---|
| **Actually healthy** | 29 (TN) | 4 (FP) |
| **Actually disease** | 8 (FN) | 20 (TP) |

- **Accuracy 0.80** — kitne total sahi: (29+20)/61.
- **Precision 0.83** — jinhe "disease" bola, unme se kitne sach me bimaar: 20/24.
- **Recall 0.71** — jo sach me bimaar the, unme se kitne pakde: 20/28.
- **F1 0.77** — precision aur recall ka balance.
- **ROC-AUC 0.89** — random bimaar aur random healthy patient liya, to 89% chance
  hai model bimaar ko zyada risk dega. Threshold pe depend nahi karta.

**Medical project me sabse important: Recall.** False negative (bimaar ko healthy
bol dena) sabse khatarnaak hai. Hamara recall 0.71 hai, matlab 8 bimaar miss hue.
**Improvement idea:** threshold 0.5 se kam karke (jaise 0.35) recall badha sakte
hain, precision thoda girega. Ye trade-off doctor/business decide karega.

---

## 5. Random Forest kaise kaam karta hai (2 line me)
Bahut saare decision trees (yahan 300) banata hai, har tree ko data ka alag
random sample aur features ka random subset milta hai. Final answer = sab trees
ka average vote. Ek tree overfit karta hai, bahut saare milke stable ho jaate hain.

Hyperparameters jo maine set kiye: `max_depth=5`, `min_samples_leaf=5` → trees
ko chhota rakha kyunki data sirf 302 rows ka hai; bade trees data ratt lete (overfitting).

---

## 6. Explanation feature ("What drives this prediction")
Har feature ke liye: patient ki value ko ek **typical healthy patient ki value**
se replace karo (healthy logon ka median/mode) aur dekho risk kitna badla.
- Laal bar = ye value risk badha rahi hai
- Hara bar = ye value risk ghata rahi hai

Isse doctor/user ko "kyun" samajh aata hai, sirf number nahi. Ye ek simple
"what-if / counterfactual" explanation hai. Advanced version SHAP hota hai (future improvement).

**Global importance** (Model performance tab): permutation importance, yaani ek
feature ki values shuffle karo aur dekho ROC-AUC kitna girta hai. Jitna zyada
gire, feature utna important.

---

## 7. Interesting medical insight (poochha ja sakta hai)
"Asymptomatic" chest pain (koi typical pain nahi) wale group me **sabse zyada**
disease hai. Sunne me ulta lagta hai, par medically "silent ischemia" common hai.
Isiliye app me ye note bhi diya hai.

---

## 8. Common interview questions

**Q: Dataset kitna bada hai? Kya ye kaafi hai?**
302 rows (1 duplicate hataya). Chhota hai, isliye CV use kiya aur model simple
rakha. Real deployment ke liye zyada aur diverse (jaise Indian patients ka) data chahiye.

**Q: Class imbalance tha?**
Nahi, ~46% disease, ~54% healthy. Balanced hai, to accuracy meaningful hai. Phir
bhi stratified split use kiya.

**Q: Purana model 83.6% accuracy de raha tha, naya 80%. To naya bura hai?**
Nahi. Purana model "healthy" ko "disease" bol raha tha aur ulte label ke against
check ho raha tha, to uski 83.6% ka matlab tha "83.6% baar ulta answer consistently".
Real duniya me woh almost har baar galat hota. Naye aur purane numbers ka fark
(80 vs 84) alag test split aur chhote test set (61 patients) ka noise hai; CV
accuracy dono ki ~84-85% hai. Asli fark direction ka hai, number ka nahi.

**Q: Agar kisi aur computer pe library ka version alag ho to?**
Saved model ke saath scikit-learn ka version bhi save hota hai. Load karte waqt
version match nahi hua ya file kharab hai, to app khud model dobara train kar
leta hai (`load_bundle()` in `src/model.py`). Isliye project kisi bhi machine
aur Streamlit Cloud pe bina manual step ke chalta hai.

**Q: GitHub Actions kya kar raha hai?**
Har push pe GitHub ek fresh machine pe libraries install karta hai, model train
karta hai aur tests chalata hai. Koi change model ko tod de to turant red cross
dikh jaata hai. Ise CI (Continuous Integration) kehte hain.

**Q: Kya ye poora code tumne khud likha?**
"Nahi, poora nahi. Project ek base version se shuru hua tha, aur debugging aur
naya code AI tool (Claude) ki madad se likha gaya. Mera role tha problem
pehchanna, har step samajhna, results verify karna, threshold ka experiment
karna, aur project ko GitHub pe upload aur deploy karna. Main har file ka kaam
aur har decision ka reason explain kar sakta hoon."

**Q: Tumne khud kya experiment kiya?**
"Maine decision threshold 0.5 se 0.35 karke dekha. Recall 0.71 se 0.79 ho
gaya aur missed patients 8 se 6, lekin false alarms 4 se 10 ho gaye aur
accuracy 0.80 se 0.74 gir gayi. ROC-AUC same raha (0.89) kyunki model wahi
tha, sirf cut-off line khiski thi. Demo ke liye maine 0.5 rakha; real
screening tool me main kam threshold choose karta taaki bimaar kam miss hon."

**Q: Overfitting kaise check kiya?**
CV score (0.91) aur untouched test score (0.89) kareeb hain. Bahut bada gap hota to overfitting hota.

**Q: Purane code me kya galat tha, label ke alawa?**
- Categorical codes (cp, thal) ko numbers ki tarah treat kiya tha.
- Missing values (ca=4, thal=0) ko real values maana.
- Koi sanity check/test nahi tha.
- Default values me `thal=3` (reversible defect, ek disease sign) set tha.

**Q: Isko production me kaise le jaoge?**
Model file + FastAPI endpoint, Docker container, cloud pe deploy. Input validation,
logging, aur time ke saath naye data pe monitoring (data drift). Streamlit Cloud pe
bhi free deploy ho sakta hai.

**Q: Limitations?**
- Chhota, purana (1988) dataset, ek hi hospital (Cleveland) ka.
- Kuch features (ca, thal) ke liye mehenge tests chahiye, early screening me hamesha available nahi.
- Ye diagnosis nahi, sirf risk estimate hai.

**Q: Aage kya improve karoge?**
- Threshold tuning for higher recall
- SHAP explanations
- Bade dataset pe train (combined UCI datasets)
- Probability calibration check
- API + deployment

---

## 9. Demo kaise dikhana hai (interview me screen share)
1. `streamlit run app.py` (ya live Streamlit Cloud link)
2. **"Fill healthy example"** dabao → ~11% Low risk. Bolo: "Purana version isko bhi at risk bolta tha."
3. **"Fill high-risk example"** → ~96%. Explanation chart dikhao: chest pain type, ST depression, thal top pe.
4. Ek value live badlo (jaise blocked vessels 3 → 0) aur risk girte dikhao. Ye interactive feel deta hai.
5. **CSV tab** → `sample_data/patients.csv` upload karo, results download karo.
6. **Model performance** tab → confusion matrix aur ROC curve samjhao.
7. **Data & fixes** tab → bug ki kahani.

---

## 10. Imandari ka note
Interview me ye mat bolna ki poora code aapne zero se khud likha. Sach bolo:
"Project ek base version se shuru hua, debugging aur naya code AI ki madad se
hua, aur maine har step samjha, verify kiya, threshold experiment kiya aur
deploy kiya."
Interviewer follow-up questions poochega; is guide ke har section ko itna
samjho ki bina dekhe explain kar sako. Har file kholo, har line padho, aur jo
samajh na aaye woh mujhse pooch lo.
