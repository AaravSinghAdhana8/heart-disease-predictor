# GitHub pe Upload + Free Live Hosting

Do cheezein alag hain:
- **GitHub** = aapka code online rehta hai (portfolio/resume ke liye).
- **Streamlit Community Cloud** = app ka **live link** banta hai jo koi bhi browser me khol sake. Free hai aur seedha GitHub se chalta hai.

---

## Part A: GitHub pe code daalo

### Pehle: account aur repository
1. https://github.com pe account banao (professional username rakho, jaise `aarav-singh-adhana`).
2. Upar right **+ → New repository**.
3. Name: `heart-disease-predictor`
4. Description: `Interactive ML app for early heart disease risk prediction with explainable results`
5. **Public** select karo.
6. README, .gitignore, license **add mat karo** (project me pehle se hain).
7. **Create repository**.

### Tareeka 1: GitHub Desktop (sabse aasaan, bina commands)
1. https://desktop.github.com se install karo, GitHub account se login karo.
2. **File → Add local repository** → project folder chuno.
3. "This directory does not appear to be a Git repository" aaye to **create a repository** pe click karo → **Create repository**.
4. Left side saari files dikhengi. Neeche Summary me likho: `Initial commit` → **Commit to main**.
5. Upar **Publish repository** → "Keep this code private" ka tick **hatao** → **Publish**.

### Tareeka 2: Git commands (interview ke liye seekhna achha hai)
Git install karo: https://git-scm.com/downloads. Phir project folder me terminal kholo:

```bash
git init
git add .
git commit -m "Heart disease predictor: fixed label bug, new pipeline, Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/heart-disease-predictor.git
git push -u origin main
```
Pehli baar push karne pe browser me GitHub login maangega, login kar do.

Pehli baar commit karte waqt "Please tell me who you are" aaye to:
```bash
git config --global user.name "Aarav Singh Adhana"
git config --global user.email "your-email@example.com"
```

### Check karo
Repository page refresh karo. README screenshots ke saath dikhna chahiye.
**Actions** tab me green tick aayega. Matlab GitHub ne khud tests chala ke pass kar diye (`.github/workflows/tests.yml` ki wajah se).

> `venv/` folder upload nahi hona chahiye. `.gitignore` isko rokta hai. Agar galti se chala gaya to repo delete karke dobara karo.

### Aage kabhi code badla to
```bash
git add .
git commit -m "kya badla, short me"
git push
```
(GitHub Desktop me: Commit → Push origin.)

---

## Part B: Live app (Streamlit Community Cloud, free)

1. https://share.streamlit.io kholo → **Continue with GitHub** se login.
2. **Create app → Deploy a public app from GitHub**.
3. Bharo:
   - Repository: `YOUR-USERNAME/heart-disease-predictor`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: jaise `heart-risk-aarav` (ye aapka link banega)
4. **Advanced settings → Python version: 3.11 ya 3.12**.
5. **Deploy**. 2-4 minute me link milega, jaise `https://heart-risk-aarav.streamlit.app`.

Pehli baar app khulne me thoda time lag sakta hai kyunki server pe model khud train hota hai (ye feature isi liye banaya hai).

**Note:** kuch din koi na khole to app "sleep" ho jata hai. Link kholne pe "Wake up" button dabao, 30 second me chalu. Interview se pehle ek baar khud khol lena.

### Live link README me daalo
`README.md` me title ke neeche ye line add karo aur push karo:
```markdown
**🔗 Live demo:** https://heart-risk-aarav.streamlit.app
```

---

## Part C: Profile ko polish karo

- Repo page pe **About** (right side ⚙️) → Website me live link daalo. Topics add karo:
  `machine-learning`, `healthcare`, `streamlit`, `scikit-learn`, `python`, `data-science`
- Apne GitHub profile pe repo ko **Pin** karo.

### Resume ke liye line
> **Early Heart Disease Prediction** — Python, scikit-learn, Streamlit
> - Debugged an existing model that flagged every patient as at risk; with AI-assisted analysis, traced it to an inverted target label in a public dataset (verified against the original UCI source), fixed the data and added regression tests.
> - Ran a decision-threshold experiment (0.5 vs 0.35) to study the recall vs false-alarm trade-off for medical screening.
> - Built a leak-free ML pipeline and selected a Random Forest via repeated cross-validation (ROC-AUC 0.89 on held-out test set).
> - Deployed an interactive app with per-patient explanations and batch CSV scoring. [Live demo] [GitHub]

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Streamlit Cloud: "ModuleNotFoundError" | Check karo `requirements.txt` repo me upload hua hai |
| Streamlit Cloud: app crash | App page pe **Manage app** (neeche right) → logs padho |
| `git push` rejected | Aapne GitHub pe README/license bana diya tha. `git pull origin main --allow-unrelated-histories` phir `git push` |
| Screenshots README me nahi dikh rahe | `docs/images/` folder upload hua hai ya nahi check karo |
