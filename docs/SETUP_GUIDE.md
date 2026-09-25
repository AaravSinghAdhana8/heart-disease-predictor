# Setup Guide — Project Kaise Chalayein (Step by Step)

Ye guide Windows ke liye hai (Mac/Linux ke commands bhi saath me diye hain).
Pehli baar setup me 10-15 minute lagenge, uske baad app 1 command se chalega.

---

## Step 0: Zaroori software (sirf ek baar)

### Python install karo
1. https://www.python.org/downloads/ se **Python 3.10, 3.11 ya 3.12** download karo.
2. Installer chalao. **Sabse pehli screen pe "Add python.exe to PATH" wala box tick karna mat bhoolna.**
3. Install ke baad check karo. Command Prompt kholo (Start → `cmd`) aur likho:
   ```
   python --version
   ```
   `Python 3.12.x` jaisa kuch dikhe to sahi hai.

### VS Code (optional, par recommended)
https://code.visualstudio.com/ se install karo. Isme code padhna aur terminal chalana aasaan hai.

---

## Tareeka 1: One-click (sabse aasaan)

1. Zip extract karo, jaise `D:\projects\heart-disease-predictor`.
2. Folder kholo aur **`run_app.bat` pe double-click** karo.
3. Pehli baar ye khud virtual environment banayega aur libraries install karega (3-5 minute).
4. Browser me app khul jayega: `http://localhost:8501`

Band karna ho to black window me `Ctrl + C` dabao ya window band kar do.

---

## Tareeka 2: Manual (interview me ye aana chahiye)

VS Code me **File → Open Folder** se project folder kholo, phir **Terminal → New Terminal**.

```bash
# 1. Virtual environment banao (sirf pehli baar)
python -m venv venv

# 2. Use activate karo (har baar naya terminal kholne pe)
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Mac/Linux

# 3. Libraries install karo (sirf pehli baar)
pip install -r requirements.txt

# 4. App chalao
streamlit run app.py
```

Activate hone ke baad terminal line ke shuru me `(venv)` dikhega.

**Virtual environment kyun?** Har project ki libraries alag rehti hain, ek project ka
version doosre ko nahi todta. Interview me ye poocha ja sakta hai.

### Baaki commands
```bash
python train.py              # model dobara train karo, saare scores print honge
python -m pytest -q          # tests chalao → "6 passed" aana chahiye
python predict_cli.py --age 60 --cp 0 --exang 1 --ca 2 --thal 3
python predict_cli.py --input-csv sample_data/patients.csv
```

---

## Agli baar app kaise chalayein

Bas `run_app.bat` pe double-click karo, ya terminal me:
```bash
venv\Scripts\activate
streamlit run app.py
```

---

## Common errors aur unka solution

| Error | Matlab | Solution |
|---|---|---|
| `'python' is not recognized` | Python PATH me nahi hai | Python reinstall karo aur **"Add to PATH"** tick karo. Ya `py` likh ke try karo: `py -m venv venv` |
| `running scripts is disabled on this system` (activate karte waqt) | PowerShell script block kar raha hai | Ek baar chalao: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` aur `Y` dabao. Ya terminal ko Command Prompt pe switch kar do |
| `'streamlit' is not recognized` | venv activate nahi hai | Pehle `venv\Scripts\activate`, ya seedha `python -m streamlit run app.py` |
| `ModuleNotFoundError: No module named 'src'` | Galat folder se chala rahe ho | `cd` karke project folder ke andar jao (jahan `app.py` hai) |
| `ModuleNotFoundError: No module named 'sklearn'` (ya pandas/plotly) | Libraries install nahi hui | `pip install -r requirements.txt` (venv activate karke) |
| `Port 8501 is already in use` | App pehle se chal raha hai | Purani window band karo, ya `streamlit run app.py --server.port 8502` |
| "Saved model was built with a different scikit-learn version; retraining..." | Error nahi hai | App khud model dobara bana raha hai, ~15 second ruko |
| `pip install` bahut slow / fail | Internet ya pip purana | `python -m pip install --upgrade pip` phir dobara try karo |

Koi aur error aaye to poora error message copy karke Claude ya Google pe daalo. Aakhri 2-3 lines me asli wajah likhi hoti hai.

---

## App me kya-kya check karna hai (demo checklist)

- [ ] **Fill healthy example** → ~11%, green
- [ ] **Fill high-risk example** → ~96%, red
- [ ] Koi value badlo (jaise blocked vessels) → risk turant badle
- [ ] **CSV tab** → `sample_data/patients.csv` upload → 3 High, 3 Low
- [ ] **Model performance** tab → charts dikhe
- [ ] `python -m pytest -q` → 6 passed
