# 🥗 CalFit  

CalFit is an **AI-driven food label analysis app** that helps users scan food labels, extract nutrition details, and assess risks (allergens, high sugar, salt, cholesterol, etc.).  
It also lets users build profiles, track meals, and get personalized health insights.  

---

## ✨ Features  

- 📷 **Food Label Scanning** – Upload a food label image, OCR extracts ingredients and nutrition.  
- 🤖 **AI Nutrition Analysis** – Uses ML + rules to detect excessive nutrients and allergens.  
- 🧾 **Risk Assessment** – Flags sugar, salt, cholesterol, oils, and allergens based on thresholds.  
- 👤 **User Profiles** – Store personal info (age, weight, medical conditions, allergies).  
- 📊 **Health Reports** – Weekly summaries, dietary overview, personalized recommendations.  
- 🥦 **Food Tracker** – Log meals manually or via scanning to track protein, carbs, fats, calories.  
- 🐳 **Dockerized** – Easy deployment with container support.  

---

## 🛠️ Tech Stack  

- **Frontend:** HTML, CSS, JS (React planned)  
- **Backend:** Flask (Python)  
- **Database:** SQLite & MongoDB (profiles, scans, thresholds)  
- **OCR:** Tesseract, OpenCV  
- **AI Models:** Random Forest, BERT/RoBERTa (NLP for label analysis)  
- **Security:** JWT Auth, CSRF protection, password hashing  
- **Deployment:** Docker, AWS/GCP (future)  

---

## ⚙️ Installation  

### 1️⃣ Clone the repo  
```bash
git clone https://github.com/BhosleAnish/CalFit.git
cd CalFit
