
# 🌱 Smart Agriculture Platform

This is a full-stack **monorepo** combining:

* 🌾 **Django backend** (REST APIs + ML models)
* 🌱 **React frontend** (farmer dashboard & chatbot)

It integrates APIs like Groq, OpenWeather, CEDA, Meersens, and Data.gov to provide predictions, recommendations, and real-time data for farmers.

---

## 🚀 Features

* 📊 Crop recommendation & yield prediction
* 🧪 Fertilizer recommendation
* 🌍 Weather, AQI, and soil data via external APIs
* 🤖 AI-powered chatbot (Groq API)
* 💹 Crop price tracker (Govt of India data)

---

## 📂 Project Structure

```
project-root/
│── frontend/         # React frontend
│── backend/          # Django backend
│   └── requirements.txt
│── ml_model/         # ML models
│── package.json      # Manages dev server for frontend + backend
│── .env.example      # Example environment variables
│── README.md         # Project setup guide
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd project-root
```

### 2. Install dependencies

* **Node dependencies** (for monorepo & frontend):

```bash
npm install
```

* **Python dependencies** (for backend):

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 3. Environment variables

Create a `.env` in the project root using `.env.example` as a template:

```bash
cp .env.example .env
```

Fill in your API keys (each teammate must use their own):

```env
OPEN_WEATHER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
CEDA_API_KEY=your_key_here
MEERSENS_API_KEY=your_key_here
API_KEY_DATA_GOV=your_key_here
```

⚠️ `.env` is already in `.gitignore` → **never commit your keys**.

### 4. Run the project

From the project root:

```bash
npm run dev
```

This will:

* Start **React frontend** (`npm run dev --prefix frontend`)
* Start **Django backend** (`python backend/manage.py runserver`)

---

## 🧪 Testing APIs

Example request to **crop recommendation**:

```bash
curl -X POST http://127.0.0.1:8000/api/crop-recommendation/ \
-d "N=90&P=40&K=40&temperature=25&humidity=70&ph=6.5&rainfall=200"
```

Example request to **chatbot**:

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ \
-H "Content-Type: application/json" \
-d '{"query": "Which crops are best for rainy season in Punjab?"}'
```

---

## 🤝 Contributing

* Create a feature branch for changes.
* Run backend & frontend locally before pushing.
* Document API changes inside `docs/`.

---

## 📜 License

MIT License.

---

👉 Do you want me to also generate a `.env.example` file for you (with placeholders for all API keys) so your teammates don’t get lost?
