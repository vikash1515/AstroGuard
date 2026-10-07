⭐ AstroGuard — Space Debris Collision Prediction & Visualization System
🚀 AI-powered orbital simulation, collision detection, and risk prediction

AstroGuard is a real-time space-object monitoring system that predicts satellite–debris collisions using orbital simulation, TLE parsing, and machine learning models.

Built for hackathons, research, and real-world aerospace applications.

🌌 Features
🛰 1. Real-Time Orbit Simulation

Propagates real TLE data using SGP4

Simulates debris clouds, satellites, swarms, and mega-constellations

3D interactive orbit visualization (Plotly)

💥 2. Collision Detection Engine

Detects close approaches using:

Euclidean separation

Relative velocity estimation

TCA (Time to Closest Approach)

DV (Delta-V for avoidance)

🤖 3. ML-Powered Collision Prediction

Trains 7 machine learning models:

Logistic Regression

SVM

KNN

Random Forest

Gradient Boosting

Neural Network (MLP)

XGBoost

Soft-Voting Ensemble

🎨 4. Beautiful Neon Visualizations

Orbit trails

Debris swarm

Constellation simulation

Collision explosion simulation

Model accuracy & F1 graphs

Confusion matrices

🌐 5. Streamlit Web App

Real-time interactive dashboard

Sidebar controls

Simulation playback

Model scoring

Downloadable HTML visualizations

Export ZIP with all outputs

          ┌────────────────────┐
          │   TLE Downloader   │
          └─────────┬──────────┘
                    │
                    ▼
         ┌───────────────────────┐
         │   Orbit Propagation   │ (SGP4)
         └──────────┬────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │   Collision Detection    │
        └──────────┬───────────────┘
                    │
                    ▼
       ┌─────────────────────────────┐
       │   Feature Engineering       │
       └──────────┬──────────────────┘
                    │
                    ▼
       ┌─────────────────────────────┐
       │  ML Training (7 Models)     │
       └──────────┬──────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Streamlit Dashboard   │
        └─────────────────────────┘
💡 Tech Stack
🧠 Machine Learning

Scikit-Learn

XGBoost

MLP Neural Network

🛰 Orbital Mechanics

SGP4

Celestrak TLE feeds

🎨 Visualizations

Plotly

3D orbit rendering

Heatmaps and bar charts

🌐 Frontend

Streamlit

Custom CSS

🚀 Getting Started
1️⃣ Clone the repo
git clone https://github.com/Athar-cell/Astroguard.git
cd Astroguard

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit app
streamlit run app/app.py

🧪 Running Simulation Notebook

Open:

notebooks/Astroguard.ipynb


Run all cells to:

Propagate TLEs

Generate debris fields

Detect close approaches

Train all ML models

Generate visualizations

Save PKL models

📸 Sample Visualizations

(You can replace these with your real images or gifs)

🛰 Orbit Simulation

💥 Collision Explosion

📊 Model Accuracy Plot

📦 Exporting Output

The Streamlit app allows you to export:

3D visualizations

Confusion matrices

Model performance plots

ML models

Scaler

ZIP bundle for hackathon submission

🧑‍💻 Author

Athar Sharma
B.Tech CSE | AI & Data Science | ML | SpaceTech
📫 atharsharma86@gmail.com

🔗 GitHub: https://github.com/Athar-cell
