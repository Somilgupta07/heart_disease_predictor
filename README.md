💓 Heart Disease Diagnostic System (AI-Driven)
An end-to-end Machine Learning and Deep Learning project designed to predict cardiovascular risks. This system utilizes a Dual-Core Inference Engine, allowing users to switch between high-performance Gradient Boosting and Neural Network backends.

**🚀 [Live Demo on Streamlit Cloud]([https://your-app-link.streamlit.app/](https://heartdiseasepredictor-jda2oaldj3jjgt89xdtzd5.streamlit.app/))**

🧠 Technical Innovations
Multi-Backend Intelligence: Integrated XGBoost (for tabular precision) and PyTorch (for non-linear deep learning patterns).

Production-Grade Preprocessing: Implemented a persistent StandardScaler pipeline to ensure mathematical feature parity between training and real-time inference.

Recall-Optimized Modeling: Specifically tuned to minimize False Negatives, ensuring higher safety in medical diagnostic scenarios.

🛠️ Tech Stack
AI/ML: PyTorch, XGBoost, Scikit-Learn, Pandas, NumPy

Visualization: Seaborn, Matplotlib (Correlation Heatmaps, ROC Curves)

Deployment: Streamlit, Git

📂 Project Architecture
app.py: Streamlit frontend and dynamic model loading logic.

model_Training.py: Advanced pipeline for model scaling and training.

models/: Serialized weights (.pth, .json, .pkl) and the critical scaler.pkl.

📈 Performance Comparison
The project compares three generations of AI:

Classic: Logistic Regression & KNN (Baseline)

Industrial: XGBoost (Optimized Gradient Boosting)

Advanced: PyTorch (3-Layer Sequential Neural Network with Dropout)

🚀 Getting Started
Install dependencies:

Bash
pip install -r requirements.txt
Train the engine:

Bash
python model_Training.py
Launch the Diagnostic UI:

Bash
streamlit run app.py
