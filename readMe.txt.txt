how to run me:
firstly) train the models
python model_training.py

secondly) run the streamlit
streamlit run app.py

what we are doing basically in this model:
firstly)
 A. Train the ML Models (model_training.py)
Load the dataset (heart.csv)

Split it into training & testing sets

Train 3 ML models

Save them using joblib.dump(...) to .pkl files

So that we don’t have to retrain the model every time the app runs

Secondly)
B. Create the Streamlit Web App (app.py)
📥 Input Collection
Users input their health data using sliders/dropdowns in the sidebar (e.g., age, sex, chest pain type, etc.)

📊 Data Visualization
Shows a preview of the dataset

Displays a bar chart of how many patients have heart disease vs. not

Shows a scatter plot of age vs. max heart rate, colored by target (disease or not)

🧠 Model Selection
User chooses a model (logistic, random forest, or KNN) from a dropdown

📈 Prediction
The app takes the user’s input and passes it into the selected model

The model outputs:

1 → person has heart disease

0 → person does not

The result is displayed using st.success() or st.error() with emojis and colors