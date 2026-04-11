# AuraHealth: Disease Prediction & Healthcare Graph System

## 📝 Project Overview
AuraHealth is an intelligent healthcare platform designed to predict diseases based on patient symptoms. What makes this project unique is its integration of Machine Learning with Graph Database technology. It doesn't just predict a disease; it connects patients, symptoms, and diagnoses in a dynamic "Knowledge Graph" using TigerGraph.

The system allows healthcare providers to:
1. Input patient symptoms through a modern web interface.
2. Get instant disease predictions (like COVID-19, Flu, or Migraine) with a confidence score.
3. Visualize the entire patient database as a connected graph.
4. Scale predictions using an automated pipeline that fetches data directly from a cloud database.

---

## 🛠️ Technologies Used

### Backend & Machine Learning
- **Python**: The core programming language used for logic and data processing.
- **Flask**: A lightweight web framework used to build the API and serve the frontend.
- **XGBoost**: A high-performance gradient boosting algorithm used for the disease prediction model.
- **Pandas & NumPy**: Used for cleaning medical data and preparing it for the AI model.
- **Joblib**: Used to save and load the trained AI model for instant use.

### Database & Graph Integration
- **TigerGraph**: A powerful graph database used to store patient records and relationships.
- **pyTigerGraph**: The connection bridge between Python and the TigerGraph database.

### Frontend (User Interface)
- **HTML5 & CSS3**: Used to create a clean, modern "Glassmorphism" styled interface.
- **JavaScript (ES6+)**: Handles the asynchronous communication between the UI and the backend.
- **Vis.js**: A specialized library used to render the interactive graph on the dashboard.

---

## 🚀 How to Start the Project

### 1. Setup the Environment
First, ensure you have Python installed. It is recommended to use a virtual environment to keep dependencies organized.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install all required libraries
pip install -r requirements.txt
```

### 2. Configure Database (Optional)
The project is built to work even without a live database connection (using its built-in fallback mode). However, to use the full TigerGraph features, you should set these environment variables:
- `TG_HOST`: Your TigerGraph instance URL.
- `TG_GRAPH_NAME`: The name of your healthcare graph.
- `TG_USERNAME` & `TG_PASSWORD`: Your database credentials.

### 3. Run the Application
Start the Flask server by running:
```bash
python app.py
```

Once the server starts, open your web browser and go to:
**http://127.0.0.1:8000**

---

## 📂 Key File Explanations
- **app.py**: The brain of the project. It handles all web requests and coordination.
- **model.py**: The training script. Use this if you want to retrain the AI on new data.
- **tigergraph_ml_integration.py**: Manages the complicated logic of talking to the graph database.
- **index.html & dashboard.html**: The visual pages you see in your browser.
- **dataset.csv**: The medical data used to teach the AI what symptoms belong to which disease.

---

## 🛡️ Special Features
- **Fallback Mode**: If the TigerGraph database is down or disconnected, AuraHealth automatically switches to a local cache or dummy data generator so the app never crashes.
- **Confidence Scores**: Every prediction comes with a percentage (e.g., 98% confidence), helping users understand how certain the AI is about a diagnosis.
- **Graph Visualization**: Instead of boring tables, you can see patients and their predictions as nodes connected in a visual graph.
