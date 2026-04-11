
# AuraHealth: Disease Prediction with TigerGraph Integration

This project is a web application that predicts diseases based on symptoms, leveraging a machine learning model integrated with a TigerGraph database. It's designed to showcase a full-stack ML application, from data fetching and prediction to visualization.

## Features

- **Symptom-based Disease Prediction**: Predicts diseases like Flu, Cold, and severity levels based on user-provided symptoms.
- **TigerGraph Integration**: Fetches patient data from a TigerGraph database, runs predictions, and writes the results back.
- **Interactive Dashboard**: Visualizes the prediction results as a graph, showing patients and their predicted health status.
- **Fallback Mechanism**: Ensures the application remains functional even if the TigerGraph database is unavailable by using a fallback dummy dataset.
- **RESTful API**: Provides endpoints to trigger predictions and fetch graph data.
- **Modular and Configurable**: The application is structured with clear separation of concerns and can be configured via environment variables.

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn (RandomForestClassifier), Pandas, NumPy, Joblib
- **Database**: TigerGraph (via pyTigerGraph)
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Local Flask server

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/krishsharma1159-1159/healthcare-project.git
   cd healthcare-project
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure TigerGraph**:
   - Set up a TigerGraph instance and create a graph with a `Patient` vertex.
   - The `Patient` vertex should have attributes for symptoms (e.g., `fever`, `cough`, `headache`, `fatigue`) and prediction results (`predicted_disease`, `prediction_score`).
   - Create a `getPatientFeatures` query in TigerGraph to fetch patient data.
   - Set the following environment variables with your TigerGraph credentials:
     ```bash
     export TG_HOST="your_tigergraph_host"
     export TG_GRAPH_NAME="your_graph_name"
     export TG_USERNAME="your_username"
     export TG_PASSWORD="your_password"
     ```

4. **Train the model (optional)**:
   - The repository includes a pre-trained model (`model.pkl`). To retrain the model, you can run:
     ```bash
     python model.py
     ```
   - This will train a new `RandomForestClassifier` on the `dataset.csv` and save it as `model.pkl`.

## Usage

1. **Run the Flask application**:
   ```bash
   python app.py
   ```
   The application will be available at `http://127.0.0.1:8000`.

2. **Access the application**:
   - Open your web browser and navigate to `http://127.0.0.1:8000` for the main prediction interface.
   - Navigate to `http://127.0.0.1:8000/dashboard` to see the graph visualization of the predictions.

## File Descriptions

- **`app.py`**: The main Flask application file. It handles routing, API endpoints, and serves the frontend.
- **`model.py`**: A script to train the machine learning model using `dataset.csv` and save it as `model.pkl`.
- **`predict.py`**: Contains the `predict_disease` function that loads the trained model and makes predictions.
- **`tigergraph_ml_integration.py`**: The core of the project. It handles the entire pipeline:
  - Connecting to TigerGraph.
  - Fetching data from TigerGraph (either from a vertex or a query).
  - Preprocessing the data.
  - Loading the ML model.
  - Running predictions.
  - Writing the prediction results back to TigerGraph.
  - Includes a fallback mechanism to use dummy data if TigerGraph is unavailable.
- **`requirements.txt`**: A list of all the Python packages required to run the project.
- **`dataset.csv`**: A sample dataset used to train the machine learning model.
- **`index.html`**: The main frontend page for the application.
- **`dashboard.html`**: The frontend page for the dashboard, which visualizes the graph data.
- **`styles.css`**: The CSS file for styling the frontend.
- **`script.js`**: The JavaScript file for the frontend, which handles API calls and graph visualization.

## API Endpoints

- **`GET /`**: Serves the main `index.html` page.
- **`GET /dashboard`**: Serves the `dashboard.html` page.
- **`POST /predict-from-tigergraph`**: Triggers the prediction pipeline. It fetches data from TigerGraph, runs predictions, and writes the results back.
- **`GET /graph-data`**: Returns the latest graph data for visualization on the dashboard.

## TigerGraph Integration

The `tigergraph_ml_integration.py` script is responsible for the seamless integration with TigerGraph. It uses the `pyTigerGraph` library to connect to the database. The `execute_prediction_pipeline` function orchestrates the entire process:

1. **Data Fetching**: It can fetch data in two modes:
   - **`vertex` mode**: Fetches all vertices of a specified type (e.g., `Patient`).
   - **`query` mode**: Executes a pre-defined installed query in TigerGraph (e.g., `getPatientFeatures`).
2. **Prediction**: The fetched data is preprocessed and fed into the trained machine learning model.
3. **Write-back**: The prediction results (disease label and score) are written back to the corresponding `Patient` vertices in TigerGraph.

## Fallback Mechanism

To ensure a smooth user experience, the application includes a robust fallback mechanism. If the TigerGraph database is not connected or fails for any reason, the application switches to a dummy data mode. It generates a small, random dataset and runs the prediction pipeline on it. This allows the user to see the application's functionality even without a live database connection.

## Future Enhancements

- **Real-time Predictions**: Implement WebSockets for real-time updates on the dashboard.
- **User Authentication**: Add user authentication to secure the application.
- **More Complex Models**: Experiment with more advanced machine learning models like neural networks.
- **Scalability**: Deploy the application on a cloud platform like AWS, Azure, or GCP for better scalability.
- **CI/CD Pipeline**: Set up a CI/CD pipeline for automated testing and deployment.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- The `pyTigerGraph` team for their excellent library.
- The Flask and Scikit-learn communities for their powerful and easy-to-use tools.
- This project was created with the assistance of GitHub Copilot.
