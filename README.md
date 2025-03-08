# DeepMed - Medical AI Assistant

DeepMed is a powerful medical AI assistant that helps healthcare professionals analyze medical data using advanced machine learning techniques. This Flask application provides a user-friendly interface for data cleaning, feature selection, model training, and prediction.

## Features

- **Data Cleaning**: Automatically clean and preprocess medical datasets
- **Feature Selection**: Identify the most important features for prediction
- **Anomaly Detection**: Detect and handle outliers in medical data
- **Model Training**: Train multiple machine learning models simultaneously
- **Model Selection**: Compare and select the best performing model
- **Prediction**: Make accurate predictions using trained models
- **AI Assistant**: Get recommendations and insights from an AI medical expert

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepMedV1.git
cd DeepMedV1
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Mac/Linux:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up the .env file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
```

## Running the Application

1. Start the Flask server:
```bash
python src/app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
DeepMedV1/
├── src/
│   ├── app.py                 # Flask application entry point
│   ├── models/                # Core model components
│   │   ├── data_cleaner.py    # Data cleaning functionality
│   │   ├── feature_selector.py # Feature selection
│   │   ├── anomaly_detector.py # Anomaly detection
│   │   ├── model_trainer.py   # Model training functionality
│   │   └── medical_assistant.py # AI assistant integration
│   ├── static/                # Static files (CSS, JS, images)
│   └── templates/             # HTML templates
├── uploads/                   # Folder for uploaded files
├── .env                       # Environment variables
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage Flow

1. **Upload Data**: Start by uploading your medical dataset (CSV or Excel)
2. **Configure Training**: Select the target variable and adjust training parameters
3. **Review Features**: Examine selected features and their importance
4. **Select Model**: Choose the best model based on performance metrics
5. **Make Predictions**: Upload new data for predictions using your selected model
6. **Chat with AI**: Ask the AI assistant for insights and recommendations

## Technologies Used

- **Flask**: Web framework
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive data visualization
- **OpenAI**: AI assistant capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the API for the medical assistant
- The scikit-learn team for machine learning tools
- The Flask team for the web framework