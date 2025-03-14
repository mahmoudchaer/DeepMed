# DeepMedVer - Medical AI Platform

A comprehensive medical AI platform for training models on medical data, making predictions, and getting AI-assisted insights.

## Features

- User authentication system
- Data cleaning and preprocessing
- Feature selection
- Anomaly detection
- Model training (classification and regression)
- Model evaluation and selection
- Prediction on new data
- AI-assisted medical recommendations
- Interactive data visualization

## Requirements

- Python 3.8+
- PostgreSQL 12+
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepMedVer.git
cd DeepMedVer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Update the PostgreSQL connection details (user, password, host, port, database name)
   - Add your OpenAI API key if you want to use the AI assistant feature

5. Set up the PostgreSQL database:
```bash
python setup_db.py
```

6. Run the application:
```bash
python app.py
```

7. Open your browser and navigate to http://localhost:5000

## Usage

1. Register a new account or log in to an existing account
2. Upload your dataset (CSV or Excel format)
3. Select the target variable and training parameters
4. Train various models and select the best one
5. Make predictions on new data
6. Use the AI assistant for medical insights

## Authentication

The platform includes a user authentication system with the following features:
- User registration with email validation
- Secure password storage (hashed)
- Login and session management
- Access control for all application features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Structure

```
DeepMedVer/
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

## Acknowledgments

- OpenAI for providing the API for the medical assistant
- The scikit-learn team for machine learning tools
- The Flask team for the web framework