# DeepMed - Medical AI Platform

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
- Azure Key Vault integration for secure secret management

## Requirements

- Python 3.8+
- PostgreSQL 12+
- Required Python packages (see requirements.txt)
- Azure subscription for Key Vault

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepMedVer.git
cd DeepMed
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

4. Configure Azure Key Vault:
   - Create an Azure Key Vault in your Azure subscription
   - Add all required secrets to your Key Vault (see AZURE_KEYVAULT.md for details)
   - Update the VAULT_URL in keyvault.py with your Key Vault URL
   - Configure authentication (Managed Identity for Azure deployments, or environment variables locally)

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
DeepMed/
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
├── keyvault.py                # Azure Key Vault integration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Azure Key Vault Integration

This application uses Azure Key Vault for secure secrets management. All configuration previously stored in .env files is now managed through Azure Key Vault, providing:

- Centralized secrets management
- Enhanced security with built-in encryption
- Role-based access control
- Detailed audit logs
- Secret rotation capability

See AZURE_KEYVAULT.md for detailed information on setup and usage.

## Technologies Used

- **Flask**: Web framework
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive data visualization
- **OpenAI**: AI assistant capabilities
- **Azure Key Vault**: Secure secrets management
- **Azure Blob Storage**: File storage

## Acknowledgments

- OpenAI for providing the API for the medical assistant
- The scikit-learn team for machine learning tools
- The Flask team for the web framework
- Microsoft Azure for cloud services and security features