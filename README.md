# DeepMed: Medical AI Platform

DeepMed is a comprehensive AI platform designed for medical professionals and researchers to leverage machine learning for healthcare applications. The platform provides tools for data processing, model training, prediction, and AI-powered medical assistance.

## 🚀 Features

- **Tabular Data Analysis**: Process and analyze medical data tables with advanced cleaning and feature selection
- **Medical Image Processing**: Train and use deep learning models for medical image analysis
- **Anomaly Detection**: Identify abnormal patterns in medical data using autoencoder-based anomaly detection
- **Data Augmentation**: Enhance medical datasets with sophisticated augmentation techniques
- **AI-Powered Chatbot**: Get medical AI assistance with a context-aware chatbot
- **Pipeline Integration**: Streamlined end-to-end workflows for medical AI applications
- **User Management**: Secure authentication and project management
- **Azure Integration**: Secure secret management via Azure Key Vault

## 🔧 Technical Architecture

DeepMed is built on a microservices architecture with the following components:

- **Flask Web Application**: Main application interface
- **Machine Learning Services**: Specialized containerized services for different ML tasks
- **Database Layer**: User data, training runs, and model storage
- **Chatbot Services**: AI assistant with RAG (Retrieval Augmented Generation) capabilities
- **Storage Services**: Management of files and model artifacts

## 📋 Requirements

- Python 3.8+
- Docker and Docker Compose
- MySQL Database
- Azure Key Vault (for production environments)

## 🛠️ Installation

1. Clone the repository
2. Create a virtual environment
```
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```
3. Install requirements
```
pip install -r requirements.txt
```
4. Configure database settings (see Configuration section)
5. Initialize the database
```
python db/init_db.py
```
6. Start the application
```
python app_api.py
```

## 🔐 Configuration

The application uses Azure Key Vault for secure configuration management. Create a `.env` file in the root directory with the following variables:

```
MYSQLUSER=your_database_user
MYSQLPASSWORD=your_database_password
MYSQLHOST=your_database_host
MYSQLPORT=your_database_port
MYSQLDB=your_database_name
SECRETKEY=your_application_secret_key
```

For production environments, these secrets should be stored in Azure Key Vault.

## 🚢 Docker Deployment

DeepMed is designed to run as a set of microservices using Docker. To deploy with Docker Compose:

```
docker-compose up -d
```

## 📚 Documentation

Comprehensive documentation is available in the `Documentation` directory, including:
- DeepMed User Guide for Chatbot
- DeepMed Business Analysis Report

## 🧪 Quality Assurance

The `quality_assurance` directory contains testing frameworks and tools for ensuring the reliability of the platform.

## 👥 Contributing

Contributions to DeepMed are welcome! Please follow the standard fork and pull request workflow.

## 📄 License

DeepMed is licensed under [Your License]. See the LICENSE file for details.

## 🔄 Services

DeepMed integrates the following services:
- Data Cleaner (port 5001)
- Feature Selector (port 5002)
- Anomaly Detector (port 5003)
- Medical Assistant (port 5005)
- Model Coordinator (port 5020)
- Model Training Service (port 5021)
- Augmentation Service (port 5023)
- Pipeline Service (port 5025)
- Anomaly Detection Service (port 5030)
- Embedding Service (port 5201)
- Vector Search Service (port 5202)
- LLM Generator Service (port 5203)

## 📝 API Documentation

API documentation is available at the `/documentation` endpoint when the application is running. 