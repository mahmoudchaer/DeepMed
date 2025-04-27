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
4. Configure Azure Key Vault (see Configuration section)
5. Initialize the database
```
python db/init_db.py
```
6. Initialize ChromaDB for vector storage (required for chatbot services)
```
python -c "import chromadb; chromadb.Client().reset()"
```
7. Start the application
```
python app_api.py
```

## 🔐 Configuration

The application uses Azure Key Vault for secure configuration management. No `.env` files are required. Instead, add the following secrets to your Azure Key Vault:

- `MYSQLUSER`: MySQL database username
- `MYSQLPASSWORD`: MySQL database password
- `MYSQLHOST`: MySQL server hostname
- `MYSQLPORT`: MySQL server port
- `MYSQLDB`: MySQL database name
- `SECRETKEY`: Flask application secret key
- `OPENAIAPIKEY`: OpenAI API key for LLM services
- `AZURESTORAGEACCOUNT`: Azure Storage account name
- `AZURESTORAGEKEY`: Azure Storage access key
- `AZURECONTAINER`: Azure Storage container name
- `SYSTEM_PROMPT_VALUE`: System prompt for the chatbot
- `VAULT_URL`: URL of your Azure Key Vault (already set in keyvault.py)

The application accesses these secrets directly from Azure Key Vault using the `keyvault.py` module, which handles authentication and secret retrieval.

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

DeepMed integrates the following microservices:

### Core Analysis Services
1. **Data Cleaner** (port 5001): Cleans and preprocesses tabular medical data
2. **Feature Selector** (port 5002): Identifies important features in medical datasets
3. **Anomaly Detector** (port 5003): Identifies anomalies in medical data
4. **Model Trainer** (port 5004): Generic model training service
5. **Medical Assistant** (port 5005): AI-powered medical information service

### Model Services
6. **Logistic Regression** (port 5010): ML model for classification tasks
7. **Decision Tree** (port 5011): Decision tree model implementation
8. **Random Forest** (port 5012): Random forest model implementation
9. **SVM** (port 5013): Support Vector Machine model implementation
10. **KNN** (port 5014): K-Nearest Neighbors model implementation 
11. **Naive Bayes** (port 5015): Naive Bayes model implementation
12. **Model Coordinator** (port 5020): Coordinates and manages all ML models

### Image Processing Services
13. **Model Training Service** (port 5021): Handles image classification model training
14. **Augmentation Service** (port 5023): Image data augmentation service
15. **Pipeline Service** (port 5025): Integrates image processing workflows

### Anomaly Detection Services
16. **Anomaly Detection Service** (port 5029): Base anomaly detection engine
17. **Anomaly Detector EEP** (port 5030): Enhanced anomaly detection platform

### Prediction Services
18. **Predictor Service** (port 5100): Image prediction service
19. **Tabular Predictor Service** (port 5101): Tabular data prediction service

### Chatbot Services
20. **Embedding Service** (port 5201): Text embedding and vectorization
21. **Vector Search Service** (port 5202): Vector similarity search for RAG
22. **LLM Generator Service** (port 5203): Language model generation service
23. **Chatbot Gateway** (port 5204): Frontend interface for the chatbot

### Monitoring Services
24. **Monitoring Service** (port 5432): System monitoring and alerting
25. **Prometheus** (port 9090): Time-series data collection
26. **Grafana** (port 3000): Data visualization and dashboards

## 📝 API Documentation

API documentation is available at the `/documentation` endpoint when the application is running.