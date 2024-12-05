# Loan Approval Classification App 🏦

A machine learning application that predicts loan approval status using Logistic Regression. The application is containerized using Docker and provides an interactive web interface built with Streamlit.

## 🚀 Quick Start

You have two options to access the application:

### Option 1: Web Browser (No Installation Required)
Visit the live application at:
```
https://loanapprovalclassfication-by-sabryfarraj.streamlit.app/
```
This option requires no setup - just click and use!

### Option 2: Run using Docker

If you prefer to run the application locally:

```bash
# Pull the image from Docker Hub
docker pull sabryfarraj/loan-approval-classifier:latest

# Run the container
docker run -p 8501:8501 sabryfarraj/loan-approval-classifier:latest
```

After running these commands, open your browser and visit:
```
http://localhost:8501
```

## 🛠️ Project Structure
```
Loan_Approval_Classification/
├── app.py                            # Main Streamlit application
├── LogisticRegression_pipeline.pkl   # Trained model pipeline
├── requirements.txt                  # Python dependencies
```

## 📋 Prerequisites
For Docker option only:
- Docker installed on your machine ([Install Docker](https://docs.docker.com/get-docker/))

## 🔧 Local Development

If you want to build the Docker image locally:

```bash
# Clone the repository
git clone https://github.com/Sabryfarraj/Loan_Approval_Classfication.git

# Navigate to project directory
cd Loan_Approval_Classfication

# Build Docker image
docker build -t loan-approval-classifier .

# Run container
docker run -p 8501:8501 loan-approval-classifier
```

## 📊 Features
- Predicts loan approval probability
- Interactive web interface
- Real-time predictions
- Available as web application and containerized application
- Pre-trained machine learning model
- Handles various input features like:
  - Marital Status
  - Number of Dependents
  - Education
  - Property Area
  - Income details
  - Loan amount
  - Credit History

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Authors
- Sabry Farraj

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments
- Built with Streamlit and scikit-learn
- Uses imbalanced-learn for handling imbalanced datasets
- Hosted on Streamlit Cloud
