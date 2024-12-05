# Use Python 3.9 as base image for better compatibility
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Copy necessary files
COPY streamlit_link.txt .
COPY LogisticRegression_pipeline.pkl .
COPY Loan_Approval_App.py .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a startup script with correct Python file name
RUN echo '#!/bin/bash\necho "You can now view your Streamlit app in your browser at: http://localhost:8501"\nstreamlit run Loan_Approval_App.py --server.address 0.0.0.0 --server.port 8501 --browser.gatherUsageStats false' > /app/start.sh
RUN chmod +x /app/start.sh

# Expose port for Streamlit
EXPOSE 8501

# Use the startup script
CMD ["/app/start.sh"]