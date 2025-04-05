#!/bin/bash

# Pull latest changes from GitHub
echo "Pulling latest changes from GitHub..."
cd /var/www/pochitlon
git pull origin main

# Deploy frontend
echo "Deploying frontend..."
cd /var/www/pochitlon
npm install
npm run build
pm2 delete pochitlon-frontend 2>/dev/null || true
pm2 serve dist 5173 --name "pochitlon-frontend"

# Deploy backend
echo "Deploying backend..."
cd /var/www/pochitlon/Backend
source ../venv/bin/activate

# Ensure environment variables are set
if [ ! -f .env ]; then
    echo "Error: .env file not found in Backend directory"
    exit 1
fi

# Install any new Python dependencies
pip install -r requirements.txt 2>/dev/null || pip install flask flask-cors python-dotenv langchain-core langchain-openai langchain-community openai langsmith google-generativeai faiss-cpu numpy typing-extensions

# Start the backend with PM2
pm2 delete pochitlon-backend 2>/dev/null || true
pm2 start app.py --name "pochitlon-backend" --interpreter python3 --env-from-file .env

# Save PM2 process list and configure PM2 to start on system boot
pm2 save
pm2 startup

# Check if services are running
echo "Checking service status..."
pm2 status

echo "Deployment completed! Application will now run persistently and start automatically on system reboot." 