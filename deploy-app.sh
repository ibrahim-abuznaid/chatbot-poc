#!/bin/bash

# Pull latest changes from GitHub
echo "Pulling latest changes from GitHub..."
cd /var/www/hilton
git pull origin main  # or your default branch name

# Deploy frontend
echo "Deploying frontend..."
cd /var/www/hilton/frontend
npm install
npm run build
pm2 delete hilton-frontend 2>/dev/null || true  # Delete if exists
pm2 serve dist 5173 --name "hilton-frontend"

# Deploy backend
echo "Deploying backend..."
cd /var/www/hilton/backend
source ../venv/bin/activate
pm2 delete hilton-backend 2>/dev/null || true  # Delete if exists
pm2 start app.py --name "hilton-backend" --interpreter python3

echo "Deployment completed!" 