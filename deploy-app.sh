#!/bin/bash

# Pull latest changes from GitHub
echo "Pulling latest changes from GitHub..."
cd /var/www/vorex
git pull origin main

# Deploy frontend
echo "Deploying frontend..."
cd /var/www/vorex
npm install
npm run build
pm2 delete vorex-frontend 2>/dev/null || true
pm2 serve dist 5173 --name "vorex-frontend"

# Deploy backend
echo "Deploying backend..."
cd /var/www/vorex/Backend
source ../venv/bin/activate
pm2 delete vorex-backend 2>/dev/null || true
pm2 start app.py --name "vorex-backend" --interpreter python3

# Save PM2 process list and configure PM2 to start on system boot
pm2 save
pm2 startup

echo "Deployment completed! Application will now run persistently and start automatically on system reboot." 