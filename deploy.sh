#!/bin/bash

# Update system
sudo apt update
sudo apt upgrade -y

# Create necessary directories
sudo mkdir -p /var/www
sudo chown -R $USER:$USER /var/www

# Install Node.js and npm
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install Python and pip
sudo apt install -y python3-pip python3-venv

# Install Nginx
sudo apt install -y nginx

# Enable Nginx to start on boot
sudo systemctl enable nginx

# Install PM2 globally
sudo npm install -g pm2

# Create application directory
sudo mkdir -p /var/www/vorex
sudo chown -R $USER:$USER /var/www/vorex

# Create Python virtual environment
python3 -m venv /var/www/vorex/venv
source /var/www/vorex/venv/bin/activate

# Install Python dependencies
pip install flask flask-cors python-dotenv langchain-core langchain-openai langchain-community openai langsmith google-generativeai faiss-cpu numpy typing-extensions

# Configure Nginx
sudo tee /etc/nginx/sites-available/poc_v2.vorexai.com << EOF
# Redirect www to non-www
server {
    listen 80;
    server_name www.poc_v2.vorexai.com;
    return 301 \$scheme://poc_v2.vorexai.com\$request_uri;
}

# Main server block
server {
    listen 80;
    server_name poc_v2.vorexai.com;

    location / {
        proxy_pass http://localhost:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

# Enable the site
sudo ln -s /etc/nginx/sites-available/poc_v2.vorexai.com /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Restart and enable Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx

# Install Certbot for SSL
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate (only for main domain)
sudo certbot --nginx -d poc_v2.vorexai.com 