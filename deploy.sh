#!/bin/bash

# Update system
sudo apt update
sudo apt upgrade -y

# Install Node.js and npm
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install Python and pip
sudo apt install -y python3-pip python3-venv

# Install Nginx
sudo apt install -y nginx

# Install PM2 globally
sudo npm install -g pm2

# Create application directory
sudo mkdir -p /var/www/hilton
sudo chown -R $USER:$USER /var/www/hilton

# Create Python virtual environment
python3 -m venv /var/www/hilton/venv
source /var/www/hilton/venv/bin/activate

# Install Python dependencies (you'll need to create requirements.txt)
pip install flask flask-cors python-dotenv langchain-core langchain-openai langchain-community openai langsmith google-generativeai faiss-cpu numpy typing-extensions

# Configure Nginx
sudo tee /etc/nginx/sites-available/hilton.dahia.ai << EOF
server {
    listen 80;
    server_name hilton.dahia.ai www.hilton.dahia.ai;

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
sudo ln -s /etc/nginx/sites-available/hilton.dahia.ai /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Install Certbot for SSL
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d hilton.dahia.ai -d www.hilton.dahia.ai 