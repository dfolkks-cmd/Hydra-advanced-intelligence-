# 🚀 Hydra Sentinel-X — VPS Deployment Guide

This guide walks you through deploying to a fresh Ubuntu 22.04 VPS from zero to a live,
TLS-secured, production API. Every step includes an explanation of *why* you're doing it,
not just *what* to type — because understanding the system means you can debug it when
something goes wrong at 2am.

---

## Prerequisites

You need: an Ubuntu 22.04 VPS (minimum 2 vCPU / 4GB RAM — the agents are LLM-heavy),
SSH access, a domain name pointed at the VPS's IP, and your `.env` file ready with all
API keys filled in.

---

## Phase 1: Harden the Server

The very first thing you do on a new VPS is lock it down. Cloud providers provision
machines with root login enabled and often with weak defaults. We fix that before
installing anything.

```bash
# SSH into your VPS as root
ssh root@YOUR_VPS_IP

# Create a non-root user for daily operations
# Running your app as root is a security risk: a compromised app = full server access
adduser hydra
usermod -aG sudo hydra       # give sudo access for installs
usermod -aG docker hydra     # allow running docker commands (added in Phase 2)

# Copy your SSH key to the new user so you can log in without a password
# Run this from your LOCAL machine, not the VPS
# ssh-copy-id hydra@YOUR_VPS_IP

# Switch to the new user for everything below
su - hydra

# Configure the firewall: only allow SSH, HTTP, and HTTPS
# Everything else is blocked by default
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status  # should show those three rules as ALLOW
```

---

## Phase 2: Install Docker

Docker is the only thing you need to install manually. Everything else — Python,
nginx, the app itself — runs inside containers.

```bash
# Update package list
sudo apt-get update

# Install Docker's GPG key (proves the package is genuinely from Docker, not tampered)
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker's official repository to apt
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine and Compose plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify the installation
docker --version        # should print Docker version 25+
docker compose version  # should print Docker Compose version 2+

# Allow the hydra user to run docker without sudo
# (we already added hydra to the docker group above)
# Log out and back in for group membership to take effect
exit && ssh hydra@YOUR_VPS_IP
```

---

## Phase 3: Deploy the Application

Now we get the code onto the server and configure it. The right way to do this in
production is via `git clone` from your repository — that way deployments are
repeatable and you can roll back by checking out a previous commit.

```bash
# Clone your repository
cd ~
git clone https://github.com/YOUR_USERNAME/hydra-sentinel-x.git
cd hydra-sentinel-x

# Create the .env file from the example and fill in your secrets
# NEVER commit .env to git — it contains your API keys
cp .env.example .env
nano .env   # add your ANTHROPIC_API_KEY and other keys here

# Create the directories that will be bind-mounted as Docker volumes.
# Docker will create these automatically, but creating them first means
# they're owned by the hydra user rather than root.
mkdir -p memory cdp nginx/ssl nginx/certbot-webroot

# Verify the directory structure looks right
ls -la
# You should see: memory/ cdp/ nginx/ Dockerfile docker-compose.yml server.py ...
```

---

## Phase 4: SSL Certificates (Let's Encrypt)

This is the step most guides gloss over, but it's what makes your API secure.
We use Certbot to get a free TLS certificate from Let's Encrypt. The certificate
proves to browsers and API clients that they're really talking to your server and
not an impersonator.

```bash
# Install Certbot
sudo apt-get install -y certbot

# Get a certificate for your domain.
# This uses the "webroot" method: Let's Encrypt verifies you control the domain
# by placing a challenge file in /var/www/certbot, which nginx serves.
# But we need nginx running first for that, so we use standalone mode initially.
sudo certbot certonly --standalone \
    --agree-tos \
    --non-interactive \
    --email your@email.com \
    -d yourdomain.com

# Certbot stores certs in /etc/letsencrypt/live/yourdomain.com/
# Copy them to where nginx expects them
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/
sudo chown hydra:hydra nginx/ssl/*

# Update nginx.conf to use your actual domain
sed -i 's/server_name _;/server_name yourdomain.com;/g' nginx/nginx.conf

# Set up automatic renewal (certificates expire every 90 days)
# Add a cron job that runs certbot renewal twice daily (standard practice)
echo "0 12 * * * root certbot renew --quiet && cp /etc/letsencrypt/live/yourdomain.com/*.pem ~/hydra-sentinel-x/nginx/ssl/" \
    | sudo tee /etc/cron.d/certbot-renew
```

---

## Phase 5: Launch

With Docker installed, the code deployed, secrets configured, and SSL certificates
in place, launching is a single command.

```bash
# Build the Docker image and start all services in detached mode (-d).
# Detached mode means the containers run in the background — your terminal
# is free, and the app keeps running after you close the SSH session.
docker compose up --build -d

# Watch the startup logs to confirm everything initialised correctly.
# You should see "System ready in X.Xs — LangGraph compiled, memory loaded."
docker compose logs -f hydra-api

# Check that all containers are running and healthy
docker compose ps
# Expected output:
#   hydra-sentinel-x   Up 30 seconds (healthy)
#   hydra-nginx        Up 25 seconds

# Test the health endpoint
curl https://yourdomain.com/
# Expected: {"status":"Hydra Sentinel-X Online","graph":"ready",...}

# Run a real analysis to confirm end-to-end functionality
curl -X POST https://yourdomain.com/api/v1/analyze \
    -H "Content-Type: application/json" \
    -d '{"query":"Analyse BTC","symbol":"BTC","portfolio_value":10000,"risk_tolerance":"moderate"}'
```

---

## Phase 6: Ongoing Operations

A deployed service needs to be maintainable. Here are the commands you'll use regularly.

```bash
# View live logs from the API (Ctrl+C to stop following)
docker compose logs -f hydra-api

# Deploy an updated version of the code
git pull origin main
docker compose up --build -d
# Docker rebuilds only the changed layers — usually takes 30-60s

# Restart the API without rebuilding (e.g. after changing .env)
docker compose restart hydra-api

# Inspect the SQLite databases directly from the host
sqlite3 memory/agent_memory.db "SELECT * FROM trade_history ORDER BY created_at DESC LIMIT 10;"

# Backup the memory databases (run from cron or before major updates)
cp -r memory memory_backup_$(date +%Y%m%d_%H%M%S)

# Stop everything (data is preserved in ./memory)
docker compose down

# Nuclear option: stop everything AND delete volume data (⚠️ irreversible)
docker compose down -v
```

---

## Architecture Summary

Once deployed, the request flow is:

```
Client (browser / app / curl)
        │
        ▼ HTTPS :443
   ┌─────────────────────────────────┐
   │         nginx container         │  ← TLS termination, rate limiting
   │    (hydra-nginx:80/443)         │    buffering, HTTP→HTTPS redirect
   └────────────┬────────────────────┘
                │ HTTP :8000 (internal Docker network)
                ▼
   ┌─────────────────────────────────┐
   │    FastAPI + Gunicorn container  │  ← 2 async Uvicorn workers
   │    (hydra-sentinel-x:8000)      │    LangGraph compiled once at startup
   └────────────┬────────────────────┘
                │
         ┌──────┴──────┐
         ▼             ▼
   ./memory/      Anthropic API  ←── all 4 specialist agents
   langgraph.db   CoinGecko API
   agent_memory.db
   (host bind mount — persists across restarts)
```

The key insight is that the **Docker volume mount** at `./memory:/app/memory` is what
makes the system stateful. Without it, every container restart would be a clean slate.
With it, the SQLite databases live on the host and the container is entirely replaceable —
you could `docker compose down`, rebuild a completely new image, and `docker compose up`,
and the agents would remember every trade and session from before.
