# Hosting De:Mark on your VPS

De:Mark is a Python/Streamlit app, so it runs on a server that executes Python —
not on the static thevisionarys.com hosting. This guide puts it on your VPS at
**https://demark.thevisionarys.com**, which the "Our Products" card links to.

Everything below is done once. Total time ~20–30 min.

---

## 1. Point a subdomain at the VPS  (DNS)

In your DNS panel (Hostinger hPanel → Domains → DNS, or your registrar) add:

| Type | Name     | Value (points to) | TTL  |
|------|----------|-------------------|------|
| A    | `demark` | `<your VPS IP>`   | 3600 |

Wait a few minutes, then confirm: `ping demark.thevisionarys.com` resolves to the VPS IP.

---

## 2. Get the code onto the VPS

SSH in (`ssh root@<VPS IP>`), then either:

**Option A — copy from your PC** (run on your PC):
```bash
scp -r "C:/Users/Chintan Kamani/Desktop/watermark-remover" root@<VPS IP>:/opt/demark
```

**Option B — via a private GitHub repo** (cleaner for updates):
```bash
# on your PC, once:
cd "C:/Users/Chintan Kamani/Desktop/watermark-remover"
git init && git add . && git commit -m "De:Mark"
# create a PRIVATE repo TheVisionarysLtd/demark on github.com, then:
git remote add origin https://github.com/TheVisionarysLtd/demark.git
git push -u origin main
# on the VPS:
git clone https://github.com/TheVisionarysLtd/demark.git /opt/demark
```

---

## 3. Install Docker (once, on the VPS)

```bash
curl -fsSL https://get.docker.com | sh
```

---

## 4. Build & run De:Mark

```bash
cd /opt/demark
docker build -t demark .
docker run -d --name demark --restart unless-stopped \
  -p 127.0.0.1:8501:8501 \
  -v demark-cache:/root/.cache \
  demark
```

- `127.0.0.1:8501` keeps the app private to the box; Nginx (next step) exposes it over HTTPS.
- The `demark-cache` volume stores the ~200 MB LaMa model so it isn't re-downloaded on restart.
- First image build takes a few minutes (installs PyTorch). Check it's healthy:
  `docker ps` → STATUS should become `healthy`, and `curl -s localhost:8501/_stcore/health` prints `ok`.

---

## 5. Put Nginx in front (reverse proxy + WebSockets)

Streamlit needs WebSocket upgrade headers and a large upload limit.

```bash
apt-get update && apt-get install -y nginx
nano /etc/nginx/sites-available/demark
```

Paste:

```nginx
server {
    listen 80;
    server_name demark.thevisionarys.com;

    client_max_body_size 2048M;      # match the 2 GB upload limit

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;      # WebSocket
        proxy_set_header Connection "upgrade";       # WebSocket
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;    # long jobs (video/PDF) don't time out
        proxy_send_timeout 3600s;
    }
}
```

Enable it and reload:
```bash
ln -s /etc/nginx/sites-available/demark /etc/nginx/sites-enabled/demark
nginx -t && systemctl reload nginx
```

At this point http://demark.thevisionarys.com should load De:Mark.

---

## 6. Add HTTPS (free, auto-renewing)

```bash
apt-get install -y certbot python3-certbot-nginx
certbot --nginx -d demark.thevisionarys.com --redirect -m you@thevisionarys.com --agree-tos
```

Now **https://demark.thevisionarys.com** is live with a valid certificate.

---

## 7. Turn the website button on

In `TVL_New-Website/src/sections/OurProducts.jsx`, near the top:

```js
const DEMARK_URL  = "https://demark.thevisionarys.com";  // already set
const DEMARK_LIVE = false;   // ← change to true
```

Set `DEMARK_LIVE = true`, then commit & push `main` — the "Launching soon" pill
becomes a live **"Launch De:Mark"** button. (Tell me when the VPS is up and I'll
flip it and push for you.)

---

## Updating De:Mark later

```bash
cd /opt/demark
git pull            # or re-scp the folder
docker build -t demark .
docker rm -f demark
docker run -d --name demark --restart unless-stopped \
  -p 127.0.0.1:8501:8501 -v demark-cache:/root/.cache demark
```

## Notes & sizing
- **RAM:** the LaMa model needs ~2 GB free while running. On a small VPS (≤1 GB),
  either add swap or set the inpaint engine to OpenCV in the app's Advanced panel —
  it still removes watermarks, just without the deep-learning fill.
- **CPU:** LaMa on CPU is a few hundred ms per image and per video frame; fine for
  a demo/product page. A busy site would want a bigger box or a GPU.
- De:Mark removes only the **visible** corner marks (Gemini sparkle, NotebookLM
  badge). It does not touch invisible provenance marks such as SynthID.
