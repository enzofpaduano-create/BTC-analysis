# Déploiement 24/7 de l'alerts runner

Ce document explique comment faire tourner `scripts/run_alerts.py` en
permanence, avec restart automatique en cas de crash, et notifications
Telegram.

---

## 1. Setup Telegram (5 min)

### 1.1 Créer un bot

1. Ouvre Telegram, cherche **@BotFather**, démarre une conversation.
2. Envoie `/newbot`, suis les prompts (nom + handle).
3. BotFather te répond avec un message qui contient :
   ```
   Use this token to access the HTTP API:
   123456789:ABCdef-GhIjklMnOPqrStUvWxYz...
   ```
   **Copie ce token**, c'est `TELEGRAM_BOT_TOKEN`.

### 1.2 Récupérer ton chat_id

1. Dans Telegram, cherche le bot que tu viens de créer et envoie-lui un
   message (n'importe lequel, par exemple `hello`).
2. Dans ton terminal :
   ```bash
   TOKEN="123456789:ABCdef-GhIjkl..."
   curl "https://api.telegram.org/bot${TOKEN}/getUpdates"
   ```
3. Dans la réponse JSON, cherche `"chat":{"id":...}` — c'est ton
   `TELEGRAM_CHAT_ID`. C'est un nombre entier (positif pour un chat
   privé, négatif pour un groupe).

### 1.3 Mettre dans `.env`

```bash
cd ~/Desktop/btc-quant
cp .env.example .env       # si pas déjà fait
$EDITOR .env
```

Renseigne :
```
TELEGRAM_BOT_TOKEN=123456789:ABCdef-...
TELEGRAM_CHAT_ID=987654321
TELEGRAM_MIN_SCORE_ABS=0.3
```

### 1.4 Tester en mode normal

```bash
uv run python -m scripts.run_alerts
```

Tu devrais voir dans les logs :
```
INFO     Telegram sink active (chat_id=987654321, min_score_abs=0.3)
```

Tu ne recevras un message Telegram QUE quand `|score| ≥ 0.3`. Aux scores
faibles (en mode flat), les logs console défilent mais Telegram reste
silencieux — c'est voulu, pour ne pas spammer.

---

## 2. Déploiement 24/7 sur Mac (launchd)

### Avantages
- ✅ Démarre automatiquement à chaque login
- ✅ Restart automatique en cas de crash
- ✅ `caffeinate -i` empêche la mise en veille système

### Limitations
- ❌ Si tu éteins ton Mac (ou s'il s'éteint pour batterie morte), le
  runner s'arrête. Pour un vrai 24/7 il faudrait un VPS — voir §3.

### Installation

```bash
cd ~/Desktop/btc-quant
bash scripts/deploy/install_macos.sh
```

Le script :
1. Crée `~/Library/LaunchAgents/com.btc-quant.alerts.plist`
2. Le charge dans `launchd`
3. Démarre le runner immédiatement

### Vérifier que ça tourne

```bash
launchctl list | grep com.btc-quant.alerts
# affiche: PID  Status  Label
```

Logs en direct :
```bash
tail -f ~/Desktop/btc-quant/logs/alerts.out.log
tail -f ~/Desktop/btc-quant/logs/alerts.err.log
```

### Arrêt / désinstallation

```bash
bash ~/Desktop/btc-quant/scripts/deploy/uninstall_macos.sh
```

### Réglages Mac à faire en plus

`launchd` lance le process, mais le Mac peut quand même se mettre en
veille s'il est sur batterie. Va dans :

**Réglages Système → Batterie / Énergie**
- Sur secteur : **"Empêcher la mise en veille automatique quand l'écran est éteint"** ✅
- Mode économie d'énergie : OFF si tu peux

`caffeinate -i` (intégré dans le plist) empêche déjà la mise en veille
système, mais cette option est une ceinture-bretelles.

---

## 3. Alternative : VPS cloud (vrai 24/7)

État réel des hébergeurs en 2025-2026 :

| Service | Réellement gratuit 24/7 ? | Effort | Notes |
|---|---|---|---|
| **Oracle Cloud Free Tier** | ✅ vraiment gratuit "à vie" | 30-45 min | 4 vCPU ARM + 24 Go RAM. Best free option |
| **GCP Always Free** | ⚠️ 1 e2-micro USA only | 30 min | RAM limitée, latence US vers Bybit |
| **Fly.io** | ❌ plus de free tier depuis fin 2024 | — | $5/mois min (~$2 pour notre charge) |
| **Render free** | ❌ sleep après 15 min | — | Inutile pour polling continu |
| **Hetzner CX22 ARM** | ❌ 3,79 €/mois | 15 min | Le moins cher avec setup le plus simple |
| **Contabo VPS S** | ❌ ~5 €/mois | 15 min | Plus de RAM, parfois moins stable |

### Choix recommandé selon ton profil

- **Tu veux tester sans dépenser un centime** → Oracle Cloud Free (§3.1)
- **Tu acceptes ~4 €/mois pour ne pas perdre 30 min de setup** → Hetzner (§3.2)

### 3.1 Oracle Cloud Free Tier (gratuit à vie)

#### A. Créer le compte et la VM

1. Aller sur [cloud.oracle.com](https://cloud.oracle.com), créer un compte gratuit. Carte bancaire requise pour la vérification (jamais débitée tant que tu restes sur les ressources free).
2. Choisir une **région proche de l'Europe** (Frankfurt, Amsterdam ou Paris) — meilleure latence vers Bybit + conformité.
3. Console → **Compute → Instances → Create Instance**.
4. Configuration :
   - **Image** : Canonical Ubuntu 24.04 (Always Free Eligible)
   - **Shape** : `VM.Standard.A1.Flex` (ARM, Always Free)
   - **OCPUs** : 1 ; **Memory** : 6 GB (large pour notre charge)
   - **SSH key** : colle ton `~/.ssh/id_ed25519.pub`. Si tu n'as pas de clé :
     ```bash
     ssh-keygen -t ed25519 -C "btc-quant-vps"
     cat ~/.ssh/id_ed25519.pub
     ```
   - **Public IP** : assigned ✅
5. Attendre ~2 min. Note l'IP publique de la VM dans les détails de l'instance.
6. Ouvrir le port 22 (SSH) si le wizard ne l'a pas fait : Networking → Default Security List → Add Ingress Rule, port 22, source 0.0.0.0/0.

#### B. Setup sur la VM

```bash
ssh ubuntu@<IP_PUBLIQUE>

# Système à jour + outils de base
sudo apt-get update && sudo apt-get install -y git curl

# Installer uv (l'install script de la doc Astral)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Cloner le repo (depuis ton GitHub privé)
git clone https://github.com/enzofpaduano-create/BTC-analysis.git btc-quant
cd btc-quant

# Configurer .env avec ton token Telegram
cp .env.example .env
nano .env   # remplis TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# Installer les dépendances une fois (uv prend ~3 min en première install)
uv sync --extra data --extra features --extra backtest
```

#### C. Service systemd (équivalent launchd Mac)

```bash
sudo tee /etc/systemd/system/btc-alerts.service > /dev/null <<EOF
[Unit]
Description=btc-quant alerts runner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/btc-quant
ExecStart=/home/ubuntu/.local/bin/uv run --no-sync python -m scripts.run_alerts
Restart=always
RestartSec=30
StandardOutput=append:/home/ubuntu/btc-quant/logs/alerts.out.log
StandardError=append:/home/ubuntu/btc-quant/logs/alerts.err.log

[Install]
WantedBy=multi-user.target
EOF

mkdir -p ~/btc-quant/logs
sudo systemctl daemon-reload
sudo systemctl enable --now btc-alerts
sudo systemctl status btc-alerts          # doit dire "active (running)"
sudo journalctl -u btc-alerts -f          # logs en direct
```

#### D. Mettre à jour quand tu push du nouveau code

```bash
ssh ubuntu@<IP>
cd btc-quant
git pull
uv sync --extra data --extra features --extra backtest   # si deps changent
sudo systemctl restart btc-alerts
```

### 3.2 Hetzner CX22 ARM (3,79 €/mois)

Pareil mais plus simple côté création de VM :

1. [hetzner.com/cloud](https://hetzner.com/cloud), créer un compte (CB).
2. Add Server → Location: Falkenstein/Helsinki/Nuremberg → Image: Ubuntu 24.04 → Type: **CX22** (ARM, 3,79 €/mois) → SSH key → Create.
3. Note l'IP publique.
4. **Le reste est identique au §3.1 B-D**.

### 3.3 Via Docker (option moderne, n'importe quel cloud)

Le projet inclut un `Dockerfile`. Pour déployer où tu veux :

```bash
# Local / serveur :
docker build -t btc-quant-alerts .
docker volume create btc_data
docker run -d --restart unless-stopped \
  --name btc-alerts \
  --env-file .env \
  -v btc_data:/app/data_store \
  -v $(pwd)/logs:/app/logs \
  btc-quant-alerts

docker logs -f btc-alerts
```

Compatible avec Fly.io, Railway, Coolify, Dokku, n'importe quel hôte qui parle Docker.

---

## 4. Que se passe-t-il quand tu reçois un message ?

Format Telegram (HTML) :
```
🟢 BUY BTCUSDT — 7/10
score +0.72 • regime bull (p=0.82)
2026-04-29 14:30 UTC
━━━━━━━━━━━━━━━━━━━━
• mean_reversion_bb_hmm: +1 @ 0.30
• trend_breakout_adx_hmm: +1 @ 0.35
```

Décodage :
- 🟢 **BUY** = direction long (🔴 SELL = short, ⚪ WAIT = pas de signal)
- **7/10** = note de conviction (calculée depuis `|score| × 10`, plancher 1, plafond 10) :
  - 3-4/10 = signal limite, à scruter
  - 5-7/10 = setup propre
  - 8-10/10 = forte conviction, les deux stratégies alignées à pleine taille
- `score +0.72` : composite signé entre -1 et +1
- `regime: bull (p=0.82)` : régime HMM courant + probabilité
- Lignes du dessous : direction et taille vol-targeted de chaque stratégie

⚠️ **AUCUNE de ces alertes n'est un ordre exécuté**. Tu décides à la
main d'agir ou non. C'est le contrat du système (cf. brief initial).

---

## 5. Changer les paramètres

Tout est éditable dans `scripts/run_alerts.py` :
- Liste des stratégies actives + leurs poids
- Symbole / timeframe
- Seuils
- Paramètres des features

Après édition, redémarre :
```bash
launchctl stop com.btc-quant.alerts   # se relancera tout seul
```

---

## 6. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| Pas d'alerte malgré gros mouvement BTC | Threshold trop haut, ou les 2 strats sont OFF | Baisse `TELEGRAM_MIN_SCORE_ABS` à 0.2 |
| Beaucoup de FLAT en console mais 0 alerte Telegram | Normal — Telegram ne fire que sur conviction forte | rien à faire |
| `Telegram POST failed: ConnectionError` | Internet down ou Telegram down | Le runner continue ; les alertes loupées vont au log JSONL |
| Runner crash en boucle | Crédentials Bybit ou pybit cassé | `tail -f logs/alerts.err.log` pour la stack trace |
| Mac dort quand même | `caffeinate` insuffisant (rare) | Réglages Système → Batterie : "Empêcher la mise en veille" |
