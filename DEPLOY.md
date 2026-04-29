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

Si tu veux que ça tourne même quand ton Mac est éteint, options :

| Service | Prix | Notes |
|---|---|---|
| **Hetzner** CX11 | ~4 €/mois | VPS Linux 2 vCPU, le moins cher fiable |
| **Contabo** VPS S | ~5 €/mois | Plus de RAM, parfois moins stable |
| **Fly.io** | gratuit (3 micro VMs) | Déploiement Docker, scale automatique |
| **Render** | gratuit (avec sleep) | Sleep après 15 min sans trafic — pas top pour nous |
| **Oracle Cloud Free** | gratuit | 2 ARM VMs, à condition d'avoir un compte vérifié |

Pour Hetzner / Contabo (le plus simple) :
1. Crée un VPS Ubuntu 24.04
2. SSH dedans, installe `git`, `uv`, clone le repo
3. Crée le `.env` (copie + colle le `.env.example`, remplis Telegram)
4. Au lieu de `launchd`, utilise **`systemd`** :

```ini
# /etc/systemd/system/btc-alerts.service
[Unit]
Description=btc-quant alerts runner
After=network-online.target

[Service]
Type=simple
User=enzo
WorkingDirectory=/home/enzo/btc-quant
ExecStart=/home/enzo/.local/bin/uv run --no-sync python -m scripts.run_alerts
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now btc-alerts
sudo systemctl status btc-alerts
journalctl -u btc-alerts -f   # logs en direct
```

C'est l'équivalent direct du setup launchd, sur Linux.

---

## 4. Que se passe-t-il quand tu reçois un message ?

Format Telegram (HTML) :
```
🟢 LONG
score +0.65
2026-04-29 14:30 UTC
regime: bull (p=0.82)
strategies:
  • mean_reversion_bb_hmm: +1 @ 0.30
  • trend_breakout_adx_hmm: +1 @ 0.35
```

Décodage :
- 🟢 = LONG (🔴 = SHORT, ⚪ = FLAT)
- `score +0.65` : entre -1 et +1, magnitude = conviction
- `regime: bull (p=0.82)` : régime HMM courant + probabilité
- `strategies` : direction (+1/-1) et taille vol-targeted de chaque strat

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
