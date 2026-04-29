# btc-quant

Système d'analyse quantitative pour le trading BTC court terme. Ingestion → features → signaux → backtest → alertes live (sans exécution auto).

> **Statut actuel :** Étape 1 — bootstrap. La connexion broker, les features, et le backtest seront ajoutés étape par étape.

---

## Stack

- **Python 3.12** (pinné — coverage ecosystem optimal pour `pandas-ta`, `vectorbt`, `hmmlearn` ; géré par `uv`)
- **Broker** : [Bybit](https://www.bybit.com) (BTCUSDT perpétuel, USDT-margined) via `pybit`
- **Données** : `pandas`, `numpy`, `polars`, `duckdb`, `pyarrow` (Parquet)
- **Features** : `pandas-ta`, `arch` (GARCH), `hmmlearn` (régimes), `ruptures` (change-point), `pykalman`
- **Backtest** : `vectorbt`
- **ML** (étape ultérieure) : `scikit-learn`, `lightgbm`
- **Outillage** : `pydantic` (config), `loguru` (logs), `pytest`, `ruff`, `black`, `mypy --strict`, `pre-commit`

## Pourquoi Bybit et pas MT5 ?

Le paquet Python `MetaTrader5` ne tourne pas sur macOS. Comme Enzo bosse depuis un Mac, on est passé à Bybit, qui :

- a une API native Mac (`pybit`),
- offre un **testnet gratuit illimité** pour le paper trading (`https://testnet.bybit.com`),
- expose un **historique BTC complet et gratuit** (klines depuis 2019),
- propose des **perpétuels avec levier** (équivalent fonctionnel d'un CFD BTC).

Si on ajoute plus tard de l'or ou d'autres actifs non-crypto, on branchera un connecteur supplémentaire (IBKR ou OANDA) — l'archi en couches le permet sans toucher aux features ni au backtest.

---

## Setup (macOS)

### 1. Installer `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # ou redémarrer le shell
```

### 2. Cloner / se placer dans le dossier

```bash
cd ~/Desktop/btc-quant
```

### 3. Installer les dépendances de l'étape courante

L'étape 1 ne nécessite que les libs de bootstrap (config, logging, tests). Les libs lourdes sont en extras et seront activées étape par étape.

```bash
uv sync                         # étape 1 : bootstrap seul
uv sync --extra data            # étape 2 : connecteur Bybit + Parquet/DuckDB
uv sync --extra data --extra features   # étape 3 : indicateurs + GARCH + HMM
uv sync --all-extras            # tout d'un coup (plus lent)
```

### 4. Configurer Bybit (uniquement à partir de l'étape 2)

1. Créer un compte sur **<https://testnet.bybit.com>** (gratuit, USDT virtuels).
2. *Settings → API → Create New API Key*. Permissions : **Read-Only** suffit pour télécharger l'historique. Ne génère une clé en écriture que quand on passera au mode alertes/exécution.
3. Copier `.env.example` vers `.env` et remplir :

```bash
cp .env.example .env
$EDITOR .env
```

### 5. Activer pre-commit

```bash
uv run pre-commit install
```

---

## Structure du repo

```
btc-quant/
├── core/         # utilitaires transverses : settings, logging
├── data/         # Étape 2 — ingestion Bybit, stockage Parquet, DuckDB
├── features/     # Étape 3 — indicateurs, vol, régimes, microstructure
├── signals/      # Étape 5 — stratégies + scoring + sizing
├── backtest/     # Étape 4 — harness vectorbt avec coûts réalistes
├── live/         # Étape 6 — boucle live, alertes (pas d'exécution auto)
├── config/       # YAML par actif / stratégie (validés par pydantic)
├── notebooks/    # Exploration uniquement, jamais de code de prod
└── tests/        # pytest, coverage > 70%
```

**Règle :** une couche ne dépend que des couches inférieures. `features` ne lit pas `signals`. `backtest` ne lit pas `live`. Tout passe par des DataFrames bien typés.

## Règles de qualité (non négociables)

1. Type hints partout, `mypy --strict` doit passer.
2. Docstrings format Google sur toute fonction publique.
3. Configs en YAML validées par `pydantic` — jamais de hardcode.
4. Logging structuré `loguru`, niveau configurable.
5. Tests unitaires sur chaque module critique, **coverage > 70 %**.
6. Tout chemin fichier passe par `pathlib.Path`.
7. Toute donnée temporelle est **timezone-aware en UTC** en interne.
8. Pas de notebook dans le code de prod — uniquement `/notebooks`.

---

## Commandes courantes

| Commande                              | Effet                                             |
| ------------------------------------- | ------------------------------------------------- |
| `uv sync`                             | Installer les deps de bootstrap                   |
| `uv sync --extra data`                | Ajouter les deps de l'étape 2                     |
| `uv run pytest`                       | Lancer les tests                                  |
| `uv run pytest --cov`                 | Tests + couverture                                |
| `uv run ruff check .`                 | Lint                                              |
| `uv run ruff format .`                | Format auto                                       |
| `uv run black .`                      | Formater (en plus de ruff-format)                 |
| `uv run mypy .`                       | Type-check strict                                 |
| `uv run pre-commit run --all-files`   | Lancer tous les hooks pre-commit manuellement     |

---

## Roadmap (étape par étape, validation à chaque palier)

- [x] **Étape 1 — Bootstrap** : structure, pyproject, README, .env.example, pre-commit, pytest, settings + logging.
- [x] **Étape 2 — Data** : connecteur Bybit (REST polling), download historique résilient, Parquet partitionné, DuckDB, qualité auto (outliers/gaps/zero-vol).
- [x] **Étape 3 — Features** : RSI/MACD/BB/ATR/EMAs, vol multi-fenêtres, GARCH(1,1) walk-forward, HMM 3-états (sortés bear/range/bull), ruptures PELT, Kalman local-linear-trend. **Test de causalité strict passé.**
- [x] **Étape 4 — Backtest** : harness custom numpy (vectorbt en backup), spread/slippage/funding réalistes, walk-forward + purge/embargo, métriques complètes (Sharpe/Sortino/Calmar/MDD/PF/expectancy), rapport HTML plotly. **Test no-look-ahead strict passé.**
- [x] **Étape 5 — Stratégie baseline** : mean-reversion Bollinger filtré HMM + framework grid-search/walk-forward + 2ème stratégie trend-following Donchian + ADX + HMM bull. Tous tests no-leak passés.
- [x] **Étape 6 — Live alertes** : `AlertsRunner` (polling Bybit → features → score composite multi-strat → console + JSONL). Aucune exécution.

Aucune exécution automatique tant que plusieurs mois de stats live ne sont pas cohérents avec le backtest.

---

## Scripts disponibles

| Commande | Effet |
|---|---|
| `uv run python -m scripts.smoke_e2e` | Smoke test data layer (1 jour de BTCUSDT M1 mainnet) |
| `DAYS_BACK=180 uv run python -m scripts.run_baseline_backtest` | Backtest baseline mean-reversion sur N jours BTCUSDT M5 + rapport HTML |
| `uv run python -m scripts.run_alerts` | Boucle live alertes (polling + scoring composite, **aucune exécution**) |
