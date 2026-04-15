# Repo Hero Web

Minimal full-stack web interface for RepoHero.

## Run locally (no Docker)

Open two terminals from the `RepoHero` directory.

### Terminal 1: Backend

```bash
cd ./RepoHero
source venv/bin/activate
pip install -r requirements.txt
python3 web/backend/app.py
```

Backend URL: `http://127.0.0.1:5001`

### Terminal 2: Frontend

```bash
cd ./RepoHero/web/frontend
npm install
npm run dev
```

Frontend URL: `http://localhost:5173/`

## Usage

1. Open `http://localhost:5173/`
2. Enter a local repository path (for example: `/Users/zhirantong/Desktop/CMPT713/nlpclass-1261-g-TokenBurgers/RepoHero`)
3. Click `Index Repository`
4. Ask questions in the chat UI
