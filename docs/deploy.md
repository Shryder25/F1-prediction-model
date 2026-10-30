# Deploying the public demo

The demo is the Docker image in this repo. It serves the HUD with the synthetic demo data
and models baked in, so it starts in seconds and needs no API keys. Two free hosts work.
Pick one; Hugging Face is the better free option because it has more memory and only
sleeps after 48 idle hours.

| | Hugging Face Space (recommended) | Render |
| --- | --- | --- |
| Free resources | 2 vCPU, 16 GB RAM | 0.1 CPU, 512 MB RAM |
| Sleeps after | 48 hours idle | 15 minutes idle (about 1 minute to wake) |
| Deploys | GitHub Action on every push to `main` | Render watches `main` |
| URL | `https://huggingface.co/spaces/<you>/f1-race-predictor` | `https://f1pred-hud.onrender.com` (or similar) |

## Hugging Face Space (about 5 minutes)

1. Sign up at https://huggingface.co/join (free).
2. Create a token at https://huggingface.co/settings/tokens: **Create new token**, type
   **Write**, name it `github-deploy`. Copy it.
3. In GitHub, open the repo's **Settings > Secrets and variables > Actions**.
   - **Secrets** tab: **New repository secret**, name `HF_TOKEN`, paste the token.
   - **Variables** tab: **New repository variable**, name `HF_SPACE`, value
     `<your-hf-username>/f1-race-predictor`.
4. Open the **Actions** tab, pick **Deploy demo to Hugging Face**, and press
   **Run workflow**. It creates the Space and uploads the app.
5. Open the Space link printed at the end of the run. The first build takes about
   10 minutes (it trains the demo models). After that, every merge to `main` redeploys.

## Render (about 5 minutes)

1. Sign up at https://render.com with your GitHub account.
2. **New > Blueprint**, pick this repository. Render reads `render.yaml` and creates the
   `f1pred-hud` web service on the free plan.
3. Press **Apply**. The first build takes about 10 minutes. The URL is on the service page.

If the build runs out of memory on the free plan, use Hugging Face instead.

## After it is live

- Add the link to the top of the README (a "Live demo" badge) and to your CV and
  LinkedIn.
- Check `/api/health` on the live URL: `models_loaded` should be `true`.
- The HUD shows a banner saying the data is synthetic. To serve real races instead, the
  real warehouse and models (from `f1pred pipeline`) need to be uploaded with the app; that
  is a follow-up once the real pipeline has run.
