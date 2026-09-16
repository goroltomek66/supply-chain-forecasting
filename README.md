# Demand Forecasting & Inventory Planning

A portfolio application that turns product demand history into forecasts and
inventory recommendations. Built with Streamlit to explore model performance,
planning priorities, and the calculations behind each recommendation.

## Key features

- CSV upload, column detection, and standardized demand data.
- Per-product forecast model selection using holdout performance, with MAE and MAPE reporting.
- Product-specific lead times, safety stock, reorder points, and daily-review inventory targets.
- Increase, maintain, or run-down recommendations, with potential pre-arrival service gaps.
- Interactive Plotly charts, calculation details, and CSV exports.
- Deterministic planning answers and optional grounded natural-language explanations through OpenAI or local Ollama.

Forecasting and inventory calculations are deterministic Python calculations, not
LLM-generated numbers. The assistant answers precise factual questions directly;
where applicable, a grounded language model explains supplied results in natural language.
Model-generated explanations can still be inaccurate and do not replace the
calculated planning outputs.

## Tech stack

Python, Streamlit, pandas, NumPy, statsmodels, Plotly, and Matplotlib.
Ollama is an optional separate local service, not a Python package dependency.

## Run locally

The dependency versions in `requirements.txt` match the tested Python 3.9 environment.
From the project directory on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell.
Open the local URL printed by Streamlit (normally `http://localhost:8501`).
Upload a CSV, choose the planning settings, and click **Run Analysis**.

The included `data/sample_demand.csv` is a small Widget example for the data and
forecasting workflow. It has no inventory snapshots, so inventory actions are
withheld. For inventory planning, provide demand history with `date`, `product`,
`demand`, and `on_hand_inventory`; include `lead_time_days` for product-specific
lead times. Explicit zero stock is different from missing stock.

Other local datasets and generated `output/` files are excluded from Git by
default. Review a dataset's contents and sharing rights before intentionally
adding it to a public repository. Running analysis recreates the output directory.

### Optional AI explanations

Set `OPENAI_API_KEY` in Streamlit Community Cloud Secrets or the environment to
use OpenAI `gpt-5-nano` through the Responses API. Keep the actual key out of
source code and Git; `.streamlit/secrets.toml` is ignored. Environment settings
have precedence over Streamlit Secrets. Calculated planning context is sent to
the selected provider only for questions that need a language-model explanation.

Each such question makes at most one model request, with SDK retries disabled,
a 30-second timeout, and a 1,200-token output budget (including reasoning).
OpenAI failure returns calculated facts rather than making another provider call.
Deterministic questions do not contact either provider.

Without an OpenAI key, the app uses local Ollama when available:

With Ollama installed, start its local service and obtain the configured model:

```bash
ollama serve
```

In another terminal:

```bash
ollama pull llama3.2:3b
```

If the Ollama desktop application already runs the service, skip `ollama serve`.
The app connects to `http://127.0.0.1:11434`. Deterministic answers do not require
Ollama; when the explanation service is unavailable, the assistant can display
calculated planning facts instead.

## Planning assumptions

Inventory position currently equals usable on-hand stock; outstanding purchase
orders and backorders are not modeled. Run-down means reducing or deferring
replenishment, not disposing of inventory. Priority is a calculated planning
order, not a measured stockout probability. See
[inventory planning details](docs/inventory_planning.md) for formulas and calendar assumptions.

## Tests

With the virtual environment active:

```bash
python -m unittest discover -s tests -v
```
