# Overview
This is Eric's playground with [DSPy](https://github.com/stanfordnlp/dspy/).

It includes sample DSPy modules for GSM8K and MMLU.  These are fully self contained e.g., you can run them by going `python ./name_of_file.py` and it just works.

Tested on Python 3.11.  Install `pip install mlflow dpsy`.

## GSM8k

1. Open `dpsy_gsm8k.py`
2. Add your OpenAI key to line 18 `os.environ["OPENAI_API_KEY"] = ...`
3. Edit lines 169-171 if you want to run on a subset of the dataset first to test
3. Start the local MLflow server with `mlflow server --host 127.0.0.1 --port 8080`
4. Execute `python ./dpsy_gsm8k.py`
5. View the runs in MLflow and/or via the console
6. To use the logged MLflow model, see `load_gsm8k_from_mlflow.py`.

Optionally, adjust `ENABLE_ARIZE_TRACING = true` if you want to see trace logs in Phoenix.  You'll need to:
- Install the packages `pip install arize-phoenix openinference-instrumentation-dspy opentelemetry-exporter-otlp`
- Start their container `docker run -p 6006:6006 arizephoenix/phoenix:latest`

## MMLU

1. Open `dpsy_gsm8k.py`
2. Add your OpenAI key to line 18 `os.environ["OPENAI_API_KEY"] = ...`
3. Edit lines 193-195 if you want to run on a subset of the dataset first to test
3. Start the local MLflow server with `mlflow server --host 127.0.0.1 --port 8080`
4. Execute `python ./dpsy_mmlu.py`
5. View the runs in MLflow and/or via the console

If you prefer to re-ETL the MMLU data, you can run `download_mmlu.py`.  However, the results of this download are saved in `mmlu_*.json`.


## Azure OpenAI

Azure OpenAI's content safety filter blocks DSPy's generated queries so this isn't fully tested. `azureopenai.py` is a working client for Azure OpenAI since the default DSPy implementation doesn't work.  TODO: Contribute this back and clean up the depedencies between the DSPy AzureOpenAI library and the client code.

```
from azureopenai import AzureOpenAI
os.environ["AZURE_OPENAI_API_KEY"] = "<your-key-here>"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-url.openai.azure.com/"

model = AzureOpenAI(
     api_provider="azure",
     model_type="chat",
     api_version="2023-07-01-preview", # version from Azure 2023-07-01-preview
     model="deployment_name",  # deployment_name from Azure - often the same as the model name e.g., chat-3.5 but not necessarly
)
dspy.configure(lm=model)
```