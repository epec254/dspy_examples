from mlflow import MlflowClient
import mlflow
import pandas as pd
import logging
import os
import dspy

os.environ["OPENAI_API_KEY"] = "<your-key-here>"

## Start Arize
# from openinference.instrumentation.dspy import DSPyInstrumentor
# from opentelemetry import trace as trace_api
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk import trace as trace_sdk
# from opentelemetry.sdk.resources import Resource
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor
#
# endpoint = "http://127.0.0.1:6006/v1/traces"
# resource = Resource(attributes={})
# tracer_provider = trace_sdk.TracerProvider(resource=resource)
# span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
# tracer_provider.add_span_processor(
#     SimpleSpanProcessor(span_exporter=span_otlp_exporter)
# )
#
# trace_api.set_tracer_provider(tracer_provider=tracer_provider)
# DSPyInstrumentor().instrument()
## End Arize


logging.basicConfig(level=logging.DEBUG)

client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# Define the run ID and artifact path
# This can be grabbed from the MLflow UI
run_id = "f6273ca57305412ea150c67f673987f2"

# If you haven't changed the sample code, this is `dspy_model`
artifact_path = "dspy_model"

# Construct the model URI
model_uri = f"runs:/{run_id}/{artifact_path}"

# Load the model
model = mlflow.pyfunc.load_model(model_uri)

inputs = pd.DataFrame(
    {
        "question": [
            "Amber, Micah, and Ahito ran 52 miles in total. Amber ran 8 miles. Micah ran 3.5 times what Amber ran. How many miles did Ahito run?"
        ],
    }
)

result = model.predict(inputs)

# print(type(result))
print(result)
