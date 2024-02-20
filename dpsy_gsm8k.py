import mlflow
import random
import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.gsm8k import GSM8K
import os
from mlflow import MlflowClient
import cloudpickle
import pandas as pd
from dspy_helpers import *

################
# Configuration
################

# OpenAI Key
os.environ["OPENAI_API_KEY"] = "<your-key-here>"

# Log the unoptimized, baseline model to MLflow?
LOG_BASELINE_MODEL = True
# Evaluate each logged model?
SHOULD_EVALUATE = True
# Try to optimize the model?
SHOULD_OPTIMIZE = True
# One of: BayesianSignatureOptimizer, SignatureOptimizer, BootstrapFewShotWithRandomSearch, BootstrapFewShot, BootstrapFewShotWithOptuna
OPTIMIZER = "BayesianSignatureOptimizer"

# enable local tracking
ENABLE_ARIZE_TRACING = False


################
### The core definition of the Signature & Model
################
class GSM8kSignature(dspy.Signature):
    """Solve grade school math word problems that are tricky."""

    question = dspy.InputField(desc="the math word problem")
    answer = dspy.OutputField(desc="The answer to the math word problem")


class EricGsm8k(dspy.Module):
    def __init__(self):
        super().__init__()
        # self.prog = dspy.ChainOfThought("question -> answer")
        self.prog = dspy.ChainOfThought(GSM8kSignature)

    def forward(self, question):
        return self.prog(question=question)


################
# Metric for optimization
################
def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split("\n")[0]

        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][
            -1
        ]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0

    return answer


def gsm8k_metric(gold, pred, trace=None):
    return int(parse_integer_answer(str(gold.answer))) == int(
        parse_integer_answer(str(pred.answer))
    )


################
# PyFunc Wrapper
################
class EricGsm8kPyfunc(mlflow.pyfunc.PythonModel):

    def load_model(self):

        self.dspy_lm = dspy.OpenAI(model="gpt-3.5-turbo")

    def load_context(self, context):
        """
        Load the DSPy model
        """
        import dspy
        import cloudpickle

        # Connect to OpenAI
        self.load_model()
        dspy.configure(lm=self.dspy_lm)

        # Load the compiled model
        with open(context.artifacts["dspy_model"], "rb") as f:
            self.compiled_model = cloudpickle.load(f)

    # TODO: automatically generate the signature
    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        question = model_input["question"][0]

        answer = self.compiled_model.forward(question=question)

        return pd.DataFrame(
            {
                "answer": [answer["answer"]],
                "rationale": [answer["rationale"]],
            }
        )


################
# Local tracking of calls sent to the LLM
################
def setup_arize_phoenx():
    # Arize Phoenix instrumentation - must start a local server
    from openinference.instrumentation.dspy import DSPyInstrumentor
    from opentelemetry import trace as trace_api
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # Trace logging
    endpoint = "http://127.0.0.1:6006/v1/traces"
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter=span_otlp_exporter)
    )

    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    DSPyInstrumentor().instrument()


################
# Main training loop
################
if __name__ == "__main__":
    if ENABLE_ARIZE_TRACING:
        setup_arize_phoenix()

    # Requires having a local MLflow tracking server running
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
    mlflow.set_experiment("dspy_gsm8k")

    # DSPY setup
    # regular OpenAI
    model = dspy.OpenAI(model="gpt-3.5-turbo")

    dspy.configure(lm=model)

    gms8k = GSM8K()

    # original model
    original = EricGsm8k()

    trainset = gms8k.train  # [:10]
    valset = gms8k.dev  # [:10]
    testset = gms8k.test  # [:200]

    # eric's run name generator so we can have {run-name}-optimized & {run-name}-unoptimized but still keep mlflow fun names
    run_name = generate_run_name()

    # Set up the evaluator, which can be used multiple times.
    metric = gsm8k_metric
    evaluate = Evaluate(
        devset=testset,
        metric=metric,
        num_threads=4,
        display_progress=True,
        display_table=0,
    )

    models_to_evaluate_and_log = []
    if LOG_BASELINE_MODEL:
        models_to_evaluate_and_log.append((original, "unoptimized"))

    if SHOULD_OPTIMIZE:
        # BayesianSignatureOptimizer, SignatureOptimizer, BootstrapFewShotWithRandomSearch, BootstrapFewShot

        if OPTIMIZER == "BayesianSignatureOptimizer":
            optimized = get_optimized_model_BayesianSignatureOptimizer(
                model=original, trainset=trainset, metric=metric
            )
        elif OPTIMIZER == "SignatureOptimizer":
            optimized = get_optimized_model_SignatureOptimizer(
                model=original, trainset=trainset, metric=metric
            )
        elif OPTIMIZER == "BootstrapFewShotWithRandomSearch":
            optimized = get_optimized_model_BootstrapFewShot(
                model=original,
                trainset=trainset,
                valset=valset,
                metric=metric,
                random_search=True,
            )
        elif OPTIMIZER == "BootstrapFewShot":
            optimized = get_optimized_model_BootstrapFewShot(
                model=original,
                trainset=trainset,
                valset=valset,
                metric=metric,
                random_search=False,
            )
        elif OPTIMIZER == "BootstrapFewShotWithOptuna":
            optimized = get_optimized_model_BootstrapFewShot(
                model=original,
                trainset=trainset,
                valset=valset,
                metric=metric,
                random_search=False,
            )
        models_to_evaluate_and_log.append((optimized, f"optimized-{OPTIMIZER}"))

    for combo in models_to_evaluate_and_log:
        name = combo[1]
        model = combo[0]

        dump_file = "dpsy_module.pkl"

        with open(dump_file, "wb") as f:
            cloudpickle.dump(model, f)

        artifacts = {"dspy_model": dump_file}

        with mlflow.start_run(run_name=f"{run_name}-{name}") as run:
            mlflow.pyfunc.log_model(
                "dspy_model",
                python_model=EricGsm8kPyfunc(),
                # TODO: Add signature to the model
                # input_example=x_train,
                # signature=signature,
                artifacts=artifacts,
                pip_requirements=["dspy", "cloudpickle"],
            )

            log_model_dump_to_mlflow(model)
            mlflow.log_param("testset_N", len(testset))
            mlflow.log_param("trainset_N", len(trainset))
            mlflow.log_param("valset_N", len(valset))

            # Evaluate our program.
            if SHOULD_EVALUATE:
                run_eval_and_log_to_mlflow(evaluator=evaluate, model_to_evaluate=model)
