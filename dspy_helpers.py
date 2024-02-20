from dspy.teleprompt import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    BayesianSignatureOptimizer,
    SignatureOptimizer,
    BootstrapFewShotWithOptuna,
)
import random
import mlflow
import re
import json


def get_optimized_model_BootstrapFewShot(
    model, trainset, valset, metric, random_search=True
):
    config_bootstrap = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
    if random_search:
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=metric, **config_bootstrap
        )
    else:
        teleprompter = BootstrapFewShot(metric=metric, **config_bootstrap)

    optimized = teleprompter.compile(model, trainset=trainset, valset=valset)
    return optimized


def get_optimized_model_BayesianSignatureOptimizer(model, trainset, metric):
    # BayesianSignatureOptimizer
    teleprompter = BayesianSignatureOptimizer(
        metric=metric, n=10, init_temperature=1.0, verbose=False, track_stats=True
    )
    kwargs = dict(num_threads=4, display_progress=True, display_table=0)

    optimized = teleprompter.compile(
        model,
        devset=trainset,
        optuna_trials_num=5,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
        eval_kwargs=kwargs,
    )

    return optimized


def get_optimized_model_SignatureOptimizer(model, trainset, metric):
    teleprompter = SignatureOptimizer(
        metric=metric, breadth=10, depth=3, init_temperature=1.4
    )
    kwargs = dict(num_threads=4, display_progress=True, display_table=0)
    optimized = teleprompter.compile(
        model.deepcopy(), devset=trainset, eval_kwargs=kwargs
    )

    return optimized


def get_optimized_model_BootstrapFewShotWithOptuna(model, trainset, valset, metric):
    # BayesianSignatureOptimizer
    teleprompter = BootstrapFewShotWithOptuna(metric=metric)
    # kwargs = dict(num_threads=4, display_progress=True, display_table=0)

    optimized = teleprompter.compile(model, trainset=trainset, valset=valset)

    return optimized


def generate_run_name():
    adjectives = [
        "bold",
        "quick",
        "lively",
        "brave",
        "calm",
        "eager",
        "fierce",
        "gentle",
        "happy",
        "jolly",
        "keen",
        "proud",
        "sly",
        "witty",
        "young",
    ]
    nouns = [
        "duck",
        "cat",
        "dog",
        "lion",
        "tiger",
        "bear",
        "wolf",
        "fox",
        "eagle",
        "hawk",
        "owl",
        "fish",
        "shark",
        "whale",
        "dolphin",
    ]

    # Select one adjective and one noun at random
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)

    # Generate a random number between 100 and 999
    number = random.randint(100, 999)

    # Combine the parts to form a run name
    run_name = f"{adjective}-{noun}-{number}"

    return run_name


def run_eval_and_log_to_mlflow(evaluator, model_to_evaluate):
    (scores, outputs) = evaluator(
        model_to_evaluate, return_all_scores=True, return_outputs=True
    )
    # print(scores)
    # print(outputs)
    # scores = 10 if scores == 0 else scores
    mlflow.log_metric("accuracy", scores)
    eval_results = []
    for output in outputs:
        # print("---")
        # print(output)
        # print("xxx")
        input_example = output[0].toDict()
        input_example["gold_answer"] = input_example.pop("answer")

        generated_answer = output[1].toDict()
        is_accurate = output[2]

        merged_dict = {**input_example, **generated_answer}
        merged_dict["correct"] = is_accurate
        # print(merged_dict)
        eval_results.append(merged_dict)
        # break
    # combined_dicts = [mlflow_dict, databricks_dict]
    dict_for_mlflow_logging = {
        key: [d[key] for d in eval_results] for key in eval_results[0]
    }

    # print(dict_for_mlflow_logging)
    mlflow.log_table(data=dict_for_mlflow_logging, artifact_file="eval_results.json")


def log_model_dump_to_mlflow(model):

    model_predictors = []
    for item in model.named_predictors():
        param_name = f"signature_{item[0]}"
        param_name = re.sub(r"[^a-zA-Z0-9]", "", param_name)
        model_predictors.append(
            {
                "param_name": item[0],
                "param_value": str(item[1].extended_signature),
            }
        )
        mlflow.log_param(
            param_name, str(item[1].extended_signature)[:5990]
        )  # mlflow has a 6000 char limit
        # i = i + 1
    named_predictors_file = "model_named_predictors.json"
    with open(named_predictors_file, "w") as f:
        json.dump(model_predictors, f)

    mlflow.log_artifact(named_predictors_file)

    dump_state_file = "dump_state.json"
    with open(dump_state_file, "w") as f:
        json.dump(model.dump_state(), f)

    mlflow.log_artifact(dump_state_file)

    mlflow.log_param("state", model.dump_state())


################
# Local tracking of calls sent to the LLM
################
def setup_arize_phoenix():
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
