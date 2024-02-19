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
import json

################
# Configuration
################

# Eric's key
os.environ["OPENAI_API_KEY"] = "<your-key-here>"

# Log the unoptimized, baseline model to MLflow?
LOG_BASELINE_MODEL = False
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

class MMLUSignature(dspy.Signature):
    """Solve tricky multiple choice problems about various subjects.  There are 57 subjects across STEM, the humanities, the social sciences, and more. It ranges in difficulty from an elementary level to an advanced professional level, and it tests both world knowledge and problem solving ability. Subjects range from traditional areas, such as mathematics and history, to more specialized areas like law and ethics. Some require you to answer a question, some require you to fill in the blank, some require you to finish the question with the correct answer."""

    subject = dspy.InputField(desc="the subject of the question")
    question = dspy.InputField(
        desc="the question to be answered with one of the choices"
    )
    choice_a = dspy.InputField(desc="the first choice you can select from")
    choice_b = dspy.InputField(desc="the second choice you can select from")
    choice_c = dspy.InputField(desc="the third choice you can select from")
    choice_d = dspy.InputField(desc="the fourth choice you can select from")
    answer = dspy.OutputField(
        desc="The answer which is always one choice_a, choice_b, choice_c, or choice_d - NOT the answer itself"
    )

class EricMMLU(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(MMLUSignature)

    def forward(self, subject, question, choice_a, choice_b, choice_c, choice_d):
        return self.prog(
            subject=subject,
            question=question,
            choice_a=choice_a,
            choice_b=choice_b,
            choice_c=choice_c,
            choice_d=choice_d,
        )


################
# Metric for optimization
################
def mmlu_metric(gold, pred, trace=None):
    """
    This function is used to calculate the metric for the MMLU model.
    We give the model credit as long as it starts w/ "choice N" or "choice_N"
    """

    choice_letter = gold.answer[-1]

    options = [f"choice_{choice_letter}", f"choice {choice_letter}"]

    modified_prediction = pred.answer[: len(options[0])].lower()

    result = False
    for option in options:
        result = modified_prediction == option
        # end early
        if result:
            return result

    return result


################
# PyFunc Wrapper
################


class EricMMLUPyfunc(mlflow.pyfunc.PythonModel):

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

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        question = model_input["question"][0]
        subject = model_input["subject"][0]
        choice_a = model_input["choice_a"][0]
        choice_b = model_input["choice_b"][0]
        choice_c = model_input["choice_c"][0]
        choice_d = model_input["choice_d"][0]

        answer = self.compiled_model.forward(
            question=question,
            subject=subject,
            choice_a=choice_a,
            choice_b=choice_b,
            choice_c=choice_c,
            choice_d=choice_d,
        )

        return pd.DataFrame(
            {
                "answer": [answer["answer"]],
                "rationale": [answer["rationale"]],
            }
        )


## Load MMLU data
def load_data():
    outputs = {}
    for split in ["dev", "validation", "test"]:
        with open(f"mmlu_{split}.jsonl", "r") as f:
            outputs[split] = [
                dspy.Example(json.loads(line)).with_inputs(
                    "subject",
                    "question",
                    "choice_a",
                    "choice_b",
                    "choice_c",
                    "choice_d",
                )
                for line in f
            ]

    return outputs


################
# Main training loop
################
if __name__ == "__main__":
    if ENABLE_ARIZE_TRACING:
        setup_arize_phoenx()

    # Requires having a local MLflow tracking server running
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
    mlflow.set_experiment("mmlu_v2")

    # DSPY setup
    # regular OpenAI
    model = dspy.OpenAI(model="gpt-3.5-turbo")

    dspy.configure(lm=model)

    data = load_data()

    # original model
    original = EricMMLU()

    trainset = data["dev"]  # [:10]
    valset = data["validation"]  # [:10]
    testset = data["test"]  # [:10]

    # eric's run name generator so we can have {run-name}-optimized & {run-name}-unoptimized but still keep mlflow fun names
    run_name = generate_run_name()

    # Set up the evaluator, which can be used multiple times.
    metric = mmlu_metric
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
                python_model=EricMMLUPyfunc(),
                # TODO: Add signature to the model
                # input_example=x_train,
                # signature=signature,
                artifacts=artifacts,
                pip_requirements=["dspy", "cloudpickle"],
            )

            for item in model.named_predictors():
                name = item[0]
                mlflow.log_param(f"signature_{item[0]}", item[1].extended_signature)

            mlflow.log_param("state", model.dump_state())
            mlflow.log_param("testset_N", len(testset))
            mlflow.log_param("trainset_N", len(trainset))
            mlflow.log_param("valset_N", len(valset))

            # Evaluate our program.
            if SHOULD_EVALUATE:
                run_eval_and_log_to_mlflow(evaluator=evaluate, model_to_evaluate=model)
