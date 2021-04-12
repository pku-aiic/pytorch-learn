import numpy as np

from typing import Dict
from typing import List

from ..protocol import MetricsOutputs
from ..protocol import MetricsProtocol
from ..protocol import InferenceOutputs
from ..constants import PREDICTIONS_KEY


class Accuracy(MetricsProtocol):
    def evaluate(self, output: InferenceOutputs) -> MetricsOutputs:
        if output.labels is None:
            raise ValueError("`labels` need to be provided for `Accuracy`")
        logits = output.forward_results[PREDICTIONS_KEY]
        predictions = logits.argmax(1)
        acc = (predictions == output.labels.ravel()).mean().item()
        return MetricsOutputs(acc, {"acc": acc})


class LogLikelihood(MetricsProtocol):
    def evaluate(self, output: InferenceOutputs) -> MetricsOutputs:
        if output.labels is None:
            raise ValueError("`labels` need to be provided for `LogLikelihood`")
        logits = output.forward_results[PREDICTIONS_KEY]
        logits -= logits.max(1, keepdims=True)
        exp = np.exp(logits)
        probabilities = exp / exp.sum(1, keepdims=True)
        selected = probabilities[range(len(logits)), output.labels.ravel()]
        log_likelihood = np.log(selected).mean().item()
        return MetricsOutputs(log_likelihood, {"log_likelihood": log_likelihood})


class MultipleMetrics(MetricsProtocol):
    def __init__(self, metrics: List[MetricsProtocol]):
        self.metrics = metrics

    def evaluate(self, output: InferenceOutputs) -> MetricsOutputs:
        scores: List[float] = []
        metrics_values: Dict[str, float] = {}
        for metric in self.metrics:
            metric_outputs = metric.evaluate(output)
            scores.append(metric_outputs.final_score)
            metrics_values.update(metric_outputs.metric_values)
        return MetricsOutputs(sum(scores) / len(scores), metrics_values)


__all__ = [
    "Accuracy",
    "LogLikelihood",
    "MultipleMetrics",
]
