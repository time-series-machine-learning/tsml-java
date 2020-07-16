package tsml.classifiers.distance_based.utils.evaluation;

import evaluation.evaluators.Evaluator;
import evaluation.evaluators.OutOfBagEvaluator;

public class OutOfBag implements EvaluationMethod {
    @Override public Evaluator buildEvaluator() {
        final OutOfBagEvaluator evaluator = new OutOfBagEvaluator();
        return evaluator;
    }
}
