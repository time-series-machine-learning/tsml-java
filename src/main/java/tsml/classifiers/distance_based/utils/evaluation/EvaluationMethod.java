package tsml.classifiers.distance_based.utils.evaluation;

import evaluation.evaluators.Evaluator;

import java.io.Serializable;

public interface EvaluationMethod extends Serializable {

    Evaluator buildEvaluator();
}
