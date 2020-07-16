package tsml.classifiers.distance_based.utils.evaluation;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;

public class CrossValidation implements EvaluationMethod {

    public CrossValidation() {

    }

    public CrossValidation(int numFolds) {
        setNumFolds(numFolds);
    }

    private int numFolds = 10;

    @Override public Evaluator buildEvaluator() {
        final CrossValidationEvaluator evaluator = new CrossValidationEvaluator();
        evaluator.setNumFolds(numFolds);
        return evaluator;
    }

    public int getNumFolds() {
        return numFolds;
    }

    public void setNumFolds(final int numFolds) {
        this.numFolds = numFolds;
    }
}
