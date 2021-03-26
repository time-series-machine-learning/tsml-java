package tsml.classifiers.distance_based.optimised;

import evaluation.evaluators.Evaluator;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.distance_based.utils.classifiers.configs.Builder;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.iteration.AbstractSearch;

public interface ParamAgent extends Agent {

    Builder<? extends TSClassifier> getClassifierBuilder();

    AbstractSearch getSearch();

    ParamSpaceBuilder getParamSpaceBuilder();

    Builder<? extends Evaluator> getEvaluatorBuilder();

    void setParamSpaceBuilder(
            final ParamSpaceBuilder paramSpaceBuilder);

    void setSearch(final AbstractSearch search);

    void setClassifierBuilder(
            final Builder<? extends TSClassifier> classifierBuilder);

    void setEvaluatorBuilder(
            final Builder<? extends Evaluator> evaluatorBuilder);
}
