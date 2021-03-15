package tsml.classifiers.distance_based.optimised;

import evaluation.evaluators.Evaluator;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.distance_based.utils.classifiers.Builder;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.iteration.AbstractSearch;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Randomizable;

import java.util.Objects;

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
