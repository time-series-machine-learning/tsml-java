package tsml.classifiers.distance_based.optimised;

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.distance_based.utils.classifiers.Builder;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.iteration.AbstractSearch;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Randomizable;

import java.util.List;
import java.util.Objects;

public class BaseParamAgent extends BaseAgent implements ParamAgent {
    
    public BaseParamAgent() {
        
    }

    private ParamSpaceBuilder paramSpaceBuilder;
    private AbstractSearch search;
    private Builder<? extends TSClassifier> classifierBuilder;
    private Builder<? extends Evaluator> evaluatorBuilder;
    private ResultsScorer scorer = ClassifierResults::getAcc; 

    @Override public void buildAgent(final TimeSeriesInstances trainData) {
        super.buildAgent(trainData);
        Objects.requireNonNull(paramSpaceBuilder);
        Objects.requireNonNull(search);
        Objects.requireNonNull(classifierBuilder);
        Objects.requireNonNull(evaluatorBuilder);
        final ParamSpace paramSpace = paramSpaceBuilder.build(trainData);
        copyRandomTo(search);
        search.buildSearch(paramSpace);
    }

    @Override public boolean hasNextExplore() {
        return search.hasNext();
    }

    @Override protected Evaluation nextExplore() {
        final ParamSet paramSet = search.next();
        final TSClassifier classifier = classifierBuilder.build();
        copySeedTo(classifier);
        ParamHandlerUtils.setParams(classifier, paramSet);
        final Evaluator evaluator = evaluatorBuilder.build();
        copySeedTo(evaluator);
        final Evaluation evaluation = buildEvaluation();
        evaluation.setClassifier(classifier);
        evaluation.setEvaluator(evaluator);
        evaluation.setScorer(scorer);
        return evaluation;
    }

    @Override public Builder<? extends TSClassifier> getClassifierBuilder() {
        return classifierBuilder;
    }

    @Override public AbstractSearch getSearch() {
        return search;
    }

    @Override public ParamSpaceBuilder getParamSpaceBuilder() {
        return paramSpaceBuilder;
    }

    @Override public Builder<? extends Evaluator> getEvaluatorBuilder() {
        return evaluatorBuilder;
    }

    @Override public void setParamSpaceBuilder(
            final ParamSpaceBuilder paramSpaceBuilder) {
        this.paramSpaceBuilder = paramSpaceBuilder;
    }

    @Override public void setSearch(final AbstractSearch search) {
        this.search = search;
    }

    @Override public void setClassifierBuilder(
            final Builder<? extends TSClassifier> classifierBuilder) {
        this.classifierBuilder = classifierBuilder;
    }

    @Override public void setEvaluatorBuilder(
            final Builder<? extends Evaluator> evaluatorBuilder) {
        this.evaluatorBuilder = evaluatorBuilder;
    }

    public ResultsScorer getScorer() {
        return scorer;
    }

    public void setScorer(final ResultsScorer scorer) {
        this.scorer = scorer;
    }
}
