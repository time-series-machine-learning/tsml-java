package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.knn.KnnConfig;
import classifiers.template.configuration.TemplateConfig;
import evaluation.storage.ClassifierResults;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class ElasticEnsembleConfig
    extends TemplateConfig<ElasticEnsembleConfig> {
    private final KnnConfig knnConfiguration = new KnnConfig();
    private boolean removeDuplicateParameterSets = true;
    private final List<ElasticEnsemble.CandidateIterator.Builder> candidateIteratorBuilders = new ArrayList<>();
    private ParameterSpaceIterationStrategy parameterSpaceIterationStrategy = ParameterSpaceIterationStrategy.RANDOM;
    private Function<ClassifierResults, Double> trainResultsMetricGetter = ClassifierResults::getAcc;

    public ElasticEnsembleConfig() {
        super();
    }

    public ElasticEnsembleConfig(final ElasticEnsembleConfig other) throws
                                                                                  Exception {
        super(other);
    }

    public KnnConfig getKnnConfiguration() {
        return knnConfiguration;
    }

    @Override
    public boolean mustResetTrain(final ElasticEnsembleConfig other) {
        return knnConfiguration.mustResetTrain(other.knnConfiguration);
    }

    @Override
    public boolean mustResetTest(final ElasticEnsembleConfig other) {
        return knnConfiguration.mustResetTest(other.knnConfiguration);
    }

    @Override
    public ElasticEnsembleConfig copy() throws
                                               Exception {
        return new ElasticEnsembleConfig(this);
    }

    @Override
    public void copyFrom(final Object object) throws
                                              Exception {
        ElasticEnsembleConfig other = (ElasticEnsembleConfig) object; // todo
        knnConfiguration.copyFrom(other.knnConfiguration);
    }

    @Override
    public void setOption(final String key, final String value) {
        knnConfiguration.setOption(key, value);
    } // todo

    public boolean isRemoveDuplicateParameterSets() {
        return removeDuplicateParameterSets;
    }

    public void setRemoveDuplicateParameterSets(final boolean removeDuplicateParameterSets) {
        this.removeDuplicateParameterSets = removeDuplicateParameterSets;
    }

    public List<ElasticEnsemble.CandidateIterator.Builder> getCandidateIteratorBuilders() {
        return candidateIteratorBuilders;
    }

    public ParameterSpaceIterationStrategy getParameterSpaceIterationStrategy() {
        return parameterSpaceIterationStrategy;
    }

    public void setParameterSpaceIterationStrategy(final ParameterSpaceIterationStrategy parameterSpaceIterationStrategy) {
        this.parameterSpaceIterationStrategy = parameterSpaceIterationStrategy;
    }

    public Function<ClassifierResults, Double> getTrainResultsMetricGetter() {
        return trainResultsMetricGetter;
    }

    public void setTrainResultsMetricGetter(final Function<ClassifierResults, Double> trainResultsMetricGetter) {
        this.trainResultsMetricGetter = trainResultsMetricGetter;
    }
}
