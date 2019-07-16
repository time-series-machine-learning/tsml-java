package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.knn.KnnConfig;
import classifiers.template.configuration.TemplateConfig;
import evaluation.storage.ClassifierResults;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

public class ElasticEnsembleConfig
    extends TemplateConfig {
    private final KnnConfig knnConfiguration = new KnnConfig();
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

    public static Comparator<ElasticEnsembleConfig> TRAIN_CONFIG_COMPARATOR =
        (config, other) -> KnnConfig.TRAIN_CONFIG_COMPARATOR.compare(config.knnConfiguration, other.knnConfiguration);

    public static Comparator<ElasticEnsembleConfig> TEST_CONFIG_COMPARATOR =
        (config, other) -> KnnConfig.TEST_CONFIG_COMPARATOR.compare(config.knnConfiguration, other.knnConfiguration);


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
