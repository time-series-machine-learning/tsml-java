package classifiers.distance_based.elastic_ensemble;

import classifiers.template.config.TemplateConfig;
import classifiers.tuning.Tuned;
import evaluation.storage.ClassifierResults;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class ElasticEnsembleConfig
    extends TemplateConfig {
    private ConstituentIterationStrategy constituentIterationStrategy = ConstituentIterationStrategy.RANDOM;
    private Function<ClassifierResults, Double> trainResultsMetricGetter = ClassifierResults::getAcc;
    private final List<Tuned> constituents = new ArrayList<>();

    public ElasticEnsembleConfig() {
        super();
    }

    public ElasticEnsembleConfig(final ElasticEnsembleConfig other) throws
                                                                                  Exception {
        super(other);
    }

    @Override
    public ElasticEnsembleConfig copy() throws
                                               Exception {
        return new ElasticEnsembleConfig(this);
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {
        // todo
    }

    @Override
    public String[] getOptions() {
        return null;
    }

    @Override
    public void copyFrom(final Object object) throws
                                              Exception {
        ElasticEnsembleConfig other = (ElasticEnsembleConfig) object; // todo
    }


    public ConstituentIterationStrategy getConstituentIterationStrategy() {
        return constituentIterationStrategy;
    }

    public void setConstituentIterationStrategy(final ConstituentIterationStrategy constituentIterationStrategy) {
        this.constituentIterationStrategy = constituentIterationStrategy;
    }

    public Function<ClassifierResults, Double> getTrainResultsMetricGetter() {
        return trainResultsMetricGetter;
    }

    public void setTrainResultsMetricGetter(final Function<ClassifierResults, Double> trainResultsMetricGetter) {
        this.trainResultsMetricGetter = trainResultsMetricGetter;
    }

    public List<Tuned> getConstituents() {
        return constituents;
    }
}
