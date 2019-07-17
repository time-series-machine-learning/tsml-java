package classifiers.tuning;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import classifiers.distance_based.elastic_ensemble.iteration.ParameterSetIterator;
import classifiers.template.classifier.TemplateClassifier;
import classifiers.template.classifier.TemplateClassifierInterface;
import classifiers.template.config.TemplateConfig;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.ContractClassifier;
import utilities.ArrayUtilities;
import utilities.StringUtilities;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

public class Tuned
    extends TemplateClassifier {
    private AbstractClassifier bestClassifier; // todo make this multiple
    private final TunedConfig config = new TunedConfig();
    private TemplateConfig classifierConfig = null;
    private AbstractIterator<AbstractClassifier> iterator;
    private ClassifierResults bestClassifierBenchmark;
    private final List<AbstractClassifier> bestClassifiers = new ArrayList<>();

    public TemplateConfig getClassifierConfig() {
        return classifierConfig;
    }

    public void setClassifierConfig(TemplateConfig classifierConfig) {
        this.classifierConfig = classifierConfig;
    }

    public Tuned() {}

    public Tuned(AbstractIterator<AbstractClassifier> iterator) {
        throw new UnsupportedOperationException();
    }

    public Tuned(Tuned other) {
        throw new UnsupportedOperationException(); // todo
    }

    @Override
    public Tuned copy() throws
                        Exception {
        throw new UnsupportedOperationException();
    }

    private ClassifierResults evalClassifier(AbstractClassifier classifier, Instances trainSet) throws Exception {
        System.out.println(StringUtilities.join(", ", classifier.getOptions()));
        ClassifierResults benchmarkResults;
        if(classifier instanceof TemplateClassifier) {
            classifier.buildClassifier(trainSet);
            benchmarkResults = ((TemplateClassifier) classifier).getTrainResults();
        } else {
            throw new UnsupportedOperationException();
        }
        return benchmarkResults;
    }

    @Override
    public void setOption(String key, String value) throws Exception {
        if(classifierConfig != null) {
            classifierConfig.setOption(key, value);
        }
        config.setOption(key, value);
    }

    @Override
    public String[] getOptions() {
        if(classifierConfig != null) {
            return ArrayUtilities.concat(config.getOptions(), classifierConfig.getOptions());
        } else {
            return config.getOptions();
        }
    }

    private void setup(Instances trainSet) {
        if(super.trainSetChanged(trainSet)) {
            getTrainStopWatch().reset();
            config.buildClassifierIterator(trainSet);
            iterator = config.getClassifierIterator();
            getTrainStopWatch().lap();
        }
    }

    @Override
    public void buildClassifier(Instances trainSet) throws Exception {
        setup(trainSet);
        while (iterator.hasNext() && withinTrainContract()) {
            AbstractClassifier classifier = iterator.next();
            iterator.remove();
            if(classifier instanceof ContractClassifier) {
                ((ContractClassifier) classifier).setTimeLimit(remainingTrainContractNanos());
            }
            classifier.setOptions(classifierConfig.getOptions());
            if(classifier instanceof TemplateClassifierInterface) {
                ((TemplateClassifierInterface) classifier).setTrainSeed(getTrainSeed());
            }
            ClassifierResults classifierBenchmark = evalClassifier(classifier, trainSet);
            int comparison = 1;
            if(bestClassifierBenchmark != null) {
                comparison = config.getComparator().compare(bestClassifierBenchmark, classifierBenchmark);
            }
            if(bestClassifiers.isEmpty() || comparison >= 0) {
                if(comparison > 0) {
                    bestClassifierBenchmark = classifierBenchmark;
                    bestClassifiers.clear();
                }
                bestClassifiers.add(classifier);
            }
            getTrainStopWatch().lap();
        }
        bestClassifier = bestClassifiers.get(getTrainRandom().nextInt(bestClassifiers.size()));
        setClassifierResultsMetaInfo(bestClassifierBenchmark);
        setTrainResults(bestClassifierBenchmark);
    }

    @Override
    public double[] distributionForInstance(Instance testInstance) throws Exception {
        return bestClassifier.distributionForInstance(testInstance);
    }
}
