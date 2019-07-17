package classifiers.tuning;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import classifiers.distance_based.elastic_ensemble.iteration.ParameterSetIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RandomIterator;
import classifiers.template.classifier.TemplateClassifier;
import classifiers.template.classifier.TemplateClassifierInterface;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.ContractClassifier;
import utilities.StringUtilities;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class Tuned
    extends TemplateClassifier {
    private AbstractClassifier bestClassifier; // todo make this multiple
    private final TunerConfig config;
    private AbstractIterator<AbstractClassifier> iterator;
    private ClassifierResults bestClassifierBenchmark;
    private final List<AbstractClassifier> bestClassifiers = new ArrayList<>();

    public Tuned(Supplier<AbstractClassifier> supplier, Function<Instances, AbstractIterator<ParameterSet>> parameterSetIteratorGetter) {
        config = new TunerConfig(instances -> new ClassifierIterator(supplier, parameterSetIteratorGetter.apply(instances)));
    }

    public Tuned(Supplier<AbstractClassifier> supplier, AbstractIterator<ParameterSet> parameterSetIterator) {
        config = new TunerConfig(new ClassifierIterator(supplier, parameterSetIterator));
    }

//    public Tuned(Supplier<AbstractClassifier> supplier, ParameterSpace parameterSpace) {
//        config = new TunerConfig(supplier, parameterSpace);
//    }

    public Tuned(Supplier<AbstractClassifier> supplier, Function<Instances, ParameterSpace> parameterSpaceGetter, AbstractIterator<Integer> iterator) {
        this(supplier, instances -> {
            ParameterSpace parameterSpace = parameterSpaceGetter.apply(instances);
            return new ParameterSetIterator(parameterSpace, iterator);
        });
    }

    public Tuned(Supplier<AbstractClassifier> supplier, ParameterSpace parameterSpace, AbstractIterator<Integer> iterator) {
        this(supplier, new ParameterSetIterator(parameterSpace, iterator));
    }

    public Tuned(AbstractIterator<AbstractClassifier> iterator) {
        config = new TunerConfig(iterator);
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

    private final ParameterSet parameterSet = new ParameterSet();

    @Override
    public void setOption(String key, String value) throws Exception {
        parameterSet.addParameter(key, value);
    }

    @Override
    public String[] getOptions() {
        if(bestClassifier == null) {
            return new String[0];
        }
        return bestClassifier.getOptions();
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
            classifier.setOptions(parameterSet.getOptions());
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
