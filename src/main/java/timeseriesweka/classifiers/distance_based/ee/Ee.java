package timeseriesweka.classifiers.distance_based.ee;

import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.TrainAccuracyEstimator;
import timeseriesweka.classifiers.distance_based.ee.selection.KBestPerTypeSelector;
import timeseriesweka.classifiers.distance_based.ee.selection.Selector;
import timeseriesweka.classifiers.distance_based.knn.Knn;
import utilities.ArrayUtilities;
import utilities.iteration.AbstractIterator;
import utilities.iteration.ClassifierIterator;
import utilities.iteration.ParameterSetIterator;
import utilities.iteration.feedback.AbstractFeedbackIterator;
import utilities.iteration.linear.RoundRobinIterator;
import utilities.iteration.random.RandomIterator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class Ee extends AbstractClassifier {
    private List<Function<Instances, ParameterSpace>> parameterSpaceFunctions = new ArrayList<>();

    private final Random trainRandom = new Random();
    private final Random testRandom = new Random();
    private Instances trainInstances;
    private Long trainSeed;
    private Long testSeed;
    private AbstractIterator<AbstractClassifier> iterator;
    private Selector<Benchmark> selector = new KBestPerTypeSelector<>();
    private List<Benchmark> constituents;
    private final AbstractIterator<AbstractIterator<AbstractClassifier>> sources = new RoundRobinIterator<>(); // todo add classifiers
    private final AbstractIterator<AbstractClassifier> improvements = new RandomIterator<>();// todo seed

    public List<Function<Instances, ParameterSpace>> getParameterSpaceFunctions() {
        return parameterSpaceFunctions;
    }

    public void setParameterSpaceFunctions(List<Function<Instances, ParameterSpace>> parameterSpaceFunctions) {
        this.parameterSpaceFunctions = parameterSpaceFunctions;
    }

    private void setup(Instances trainInstances) {
        this.trainInstances = trainInstances;
        // todo selector random
        List<AbstractIterator<AbstractClassifier>> classifierIterators = new ArrayList<>();
        for(Function<Instances, ParameterSpace> function : parameterSpaceFunctions) {
            ParameterSpace parameterSpace = function.apply(trainInstances);
            parameterSpace.removeDuplicateParameterSets();
            if(parameterSpace.size() > 0) {
                AbstractIterator<Integer> iterator = new RandomIterator<>(trainRandom); // todo set seed
                iterator.addAll(ArrayUtilities.sequence(parameterSpace.size()));
                ParameterSetIterator parameterSetIterator = new ParameterSetIterator(parameterSpace, iterator);
                ClassifierIterator classifierIterator = new ClassifierIterator();
                classifierIterator.setParameterSetIterator(parameterSetIterator);
                classifierIterator.setSupplier(() -> {
                    Knn knn = new Knn();
                    knn.setTrainSize(2);
                    return knn;
                });
                classifierIterators.add(classifierIterator);
            }
        }
        iterator = new AbstractIterator<AbstractClassifier>() {

            @Override
            public void add(AbstractClassifier item) {
                throw new UnsupportedOperationException();
            }

            @Override
            public AbstractFeedbackIterator<AbstractClassifier, Benchmark> iterator() {
                throw new UnsupportedOperationException();
            }

            @Override
            public boolean hasNext() {
                return sources.hasNext() || improvements.hasNext();
            }

            private AbstractClassifier nextSource() {
                AbstractIterator<AbstractClassifier> classifierIterator = sources.next();
                AbstractClassifier classifier = classifierIterator.next();
                if(!classifierIterator.hasNext()) {
                    sources.remove();
                }
                return classifier;
            }

            private AbstractClassifier nextImprovement() {
                return improvements.next();
            }

            @Override
            public AbstractClassifier next() {
                boolean anotherSource = sources.hasNext();
                boolean anotherImprovement = improvements.hasNext();
                boolean choice = anotherSource;
                if(anotherSource && anotherImprovement) {
                    choice = trainRandom.nextBoolean();
                }
                if(choice) {
                    return nextSource();
                } else {
                    return nextImprovement();
                }
            }
        };
    }

    private AbstractClassifier improve(AbstractClassifier classifier) {
        if(classifier instanceof Knn) {
            Knn knn = (Knn) classifier;
            int trainSize = knn.getTrainSize();
            if(trainSize + 1 <= trainInstances.size()) {
                knn.setTrainSize(trainSize);
                return knn;
            } else {
                return null;
            }
        }
        throw new UnsupportedOperationException();
    }

    private void feedback(Benchmark benchmark) {
        AbstractClassifier classifier = benchmark.getClassifier();
        AbstractClassifier improved = improve(classifier);
        if (improved != null) {
            improvements.add(improved);
        }
    }

    @Override
    public void buildClassifier(Instances trainInstances) throws Exception {
        setup(trainInstances);
        while (iterator.hasNext()) {
            AbstractClassifier classifier = iterator.next();
            classifier.buildClassifier(trainInstances);
            ClassifierResults results;
            if(classifier instanceof TrainAccuracyEstimator) {
                results = ((TrainAccuracyEstimator) classifier).getTrainResults();
            } else {
                throw new UnsupportedOperationException();
            }
            Benchmark benchmark = new Benchmark(classifier, results);
            feedback(benchmark);
            selector.add(benchmark);
        }
        constituents = selector.getSelected();
    }

    @Override
    public double[] distributionForInstance(Instance testInstance) throws Exception {
        double[] distribution = new double[testInstance.numClasses()];
        for(Benchmark constituent : constituents) {
            double weight = constituent.getResults().getAcc();
            double[] constituentDistribution = constituent.getClassifier().distributionForInstance(testInstance);
            ArrayUtilities.multiplyInPlace(constituentDistribution, weight);
            ArrayUtilities.addInPlace(distribution, constituentDistribution);
        }
        ArrayUtilities.normaliseInPlace(distribution);
        return distribution;
    }

    @Override
    public double classifyInstance(Instance testInstance) throws Exception {
        double[] distribution = distributionForInstance(testInstance);
        return ArrayUtilities.bestIndex(Arrays.asList(ArrayUtilities.box(distribution)), testRandom);
    }
}
