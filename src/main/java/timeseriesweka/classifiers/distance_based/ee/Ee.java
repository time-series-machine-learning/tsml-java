package timeseriesweka.classifiers.distance_based.ee;

import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import evaluation.tuning.ParameterSpaceBuilder;
import timeseriesweka.classifiers.Seedable;
import timeseriesweka.classifiers.TrainAccuracyEstimator;
import timeseriesweka.classifiers.distance_based.distances.ddtw.DdtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.ddtw.FullDdtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.dtw.DtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.dtw.EdParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.dtw.FullDtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.erp.ErpParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.lcss.LcssParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.msm.MsmParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.twed.TwedParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.wddtw.WddtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.wdtw.WdtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.ee.selection.KBestPerTypeSelector;
import timeseriesweka.classifiers.distance_based.ee.selection.Selector;
import timeseriesweka.classifiers.distance_based.knn.Knn;
import utilities.ArrayUtilities;
import utilities.StringUtilities;
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

import static experiments.data.DatasetLoading.sampleDataset;
import static utilities.GenericTools.indexOfMax;

public class Ee
    extends AbstractClassifier
    implements TrainAccuracyEstimator {

    private final Random trainRandom = new Random();
    private final Random testRandom = new Random();
    private final AbstractIterator<AbstractIterator<AbstractClassifier>> sources = new RoundRobinIterator<>(); //
    // todo add classifiers
    private final AbstractIterator<AbstractClassifier> improvements = new RandomIterator<>();// todo seed
    private List<Function<Instances, ParameterSpace>> parameterSpaceFunctions = new ArrayList<>(Arrays.asList(
        i -> new EdParameterSpaceBuilder().build(),
        i -> new DtwParameterSpaceBuilder().build(i),
        i -> new FullDtwParameterSpaceBuilder().build(),
        i -> new DdtwParameterSpaceBuilder().build(i),
        i -> new FullDdtwParameterSpaceBuilder().build(),
        i -> new WdtwParameterSpaceBuilder().build(),
        i -> new WddtwParameterSpaceBuilder().build(),
        i -> new LcssParameterSpaceBuilder().build(i),
        i -> new MsmParameterSpaceBuilder().build(),
        i -> new ErpParameterSpaceBuilder().build(i),
        i -> new TwedParameterSpaceBuilder().build()
                                                                                                             ));
    private Instances trainInstances;
    private Long trainSeed;
    private Long testSeed;
    private AbstractIterator<AbstractClassifier> iterator;
    private Selector<Benchmark> selector = new KBestPerTypeSelector<>(benchmark -> benchmark.getResults().getAcc());
    private List<Benchmark> constituents;
    private boolean estimateTrain = true;
    private String trainResultsPath;
    private ClassifierResults trainResults;

    public static void main(String[] args) throws
                                           Exception {
        long seed = 0;
        Instances[] dataset = sampleDataset("/home/vte14wgu/Projects/datasets/Univariate2018/", "GunPoint", (int) seed);
        Instances train = dataset[0];
        Instances test = dataset[1];
        Ee ee = new Ee();
        ee.setTrainSeed(seed);
        ee.setTestSeed(seed);
        ee.buildClassifier(train);
        ClassifierResults trainResults = ee.getTrainResults();
        System.out.println("train acc: " + trainResults.getAcc());
        System.out.println("-----");
        ClassifierResults testResults = new ClassifierResults();
        for (Instance testInstance : test) {
            long time = System.nanoTime();
            double[] distribution = ee.distributionForInstance(testInstance);
            double prediction = indexOfMax(distribution);
            time = System.nanoTime() - time;
            testResults.addPrediction(testInstance.classValue(), distribution, prediction, time, null);
        }
        System.out.println(testResults.getAcc());
    }

    @Override
    public void buildClassifier(Instances trainInstances) throws
                                                          Exception {
        setup(trainInstances);
        while (iterator.hasNext()) {
            AbstractClassifier classifier = iterator.next();
            ClassifierResults results;
            if (classifier instanceof TrainAccuracyEstimator) {
                ((TrainAccuracyEstimator) classifier).setFindTrainAccuracyEstimate(true);
                if(classifier instanceof Seedable) {
                    ((Seedable) classifier).setTrainSeed(trainSeed);
                    ((Seedable) classifier).setTestSeed(testSeed);
                }
                classifier.buildClassifier(trainInstances);
                results = ((TrainAccuracyEstimator) classifier).getTrainResults();
            } else {
                throw new UnsupportedOperationException();
            }
            Benchmark benchmark = new Benchmark(classifier, results);
            feedback(benchmark);
            selector.add(benchmark);
        }
        constituents = selector.getSelected();
        if(estimateTrain) {
            trainResults = new ClassifierResults();
            for(int i = 0; i < trainInstances.size(); i++) {
                double[] distribution = new double[trainInstances.numClasses()];
                for(Benchmark constituent : constituents) {
                    ClassifierResults constituentTrainResults = constituent.getResults();
                    double[] constituentDistribution = constituentTrainResults.getProbabilityDistribution(i);
                    ArrayUtilities.multiplyInPlace(constituentDistribution, constituentTrainResults.getAcc());
                    ArrayUtilities.addInPlace(distribution, constituentDistribution);
                }
            }
            if(trainResultsPath != null) {
                trainResults.writeFullResultsToFile(trainResultsPath);
            }
        }
    }

    private void setup(Instances trainInstances) {
        if(trainInstances.isEmpty()) {
            throw new IllegalArgumentException("train instances empty");
        }
        if(parameterSpaceFunctions.isEmpty()) {
            throw new IllegalStateException("no constituents given");
        }
        this.trainInstances = trainInstances;
        // todo selector random
        for (Function<Instances, ParameterSpace> function : parameterSpaceFunctions) {
            ParameterSpace parameterSpace = function.apply(trainInstances);
            parameterSpace.removeDuplicateParameterSets();
            if (parameterSpace.size() > 0) {
                AbstractIterator<Integer> iterator = new RandomIterator<>(trainRandom);
                ParameterSetIterator parameterSetIterator = new ParameterSetIterator(parameterSpace, iterator);
                ClassifierIterator classifierIterator = new ClassifierIterator();
                classifierIterator.setParameterSetIterator(parameterSetIterator);
                classifierIterator.setSupplier(() -> {
                    Knn knn = new Knn();
                    knn.setTrainSize(2);
                    return knn;
                });
                sources.add(classifierIterator);
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

            @Override
            public AbstractClassifier next() {
                boolean anotherSource = sources.hasNext();
                boolean anotherImprovement = improvements.hasNext();
                boolean choice = anotherSource;
                if (anotherSource && anotherImprovement) {
                    choice = trainRandom.nextBoolean();
                }
                if (choice) {
                    return nextSource();
                } else {
                    return nextImprovement();
                }
            }

            private AbstractClassifier nextSource() {
                AbstractIterator<AbstractClassifier> classifierIterator = sources.next();
                AbstractClassifier classifier = classifierIterator.next();
                classifierIterator.remove();
                if (!classifierIterator.hasNext()) {
                    sources.remove();
                }
                return classifier;
            }

            private AbstractClassifier nextImprovement() {
                AbstractClassifier classifier = improvements.next();
                improvements.remove();
                classifier = improve(classifier);
                return classifier;
            }
        };
    }

    private void feedback(Benchmark benchmark) {
        AbstractClassifier classifier = benchmark.getClassifier();
        if (canImprove(classifier)) {
            improvements.add(classifier);
        }
    }

    private boolean canImprove(AbstractClassifier classifier) {
        if (classifier instanceof Knn) {
            Knn knn = (Knn) classifier;
            int trainSize = knn.getTrainSize();
            return trainSize + 1 <= trainInstances.size();
        }
        throw new UnsupportedOperationException();
    }

    private AbstractClassifier improve(AbstractClassifier classifier) {
        if (classifier instanceof Knn) {
            Knn knn = (Knn) classifier;
            int trainSize = knn.getTrainSize();
            knn.setTrainSize(trainSize);
            return knn;
        }
        throw new UnsupportedOperationException();
    }

    public List<Function<Instances, ParameterSpace>> getParameterSpaceFunctions() {
        return parameterSpaceFunctions;
    }

    public void setParameterSpaceFunctions(List<Function<Instances, ParameterSpace>> parameterSpaceFunctions) {
        this.parameterSpaceFunctions = parameterSpaceFunctions;
    }

    @Override
    public double classifyInstance(Instance testInstance) throws
                                                          Exception {
        double[] distribution = distributionForInstance(testInstance);
        return ArrayUtilities.bestIndex(Arrays.asList(ArrayUtilities.box(distribution)), testRandom);
    }

    @Override
    public double[] distributionForInstance(Instance testInstance) throws
                                                                   Exception {
        double[] distribution = new double[testInstance.numClasses()];
        for (Benchmark constituent : constituents) {
            double weight = constituent.getResults()
                                       .getAcc();
            double[] constituentDistribution = constituent.getClassifier()
                                                          .distributionForInstance(testInstance);
            ArrayUtilities.multiplyInPlace(constituentDistribution, weight);
            ArrayUtilities.addInPlace(distribution, constituentDistribution);
        }
        ArrayUtilities.normaliseInPlace(distribution);
        return distribution;
    }

    public Long getTrainSeed() {
        return trainSeed;
    }

    public void setTrainSeed(final Long trainSeed) {
        this.trainSeed = trainSeed;
    }

    public Long getTestSeed() {
        return testSeed;
    }

    public void setTestSeed(final Long testSeed) {
        this.testSeed = testSeed;
    }

    @Override
    public void setFindTrainAccuracyEstimate(final boolean estimateTrain) {
        this.estimateTrain = estimateTrain;
    }

    @Override
    public void writeTrainEstimatesToFile(final String path) {
        trainResultsPath = path;
    }

    @Override
    public ClassifierResults getTrainResults() {
        return trainResults;
    }

    @Override
    public String getParameters() {
        return StringUtilities.join(",", getOptions());
    }
}
