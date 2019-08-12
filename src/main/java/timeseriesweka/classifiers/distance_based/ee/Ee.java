package timeseriesweka.classifiers.distance_based.ee;

import evaluation.evaluators.BespokeTrainEstimateEvaluator;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.Seedable;
import timeseriesweka.classifiers.TrainAccuracyEstimator;
import timeseriesweka.classifiers.distance_based.distances.DistanceMeasure;
import timeseriesweka.classifiers.distance_based.distances.ddtw.DdtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.ddtw.FullDdtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.dtw.Dtw;
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
import timeseriesweka.classifiers.distance_based.ee.selection.KBestSelector;
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

import java.util.*;
import java.util.function.Function;

import static experiments.data.DatasetLoading.sampleDataset;
import static utilities.GenericTools.indexOfMax;

public class Ee
    extends AbstractClassifier
    implements TrainAccuracyEstimator {

    private final Random trainRandom = new Random();
    private final Random testRandom = new Random();
    // todo add classifiers
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
    private Long trainSeed;
    private Long testSeed;
    private AbstractIterator<Member> memberIterator;
    private List<Benchmark> constituents;
    private List<Member> members;
    private boolean estimateTrain = true;
    private String trainResultsPath;
    private ClassifierResults trainResults;
    private int trainInstancesSize;

    public class Member {

        private final AbstractIterator<AbstractClassifier> source;

        public AbstractIterator<AbstractClassifier> getImprovement() {
            return improvement;
        }

        private final AbstractIterator<AbstractClassifier> improvement;
        private final IncrementalTuner tuner;

        public Member(final AbstractIterator<AbstractClassifier> source,
                      final AbstractIterator<AbstractClassifier> improvement,
                      final KBestSelector<Benchmark, Double> selector,
                      final Instances trainInstances) {
            this.source = source;
            this.improvement = improvement;
            tuner = new IncrementalTuner();
            tuner.setIterator(new Iterator<AbstractClassifier>() {
                @Override
                public boolean hasNext() {
                    return source.hasNext() || improvement.hasNext();
                }

                @Override
                public AbstractClassifier next() {
                    boolean anotherSource = source.hasNext();
                    boolean anotherImprovement = improvement.hasNext();
                    boolean choice = anotherSource;
                    AbstractClassifier classifier;
                    if (anotherSource && anotherImprovement) {
                        choice = trainRandom.nextBoolean();
                    }
                    if (choice) {
                        classifier = nextSource();
                    } else {
                        classifier = nextImprovement();
                    }
                    if(classifier instanceof Seedable) {
                        ((Seedable) classifier).setTrainSeed(trainSeed);
                        ((Seedable) classifier).setTestSeed(testSeed);
                    }
                    return classifier;
                }
            });
            tuner.setSelector(selector);
            tuner.setEvaluator(new BespokeTrainEstimateEvaluator());
            tuner.setInstances(trainInstances);
        }

        private AbstractClassifier nextSource() {
            AbstractClassifier classifier = source.next();
            source.remove();
            return classifier;
        }

        private AbstractClassifier nextImprovement() {
            AbstractClassifier classifier = improvement.next();
            improvement.remove();
            classifier = improve(classifier);
            return classifier;
        }

        public IncrementalTuner getTuner() {
            return tuner;
        }
    }

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
        while (memberIterator.hasNext()) {
            Member member = memberIterator.next();
            IncrementalTuner tuner = member.getTuner();
            Benchmark benchmark = tuner.next();
            tuner.remove();
            if(!tuner.hasNext()) {
                memberIterator.remove();
            }
            feedback(member, benchmark);
        }
        constituents = new ArrayList<>();
        for(Member member : members) {
            List<Benchmark> selected = member.getTuner().getSelector().getSelectedAsList();
            if(selected.size() > 1) {
                // todo rand trim
            }
            constituents.addAll(selected);
        }
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
        // todo is train seed set / test seed
        if(trainInstances.isEmpty()) {
            throw new IllegalArgumentException("train instances empty");
        }
        if(parameterSpaceFunctions.isEmpty()) {
            throw new IllegalStateException("no constituents given");
        }
        trainInstancesSize = trainInstances.size();
        members = new ArrayList<>();
        memberIterator = new RandomIterator<>(trainRandom);
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
                KBestSelector<Benchmark, Double> selector = new KBestSelector<>((aDouble, t1) -> Double.compare(t1, aDouble));
                selector.setLimit(1);
                selector.setExtractor(benchmark -> benchmark.getResults().getAcc());
                Member member = new Member(classifierIterator, new RandomIterator<>(trainRandom), selector, trainInstances);
                memberIterator.add(member);
                members.add(member);
            }
        }
    }

    private void feedback(Member member, Benchmark benchmark) {
        AbstractClassifier classifier = benchmark.getClassifier();
        if (canImprove(classifier)) {
            member.getImprovement().add(classifier);
        }
    }

    private boolean canImprove(AbstractClassifier classifier) {
        if (classifier instanceof Knn) {
            Knn knn = (Knn) classifier;
            int trainSize = knn.getTrainSize();
            return trainSize + 1 <= trainInstancesSize;
        }
        throw new UnsupportedOperationException();
    }

    private AbstractClassifier improve(AbstractClassifier classifier) {
        if (classifier instanceof Knn) {
            Knn knn = (Knn) classifier;
            try {
                knn = knn.shallowCopy();
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
            int trainSize = knn.getTrainSize();
            knn.setTrainSize(trainSize + 1);
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
