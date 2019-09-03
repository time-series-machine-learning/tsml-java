package timeseriesweka.classifiers.distance_based.ee;

import evaluation.evaluators.BespokeTrainEstimateEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.*;
import timeseriesweka.classifiers.distance_based.distance_measures.DistanceMeasure;
import timeseriesweka.classifiers.distance_based.distance_measures.Ddtw;
import timeseriesweka.classifiers.distance_based.distance_measures.DistanceMeasureParameterSpaces;
import timeseriesweka.classifiers.distance_based.distance_measures.Wddtw;
import timeseriesweka.classifiers.distance_based.ee.selection.KBestSelector;
import timeseriesweka.classifiers.distance_based.knn.Knn;
import utilities.*;
import utilities.cache.CachedFunction;
import utilities.iteration.AbstractIterator;
import utilities.iteration.ClassifierIterator;
import utilities.iteration.ParameterSetIterator;
import utilities.iteration.random.RandomIterator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.logging.Logger;

import static experiments.data.DatasetLoading.sampleDataset;
import static utilities.GenericTools.indexOfMax;

public class Ee
    extends AbstractClassifier
    implements TrainAccuracyEstimator,
               TrainTimeContractable,
               TestTimeContractable,
               Checkpointable,
               Copyable,
               Serializable,
               SeedableClassifier {

    private Random trainRandom = new Random();
    private Random testRandom = new Random();

    private CachedFunction<Instance, Instance> derivativeCache;

    private List<Function<Instances, ParameterSpace>> parameterSpaceFunctions = new ArrayList<>(Arrays.asList(
        i -> DistanceMeasureParameterSpaces.buildEdParameterSpace(),
        DistanceMeasureParameterSpaces::buildDtwParameterSpace,
        i -> DistanceMeasureParameterSpaces.buildFullDtwParameterSpace(),
        DistanceMeasureParameterSpaces::buildDdtwParameterSpace,
        i -> DistanceMeasureParameterSpaces.buildFullDdtwParameterSpace(),
        i -> DistanceMeasureParameterSpaces.buildWdtwParameterSpace(),
        i -> DistanceMeasureParameterSpaces.buildWddtwParameterSpace(),
        DistanceMeasureParameterSpaces::buildLcssParameterSpace,
        i -> DistanceMeasureParameterSpaces.buildMsmParameterSpace(),
        DistanceMeasureParameterSpaces::buildErpParameterSpace,
        i -> DistanceMeasureParameterSpaces.buildTwedParameterSpace()
                                                                                                             ));
    private Long trainSeed;
    private Long testSeed;
    private AbstractIterator<Member> memberIterator;
    private List<Benchmark> constituents;
    private List<Member> members;
    private boolean estimateTrainEnabled = true;
    private transient String trainResultsPath;
    private ClassifierResults trainResults;
    private int minTrainNeighbourhoodSizeLimit = 2;
    private int trainNeighbourhoodSizeLimit = -1;
    private double trainNeighbourhoodSizeLimitPercentage = -1;
    private transient String offlineBuildClassifierResultsDirPath;
    private transient List<String> offlineBuildClassifierNames;
    private boolean resetTrainEnabled = true;
    private boolean resetTestEnabled = true;
    private Instances trainInstances;
    private transient Logger logger = Logger.getLogger(Ee.class.getCanonicalName());
    private transient String checkpointDirPath;
    private transient boolean checkpointing;
    private transient long lastCheckpointTimestamp = 0;
    private transient long checkpointIntervalNanos = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    private long testTimeLimitNanos = -1;
    private long trainTimeLimitNanos = -1;
    private StopWatch trainTimer = new StopWatch();
    private StopWatch testTimer = new StopWatch();
    private transient boolean offlineBuild = false;

    public void setTrainRandom(final Random trainRandom) {
        this.trainRandom = trainRandom;
    }

    public void setTestRandom(final Random testRandom) {
        this.testRandom = testRandom;
    }

    public void setDerivativeCache(final CachedFunction<Instance, Instance> derivativeCache) {
        this.derivativeCache = derivativeCache;
    }

    public void setEstimateTrainEnabled(final boolean estimateTrainEnabled) {
        this.estimateTrainEnabled = estimateTrainEnabled;
    }

    public void setTrainResultsPath(final String trainResultsPath) {
        this.trainResultsPath = trainResultsPath;
    }

    public void setTrainNeighbourhoodSizeLimit(final int trainNeighbourhoodSizeLimit) {
        this.trainNeighbourhoodSizeLimit = trainNeighbourhoodSizeLimit;
    }

    public void setTrainNeighbourhoodSizeLimitPercentage(final double trainNeighbourhoodSizeLimitPercentage) {
        this.trainNeighbourhoodSizeLimitPercentage = trainNeighbourhoodSizeLimitPercentage;
    }

    public void setLogger(final Logger logger) {
        this.logger = logger;
    }

    public void setCheckpointDirPath(final String checkpointDirPath) {
        this.checkpointDirPath = checkpointDirPath;
    }

    @Override
    public void setCheckpointInterval(final long amount, final TimeUnit unit) {
        checkpointIntervalNanos = TimeUnit.NANOSECONDS.convert(amount, unit);
    }

    public Random getTrainRandom() {
        return trainRandom;
    }

    public Random getTestRandom() {
        return testRandom;
    }

    public CachedFunction<Instance, Instance> getDerivativeCache() {
        return derivativeCache;
    }

    public boolean isEstimateTrainEnabled() {
        return estimateTrainEnabled;
    }

    public String getTrainResultsPath() {
        return trainResultsPath;
    }

    public int getMinTrainNeighbourhoodSizeLimit() {
        return minTrainNeighbourhoodSizeLimit;
    }

    public int getTrainNeighbourhoodSizeLimit() {
        return trainNeighbourhoodSizeLimit;
    }

    public double getTrainNeighbourhoodSizeLimitPercentage() {
        return trainNeighbourhoodSizeLimitPercentage;
    }

    public String getCheckpointDirPath() {
        return checkpointDirPath;
    }

    public boolean isCheckpointing() {
        return checkpointing;
    }

    public long getCheckpointIntervalNanos() {
        return checkpointIntervalNanos;
    }

    private void checkpoint() throws
            IOException {
        checkpoint(false);
    }

    private boolean withinCheckpointInterval() {
        return System.nanoTime() - lastCheckpointTimestamp < checkpointIntervalNanos;
    }

    private void checkpoint(boolean force) throws
                                           IOException {
        if(checkpointing &&
           (
               (hasTrainTimeLimit() && !withinTrainTimeLimit()) ||
               (!hasTrainTimeLimit() && !withinCheckpointInterval()) ||
               force
           )
        ) {
            saveToFile(getCheckpointFilePath());
            lastCheckpointTimestamp = System.nanoTime();
        }
    }

    private String getTrainSeedAsString() {
        return this.trainSeed == null ? "" :
                String.valueOf(this.trainSeed);
    }

    private String getCheckpointFilePath() {
        return checkpointDirPath + "/checkpoint" + getTrainSeedAsString() + ".ser";
    }

    private void loadFromCheckpoint() {
        if(checkpointing) {
            // keep copy of current checkpointing config
            String currentCheckpointDirPath = checkpointDirPath;
            long currentCheckpointIntervalNanos = checkpointIntervalNanos;
            String currentTrainResultsPath = trainResultsPath;
            try {
                // load from checkpoint file, carrying across checkpointing config
                loadFromFile(getCheckpointFilePath());
                // reapply current checkpointing config
                setCheckpointInterval(currentCheckpointIntervalNanos, TimeUnit.NANOSECONDS);
                setCheckpointDirPath(currentCheckpointDirPath);
                setCheckpointing(true);
                setTrainResultsPath(currentTrainResultsPath);
                lastCheckpointTimestamp = System.nanoTime();
            } catch (Exception e) {

            }
        }
    }

    @Override
    public void setCheckpointing(final boolean on) {
        checkpointing = on;
    }

    public Logger getLogger() {
        return logger;
    }

    public String getOfflineBuildClassifierResultsDirPath() {
        return offlineBuildClassifierResultsDirPath;
    }

    public void setOfflineBuildClassifierResultsDirPath(final String offlineBuildClassifierResultsDirPath) {
        this.offlineBuildClassifierResultsDirPath = offlineBuildClassifierResultsDirPath;
    }

    public boolean isResetTrainEnabled() {
        return resetTrainEnabled;
    }

    public void setResetTrainEnabled(final boolean resetTrainEnabled) {
        this.resetTrainEnabled = resetTrainEnabled;
    }

    @Override
    public void setSavePath(final String path) {
        checkpointDirPath = path;
    }

    @Override
    public void copyFromSerObject(final Object obj) throws
                                                    Exception {
        // todo
    }

    public boolean hasTrainTimeLimit() {
        return trainTimeLimitNanos >= 0;
    }

    public boolean hasTestTimeLimit() {
        return testTimeLimitNanos >= 0;
    }

    private boolean withinTestTimeLimit() {
        return hasTestTimeLimit() && testTimer.getTimeNanos() < testTimeLimitNanos;
    }

    private boolean withinTrainTimeLimit() {
        return !hasTrainTimeLimit() || trainTimer.getTimeNanos() < trainTimeLimitNanos;
    }

    @Override
    public void setTestTimeLimit(final TimeUnit time, final long amount) {
        testTimeLimitNanos = TimeUnit.NANOSECONDS.convert(amount, time);
    }

    @Override
    public void setTrainTimeLimit(final TimeUnit time, final long amount) {
        trainTimeLimitNanos = TimeUnit.NANOSECONDS.convert(amount, time);
    }

    public boolean isOfflineBuild() {
        return offlineBuild;
    }

    public void setOfflineBuild(final boolean offlineBuild) {
        this.offlineBuild = offlineBuild;
    }

    public List<String> getOfflineBuildClassifierNames() {
        return offlineBuildClassifierNames;
    }

    public void setOfflineBuildClassifierNames(final List<String> offlineBuildClassifierNames) {
        this.offlineBuildClassifierNames = offlineBuildClassifierNames;
    }

    public class Member {

        private final AbstractIterator<AbstractClassifier> source;
        private final AbstractIterator<AbstractClassifier> improvement;

        public AbstractIterator<AbstractClassifier> getImprovement() {
            return improvement;
        }

        private final AbstractIterator<AbstractClassifier> iterator;
        private final KBestSelector<Benchmark, Double> selector;
        private final Evaluator evaluator;

        public Member(final AbstractIterator<AbstractClassifier> source,
                      final AbstractIterator<AbstractClassifier> improvement,
                      final KBestSelector<Benchmark, Double> selector) {
            this.source = source;
            this.improvement = improvement;
            this.iterator = new AbstractIterator<AbstractClassifier>() {

                private AbstractIterator<AbstractClassifier> previous;

                @Override
                public void add(final AbstractClassifier item) {
                    throw new UnsupportedOperationException();
                }

                @Override
                public AbstractIterator<AbstractClassifier> iterator() {
                    throw new UnsupportedOperationException();
                }

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
                        previous = source;
                        classifier = nextSource();
                    } else {
                        previous = improvement;
                        classifier = nextImprovement();
                    }
                    return classifier;
                }

                @Override
                public void remove() {
                    previous.remove();
                }
            };
            this.selector = selector;
            this.evaluator = new BespokeTrainEstimateEvaluator();
        }

        private AbstractClassifier nextSource() {
            AbstractClassifier classifier = source.next();
            if(classifier instanceof Knn) {
                Knn knn = (Knn) classifier;
                DistanceMeasure distanceMeasure = knn.getDistanceMeasure();
                if(distanceMeasure instanceof Ddtw || distanceMeasure instanceof Wddtw) {
                    if(derivativeCache == null) {
                        derivativeCache = new Ddtw().getDerivativeCache();
                    }
                    if(distanceMeasure instanceof Ddtw) {
                        ((Ddtw) distanceMeasure).setDerivativeCache(derivativeCache);
                    }
                    if(distanceMeasure instanceof Wddtw) {
                        ((Wddtw) distanceMeasure).setDerivativeCache(derivativeCache);
                    }
                }
            }
            return classifier;
        }

        private AbstractClassifier nextImprovement() {
            AbstractClassifier classifier = improvement.next();
            classifier = improveClassifier(classifier);
            return classifier;
        }

        public AbstractIterator<AbstractClassifier> getIterator() {
            return iterator;
        }

        public KBestSelector<Benchmark, Double> getSelector() {
            return selector;
        }

        public Evaluator getEvaluator() {
            return evaluator;
        }
    }

    public static void main(String[] args) throws
                                           Exception {
//        long seed = 0;
//        String username = "goastler";
//        Instances[] dataset = sampleDataset("/home/" + username + "/Projects/datasets/Univariate2018/", "GunPoint", (int) seed);
//        Instances train = dataset[0];
//        Instances test = dataset[1];
//        Ee ee = new Ee();
////        ee.getLogger().setLevel(Level.OFF);
//        ee.setTrainSeed(seed);
//        ee.setTestSeed(seed);
//        ee.setFindTrainAccuracyEstimate(true);
//        ee.buildClassifier(train);
//        ClassifierResults trainResults = ee.getTrainResults();
//        System.out.println("train acc: " + trainResults.getAcc());
//        System.out.println("-----");
//        ClassifierResults testResults = new ClassifierResults();
//        for (Instance testInstance : test) {
//            long time = System.nanoTime();
//            double[] distribution = ee.distributionForInstance(testInstance);
//            double prediction = indexOfMax(distribution);
//            time = System.nanoTime() - time;
//            testResults.addPrediction(testInstance.classValue(), distribution, prediction, time, null);
//        }
//        System.out.println(testResults.getAcc());
        long seed = 0;
        String username = "vte14wgu";
        Instances[] dataset = sampleDataset("/home/" + username + "/Projects/datasets/Univariate2018/", "GunPoint", (int) seed);
        Instances train = dataset[0];
        Instances test = dataset[1];
        List<String> names = Arrays.asList("TUNED_DTW_1NN", "TUNED_DDTW_1NN", "TUNED_WDTW_1NN", "TUNED_WDDTW_1NN", "TUNED_MSM_1NN", "TUNED_LCSS_1NN", "TUNED_ERP_1NN", "TUNED_TWED_1NN", "ED_1NN", "DTW_1NN", "DDTW_1NN");
        names = new ArrayList<>(names);
        for(int i = 0; i < names.size(); i++) {
            String name = names.get(i);
            name += ",trnslp,1.0";
            names.set(i, name);
        }
        Ee ee = new Ee();
        ee.setOfflineBuildClassifierResultsDirPath("/home/vte14wgu/Projects/tsml/results/");
//        ee.getLogger().setLevel(Level.OFF);
        ee.setTrainSeed(seed);
        ee.setTestSeed(seed);
        ee.setOfflineBuild(true);
        ee.setOfflineBuildClassifierNames(names);
        ee.setFindTrainAccuracyEstimate(true);
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

    private void buildTrainEstimate() throws
                                                              Exception {
        trainTimer.lap();
        if(estimateTrainEnabled) {
            trainResults = new ClassifierResults();
            for(int i = 0; i < trainInstances.size(); i++) {
                long time = System.nanoTime();
                double[] distribution = new double[trainInstances.numClasses()];
                for(Benchmark constituent : constituents) {
                    ClassifierResults constituentTrainResults = constituent.getResults();
                    double[] constituentDistribution = constituentTrainResults.getProbabilityDistribution(i);
                    ArrayUtilities.multiplyInPlace(constituentDistribution, constituentTrainResults.getAcc());
                    ArrayUtilities.addInPlace(distribution, constituentDistribution);
                }
                ArrayUtilities.normaliseInPlace(distribution);;
                int prediction = ArrayUtilities.bestIndex(Arrays.asList(ArrayUtilities.box(distribution)), trainRandom);
                time = System.nanoTime() - time;
//                Instance trainInstance = trainInstances.get(i);
                trainResults.addPrediction(
//                    trainInstance.classValue(),
                    constituents.get(0).getResults().getTrueClassValue(i),
                                           distribution,
                                           prediction,
                                           time,
                                           null);
            }
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            // todo mem / build time / etc
            if(trainResultsPath != null) {
                trainResults.writeFullResultsToFile(trainResultsPath);
                trainTimer.resetClock();
            }
            trainTimer.lap();
        }
    }

    private void buildClassifierOnline() throws
                                         Exception {
        loadFromCheckpoint();
        while (memberIterator.hasNext() && withinTrainTimeLimit()) {
            Member member = memberIterator.next();
            AbstractIterator<AbstractClassifier> iterator = member.getIterator();
            AbstractClassifier classifier = iterator.next();
            iterator.remove();
            ClassifierResults trainResults = member.getEvaluator().evaluate(classifier, trainInstances);
            Benchmark benchmark = new Benchmark(classifier, trainResults);
            logger.info(trainResults.getAcc() + " for " + classifier.toString() + " " + StringUtilities.join(", ", classifier.getOptions()));
            member.getSelector().add(benchmark);
            feedback(member, benchmark);
            if(!iterator.hasNext()) {
                memberIterator.remove();
            }
            trainTimer.lap();
            checkpoint();
            trainTimer.resetClock();
        }
    }

    private void buildClassifierOffline() throws
                                          Exception {
        trainTimer.lap();
        String postfix = "/Predictions/" + trainInstances.relationName() + "/";
        for(String name : offlineBuildClassifierNames) {
            File resultsDir = new File(offlineBuildClassifierResultsDirPath, name + postfix);
            File[] files = resultsDir.listFiles(file -> {
                String name1 = file.getName();
                return name1.startsWith("fold" + trainSeed + "_") && file.isFile();
            });
            if(files == null || files.length == 0) {
                // no parameters, therefore should already have train file
                files = new File[] {new File(resultsDir, "trainFold" + trainSeed + ".csv")};
            }
            trainTimer.resetClock();
            KBestSelector<Benchmark, Double> selector = buildSelector();
            Member member = new Member(null, null, selector);
            members.add(member);
            for(File file : files) {
                ClassifierResults trainResults = new ClassifierResults();
                trainResults.loadResultsFromFile(file.getPath());
                AbstractClassifier classifier = buildKnn();
                classifier.setOptions(trainResults.getParas().split(","));
                trainTimer.add(trainResults.getBuildTimeInNanos());
                trainTimer.resetClock();
                selector.add(new Benchmark(classifier, trainResults));
                trainTimer.lap();
            }
        }
        trainTimer.resetClock();
    }

    private void pickConstituents() throws
                                    Exception {
        constituents = new ArrayList<>();
        for(Member member : members) {
            List<Benchmark> selected = member.getSelector().getSelectedAsList();
            Benchmark choice = ArrayUtilities.randomChoice(selected, trainRandom);
            constituents.add(choice);
        }
    }

    private void buildOfflineConstituents() throws
                                            Exception {
        if(offlineBuild) {
            trainTimer.lap();
            for(Benchmark benchmark : constituents) {
                Classifier classifier = benchmark.getClassifier();
                if(classifier instanceof TrainAccuracyEstimator) {
                    ((TrainAccuracyEstimator) classifier).setFindTrainAccuracyEstimate(false);
                    classifier.buildClassifier(trainInstances);
                    ((TrainAccuracyEstimator) classifier).setFindTrainAccuracyEstimate(true);
                } else {
                    classifier.buildClassifier(trainInstances);
                }
            }
            trainTimer.resetClock();
        }
    }

    @Override
    public void buildClassifier(Instances trainInstances) throws
                                                          Exception {
        setup(trainInstances);
        if(offlineBuild) {
            buildClassifierOffline();
        } else {
            buildClassifierOnline();
        }
        pickConstituents();
        buildOfflineConstituents();
        buildTrainEstimate();
        trainTimer.lap();
    }

    private void setup(Instances trainInstances) {
        if(resetTrainEnabled) {
            trainTimer.resetTime();
            trainTimer.resetClock();
            resetTrainEnabled = false;
            if(trainSeed == null) {
                logger.warning("train seed not set");
            }
            if(testSeed == null) {
                logger.warning("test seed not set");
            }
            if(trainInstances.isEmpty()) {
                throw new IllegalArgumentException("train instances empty");
            }
            if(parameterSpaceFunctions.isEmpty()) {
                throw new IllegalStateException("no constituents given");
            }
            if(Checks.isValidPercentage(trainNeighbourhoodSizeLimitPercentage)) {
                trainNeighbourhoodSizeLimit = (int) (trainNeighbourhoodSizeLimitPercentage * trainInstances.size());
            }
            derivativeCache = null;
            this.trainInstances = trainInstances;
            members = new ArrayList<>();
            if(!offlineBuild) {
                memberIterator = new RandomIterator<>(trainRandom);
                for (Function<Instances, ParameterSpace> function : parameterSpaceFunctions) {
                    ParameterSpace parameterSpace = function.apply(trainInstances);
                    parameterSpace.removeDuplicateParameterSets();
                    if (parameterSpace.size() > 0) {
                        AbstractIterator<Integer> iterator = new RandomIterator<>(trainRandom);
                        ParameterSetIterator parameterSetIterator = new ParameterSetIterator(parameterSpace, iterator);
                        ClassifierIterator classifierIterator = new ClassifierIterator();
                        classifierIterator.setParameterSetIterator(parameterSetIterator);
                        classifierIterator.setSupplier(this::buildKnn);
                        Member member = new Member(classifierIterator, new RandomIterator<>(trainRandom), buildSelector());
                        memberIterator.add(member);
                        members.add(member);
                    }
                }
            }
            trainTimer.lap();
        }
    }

    private AbstractClassifier buildKnn() {
        Knn knn = new Knn();
        knn.setTrainNeighbourhoodSizeLimit(minTrainNeighbourhoodSizeLimit);
        if(trainSeed != null) knn.setTrainSeed(trainSeed);
        if(testSeed != null) knn.setTestSeed(testSeed);
        return knn;
    }

    private KBestSelector<Benchmark, Double> buildSelector() {
        KBestSelector<Benchmark, Double> selector = new KBestSelector<>(Double::compare);
        selector.setLimit(1);
        selector.setExtractor(benchmark -> benchmark.getResults().getAcc());
        return selector;
    }

    private void feedback(Member member, Benchmark benchmark) {
        AbstractClassifier classifier = benchmark.getClassifier();
        if (canImproveClassifier(classifier)) {
            member.getImprovement().add(classifier);
        }
    }

    private boolean canImproveClassifier(AbstractClassifier classifier) {
        if (classifier instanceof Knn) {
            Knn knn = (Knn) classifier;
            int trainSize = knn.getTrainNeighbourhoodSizeLimit();
            return (trainSize + 1 <= trainNeighbourhoodSizeLimit || trainSize + 1 < trainInstances.size()) && trainSize >= 0;
        }
        throw new UnsupportedOperationException();
    }

    private AbstractClassifier improveClassifier(AbstractClassifier classifier) {
        if (classifier instanceof Knn) {
            Knn knn = (Knn) classifier;
            int trainSize = knn.getTrainNeighbourhoodSizeLimit();
            knn.setTrainNeighbourhoodSizeLimit(trainSize + 1);
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

    private void setupTest() {
        if(resetTestEnabled) {
            resetTestEnabled = false;
            if(testSeed != null) testRandom.setSeed(testSeed);
        }
    }

    @Override
    public double[] distributionForInstance(Instance testInstance) throws
                                                                   Exception {
        // todo test contract, how best to break up constituents...
        setupTest();
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

    @Override
    public void setTrainSeed(final long seed) {
        trainSeed = seed;
    }

    @Override
    public void setTestSeed(final long seed) {
        testSeed = seed;
    }

    public Long getTrainSeed() {
        return trainSeed;
    }

    public Long getTestSeed() {
        return testSeed;
    }

    @Override
    public Object shallowCopy() throws
                                Exception {
        Ee ee = new Ee();
        ee.shallowCopyFrom(this);
        return ee;
    }

    @Override
    public void shallowCopyFrom(final Object object) throws
                                                     Exception {
        Ee other = (Ee) object;
        // generic fields
        trainRandom = other.trainRandom;
        testRandom = other.testRandom;
        testSeed = other.testSeed;
        trainSeed = other.trainSeed;
        estimateTrainEnabled = other.estimateTrainEnabled;
        resetTrainEnabled = other.resetTrainEnabled;
        resetTestEnabled = other.resetTestEnabled;
        logger = other.logger;
        checkpointing = other.checkpointing;
        checkpointIntervalNanos = other.checkpointIntervalNanos;
        lastCheckpointTimestamp = other.lastCheckpointTimestamp;
        checkpointDirPath = other.checkpointDirPath;
        trainTimeLimitNanos = other.trainTimeLimitNanos;
        testTimeLimitNanos = other.testTimeLimitNanos;
        trainTimer = other.trainTimer;
        testTimer = other.testTimer;
        trainResults = other.trainResults;
        trainInstances = other.trainInstances;
        trainResultsPath = other.trainResultsPath;
        // bespoke fields
        memberIterator = other.memberIterator;
        constituents = other.constituents;
        minTrainNeighbourhoodSizeLimit = other.minTrainNeighbourhoodSizeLimit;
        trainNeighbourhoodSizeLimit = other.trainNeighbourhoodSizeLimit;
        trainNeighbourhoodSizeLimitPercentage = other.trainNeighbourhoodSizeLimitPercentage;
        offlineBuildClassifierResultsDirPath = other.offlineBuildClassifierResultsDirPath;
        offlineBuildClassifierNames = other.offlineBuildClassifierNames;
        offlineBuild = other.offlineBuild;
    }

    @Override
    public void setFindTrainAccuracyEstimate(final boolean estimateTrain) {
        this.estimateTrainEnabled = estimateTrain;
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
