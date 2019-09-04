package timeseriesweka.classifiers.distance_based.ee;

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import net.sourceforge.sizeof.SizeOf;
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
import utilities.iteration.limited.LimitedIterator;
import utilities.iteration.linear.QueueIterator;
import utilities.iteration.random.RandomIterator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;

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
               SeedableClassifier,
               Options {

    private Random trainRandom = new Random();
    private Random testRandom = new Random();

    private CachedFunction<Instance, Instance> derivativeCache;

    public List<Member> getMembers() {
        return members;
    }

    public void setMembers(List<Member> members) {
        this.members = members;
    }

    private final static List<Function<Instances, ParameterSpace>> TRADITIONAL_CONFIG_PARAMETER_SPACE_FUNCTIONS = new ArrayList<>(Arrays.asList(
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
    private List<Member> members = new ArrayList<>();
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
    public static final String PARAMETER_SPACE_SIZE_LIMIT_KEY = "pssl";
    public static final String PARAMETER_SPACE_SIZE_LIMIT_PERCENTAGE_KEY = "psslp";
    private int parameterSpaceSizeLimit = -1;
    private double parameterSpaceSizeLimitPercentage = -1;

    // todo getOptions


    @Override
    public String[] getOptions() {
        return new String[] {
            Knn.TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY,
            String.valueOf(trainNeighbourhoodSizeLimit),
            Knn.TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY,
            String.valueOf(trainNeighbourhoodSizeLimitPercentage),
            PARAMETER_SPACE_SIZE_LIMIT_KEY,
            String.valueOf(parameterSpaceSizeLimit),
            PARAMETER_SPACE_SIZE_LIMIT_PERCENTAGE_KEY,
            String.valueOf(parameterSpaceSizeLimitPercentage)
        };
    }

    @Override
    public void setOption(final String key, final String value) {
        switch (key) {
            case Knn.TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY:
                setTrainNeighbourhoodSizeLimitPercentage(Double.parseDouble(value));
                break;
            case Knn.TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY:
                setTrainNeighbourhoodSizeLimit(Integer.parseInt(value));
                break;
            case PARAMETER_SPACE_SIZE_LIMIT_KEY:
                setParameterSpaceSizeLimit(Integer.parseInt(value));
                break;
            case PARAMETER_SPACE_SIZE_LIMIT_PERCENTAGE_KEY:
                setParameterSpaceSizeLimitPercentage(Double.parseDouble(value));
                break;
        }
    }

    public void setOptions(String[] options) throws
                                              Exception {
        StringUtilities.forEachPair(options, this::setOption);
    }

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

    public int getParameterSpaceSizeLimit() {
        return parameterSpaceSizeLimit;
    }

    public void setParameterSpaceSizeLimit(final int parameterSpaceSizeLimit) {
        this.parameterSpaceSizeLimit = parameterSpaceSizeLimit;
    }

    public double getParameterSpaceSizeLimitPercentage() {
        return parameterSpaceSizeLimitPercentage;
    }

    public void setParameterSpaceSizeLimitPercentage(final double parameterSpaceSizeLimitPercentage) {
        this.parameterSpaceSizeLimitPercentage = parameterSpaceSizeLimitPercentage;
    }

    public class Member {

        private AbstractIterator<AbstractClassifier> source;
        private AbstractIterator<AbstractClassifier> improvement;
        private String offlinePath;
        private Function<Instances, ParameterSpace> parameterSpaceFunction;

        public AbstractIterator<AbstractClassifier> getImprovement() {
            return improvement;
        }

        private AbstractIterator<AbstractClassifier> iterator;
        private KBestSelector<Benchmark, Double> selector;
        private Evaluator evaluator;
        private LimitedIterator<?> limitedIterator;

        public AbstractIterator<AbstractClassifier> getSource() {
            return source;
        }

        public void setSource(AbstractIterator<AbstractClassifier> source) {
            this.source = source;
        }

        public void setImprovement(AbstractIterator<AbstractClassifier> improvement) {
            this.improvement = improvement;
        }

        public void setIterator(AbstractIterator<AbstractClassifier> iterator) {
            this.iterator = iterator;
        }

        public void setSelector(KBestSelector<Benchmark, Double> selector) {
            this.selector = selector;
        }

        public void setEvaluator(Evaluator evaluator) {
            this.evaluator = evaluator;
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

        public String getOfflinePath() {
            return offlinePath;
        }

        public void setOfflinePath(String offlinePath) {
            this.offlinePath = offlinePath;
        }

        public Function<Instances, ParameterSpace> getParameterSpaceFunction() {
            return parameterSpaceFunction;
        }

        public void setParameterSpaceFunction(Function<Instances, ParameterSpace> parameterSpaceFunction) {
            this.parameterSpaceFunction = parameterSpaceFunction;
        }

        public LimitedIterator<?> getLimitedIterator() {
            return limitedIterator;
        }

        public void setLimitedIterator(final LimitedIterator<?> limitedIterator) {
            this.limitedIterator = limitedIterator;
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
        String username = "goastler";
        Instances[] dataset = sampleDataset("/home/" + username + "/Projects/datasets/Univariate2018/", "GunPoint", (int) seed);
        Instances train = dataset[0];
        Instances test = dataset[1];
        List<String> names = Arrays.asList(
                "ED_1NN",
                "TUNED_DTW_1NN",
                "DTW_1NN",
                "TUNED_DDTW_1NN",
                "DDTW_1NN",
                "TUNED_WDTW_1NN",
                "TUNED_WDDTW_1NN",
                "TUNED_LCSS_1NN",
                "TUNED_MSM_1NN",
                "TUNED_ERP_1NN",
                "TUNED_TWED_1NN"
        );
        names = new ArrayList<>(names);
        List<Function<Instances, ParameterSpace>> functions = Arrays.asList(
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
        );
        Ee ee = new Ee();
        String option = "trnslp,1.0";
        ee.setTrainNeighbourhoodSizeLimitPercentage(1.0);
//        ee.setOptions(option.split(",")); // todo
        String resultsPath = "/home/goastler/Projects/tsml/results3/";
        for(int i = 0; i < names.size(); i++) {
            String name = names.get(i);
            name += "," + option;
            name += "/Predictions/" + train.relationName();
            Member member = ee.new Member();
            member.setParameterSpaceFunction(functions.get(i));
            member.setOfflinePath(resultsPath + name);
            ee.getMembers().add(member);
        }
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
            trainResults.setBuildTime(trainTimer.getTimeNanos());
            trainResults.setParas(StringUtilities.join(",", getOptions()));
            trainTimer.lap();
            try {
                trainResults.setMemory(SizeOf.deepSizeOf(this));
            } catch (Exception ignored) {

            }
            if(trainResultsPath != null) {
                trainResults.writeFullResultsToFile(trainResultsPath);
            }
            if(trainResultsPath != null) {
                trainResults.writeFullResultsToFile(trainResultsPath);
                trainTimer.resetClock();
            }
        }
    }

    private void build() throws
                                         Exception {
        loadFromCheckpoint();
        while (memberIterator.hasNext() && withinTrainTimeLimit()) {
            Member member = memberIterator.next();
            AbstractIterator<AbstractClassifier> iterator = member.getIterator();
            AbstractClassifier classifier = iterator.next();
            iterator.remove();
            ClassifierResults trainResults;
            if(offlineBuild) {
                trainTimer.lap();
                trainResults = lookupTrainResults(classifier, member.getOfflinePath());
                trainTimer.add(trainResults.getBuildTime());
                trainTimer.resetClock();
            } else {
                trainResults = member.getEvaluator().evaluate(classifier, trainInstances);
            }
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

    private ClassifierResults lookupTrainResults(AbstractClassifier classifier, String path) throws Exception {
        String[] options = classifier.getOptions();
        if(path == null) {
            throw new IllegalStateException("null path");
        }
        File dir = new File(path);
        File[] files = dir.listFiles(file -> {
            String name1 = file.getName();
            return name1.startsWith("fold" + trainSeed + "_") && file.isFile();
        });
        if(files == null || files.length == 0) {
            // no parameters, therefore should already have train file
            files = new File[] {new File(dir, "trainFold" + trainSeed + ".csv")};
        }
        for(File file : files) {
            ClassifierResults trainResults = new ClassifierResults();
            trainResults.loadResultsFromFile(file.getPath());
//                System.out.println(trainResults.getParas() + " vs " + StringUtilities.join(",", options));
            if(StringUtilities.equalPairs(trainResults.getParas().split(","), options)) {
                return trainResults;
            }
        }
        throw new IllegalStateException("offline train results not found for " + StringUtilities.join(",", options));
    }

    private void pickConstituents() throws
                                    Exception {
        constituents = new ArrayList<>();
        for(Member member : members) {
            List<Benchmark> selected = member.getSelector().getSelectedAsList(trainRandom);
            Benchmark choice = ArrayUtilities.randomChoice(selected, trainRandom);
            constituents.add(choice);
        }
    }

    private void buildOfflineConstituents() throws
                                            Exception {
        trainTimer.lap();
        if(offlineBuild) {
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
        }
        trainTimer.resetClock();
    }

    @Override
    public void buildClassifier(Instances trainInstances) throws
                                                          Exception {
        setupTrain(trainInstances);
        build();
        pickConstituents();
        buildOfflineConstituents();
        buildTrainEstimate();
        trainTimer.lap();
    }

    private void setupTrain(Instances trainInstances) {
        trainTimer.resetClock();
        if(resetTrainEnabled) {
            trainTimer.resetTime();;
        }
        if(Checks.isValidPercentage(trainNeighbourhoodSizeLimitPercentage)) {
            trainNeighbourhoodSizeLimit = (int) (trainNeighbourhoodSizeLimitPercentage * trainInstances.size());
        }
        if(Checks.isValidPercentage(parameterSpaceSizeLimitPercentage)) {
            parameterSpaceSizeLimit = (int) (parameterSpaceSizeLimitPercentage * 100);
        }
        if(resetTrainEnabled) {
            resetTrainEnabled = false;
            if(trainSeed == null) {
                logger.warning("train seed not set");
            } else {
                trainRandom.setSeed(trainSeed);
            }
            if(trainInstances.isEmpty()) {
                throw new IllegalArgumentException("train instances empty");
            }
            if(members.isEmpty()) {
                throw new IllegalStateException("no constituents given");
            }
            derivativeCache = null;
            this.trainInstances = trainInstances;
            memberIterator = new QueueIterator<>();//new RandomIterator<>(trainRandom);
            for (Member member : members) {
                ParameterSpace parameterSpace = member.getParameterSpaceFunction().apply(trainInstances);
                parameterSpace.removeDuplicateParameterSets();
                if (parameterSpace.size() > 0) {
                    AbstractIterator<Integer> iterator = new RandomIterator<>(trainRandom);
                    ParameterSetIterator parameterSetIterator = new ParameterSetIterator(parameterSpace, iterator);
                    LimitedIterator<AbstractClassifier> limitedIterator = new LimitedIterator<>();
                    ClassifierIterator classifierIterator = new ClassifierIterator();
                    limitedIterator.setIterator(classifierIterator);
                    classifierIterator.setParameterSetIterator(parameterSetIterator);
                    classifierIterator.setSupplier(this::buildKnn);
                    member.setSource(limitedIterator);
                    member.setImprovement(new RandomIterator<>(trainRandom));
                    member.setLimitedIterator(limitedIterator);
                    member.setIterator(new AbstractIterator<AbstractClassifier>() {

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
                            return member.getSource().hasNext() || member.getImprovement().hasNext();
                        }

                        @Override
                        public AbstractClassifier next() {
                            boolean anotherSource = member.getSource().hasNext();
                            boolean anotherImprovement = member.getImprovement().hasNext();
                            boolean choice = anotherSource;
                            AbstractClassifier classifier;
                            if (anotherSource && anotherImprovement) {
                                choice = trainRandom.nextBoolean();
                            }
                            if (choice) {
                                previous = member.getSource();
                                classifier = member.nextSource();
                            } else {
                                previous = member.getImprovement();
                                classifier = member.nextImprovement();
                            }
                            return classifier;
                        }

                        @Override
                        public void remove() {
                            previous.remove();
                        }
                    });
                    member.setSelector(buildSelector());
                }
            }
            memberIterator.addAll(members);
        }
        for(Member member : members) {
            member.getLimitedIterator().setLimit(parameterSpaceSizeLimit);
        }
        trainTimer.lap();
    }

    private AbstractClassifier buildKnn() {
        Knn knn = new Knn();
        if(Checks.isValidPercentage(trainNeighbourhoodSizeLimitPercentage) || trainNeighbourhoodSizeLimit >= 0) {
            knn.setTrainNeighbourhoodSizeLimit(trainNeighbourhoodSizeLimit);
            knn.setTrainNeighbourhoodSizeLimitPercentage(trainNeighbourhoodSizeLimitPercentage);
        } else {
            knn.setTrainNeighbourhoodSizeLimit(minTrainNeighbourhoodSizeLimit);
        }
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
            return (trainNeighbourhoodSizeLimit < 0 || trainSize + 1 <= trainNeighbourhoodSizeLimit) &&
                   (trainSize + 1 < trainInstances.size() && trainSize >= 0);
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
