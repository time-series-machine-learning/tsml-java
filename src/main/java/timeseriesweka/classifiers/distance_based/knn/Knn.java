package timeseriesweka.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import net.sourceforge.sizeof.SizeOf;
import timeseriesweka.classifiers.*;
import timeseriesweka.classifiers.distance_based.distance_measures.DistanceMeasure;
import timeseriesweka.classifiers.distance_based.distance_measures.Dtw;
import timeseriesweka.classifiers.distance_based.ee.selection.KBestSelector;
import timeseriesweka.classifiers.distance_based.knn.sampling.DistributedRandomSampler;
import utilities.cache.Cache;
import utilities.cache.DupeCache;
import utilities.*;
import utilities.iteration.AbstractIterator;
import utilities.iteration.linear.LinearIterator;
import utilities.iteration.linear.RoundRobinIterator;
import utilities.iteration.random.RandomIterator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import static experiments.data.DatasetLoading.sampleDataset;
import static timeseriesweka.classifiers.distance_based.distance_measures.DistanceMeasure.DISTANCE_MEASURE_KEY;

public class Knn extends AbstractClassifier implements Options,
                                                       SeedableClassifier, TrainTimeContractable, TestTimeContractable, Copyable, Serializable, TrainAccuracyEstimator,
                                                       Checkpointable {

    private static final String K_KEY = "k";
    private Random trainRandom = new Random();
    private Long testSeed;
    private Random testRandom = new Random();
    private boolean estimateTrainEnabled = true;
    private boolean resetTrainEnabled = true;
    private boolean resetTestEnabled = true;
    private int k = 1;
    private DistanceMeasure distanceMeasure = new Dtw();
    private long trainTimeLimitNanos = -1;
    private long testTimeLimitNanos = -1;
    private Long trainSeed = null;
    private transient String checkpointDirPath;
    private transient long lastCheckpointTimestamp = 0;
    private transient long checkpointIntervalNanos = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    private AbstractIterator<Instance> trainInstanceIterator;
    private AbstractIterator<Instance> trainEstimatorIterator;
    private Instances trainInstances;
    private List<Searcher> trainSearchers;
    private List<Instance> neighbourhood;
    private StopWatch trainTimer = new StopWatch();
    private StopWatch testTimer = new StopWatch();
    private int trainNeighbourhoodSizeLimit = -1;
    private double trainNeighbourhoodSizeLimitPercentage = -1;
    private int trainEstimateSizeLimit = -1;
    private transient String trainResultsPath;
    private ClassifierResults trainResults;
    private Cache<Instance, Instance, Double> distanceCache;
    private boolean distanceCacheEnabled = true;
    public static final String DISTANCE_CACHE_ENABLED_KEY = "dce";
    public static final String TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY = "trnsl";
    public static final String TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY = "trnslp";
    public static final String TRAIN_ESTIMATE_SIZE_LIMIT_KEY = "tresl";
    private boolean earlyAbandonEnabled = true;
    private transient boolean checkpointing = false;
    private transient Logger logger = Logger.getLogger(Knn.class.getCanonicalName());
    private NeighbourIterationStrategy neighbourIterationStrategy = NeighbourIterationStrategy.RANDOM; // todo copy field

    public void setTrainRandom(final Random trainRandom) {
        this.trainRandom = trainRandom;
    }

    public void setTestRandom(final Random testRandom) {
        this.testRandom = testRandom;
    }

    public void setEstimateTrainEnabled(final boolean estimateTrainEnabled) {
        this.estimateTrainEnabled = estimateTrainEnabled;
    }

    public void setDistanceCache(final Cache<Instance, Instance, Double> distanceCache) {
        this.distanceCache = distanceCache;
    }

    public void setLogger(final Logger logger) {
        this.logger = logger;
    }

    public Random getTrainRandom() {
        return trainRandom;
    }

    public Random getTestRandom() {
        return testRandom;
    }

    public boolean isEstimateTrainEnabled() {
        return estimateTrainEnabled;
    }

    public Cache<Instance, Instance, Double> getDistanceCache() {
        return distanceCache;
    }

    public String getCheckpointDirPath() {
        return checkpointDirPath;
    }

    public void setCheckpointDirPath(final String checkpointDirPath) {
        this.checkpointDirPath = checkpointDirPath;
    }

    public long getCheckpointIntervalNanos() {
        return checkpointIntervalNanos;
    }

    public void setCheckpointIntervalNanos(final long checkpointIntervalNanos) {
        this.checkpointIntervalNanos = checkpointIntervalNanos;
    }

    public boolean isCheckpointing() {
        return checkpointing;
    }

    public Logger getLogger() {
        return logger;
    }

    public void setTrainResultsPath(final String trainResultsPath) {
        this.trainResultsPath = trainResultsPath;
    }

    public boolean isEarlyAbandonEnabled() {
        return earlyAbandonEnabled;
    }

    public void setEarlyAbandonEnabled(boolean earlyAbandonEnabled) {
        this.earlyAbandonEnabled = earlyAbandonEnabled;
    }

    public boolean isDistanceCacheEnabled() {
        return distanceCacheEnabled;
    }

    public void setDistanceCacheEnabled(boolean distanceCacheEnabled) {
        this.distanceCacheEnabled = distanceCacheEnabled;
    }

    public Knn() {

    }

    @Override
    public void setTestSeed(final long seed) {
        testSeed = seed;
    }

    @Override
    public Long getTrainSeed() {
        return trainSeed;
    }

    @Override
    public Long getTestSeed() {
        return testSeed;
    }

    @Override
    public void setTrainSeed(final long seed) {
        trainSeed = seed;
    }

    public static void main(String[] args) throws
            Exception {
        int seed = 0;
        String user = "vte14wgu";
        Instances[] dataset = sampleDataset("/home/" + user + "/Projects/datasets/Univariate2018/", "DistalPhalanxTW", seed);
        Instances train = dataset[0];
        Instances test = dataset[1];
//        ParameterSpace parameterSpace = new DtwParameterSpaceBuilder().build(train);
//        for(int i = 0; i < parameterSpace.size(); i++) {
//            ParameterSet parameterSet = parameterSpace.get(i);
//            System.out.println(parameterSet);
//            Knn knn = new Knn();
//            knn.setOptions(parameterSet.getOptions());
//            knn.setFindTrainAccuracyEstimate(true);
//            knn.setTrainSeed(seed);
//            knn.setTestSeed(seed);
////            knn.setTrainNeighbourhoodSizeLimit(10);
////            knn.setTrainEstimateSizeLimit(10);
//            knn.buildClassifier(train);
//            ClassifierResults trainResults = knn.getTrainResults();
//            System.out.println(i + " " + trainResults.getAcc());
//        }
        Knn knn = new Knn();
        Dtw dtw = new Dtw();
        dtw.setWarpingWindow(-1);
//        knn.setDistanceCacheEnabled(false);
//        knn.setEarlyAbandonEnabled(false);
        knn.setDistanceMeasure(dtw);
        knn.setTrainSeed(seed);
        knn.setTestSeed(seed);
        knn.buildClassifier(train);
        ClassifierResults trainResults = knn.getTrainResults();
        System.out.println("train acc: " + trainResults.getAcc());
        System.out.println("-----");
        ClassifierResults testResults = new ClassifierResults();
        for (Instance testInstance : test) {
            long time = System.nanoTime();
            double[] distribution = knn.distributionForInstance(testInstance);
            double prediction = knn.classifyInstance(testInstance);
            time = System.nanoTime() - time;
            testResults.addPrediction(testInstance.classValue(), distribution, prediction, time, null);
        }
        System.out.println(testResults.getAcc());
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        Options.super.setOptions(options);
        distanceMeasure.setOptions(options);
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

    @Override
    public void buildClassifier(final Instances trainInstances) throws
            Exception {
        loadFromCheckpoint();
        setupTrain(trainInstances);
        if(estimateTrainEnabled) {
            boolean hasRemainingTrainNeighbours = hasRemainingTrainNeighbours();
            boolean hasRemainingTrainSearchers = hasRemainingTrainSearchers();
            trainTimer.lap();
            while ((hasRemainingTrainSearchers || hasRemainingTrainNeighbours) && withinTrainTimeLimit()) {
                boolean choice = hasRemainingTrainSearchers;
                if(trainSearchers.size() < trainInstances.numClasses()) { // impose minimum of # classes of train instances to produce train estimate
                    choice = true;
                }
                else if(neighbourhood.size() < trainInstances.numClasses()) { // impose min number of neighbours
                    choice = false;
                }
                else if (hasRemainingTrainNeighbours && hasRemainingTrainSearchers) {
                    choice = trainRandom.nextBoolean();
                }
                if (choice) {
                    nextTrainSearcher();
                } else {
                    nextTrainInstance();
                }
                hasRemainingTrainNeighbours = hasRemainingTrainNeighbours();
                hasRemainingTrainSearchers = hasRemainingTrainSearchers();
                trainTimer.lap();
                checkpoint();
                trainTimer.resetClock();
            }
            buildTrainEstimate();
        }
        checkpoint(true);
    }

    private Iterator<Instance> buildTestNeighbourIterator() {
        RandomIterator<Instance> iterator = new RandomIterator<>(testRandom);
        iterator.addAll(trainInstances);
        return iterator;
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
            Exception {
        setupTest();
        Searcher searcher = new Searcher(testInstance, false);
        Iterator<Instance> iterator = buildTestNeighbourIterator();
        while (withinTestTimeLimit() && iterator.hasNext()) {
            Instance trainInstance = iterator.next();
            iterator.remove();
            searcher.add(trainInstance);
        }
        double[] distribution = searcher.predict();
        testTimer.lap();
        return distribution;
    }

    @Override
    public String[] getOptions() {
        return ArrayUtilities.concat(distanceMeasure.getOptions(), new String[]{
                TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY,
                String.valueOf(trainNeighbourhoodSizeLimitPercentage),
                TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY,
                String.valueOf(trainNeighbourhoodSizeLimit),
                TRAIN_ESTIMATE_SIZE_LIMIT_KEY,
                String.valueOf(trainEstimateSizeLimit),
                DISTANCE_MEASURE_KEY,
                String.valueOf(distanceMeasure),
                K_KEY,
                String.valueOf(k),
                TRAIN_TIME_CONTRACT_KEY,
                String.valueOf(trainTimeLimitNanos),
                TEST_TIME_CONTRACT_KEY,
                String.valueOf(testTimeLimitNanos),
                DISTANCE_CACHE_ENABLED_KEY,
                String.valueOf(distanceCacheEnabled),
        });
    }

    private void setupTrain(Instances trainInstances) {
        trainTimer.resetClock();
        if (resetTrainEnabled) {
            resetTrainEnabled = false;
            trainTimer.resetTime();
            if (trainSeed != null) {
                trainRandom.setSeed(trainSeed);
            } else {
                System.err.println("train seed not set");
            }
            this.trainInstances = trainInstances;
            if(estimateTrainEnabled) {
                neighbourhood = new ArrayList<>();
                if(distanceCacheEnabled) {
                    InstanceTools.indexInstances(trainInstances);
                    if(distanceCache == null) {
                        if(distanceMeasure.isSymmetric() && distanceCache == null) {
                            distanceCache = new DupeCache<>();
                        } else {
                            distanceCache = new Cache<>();
                        }
                    }
                }
                trainSearchers = new ArrayList<>();
                trainInstanceIterator = buildTrainInstanceIterator();
                trainEstimatorIterator = buildTrainEstimatorIterator();
            }
        }
        if(estimateTrainEnabled) {
            if(Checks.isValidPercentage(trainNeighbourhoodSizeLimitPercentage)) {
                trainNeighbourhoodSizeLimit = (int) (trainInstances.size() * trainNeighbourhoodSizeLimitPercentage);
            }
        }
        trainTimer.lap();
    }

    private boolean withinTrainEstimateSizeLimit() {
        return trainEstimateSizeLimit < 0 || trainSearchers.size() < trainEstimateSizeLimit;
    }

    private boolean withinNeighbourhoodSizeLimit() {
        return trainNeighbourhoodSizeLimit < 0 || neighbourhood.size() < trainNeighbourhoodSizeLimit;
    }

    private boolean hasRemainingTrainNeighbours() {
        return trainInstanceIterator.hasNext() && withinNeighbourhoodSizeLimit();
    }

    private boolean hasRemainingTrainSearchers() {
        return trainEstimatorIterator.hasNext() && withinTrainEstimateSizeLimit();
    }

    private boolean withinTrainTimeLimit() {
        return !hasTrainTimeLimit() || trainTimer.getTimeNanos() < trainTimeLimitNanos;
    }

    private void nextTrainSearcher() {
        Instance trainInstance = trainEstimatorIterator.next();
        trainEstimatorIterator.remove();
        Searcher searcher = new Searcher(trainInstance, true);
        searcher.addAll(neighbourhood);
        trainSearchers.add(searcher);
    }

    private void nextTrainInstance() {
        Instance trainInstance = trainInstanceIterator.next();
        trainInstanceIterator.remove();
        for (Searcher trainSearcher : trainSearchers) {
            trainSearcher.add(trainInstance);
        }
        neighbourhood.add(trainInstance);
    }

    private void buildTrainEstimate() throws
            Exception {
        if(estimateTrainEnabled) {
            trainResults = new ClassifierResults();
            for (Searcher searcher : trainSearchers) {
                long time = System.nanoTime();
                double[] distribution = searcher.predict();
                ArrayUtilities.normaliseInPlace(distribution);
                int prediction = ArrayUtilities.bestIndex(Arrays.asList(ArrayUtilities.box(distribution)), trainRandom);
                time = System.nanoTime() - time;
                int index = ((InstanceTools.IndexedInstance) searcher.getTarget()).getIndex();
                String description = String.valueOf(index);
                trainResults.addPrediction(searcher.getTarget().classValue(),
                                           distribution,
                                           prediction,
                                           time,
                                           description);
            }
            trainTimer.lap();
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            trainResults.setBuildTime(trainTimer.getTimeNanos());
            trainResults.setParas(StringUtilities.join(",", getOptions()));
            try {
                trainResults.setMemory(SizeOf.deepSizeOf(this));
            } catch (Exception ignored) {

            }
            if(trainResultsPath != null) {
                trainResults.writeFullResultsToFile(trainResultsPath);
            }
        }
    }

    private void setupTest() {
        testTimer.reset();
        if (resetTestEnabled) {
            resetTestEnabled = false;
            if (testSeed != null) {
                testRandom.setSeed(testSeed);
            } else {
                logger.warning("test seed not set");
            }
        }
    }

    private AbstractIterator<Instance> buildTrainInstanceIterator() {
        AbstractIterator<Instance> result;
        switch (neighbourIterationStrategy) {
            case RANDOM:
                RandomIterator<Instance> iterator = new RandomIterator<>();
                iterator.setSeed(trainRandom.nextLong());
                result = iterator;
                break;
            case LINEAR:
                result = new LinearIterator<>();
                break;
            case ROUND_ROBIN:
                result = new RoundRobinIterator<>();
                break;
//            case DISTRIBUTED:
//                result = new DistributedRandomSampler();
//                break;
//            case REGRESSED:
//
//                break;
            default:
                throw new UnsupportedOperationException();
        }
//        LinearIterator<Instance> iterator = new LinearIterator<>();
        result.addAll(trainInstances);
        return result;
    }

    private AbstractIterator<Instance> buildTrainEstimatorIterator() {
        RandomIterator<Instance> iterator = new RandomIterator<>();
        iterator.setSeed(trainRandom.nextLong());
//        LinearIterator<Instance> iterator = new LinearIterator<>();
        iterator.addAll(trainInstances);
        return iterator;
    }

    public boolean hasTrainTimeLimit() {
        return trainTimeLimitNanos >= 0;
    }

    @Override
    public void setFindTrainAccuracyEstimate(final boolean estimateTrain) {
        this.estimateTrainEnabled = estimateTrain;
    }

    public ClassifierResults getTrainResults() {
        return trainResults;
    }

    public boolean isResetTrainEnabled() {
        return resetTrainEnabled;
    }

    public void setResetTrainEnabled(final boolean resetTrainEnabled) {
        this.resetTrainEnabled = resetTrainEnabled;
    }

    public int getK() {
        return k;
    }

    public void setK(final int k) {
        this.k = k;
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
    }

    public boolean isResetTestEnabled() {
        return resetTestEnabled;
    }

    public void setResetTestEnabled(final boolean resetTestEnabled) {
        this.resetTestEnabled = resetTestEnabled;
    }

    @Override
    public void setOption(String key, String value) {
        switch (key) {
            case DISTANCE_MEASURE_KEY:
                setDistanceMeasure(DistanceMeasure.fromString(value));
                break;
            case K_KEY:
                setK(Integer.parseInt(value));
                break;
            case TRAIN_SEED_KEY:
                setTrainSeed(Long.parseLong(value));
                break;
            case TEST_SEED_KEY:
                setTestSeed(Long.parseLong(value));
                break;
            case TRAIN_TIME_CONTRACT_KEY:
                setTrainTimeLimit(Long.parseLong(value));
                break;
            case TEST_TIME_CONTRACT_KEY:
                setTestTimeLimit(Long.parseLong(value));
                break;
            case DISTANCE_CACHE_ENABLED_KEY:
                setDistanceCacheEnabled(Boolean.parseBoolean(value));
                break;
            case TRAIN_ESTIMATE_SIZE_LIMIT_KEY:
                setTrainEstimateSizeLimit(Integer.parseInt(value));
                break;
            case TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY:
                setTrainNeighbourhoodSizeLimit(Integer.parseInt(value));
                break;
            case TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY:
                setTrainNeighbourhoodSizeLimitPercentage(Double.parseDouble(value));
        }
    }

    @Override
    public void setTestTimeLimit(TimeUnit time, long amount) {
        testTimeLimitNanos = TimeUnit.NANOSECONDS.convert(amount, time);
    }

    @Override
    public void setTrainTimeLimit(TimeUnit time, long amount) {
        trainTimeLimitNanos = TimeUnit.NANOSECONDS.convert(amount, time);
    }

    @Override
    public Knn shallowCopy() throws Exception {
        Knn knn = new Knn();
        knn.shallowCopyFrom(this);
        return knn;
    }

    @Override
    public void shallowCopyFrom(Object object) throws Exception {
        Knn other = (Knn) object;
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
        distanceCache = other.distanceCache;
        distanceCacheEnabled = other.distanceCacheEnabled;
        trainSearchers = other.trainSearchers;
        neighbourhood = other.neighbourhood;
        trainNeighbourhoodSizeLimit = other.trainNeighbourhoodSizeLimit;
        trainEstimateSizeLimit = other.trainEstimateSizeLimit;
        earlyAbandonEnabled = other.earlyAbandonEnabled;
        trainInstanceIterator = other.trainInstanceIterator;
        trainEstimatorIterator = other.trainEstimatorIterator;
        k = other.k;
        distanceMeasure = other.distanceMeasure;
    }

    public int getTrainNeighbourhoodSizeLimit() {
        return trainNeighbourhoodSizeLimit;
    }

    public void setTrainNeighbourhoodSizeLimit(int trainNeighbourhoodSizeLimit) {
        this.trainNeighbourhoodSizeLimit = trainNeighbourhoodSizeLimit;
    }

    private boolean withinTestTimeLimit() {
        return !hasTestTimeLimit() || testTimer.getTimeNanos() < testTimeLimitNanos;
    }

    public boolean hasTestTimeLimit() {
        return testTimeLimitNanos >= 0;
    }

    public int getTrainEstimateSizeLimit() {
        return trainEstimateSizeLimit;
    }

    public void setTrainEstimateSizeLimit(int trainEstimateSizeLimit) {
        this.trainEstimateSizeLimit = trainEstimateSizeLimit;
    }

    public double getTrainNeighbourhoodSizeLimitPercentage() {
        return trainNeighbourhoodSizeLimitPercentage;
    }

    public void setTrainNeighbourhoodSizeLimitPercentage(double trainNeighbourhoodSizeLimitPercentage) {
        this.trainNeighbourhoodSizeLimitPercentage = trainNeighbourhoodSizeLimitPercentage;
    }

    @Override
    public void setSavePath(final String path) {
        checkpointDirPath = path;
    }

    @Override
    public void copyFromSerObject(final Object obj) throws
                                                    Exception {
        shallowCopyFrom(obj);
    }

    @Override
    public void writeTrainEstimatesToFile(String path) {
        trainResultsPath = path;
    }

    public NeighbourIterationStrategy getNeighbourIterationStrategy() {
        return neighbourIterationStrategy;
    }

    public void setNeighbourIterationStrategy(final NeighbourIterationStrategy neighbourIterationStrategy) {
        this.neighbourIterationStrategy = neighbourIterationStrategy;
    }

    private static class Neighbour {
        private final Instance instance;
        private final double distance;

        private Neighbour(final Instance instance, final double distance) {
            this.instance = instance;
            this.distance = distance;
        }

        public double getDistance() {
            return distance;
        }

        public Instance getInstance() {
            return instance;
        }
    }

    private class Searcher {
        private final Instance target;
        private final boolean train;
        private final KBestSelector<Neighbour, Double> selector;

        private Searcher(final Instance target, boolean train) {
            this.target = target;
            selector = new KBestSelector<>((a, b) -> Double.compare(b, a));
            selector.setLimit(k);
            this.train = train;
            selector.setExtractor(Neighbour::getDistance);
        }

        public Instance getTarget() {
            return target;
        }

        public void add(Instance instance, double distance) {
            if (!instance.equals(target)) {
                addUnchecked(instance, distance);
            }
        }

        private void addUnchecked(Instance instance, double distance) {
            Neighbour neighbour = new Neighbour(instance, distance);
            selector.add(neighbour);
        }

        public double[] predict() {
            double[] distribution = new double[target.numClasses()];
            TreeMap<Double, List<Neighbour>> map = selector.getSelectedAsMap(trainRandom);
            int i = 0;
            for (Map.Entry<Double, List<Neighbour>> entry : map.entrySet()) {
                for (Neighbour neighbour : entry.getValue()) {
                    distribution[(int) neighbour.getInstance().classValue()]++;
                    i++;
                }
            }
            assert i == k; // todo what if dataset size < k? probs should set k to same size
            return distribution;
        }

        public void addAll(final List<Instance> instances) {
            for (Instance instance : instances) {
                add(instance);
            }
        }

        private double findDistance(Instance instance){
            Double max = selector.getWorstValue();
            if (max == null || !earlyAbandonEnabled) {
                max = Double.POSITIVE_INFINITY;
            }
            return distanceMeasure.distance(target, instance, max);
        }

        public void add(Instance instance) {
            if (!instance.equals(target)) {
                double distance;
                if(distanceCacheEnabled && train) {
                    if(!(instance instanceof InstanceTools.IndexedInstance)) {
                        logger.warning("instance not hashed / indexed, therefore caching will be unreliable if serialising!");
                    }
                    distance = distanceCache.getAndPut(target, instance, () -> findDistance(instance));
                } else {
                    distance = findDistance(instance);
                }
                addUnchecked(instance, distance);
            }
        }
    }

    @Override
    public String toString() {
        return "KNN";
    }
}
