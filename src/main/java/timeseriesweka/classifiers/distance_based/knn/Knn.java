package timeseriesweka.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.Seedable;
import timeseriesweka.classifiers.TestTimeContractable;
import timeseriesweka.classifiers.TrainAccuracyEstimator;
import timeseriesweka.classifiers.TrainTimeContractable;
import timeseriesweka.classifiers.distance_based.distances.DistanceMeasure;
import timeseriesweka.classifiers.distance_based.distances.ddtw.DdtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.dtw.Dtw;
import timeseriesweka.classifiers.distance_based.distances.dtw.DtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.ee.selection.KBestSelector;
import timeseriesweka.filters.cache.Cache;
import timeseriesweka.filters.cache.DupeCache;
import utilities.*;
import utilities.iteration.AbstractIterator;
import utilities.iteration.linear.LinearIterator;
import utilities.iteration.random.RandomIterator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import static experiments.data.DatasetLoading.sampleDataset;
import static timeseriesweka.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_KEY;

public class Knn extends AbstractClassifier implements Options, Seedable, TrainTimeContractable, TestTimeContractable, Copyable, Serializable,
                                                       TrainAccuracyEstimator {

    private static final String K_KEY = "k";
    private Random trainRandom = new Random();
    private Long testSeed;
    private Random testRandom = new Random();
    private boolean trainEstimateEnabled = true;
    private String trainResultsPath;
    private boolean resetTrainEnabled = true;
    private boolean resetTestEnabled = true;
    private int k = 1;
    private DistanceMeasure distanceMeasure = new Dtw();
    private long trainTimeLimitNanos = -1;
    private long testTimeLimitNanos = -1;
    private Long trainSeed = null;
    private AbstractIterator<Instance> trainInstanceIterator;
    private AbstractIterator<Instance> trainEstimatorIterator;
    private Instances trainInstances;
    private List<Searcher> trainSearchers;
    private List<Instance> neighbourhood;
    private StopWatch trainTimer = new StopWatch();
    private StopWatch testTimer = new StopWatch();
    private int neighbourhoodSizeLimit = -1;
    private double neighbourhoodSizeLimitPercentage = -1;
    private int trainEstimateSizeLimit = -1;
    private ClassifierResults trainResults;
    private Cache<Instance, Instance, Double> distanceCache;
    private boolean distanceCacheEnabled = true;
    private static final String NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY = "neighbourhoodSizeLimitPercentage";
    private static final String DISTANCE_CACHE_ENABLED_KEY = "distanceCacheEnabled";
    private static final String NEIGHBOURHOOD_SIZE_LIMIT_KEY = "neighbourhoodSizeLimit";
    private static final String TRAIN_ESTIMATE_SIZE_LIMIT_KEY = "trainEstimateSizeLimit";
    private boolean earlyAbandonEnabled = true;
    private final Logger logger = Logger.getLogger(Knn.class.getCanonicalName());

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
        String user = "goastler";
        Instances[] dataset = sampleDataset("/home/" + user + "/Projects/datasets/Univariate2018/", "StarLightCurves", seed);
        Instances train = dataset[0];
        Instances test = dataset[1];
        ParameterSpace parameterSpace = new DtwParameterSpaceBuilder().build(train);
        for(int i = 0; i < parameterSpace.size(); i++) {
            ParameterSet parameterSet = parameterSpace.get(i);
            System.out.println(parameterSet);
            Knn knn = new Knn();
            knn.setOptions(parameterSet.getOptions());
            knn.setFindTrainAccuracyEstimate(true);
            knn.setTrainSeed(seed);
            knn.setTestSeed(seed);
            knn.setNeighbourhoodSizeLimit(10);
            knn.setTrainEstimateSizeLimit(10);
            knn.buildClassifier(train);
            ClassifierResults trainResults = knn.getTrainResults();
            System.out.println(i + " " + trainResults.getAcc());
        }
//        Knn knn = new Knn();
//        knn.setTrainSeed(seed);
//        knn.setTestSeed(seed);
//        knn.buildClassifier(train);
//        ClassifierResults trainResults = knn.getTrainResults();
//        System.out.println("train acc: " + trainResults.getAcc());
//        System.out.println("-----");
//        ClassifierResults testResults = new ClassifierResults();
//        for (Instance testInstance : test) {
//            long time = System.nanoTime();
//            double[] distribution = knn.distributionForInstance(testInstance);
//            double prediction = indexOfMax(distribution);
//            time = System.nanoTime() - time;
//            testResults.addPrediction(testInstance.classValue(), distribution, prediction, time, null);
//        }
//        System.out.println(testResults.getAcc());
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        Options.super.setOptions(options);
        distanceMeasure.setOptions(options);
    }

    @Override
    public void buildClassifier(final Instances trainingSet) throws
            Exception {
        setupTrain(trainingSet);
        if(trainEstimateEnabled) {
            boolean hasRemainingTrainNeighbours = hasRemainingTrainNeighbours();
            boolean hasRemainingTrainSearchers = hasRemainingTrainSearchers();
            while ((hasRemainingTrainSearchers || hasRemainingTrainNeighbours) && withinTrainTimeLimit()) {
                boolean choice = hasRemainingTrainSearchers;
                if (hasRemainingTrainNeighbours && hasRemainingTrainSearchers) {
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
            }
            buildTrainResults();
        }
        trainTimer.lap();
        if(trainEstimateEnabled && trainResultsPath != null) {
            trainResults.writeFullResultsToFile(trainResultsPath);
        }
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
            Exception {
//        testTimer.reset(); // todo
        setupTest();
//        resetTest = true;
        Searcher searcher = new Searcher(testInstance, false);
        searcher.addAll(trainInstances);
        return searcher.predict();
    }

    @Override
    public String[] getOptions() {
        return ArrayUtilities.concat(distanceMeasure.getOptions(), new String[]{
                NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY,
                String.valueOf(neighbourhoodSizeLimitPercentage),
                NEIGHBOURHOOD_SIZE_LIMIT_KEY,
                String.valueOf(neighbourhoodSizeLimit),
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

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    private void setupTrain(Instances trainInstances) {
        if (resetTrainEnabled) {
            trainTimer.reset();
            trainTimer.start();
            if (trainSeed != null) {
                trainRandom.setSeed(trainSeed);
            } else {
                System.err.println("train seed not set");
            }
            if(trainEstimateEnabled) {
                if(neighbourhoodSizeLimitPercentage >= 0 && neighbourhoodSizeLimitPercentage <= 1) {
                    neighbourhoodSizeLimit = (int) (trainInstances.size() * neighbourhoodSizeLimitPercentage);
                }
                neighbourhood = new ArrayList<>();
                if(distanceCacheEnabled) {
                    InstanceTools.indexInstances(trainInstances);
                    if(distanceMeasure.isSymmetric() && distanceCache == null) {
                        distanceCache = new DupeCache<>();
                    } else {
                        distanceCache = new Cache<>();
                    }
                }
                trainSearchers = new ArrayList<>();
                this.trainInstances = trainInstances;
                trainInstanceIterator = buildTrainInstanceIterator();
                trainEstimatorIterator = buildTrainEstimatorIterator();
            }
            trainTimer.lap();
        }
    }

    private boolean withinTrainEstimateSizeLimit() {
        return trainEstimateSizeLimit < 0 || trainSearchers.size() < trainEstimateSizeLimit;
    }

    private boolean withinNeighbourhoodSizeLimit() {
        return neighbourhoodSizeLimit < 0 || neighbourhood.size() < neighbourhoodSizeLimit;
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

    private void buildTrainResults() throws
            Exception {
        if(trainEstimateEnabled) {
            trainResults = new ClassifierResults();
            for (Searcher searcher : trainSearchers) {
                long time = System.nanoTime();
                double[] distribution = searcher.predict();
                ArrayUtilities.normaliseInPlace(distribution);
                int prediction = ArrayUtilities.bestIndex(Arrays.asList(ArrayUtilities.box(distribution)), trainRandom);
                time = System.nanoTime() - time;
                trainResults.addPrediction(searcher.getTarget().classValue(),
                                           distribution,
                                           prediction,
                                           time,
                                           null);
            }
//        setClassifierResultsMetaInfo(trainResults);
        }
    }

    private void setupTest() {
        if (resetTestEnabled) {
            resetTestEnabled = false;
            if (testSeed != null) {
                testRandom.setSeed(testSeed);
            } else {
                System.err.println("test seed not set");
            }
        }
    }

    private AbstractIterator<Instance> buildTrainInstanceIterator() {
        RandomIterator<Instance> iterator = new RandomIterator<>();
        iterator.setSeed(trainRandom.nextLong());
//        LinearIterator<Instance> iterator = new LinearIterator<>();
        iterator.addAll(trainInstances);
        return iterator;
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
        this.trainEstimateEnabled = estimateTrain;
    }

    @Override
    public void writeTrainEstimatesToFile(final String path) {
        trainResultsPath = path;
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
            case NEIGHBOURHOOD_SIZE_LIMIT_KEY:
                setNeighbourhoodSizeLimit(Integer.parseInt(value));
                break;
            case NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY:
                setNeighbourhoodSizeLimitPercentage(Double.parseDouble(value));
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
        trainRandom = other.trainRandom;
        testRandom = other.testRandom;
        testSeed = other.testSeed;
        trainSeed = other.trainSeed;
        trainEstimateEnabled = other.trainEstimateEnabled;
        trainResultsPath = other.trainResultsPath;
        resetTrainEnabled = other.resetTrainEnabled;
        resetTestEnabled = other.resetTestEnabled;
        k = other.k;
        distanceMeasure = other.distanceMeasure;
        trainTimeLimitNanos = other.trainTimeLimitNanos;
        testTimeLimitNanos = other.testTimeLimitNanos;
        trainInstanceIterator = other.trainInstanceIterator;
        trainEstimatorIterator = other.trainEstimatorIterator;
        trainInstances = other.trainInstances;
        distanceCache = other.distanceCache;
        distanceCacheEnabled = other.distanceCacheEnabled;
        trainSearchers = other.trainSearchers;
        neighbourhood = other.neighbourhood;
        trainTimer = other.trainTimer;
        testTimer = other.testTimer;
        neighbourhoodSizeLimit = other.neighbourhoodSizeLimit;
        trainResults = other.trainResults;
    }

    public int getNeighbourhoodSizeLimit() {
        return neighbourhoodSizeLimit;
    }

    public void setNeighbourhoodSizeLimit(int neighbourhoodSizeLimit) {
        this.neighbourhoodSizeLimit = neighbourhoodSizeLimit;
    }

    @Override
    public String getParameters() {
        return StringUtilities.join(",", getOptions());
    }

    private boolean withinTestTimeLimit() {
        return hasTestTimeLimit() && testTimer.getTimeNanos() < testTimeLimitNanos;
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

    public double getNeighbourhoodSizeLimitPercentage() {
        return neighbourhoodSizeLimitPercentage;
    }

    public void setNeighbourhoodSizeLimitPercentage(double neighbourhoodSizeLimitPercentage) {
        this.neighbourhoodSizeLimitPercentage = neighbourhoodSizeLimitPercentage;
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
            TreeMap<Double, List<Neighbour>> map = selector.getSelectedAsMap();
            int i = 0;
            for (Map.Entry<Double, List<Neighbour>> entry : map.entrySet()) {
                for (Neighbour neighbour : entry.getValue()) {
                    distribution[(int) neighbour.getInstance().classValue()]++;
                    i++;
                }
            }
            if(i > 1) {
                System.out.println(i);
                throw new UnsupportedOperationException(); // todo need to limit to k
            }
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
