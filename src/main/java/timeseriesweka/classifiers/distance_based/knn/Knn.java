package timeseriesweka.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.Seedable;
import timeseriesweka.classifiers.TestTimeContractable;
import timeseriesweka.classifiers.TrainAccuracyEstimator;
import timeseriesweka.classifiers.TrainTimeContractable;
import timeseriesweka.classifiers.distance_based.distances.DistanceMeasure;
import timeseriesweka.classifiers.distance_based.distances.dtw.Dtw;
import timeseriesweka.classifiers.distance_based.ee.selection.KBestSelector;
import utilities.*;
import utilities.iteration.AbstractIterator;
import utilities.iteration.random.RandomIterator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

import static experiments.data.DatasetLoading.sampleDataset;
import static timeseriesweka.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_KEY;
import static utilities.GenericTools.indexOfMax;

public class Knn extends AbstractClassifier implements Options, Seedable, TrainTimeContractable, TestTimeContractable, Copyable, Serializable,
                                                       TrainAccuracyEstimator {

    @Override
    public void setTrainSeed(final long seed) {
        trainSeed = seed;
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

    private Long testSeed;

    @Override
    public void setFindTrainAccuracyEstimate(final boolean estimateTrain) {
        this.estimateTrain = estimateTrain;
    }

    @Override
    public void writeTrainEstimatesToFile(final String path) {
        trainResultsPath = path;
    }

    private boolean estimateTrain = true;
    private String trainResultsPath;

    public ClassifierResults getTrainResults() {
        return trainResults;
    }

    private static final String K_KEY = "k";
    private boolean resetTrain = true;
    private boolean resetTest = true;
    private int k = 1;

    public boolean isResetTrain() {
        return resetTrain;
    }

    public void setResetTrain(final boolean resetTrain) {
        this.resetTrain = resetTrain;
    }

    public int getK() {
        return k;
    }

    public void setK(final int k) {
        this.k = k;
    }

    private DistanceMeasure distanceMeasure = new Dtw();

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
    }

    public boolean isResetTest() {
        return resetTest;
    }

    public void setResetTest(final boolean resetTest) {
        this.resetTest = resetTest;
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
        }
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        Options.super.setOptions(options);
        distanceMeasure.setOptions(options);
    }

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String[] getOptions() {
        return ArrayUtilities.concat(distanceMeasure.getOptions(), new String[]{
                DISTANCE_MEASURE_KEY,
                String.valueOf(distanceMeasure),
                K_KEY,
                String.valueOf(k),
                TRAIN_TIME_CONTRACT_KEY,
                String.valueOf(trainTimeLimitNanos),
                TEST_TIME_CONTRACT_KEY,
                String.valueOf(testTimeLimitNanos),
        });
    }

    @Override
    public void setTestTimeLimit(TimeUnit time, long amount) {
        testTimeLimitNanos = TimeUnit.NANOSECONDS.convert(amount, time);
    }

    @Override
    public void setTrainTimeLimit(TimeUnit time, long amount) {
        trainTimeLimitNanos = TimeUnit.NANOSECONDS.convert(amount, time);
    }

    public Knn() {

    }

    public Knn(Knn other) throws Exception {
        copyFrom(other);
    }

    @Override
    public Object copy() throws Exception {
        throw new UnsupportedOperationException(); // todo
    }

    @Override
    public void copyFrom(Object object) throws Exception {
        throw new UnsupportedOperationException(); // todo
    }

    public int getTrainSize() {
        return trainSize;
    }

    public void setTrainSize(int trainSize) {
        this.trainSize = trainSize;
    }

    @Override
    public String getParameters() {
        return StringUtilities.join(",", getOptions());
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

        public Instance getTarget() {
            return target;
        }

        private Searcher(final Instance target, boolean train) {
            this.target = target;
            selector = new KBestSelector<>();
            selector.setLimit(k);
            this.train = train;
            if (train) {
                selector.setRandom(trainRandom);
            } else {
                selector.setRandom(testRandom);
            }
            selector.setExtractor(Neighbour::getDistance);
        }

        public void add(Instance instance) {
            if (!instance.equals(target)) {
                distanceMeasure.setCandidate(instance);
                distanceMeasure.setTarget(target);
                Double max = selector.getWorstValue();
                if (max != null) {
                    distanceMeasure.setLimit(max);
                }
                double distance;
                if (train) {
                    distance = findDistance(instance, target, distanceMeasure::distance);
                } else {
                    distance = distanceMeasure.distance();
                }
                addUnchecked(instance, distance);
            }
        }

        public void add(Instance instance, double distance) {
            if (instance != target) {
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
            for (Map.Entry<Double, List<Neighbour>> entry : map.entrySet()) {
                for (Neighbour neighbour : entry.getValue()) {
                    distribution[(int) neighbour.getInstance().classValue()]++;
                }
            }
            return distribution;
        }

        public void addAll(final List<Instance> instances) {
            for (Instance instance : instances) {
                add(instance);
            }
        }
    }

    private long trainTimeLimitNanos = -1;
    private long testTimeLimitNanos = -1;
    private Long trainSeed = null;
    private AbstractIterator<Instance> trainInstanceIterator;
    private AbstractIterator<Instance> trainEstimatorIterator;
    private Instances trainInstances;
    private Map<Instance, Map<Instance, Double>> cache;
    private List<Searcher> trainSearchers;
    private List<Instance> neighbourhood;
    private final Random trainRandom = new Random();
    private final Random testRandom = new Random();
    private StopWatch trainTimer = new StopWatch();
    private StopWatch testTimer = new StopWatch();
    private int trainSize = -1;

    private void setupTrain(Instances trainInstances) {
        if (resetTrain) {
            trainTimer.reset();
            trainTimer.start();
            if (trainSeed != null) {
                trainRandom.setSeed(trainSeed);
            } else {
                System.err.println("train seed not set");
            }
            if(estimateTrain) {
                neighbourhood = new ArrayList<>();
                cache = new HashMap<>();
                trainSearchers = new ArrayList<>();
                this.trainInstances = trainInstances;
                trainInstanceIterator = buildTrainInstanceIterator();
                trainEstimatorIterator = buildTrainEstimatorIterator();
            }
            trainTimer.lap();
        }
    }

    private AbstractIterator<Instance> buildTrainInstanceIterator() {
        RandomIterator<Instance> iterator = new RandomIterator<>();
        iterator.setSeed(trainRandom.nextLong());
        iterator.addAll(trainInstances);
        return iterator;
    }

    private AbstractIterator<Instance> buildTrainEstimatorIterator() {
        RandomIterator<Instance> iterator = new RandomIterator<>();
        iterator.setSeed(trainRandom.nextLong());
        iterator.addAll(trainInstances);
        return iterator;
    }


    private Double findAndRemoveCachedDistanceOrdered(Instance a, Instance b) {
        Map<Instance, Double> subCache = cache.get(a);
        if (subCache != null) {
            Double distance = subCache.get(b);
            if (distance != null) {
                subCache.remove(b);
                if (subCache.isEmpty()) {
                    cache.remove(a);
                }
            }
            return distance;
        }
        return null;
    }

    private Double findAndRemoveCachedDistance(Instance a, Instance b) {
        Double cachedDistance = findAndRemoveCachedDistanceOrdered(a, b);
        if (cachedDistance == null) {
            cachedDistance = findAndRemoveCachedDistanceOrdered(b, a);
        }
        return cachedDistance;
    }

    private double findDistance(Instance a, Instance b, Supplier<Double> supplier) {
        Double distance = findAndRemoveCachedDistance(a, b);
        if (distance == null) {
            distance = supplier.get();
            cache.computeIfAbsent(a, x -> new HashMap<>()).put(b, distance);
        }
        return distance;
    }

    private void nextTrainInstance() {
        Instance trainInstance = trainInstanceIterator.next();
        trainInstanceIterator.remove();
        for (Searcher trainSearcher : trainSearchers) {
            trainSearcher.add(trainInstance);
        }
        neighbourhood.add(trainInstance);
    }

    private void nextTrainSearcher() {
        Instance trainInstance = trainEstimatorIterator.next();
        trainEstimatorIterator.remove();
        Searcher searcher = new Searcher(trainInstance, true);
        searcher.addAll(neighbourhood);
        trainSearchers.add(searcher);
    }

    private boolean hasRemainingTrainSearchers() {
        return trainEstimatorIterator.hasNext();
    }

    private boolean hasRemainingTrainNeighbours() {
        return trainInstanceIterator.hasNext();
    }

    public boolean hasTrainTimeLimit() {
        return trainTimeLimitNanos >= 0;
    }

    public boolean hasTestTimeLimit() {
        return testTimeLimitNanos >= 0;
    }

    private boolean withinTrainTimeLimit() {
        return !hasTrainTimeLimit() || trainTimer.getTimeNanos() < trainTimeLimitNanos;
    }

    private boolean withinTestTimeLimit() {
        return hasTestTimeLimit() && testTimer.getTimeNanos() < testTimeLimitNanos;
    }

    @Override
    public void buildClassifier(final Instances trainingSet) throws
            Exception {
        setupTrain(trainingSet);
        if(estimateTrain) {
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
    }

    private void buildTrainResults() throws
            Exception {
        trainResults = new ClassifierResults();
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

    private ClassifierResults trainResults;

    private void setupTest() {
        if (resetTest) {
            resetTest = false;
            if (testSeed != null) {
                testRandom.setSeed(testSeed);
            } else {
                System.err.println("test seed not set");
            }
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

    public static void main(String[] args) throws
            Exception {
        int seed = 0;
        Instances[] dataset = sampleDataset("/home/vte14wgu/Projects/datasets/Univariate2018/", "GunPoint", seed);
        Instances train = dataset[0];
        Instances test = dataset[1];
        Knn knn = new Knn();
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
            double prediction = indexOfMax(distribution);
            time = System.nanoTime() - time;
            testResults.addPrediction(testInstance.classValue(), distribution, prediction, time, null);
        }
        System.out.println(testResults.getAcc());
    }
}
