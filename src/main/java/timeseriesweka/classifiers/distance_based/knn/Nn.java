package timeseriesweka.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.Seedable;
import timeseriesweka.classifiers.distance_based.distances.DistanceMeasure;
import timeseriesweka.classifiers.distance_based.distances.dtw.Dtw;
import timeseriesweka.classifiers.distance_based.ee.selection.BestKSelector;
import utilities.ArrayUtilities;
import utilities.iteration.AbstractIterator;
import utilities.iteration.random.RandomIterator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.function.Supplier;

import static experiments.data.DatasetLoading.sampleDataset;
import static utilities.GenericTools.indexOfMax;

public class Nn extends AbstractClassifier implements Seedable {

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

    public ClassifierResults getTrainResults() {
        return trainResults;
    }

    public static class Config {
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
        private final BestKSelector<Neighbour, Double> selector;

        public Instance getTarget() {
            return target;
        }

        private Searcher(final Instance target, boolean train) {
            this.target = target;
            selector = new BestKSelector<>();
            selector.setLimit(config.getK());
            this.train = train;
            if(train) {
                selector.setRandom(trainRandom);
            } else {
                selector.setRandom(testRandom);
            }
            selector.setExtractor(Neighbour::getDistance);
        }

        public void add(Instance instance) {
            if(!instance.equals(target)) {
                DistanceMeasure distanceMeasure = config.getDistanceMeasure();
                distanceMeasure.setCandidate(instance);
                distanceMeasure.setTarget(target);
                Double max = selector.getLargestValue();
                if(max != null) {
                    distanceMeasure.setLimit(max);
                }
                double distance;
                if(train) {
                    distance = findDistance(instance, target, distanceMeasure::distance);
                } else {
                    distance = distanceMeasure.distance();
                }
                addUnchecked(instance, distance);
            }
        }

        public void add(Instance instance, double distance) {
            if(instance != target) {
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
            for(Instance instance : instances) {
                add(instance);
            }
        }
    }

    private Config config = new Config();
    private Long trainSeed = null;
    private AbstractIterator<Instance> trainInstanceIterator;
    private AbstractIterator<Instance> trainEstimatorIterator;
    private Instances trainInstances;
    private Map<Instance, Map<Instance, Double>> cache;
    private List<Searcher> trainSearchers;
    private List<Instance> neighbourhood;
    private final Random trainRandom = new Random();
    private final Random testRandom = new Random();

    private void setupTrain(Instances trainInstances) {
        if(config.isResetTrain()) {
            if(trainSeed != null) {
                trainRandom.setSeed(trainSeed);
            } else {
                System.err.println("train seed not set");
            }
            neighbourhood = new ArrayList<>();
            cache = new HashMap<>();
            trainSearchers = new ArrayList<>();
            this.trainInstances = trainInstances;
            trainInstanceIterator = buildTrainInstanceIterator();
            trainEstimatorIterator = buildTrainEstimatorIterator();
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
        if(subCache != null) {
            Double distance = subCache.get(b);
            if(distance != null) {
                subCache.remove(b);
                if(subCache.isEmpty()){
                    cache.remove(a);
                }
            }
            return distance;
        }
        return null;
    }

    private Double findAndRemoveCachedDistance(Instance a, Instance b) {
        Double cachedDistance = findAndRemoveCachedDistanceOrdered(a, b);
        if(cachedDistance == null) {
            cachedDistance = findAndRemoveCachedDistanceOrdered(b, a);
        }
        return cachedDistance;
    }

    private double findDistance(Instance a, Instance b, Supplier<Double> supplier) {
        Double distance = findAndRemoveCachedDistance(a, b);
        if(distance == null) {
            distance = supplier.get();
            cache.computeIfAbsent(a, x -> new HashMap<>()).put(b, distance);
        }
        return distance;
    }

    private void nextTrainInstance() {
        Instance trainInstance = trainInstanceIterator.next();
        trainInstanceIterator.remove();
        for(Searcher trainSearcher : trainSearchers) {
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

    @Override
    public void buildClassifier(final Instances trainingSet) throws
                                                      Exception {
        setupTrain(trainingSet);
        boolean hasRemainingTrainNeighbours = hasRemainingTrainNeighbours();
        boolean hasRemainingTrainSearchers = hasRemainingTrainSearchers();
        while ((hasRemainingTrainSearchers || hasRemainingTrainNeighbours)) {
            boolean choice = hasRemainingTrainSearchers;
            if(hasRemainingTrainNeighbours && hasRemainingTrainSearchers) {
                choice = trainRandom.nextBoolean();
            }
            if(choice) {
                nextTrainSearcher();
            } else {
                nextTrainInstance();
            }
            hasRemainingTrainNeighbours = hasRemainingTrainNeighbours();
            hasRemainingTrainSearchers = hasRemainingTrainSearchers();
        }
        cache = null;
        buildTrainResults();
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
        if(config.isResetTest()) {
            if(testSeed != null) {
                testRandom.setSeed(testSeed);
            } else {
                System.err.println("test seed not set");
            }
        }
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                     Exception {
        setupTest();
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
        Nn nn = new Nn();
        nn.setTrainSeed(seed);
        nn.setTestSeed(seed);
        nn.buildClassifier(train);
        ClassifierResults trainResults = nn.getTrainResults();
        System.out.println("train acc: " + trainResults.getAcc());
        System.out.println("-----");
        ClassifierResults testResults = new ClassifierResults();
        for(Instance testInstance : test) {
            long time = System.nanoTime();
            double[] distribution = nn.distributionForInstance(testInstance);
            double prediction = indexOfMax(distribution);
            time = System.nanoTime() - time;
            testResults.addPrediction(testInstance.classValue(), distribution, prediction, time, null);
        }
        System.out.println(testResults.getAcc());
    }
}
