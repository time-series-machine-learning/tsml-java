package classifiers.distance_based.knn;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.knn.sampling.*;
import classifiers.template_classifier.TemplateClassifier;
import distances.DistanceMeasure;
import distances.time_domain.dtw.Dtw;
import evaluation.storage.ClassifierResults;
import utilities.ArrayUtilities;
import utilities.StringUtilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

import static java.lang.Double.max;
import static utilities.ArrayUtilities.argMax;

public class Knn
    extends TemplateClassifier {

    public static final int DEFAULT_K = 1;
    public static final boolean DEFAULT_EARLY_ABANDON = true;
    public static final String K_KEY = "k";
    public static final int DEFAULT_NEIGHBOURHOOD_SIZE = -1;
    public static final String NEIGHBOURHOOD_SIZE_KEY = "neighbourhoodSize";
    public static final String NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY = "neighbourhoodSizePercentage";
    public static final String DISTANCE_MEASURE_KEY = "distanceMeasure";
    public static final String NEIGHBOUR_SEARCH_STRATEGY_KEY = "neighbourSearchStrategy";
    public static final String TRAIN_SET_SIZE_KEY = "trainSubSetSize";
    public static final String TRAIN_SET_SIZE_PERCENTAGE_KEY = "trainSubSetSizePercentage";
    public static final String EARLY_ABANDON_KEY = "earlyAbandon";
    public static final int DEFAULT_TRAIN_SET_SIZE = -1;
    public static final int DEFAULT_TRAIN_SET_SIZE_PERCENTAGE = -1;
    public static final double DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE = -1;
    private final List<NearestNeighbourSet> trainNearestNeighbourSets = new ArrayList<>();
    private final List<NearestNeighbourSet> testNearestNeighbourSets = new ArrayList<>();
    private final List<Instance> untestedTrainInstances = new ArrayList<>(); // todo needed?
    private final List<Instance> trainSubSet = new ArrayList<>();
    private int k;
    private DistanceMeasure distanceMeasure;
    private boolean earlyAbandon;
    private int neighbourhoodSize;
    private double trainSubSetSizePercentage;
    private int trainSubSetSize;
    private NeighbourSearchStrategy neighbourSearchStrategy = NeighbourSearchStrategy.RANDOM;
    private long maxPhaseTime = 0;
    private DynamicIterator<Instance, ?> neighbourhoodSampler;
    private DynamicIterator<Instance, ?> trainSubSetSampler;
    private final List<Instance> neighbourhood = new ArrayList<>();
    private double neighbourhoodSizePercentage = DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE;
    private TrainSetSampleStrategy trainSetSampleStrategy = TrainSetSampleStrategy.RANDOM;
    private Instances trainSet;

    public Knn() {
        setDistanceMeasure(new Dtw(0));
        setK(DEFAULT_K);
        setNeighbourhoodSize(DEFAULT_NEIGHBOURHOOD_SIZE);
        setEarlyAbandon(DEFAULT_EARLY_ABANDON);
        setTrainSubSetSizePercentage(DEFAULT_TRAIN_SET_SIZE_PERCENTAGE);
        setTrainSubSetSize(DEFAULT_TRAIN_SET_SIZE);
        setNeighbourSearchStrategy(NeighbourSearchStrategy.RANDOM);
    }

    @Override
    public String toString() {
        return "knn";
    }

    public Knn copy() {
        Knn knn = new Knn();
        try {
            knn.copyFromSerObject(this);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return knn;
    }

    public int getK() {
        return k;
    }

    public String[] getOptions() {
        return ArrayUtilities.concat(super.getOptions(), new String[] {
            K_KEY,
            String.valueOf(k),
            NEIGHBOURHOOD_SIZE_KEY,
            String.valueOf(neighbourhoodSize),
            NEIGHBOUR_SEARCH_STRATEGY_KEY,
            neighbourSearchStrategy.name(),
            TRAIN_SET_SIZE_KEY,
            String.valueOf(trainSubSetSize),
            EARLY_ABANDON_KEY,
            String.valueOf(earlyAbandon),
            NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY,
            String.valueOf(neighbourhoodSizePercentage),
            TRAIN_SET_SIZE_PERCENTAGE_KEY,
            String.valueOf(trainSubSetSizePercentage),
            DISTANCE_MEASURE_KEY,
            distanceMeasure.toString(),
            StringUtilities.join(",", distanceMeasure.getOptions()),
            });
    }

    public void setK(final int k) {
        this.k = k;
    }

    public void setOptions(final String[] options) throws
                                                   Exception {
        super.setOptions(options);
        for (int i = 0; i < options.length - 1; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            if (key.equals(K_KEY)) {
                setK(Integer.parseInt(value));
            } else if (key.equals(NEIGHBOURHOOD_SIZE_KEY)) {
                setK(Integer.parseInt(value));
            } else if (key.equals(DISTANCE_MEASURE_KEY)) {
                setDistanceMeasure(DistanceMeasure.fromString(value));
            } else if (key.equals(NEIGHBOUR_SEARCH_STRATEGY_KEY)) {
                setNeighbourSearchStrategy(NeighbourSearchStrategy.fromString(value));
            } else if (key.equals(EARLY_ABANDON_KEY)) {
                setEarlyAbandon(Boolean.parseBoolean(value));
            } else if (key.equals(TRAIN_SET_SIZE_KEY)) {
                setTrainSubSetSize(Integer.parseInt(value));
            } else if (key.equals(TRAIN_SET_SIZE_PERCENTAGE_KEY)) {
                setTrainSubSetSizePercentage(Double.parseDouble(value));
            } else if (key.equals(NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY)) {
                setNeighbourhoodSizePercentage(Double.valueOf(value));
            }
        }
        distanceMeasure.setOptions(options);
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
    }

    public boolean getEarlyAbandon() {
        return earlyAbandon;
    }

    public void setEarlyAbandon(final boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
    }

    public int getNeighbourhoodSize() {
        return neighbourhoodSize;
    }

    public void setNeighbourhoodSize(final int neighbourhoodSize) {
        this.neighbourhoodSize = neighbourhoodSize;
    }

    public NeighbourSearchStrategy getNeighbourSearchStrategy() {
        return neighbourSearchStrategy;
    }

    public void setNeighbourSearchStrategy(final NeighbourSearchStrategy neighbourSearchStrategy) {
        this.neighbourSearchStrategy = neighbourSearchStrategy;
    }

    public double getTrainSubSetSizePercentage() {
        return trainSubSetSizePercentage;
    }

    public int getTrainSubSetSize() {
        return trainSubSetSize;
    }

    public void setTrainSubSetSize(final int trainSubSetSize) {
        this.trainSubSetSize = trainSubSetSize;
    }

    public void setTrainSubSetSizePercentage(final double trainSubSetSizePercentage) {
        this.trainSubSetSizePercentage = trainSubSetSizePercentage;
    }

    public double getNeighbourhoodSizePercentage() {
        return neighbourhoodSizePercentage;
    }

    public void setNeighbourhoodSizePercentage(double neighbourhoodSizePercentage) {
        this.neighbourhoodSizePercentage = neighbourhoodSizePercentage;
    }

    private boolean hasRemainingTrainSubSet() {
        return (trainSubSet.size() < trainSubSetSize || (trainSubSetSize < 0 && trainSubSet.size() < trainSet.size()))
               && trainSubSetSampler.hasNext();
    }

    private boolean hasRemainingNeighbours() {
        return (neighbourhood.size() < neighbourhoodSize || neighbourhoodSize < 0)
               && neighbourhoodSampler.hasNext();
    }

    @Override
    public void buildClassifier(Instances trainSet) throws
                                                           Exception {
        if (trainSetChanged(trainSet)) { // todo move to reset func and call in setters if exceeding certain
            // stuff, e.g. neighbourhood size inc
            getTrainStopWatch().reset();
            this.trainSet = trainSet;
            trainSubSet.clear();
            trainNearestNeighbourSets.clear();
            neighbourhood.clear();
            setupNeighbourhoodSize();
            setupTrainSetSize();
            setupTrainSetSampleStrategy();
            setupNeighbourSearchStrategy();
        }
        boolean remainingTrainSubSet = hasRemainingTrainSubSet();
        boolean remainingNeighbours = hasRemainingNeighbours();
        while ((remainingTrainSubSet || remainingNeighbours)
               && maxPhaseTime < remainingTrainContractNanos()) {
            boolean choice = true;
            if(remainingTrainSubSet && remainingNeighbours) {
                choice = getTrainRandom().nextBoolean(); // todo change to strategy
            } else if(remainingNeighbours) { // todo empty train set should be rand predict
                choice = false;
            }
            if(choice) {
                // expand train subset
                Instance trainInstance = trainSubSetSampler.next();
                trainSubSetSampler.remove();
                NearestNeighbourSet nearestNeighbourSet = new NearestNeighbourSet(trainInstance);
                trainNearestNeighbourSets.add(nearestNeighbourSet);
                trainSubSet.add(trainInstance);
                nearestNeighbourSet.addAll(neighbourhood);
                neighbourhoodSampler.add(trainInstance);
            } else {
                // expand neighbourhood
                long startTime = System.nanoTime();
                Instance neighbour = neighbourhoodSampler.next();
                neighbourhoodSampler.remove();
                neighbourhood.add(neighbour);
                untestedTrainInstances.add(neighbour);
                for (NearestNeighbourSet nearestNeighbourSet : trainNearestNeighbourSets) {
                    if (!nearestNeighbourSet.getTarget()
                                            .equals(neighbour)) {
                        nearestNeighbourSet.add(neighbour);
                    }
                }
                long phaseTime = System.nanoTime() - startTime;
                maxPhaseTime = Long.max(phaseTime, maxPhaseTime);
            }
            remainingTrainSubSet = hasRemainingTrainSubSet();
            remainingNeighbours = hasRemainingNeighbours();
            getTrainStopWatch().lap();
        }
        ClassifierResults trainResults = new ClassifierResults();
        setTrainResults(trainResults);
        for (NearestNeighbourSet nearestNeighbourSet : trainNearestNeighbourSets) {
            nearestNeighbourSet.trim();
            double[] distribution = nearestNeighbourSet.predict();
            trainResults.addPrediction(nearestNeighbourSet.getTarget().classValue(), distribution, argMax(distribution), // todo random
                                       nearestNeighbourSet.getTime(), null);
        }
        getTrainStopWatch().lap();
        setClassifierResultsMetaInfo(trainResults);
    }

    private void setupNeighbourSearchStrategy() {
        switch (neighbourSearchStrategy) {
            case RANDOM:
                neighbourhoodSampler = new RandomSampler(getTrainRandom());
                break;
            case LINEAR:
                neighbourhoodSampler = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                neighbourhoodSampler = new RoundRobinRandomSampler(getTrainRandom());
                break;
            case DISTRIBUTED_RANDOM:
                neighbourhoodSampler = new DistributedRandomSampler(getTrainRandom());
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    private void setupNeighbourhoodSize() {
        if (neighbourhoodSizePercentage >= 0) {
            setNeighbourhoodSize((int) (trainSet.size() * neighbourhoodSizePercentage));
        }
    }

    private void setupTrainSetSize() {
        if (trainSubSetSizePercentage >= 0) {
            setTrainSubSetSize((int) (trainSet.size() * trainSubSetSizePercentage));
        }
    }

    private void setupTrainSetSampleStrategy() {
        switch (trainSetSampleStrategy) {
            case RANDOM:
                trainSubSetSampler = new RandomSampler(getTrainRandom());
                break;
            case LINEAR:
                trainSubSetSampler = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                trainSubSetSampler = new RoundRobinRandomSampler(getTrainRandom());
                break;
            case DISTRIBUTED_RANDOM:
                trainSubSetSampler = new DistributedRandomSampler(getTrainRandom());
                break;
            default:
                throw new UnsupportedOperationException();
        }
        for(Instance trainInstance : trainSet) {
            trainSubSetSampler.add(trainInstance);
        }
    }

    @Override
    public void copyFromSerObject(final Object obj) throws
                                                    Exception {
        super.copyFromSerObject(obj);
        Knn other = (Knn) obj;
        trainNearestNeighbourSets.clear();
        for (NearestNeighbourSet nearestNeighbourSet : other.trainNearestNeighbourSets) {
            trainNearestNeighbourSets.add(new NearestNeighbourSet(nearestNeighbourSet));
        }
        testNearestNeighbourSets.clear();
        for (NearestNeighbourSet nearestNeighbourSet : other.testNearestNeighbourSets) {
            testNearestNeighbourSets.add(new NearestNeighbourSet(nearestNeighbourSet));
        }
        untestedTrainInstances.clear();
        untestedTrainInstances.addAll(other.untestedTrainInstances);
        setK(other.getK());
        setDistanceMeasure(DistanceMeasure.fromString(other.getDistanceMeasure()
                                                           .toString()));
        distanceMeasure.setOptions(other.getDistanceMeasure()
                                        .getOptions());
        setEarlyAbandon(other.getEarlyAbandon());
        setNeighbourhoodSize(other.getNeighbourhoodSize());
        setNeighbourSearchStrategy(other.getNeighbourSearchStrategy());
        setTrainSubSetSizePercentage(other.getTrainSubSetSizePercentage());
        setTrainSubSetSize(other.getTrainSubSetSize());
        trainSubSet.clear();
        trainSubSet.addAll(other.trainSubSet);
        neighbourhood.clear();
        neighbourhood.addAll(other.neighbourhood);
        trainSubSetSampler = other.trainSubSetSampler.iterator();
        neighbourhoodSampler = other.neighbourhoodSampler.iterator();
        trainSet = other.trainSet;
    }

    public ClassifierResults getTestResults(Instances testInstances) throws
                                                                     Exception {
        if (testSetChanged(testInstances)) {
            getTestStopWatch().reset();
            testNearestNeighbourSets.clear();
            for (Instance testInstance : testInstances) {
                testNearestNeighbourSets.add(new NearestNeighbourSet(testInstance));
            }
        }
        if (!untestedTrainInstances.isEmpty()) {
            do {
                Instance neighbour = untestedTrainInstances.remove(0);
                for (NearestNeighbourSet nearestNeighbourSet : testNearestNeighbourSets) {
                    nearestNeighbourSet.add(neighbour);
                }
            }
            while (!untestedTrainInstances.isEmpty());
            for (NearestNeighbourSet nearestNeighbourSet : testNearestNeighbourSets) {
                nearestNeighbourSet.trim();
            }
        }
        ClassifierResults results = new ClassifierResults();
        for (NearestNeighbourSet nearestNeighbourSet : testNearestNeighbourSets) {
            double[] distribution = nearestNeighbourSet.predict();
            long time = nearestNeighbourSet.getTime();
            results.addPrediction(nearestNeighbourSet.getTarget()
                                                     .classValue(), distribution, argMax(distribution), time, null);
        }
        getTestStopWatch().lap();
        setClassifierResultsMetaInfo(results);
        if (getTrainResultsPath() != null) { // todo hacky implementation, update when train estimate api overhauled
            getTrainResults().writeFullResultsToFile(getTrainResultsPath());
        }
        return results;
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                         Exception {
        NearestNeighbourSet testNearestNeighbourSet = new NearestNeighbourSet(testInstance);
        testNearestNeighbourSet.addAll(neighbourhood);
        testNearestNeighbourSet.trim();
        return testNearestNeighbourSet.predict();
    }

    public enum NeighbourSearchStrategy {
        RANDOM,
        LINEAR,
        ROUND_ROBIN_RANDOM,
        DISTRIBUTED_RANDOM;

        public static NeighbourSearchStrategy fromString(String str) {
            for (NeighbourSearchStrategy s : NeighbourSearchStrategy.values()) {
                if (s.name()
                     .equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    public enum TrainSetSampleStrategy {
        RANDOM,
        LINEAR,
        ROUND_ROBIN_RANDOM,
        DISTRIBUTED_RANDOM;

        public static TrainSetSampleStrategy fromString(String str) {
            for (TrainSetSampleStrategy s : TrainSetSampleStrategy.values()) {
                if (s.name()
                     .equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    private class NearestNeighbourSet {
        private final Instance target;
        private final TreeMap<Double, List<Instance>> neighbours = new TreeMap<>();
        private TreeMap<Double, List<Instance>> trimmedNeighbours = null;
        private double maxDistance = Double.POSITIVE_INFINITY;
        private int size = 0;
        private long time = 0;
        private long predictTime = 0;

        private NearestNeighbourSet(final Instance target) {
            this.target = target;
        }

        public NearestNeighbourSet(NearestNeighbourSet other) {
            this(other.target);
            for (Map.Entry<Double, List<Instance>> entry : other.neighbours.entrySet()) {
                neighbours.put(entry.getKey(), new ArrayList<>(entry.getValue()));
            }
            size = other.size;
            time = other.time;
            predictTime = other.predictTime;
            trim();
        }

        public void trim() {
            long startTime = System.nanoTime();
            int size = this.size;
            if (size == 0 || size == k || k <= 0) {
                trimmedNeighbours = neighbours;
            } else {
                trimmedNeighbours = new TreeMap<>();
                for (Map.Entry<Double, List<Instance>> entry : neighbours.entrySet()) {
                    trimmedNeighbours.put(entry.getKey(), entry.getValue());
                }
                Map.Entry<Double, List<Instance>> last = trimmedNeighbours.lastEntry();
                List<Instance> furthestNeighbours = new ArrayList<>(last.getValue());
                trimmedNeighbours.put(last.getKey(), furthestNeighbours);
                while (size > k) {
                    size--;
                    int index = getTrainRandom().nextInt(furthestNeighbours.size());
                    furthestNeighbours.remove(index);
                }
            }
            time += System.nanoTime() - startTime;
        }

        public Instance getTarget() {
            return target;
        }

        public int size() {
            return size;
        }

        public long getTime() {
            return time + predictTime;
        }

        public double[] predict() {
            long startTime = System.nanoTime();
            double[] distribution = new double[target.numClasses()];
            TreeMap<Double, List<Instance>> neighbours = trimmedNeighbours;
            if (neighbours.size() == 0) {
                distribution[getTestRandom().nextInt(distribution.length)]++;
                return distribution;
            }
            for (Map.Entry<Double, List<Instance>> entry : neighbours.entrySet()) {
                for (Instance instance : entry.getValue()) {
                    // todo weighted
                    distribution[(int) instance.classValue()]++;
                }
            }
            ArrayUtilities.normaliseInplace(distribution);
            predictTime = System.nanoTime() - startTime;
            return distribution;
        }

        public void addAll(final List<Instance> instances) {
            for (Instance instance : instances) {
                add(instance);
            }
        }

        public double add(Instance instance) {
            long startTime = System.nanoTime();
            double distance = distanceMeasure.distance(target, instance, maxDistance);
            time += System.nanoTime() - startTime;
            add(instance, distance);
            return distance;
        }

        public void add(Instance instance, double distance) {
            long startTime = System.nanoTime();
            List<Instance> equalDistanceNeighbours = neighbours.get(distance);
            if (equalDistanceNeighbours == null) {
                equalDistanceNeighbours = new ArrayList<>();
                neighbours.put(distance, equalDistanceNeighbours);
                if (earlyAbandon) {
                    maxDistance = max(maxDistance, distance);
                }
            }
            equalDistanceNeighbours.add(instance);
            size++;
            if (k > 0) {
                int lastEntrySize = neighbours.lastEntry()
                                              .getValue()
                                              .size();
                if (size - k >= lastEntrySize) {
                    neighbours.pollLastEntry();
                    size -= lastEntrySize;
                    if (earlyAbandon) {
                        maxDistance = max(maxDistance, neighbours.lastEntry()
                                                                 .getKey());
                    }
                }
            }
            trimmedNeighbours = null;
            time += System.nanoTime() - startTime;
        }
    }






}
