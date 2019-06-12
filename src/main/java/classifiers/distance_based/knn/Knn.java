package classifiers.distance_based.knn;

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
    public static final boolean DEFAULT_EARLY_ABANDON = false;
    public static final String K_KEY = "k";
    public static final int DEFAULT_SAMPLE_SIZE = -1;
    public static final String NEIGHBOURHOOD_SIZE_KEY = "neighbourhoodSize";
    public static final String DISTANCE_MEASURE_KEY = "distanceMeasure";
    public static final String NEIGHBOUR_SEARCH_STRATEGY_KEY = "neighbourSearchStrategy";
    private final List<NearestNeighbourSet> trainNearestNeighbourSets = new ArrayList<>();
    private final List<NearestNeighbourSet> testNearestNeighbourSets = new ArrayList<>();
    private final List<Instance> untestedTrainInstances = new ArrayList<>();
    private final List<Instance> remainingTrainInstances = new ArrayList<>();
    private List<Instance> trainInstances;
    private int k;
    private DistanceMeasure distanceMeasure;
    private boolean earlyAbandon;
    private int neighbourhoodSize;
    private NeighbourSearchStrategy neighbourSearchStrategy = NeighbourSearchStrategy.RANDOM;
    private long maxPhaseTime = 0;

    public Knn() {
        setDistanceMeasure(new Dtw(0));
        setK(DEFAULT_K);
        setNeighbourhoodSize(DEFAULT_SAMPLE_SIZE);
        setEarlyAbandon(DEFAULT_EARLY_ABANDON);
        setNeighbourSearchStrategy(NeighbourSearchStrategy.RANDOM);
    }

    @Override
    public String toString() {
        return "knn";
    }

    public String[] getOptions() {
        return ArrayUtilities.concat(super.getOptions(), new String[] {
            K_KEY,
            String.valueOf(k),
            NEIGHBOURHOOD_SIZE_KEY,
            String.valueOf(neighbourhoodSize),
            NEIGHBOUR_SEARCH_STRATEGY_KEY,
            neighbourSearchStrategy.name(),
            DISTANCE_MEASURE_KEY,
            distanceMeasure.toString(),
            StringUtilities.join(",", distanceMeasure.getOptions()),
            });
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

    public void setK(final int k) {
        this.k = k;
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
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
            }
        }
        distanceMeasure.setOptions(options);
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

    public NeighbourSearchStrategy getNeighbourSearchStrategy() {
        return neighbourSearchStrategy;
    }

    public void setNeighbourSearchStrategy(final NeighbourSearchStrategy neighbourSearchStrategy) {
        this.neighbourSearchStrategy = neighbourSearchStrategy;
    }

    public void setNeighbourhoodSize(final int neighbourhoodSize) {
        this.neighbourhoodSize = neighbourhoodSize;
    }

    public boolean usesAllNeighbours() {
        return k <= 0;
    } // todo use this!

    @Override
    public void buildClassifier(Instances data) throws
                                                Exception {
        if (trainSetChanged(data)) {
            getTrainStopWatch().reset();
            trainInstances = data;
            remainingTrainInstances.clear();
            remainingTrainInstances.addAll(data);
            trainNearestNeighbourSets.clear();
            for (Instance instance : data) {
                trainNearestNeighbourSets.add(new NearestNeighbourSet(instance));
            }
            getTrainStopWatch().lap();
        }
        while (withinSampleSize() && samplesTrainInstances()
               && !remainingTrainInstances.isEmpty()
               && maxPhaseTime < remainingTrainContractNanos()) {
            long startTime = System.nanoTime();
            Instance instance = sampleInstance();
            untestedTrainInstances.add(instance);
            for (NearestNeighbourSet nearestNeighbourSet : trainNearestNeighbourSets) {
                nearestNeighbourSet.add(instance);
            }
            long phaseTime = System.nanoTime() - startTime;
            maxPhaseTime = Long.max(phaseTime, maxPhaseTime);
            getTrainStopWatch().lap();
        }
        ClassifierResults trainResults = new ClassifierResults();
        setTrainResults(trainResults);
        for (NearestNeighbourSet nearestNeighbourSet : trainNearestNeighbourSets) {
            double[] distribution = nearestNeighbourSet.predict();
            trainResults.addPrediction(nearestNeighbourSet.getTarget()
                                                          .classValue(), distribution, argMax(distribution),
                                       nearestNeighbourSet.getTime(), null);
        }
        getTrainStopWatch().lap();
        setClassifierResultsMetaInfo(trainResults);
    }

    @Override
    public void copyFromSerObject(final Object obj) throws
                                                    Exception {
        super.copyFromSerObject(obj);
        Knn other = (Knn) obj;
        trainNearestNeighbourSets.clear();
        for (NearestNeighbourSet nearestNeighbourSet : other.trainNearestNeighbourSets) {
            trainNearestNeighbourSets.add(nearestNeighbourSet.copy());
        }
        testNearestNeighbourSets.clear();
        for (NearestNeighbourSet nearestNeighbourSet : other.testNearestNeighbourSets) {
            testNearestNeighbourSets.add(nearestNeighbourSet.copy());
        }
        untestedTrainInstances.clear();
        untestedTrainInstances.addAll(other.untestedTrainInstances);
        setK(other.getK());
        setDistanceMeasure(other.getDistanceMeasure());
        setEarlyAbandon(other.getEarlyAbandon());
        setNeighbourhoodSize(other.getNeighbourhoodSize());
        setNeighbourSearchStrategy(other.getNeighbourSearchStrategy());
        remainingTrainInstances.clear();
        remainingTrainInstances.addAll(other.remainingTrainInstances);
        trainInstances = other.trainInstances;
        remainingTrainInstances.clear();
        remainingTrainInstances.addAll(other.remainingTrainInstances);
    }

    private boolean withinSampleSize() {
        return trainInstances.size() - remainingTrainInstances.size() < neighbourhoodSize;
    }

    public boolean samplesTrainInstances() {
        return neighbourhoodSize > 0;
    }

    private Instance sampleInstance() {
        List<Instance> instances = remainingTrainInstances;
        if (neighbourSearchStrategy.equals(NeighbourSearchStrategy.RANDOM)) {
            return instances.remove(getTrainRandom().nextInt(instances.size()));
        } else if (neighbourSearchStrategy.equals(NeighbourSearchStrategy.LINEAR)) {
            return instances.remove(0);
        } else if (neighbourSearchStrategy.equals(NeighbourSearchStrategy.ROUND_ROBIN_RANDOM)) {
//            return instances.remove(0); todo
            throw new UnsupportedOperationException();
        } else if (neighbourSearchStrategy.equals(NeighbourSearchStrategy.DISTRIBUTED_RANDOM)) {
//            return instances.remove(0); todo
            throw new UnsupportedOperationException();
        } else {
            throw new UnsupportedOperationException();
        }
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                         Exception {
        NearestNeighbourSet testNearestNeighbourSet = new NearestNeighbourSet(testInstance);
        for (NearestNeighbourSet trainNearestNeighbourSet : trainNearestNeighbourSets) {
            testNearestNeighbourSet.add(trainNearestNeighbourSet.getTarget());
        }
        return testNearestNeighbourSet.predict();
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
        boolean change = !untestedTrainInstances.isEmpty();
        while (!untestedTrainInstances.isEmpty()) {
            Instance instance = untestedTrainInstances.remove(0);
            for (NearestNeighbourSet nearestNeighbourSet : testNearestNeighbourSets) {
                nearestNeighbourSet.add(instance);
            }
        }
        if (change) {
            for (NearestNeighbourSet nearestNeighbourSet : testNearestNeighbourSets) {
                nearestNeighbourSet.trimNeighbours();
            }
        }
        ClassifierResults results = new ClassifierResults();
        for (int i = 0; i < testNearestNeighbourSets.size(); i++) {
            NearestNeighbourSet nearestNeighbourSet = testNearestNeighbourSets.get(i);
            double[] distribution = nearestNeighbourSet.predict();
            long time = nearestNeighbourSet.getTime();
            results.addPrediction(nearestNeighbourSet.getTarget()
                                                     .classValue(), distribution, argMax(distribution), time, null);
        }
        getTestStopWatch().lap();
        setClassifierResultsMetaInfo(results);
        if(getTrainResultsPath() != null) {
            getTrainResults().writeFullResultsToFile(getTrainResultsPath());
        }
        return results;
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

    private class NearestNeighbourSet {
        private final Instance target;
        private final TreeMap<Double, List<Instance>> neighbours = new TreeMap<>();
        private TreeMap<Double, List<Instance>> trimmedNeighbours = null;
        private double maxDistance = Double.POSITIVE_INFINITY;
        private int size = 0;
        private long time = 0;
        private long predictTime = 0;

        private NearestNeighbourSet(final Instance target) {this.target = target;}

        public NearestNeighbourSet copy() {
            NearestNeighbourSet nearestNeighbourSet = new NearestNeighbourSet(target);
            for (Map.Entry<Double, List<Instance>> entry : neighbours.entrySet()) {
                nearestNeighbourSet.neighbours.put(entry.getKey(), new ArrayList<>(entry.getValue()));
            }
            nearestNeighbourSet.size = size;
            nearestNeighbourSet.time = time;
            nearestNeighbourSet.predictTime = predictTime;
            nearestNeighbourSet.trimNeighbours();
            return nearestNeighbourSet;
        }

        public void trimNeighbours() {
            long startTime = System.nanoTime();
            int size = this.size;
            if (size == 0 || size == k) {
                trimmedNeighbours = neighbours;
                return;
            }
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
            if (trimmedNeighbours == null) {
                trimNeighbours();
            }
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
            ArrayUtilities.normalise_inplace(distribution);
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
                if (earlyAbandon) { maxDistance = max(maxDistance, distance); }
            }
            equalDistanceNeighbours.add(instance);
            size++;
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
            trimmedNeighbours = null;
            time += System.nanoTime() - startTime;
        }
    }






}
