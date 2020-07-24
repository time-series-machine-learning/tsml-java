package tsml.classifiers.distance_based.knn;

import tsml.classifiers.distance_based.proximity.ProximitySplit;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.BaseCheckpointer;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointer;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap;
import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.system.timing.TimedTest;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrain;
import utilities.ArrayUtilities;
import weka.core.DistanceFunction;
import weka.core.Instance;

import java.io.Serializable;
import java.util.Comparator;
import java.util.Random;
import java.util.logging.Logger;

public class K  extends BaseClassifier
        implements ContractedTest, ContractedTrain, TimedTrain, TimedTest, WatchedMemory, Checkpointed {

    private static final long serialVersionUID = 1;
    // train timer
    private final StopWatch trainTimer = new StopWatch();
    // test / predict timer
    private final StopWatch testTimer = new StopWatch();
    // memory watcher
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    // train stage timer used for predicting whether there's enough time for more training within the train contract
    private final StopWatch trainStageTimer = new StopWatch();
    // test stage timer used for predicting whether there's enough time for more prediction within the test contract
    private final StopWatch testStageTimer = new StopWatch();
    // the train time limit / contract
    private transient long trainTimeLimit;
    // the test time limit / contract
    private transient long testTimeLimit;
    // checkpoint config
    private transient final Checkpointer checkpointer = new BaseCheckpointer(this);
    // whether to early abandon the distance measurements
    private boolean earlyAbandon;
    // the distance function used to compare instances
    private DistanceFunction distanceFunction;

    public boolean isEarlyAbandon() {
        return earlyAbandon;
    }

    public void setEarlyAbandon(final boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public void setDistanceFunction(final DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }

    public int getK() {
        return k;
    }

    public void setK(final int k) {
        this.k = k;
    }

    // number of neighbours
    private int k = 1;

    @Override public double[] distributionForInstance(final Instance instance) throws Exception {
        throw new UnsupportedOperationException();
    }


    /**
     * Searcher class to find the set of nearest neighbours for a given instance.
     */
    public class Searcher implements Serializable {
        // map of distance to instances, allowing multiple instances to have the same distance
        private final PrunedMultimap<Double, Instance> prunedMap;
        // the target instance we're trying to find the closest neighbour to
        private final Instance instance;
        // distance limit if we're early abandoning
        private double limit = Double.POSITIVE_INFINITY;
        // timer to record comparison time
        private final StopWatch comparisonTimer = new StopWatch();
        // timer to record the prediction time
        private final StopWatch predictTimer = new StopWatch();

        public Instance getInstance() {
            return instance;
        }

        public Searcher(Instance instance) {
            this.prunedMap = PrunedMultimap.asc();
            // set the map to look for the k closest neighbours but keep neighbours which draw (e.g. both have a
            // distance of 2, say
            prunedMap.setSoftLimit(k);
            this.instance = instance;
        }

        // add an instance, finding the distance between the target instance and the given instance
        public double add(Instance neighbour) {
            StopWatch timer = new StopWatch();
            timer.start();
            final double distance = distanceFunction.distance(this.instance, neighbour, limit);
            timer.stop();
            add(neighbour, distance, timer.getTime());
            return distance;
        }

        // add an instance given a precomputed distance and corresponding time it took to find that distance
        public void add(Instance neighbour, double distance, long distanceMeasurementTime) {
            comparisonTimer.start();
            prunedMap.put(distance, neighbour);
            if(earlyAbandon) {
                limit = prunedMap.lastKey();
            }
            comparisonTimer.add(distanceMeasurementTime);
            comparisonTimer.stop();
        }

        public double[] predict() {
            predictTimer.resetAndStart();
            final PrunedMultimap<Double, Instance> nearestNeighbourMap = prunedMap;
            final Random random = null; //getRandom();
            final Logger logger = getLogger();
            final double[] distribution = new double[instance.numClasses()];
            if(nearestNeighbourMap.isEmpty()) {
                logger.info("no neighbours available, random guessing");
                distribution[random.nextInt(distribution.length)]++;
            } else {
                for(final Double key : nearestNeighbourMap.keys()) {
                    for(final Instance nearestNeighbour : nearestNeighbourMap.get(key)) {
                        distribution[(int) nearestNeighbour.classValue()]++;
                    }
                }
                ArrayUtilities.normaliseInPlace(distribution);
            }
            predictTimer.stop();
            return distribution;
        }

        public long getTimeInNanos() {
            // the time taken to find the nearest neighbours and make a prediction for the target instance
            return predictTimer.getTime() + comparisonTimer.getTime();
        }

        public double getLimit() {
            return limit;
        }
    }

    @Override public StopWatch getTrainTimer() {
        return trainTimer;
    }

    @Override public StopWatch getTestTimer() {
        return testTimer;
    }

    @Override public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    @Override public Checkpointer getCheckpointer() {
        return checkpointer;
    }

    public long getTrainTimeLimit() {
        return trainTimeLimit;
    }

    public void setTrainTimeLimit(final long trainTimeLimit) {
        this.trainTimeLimit = trainTimeLimit;
    }

    public long getTestTimeLimit() {
        return testTimeLimit;
    }

    public void setTestTimeLimit(final long testTimeLimit) {
        this.testTimeLimit = testTimeLimit;
    }
}
