package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.*;
import tsml.classifiers.distance_based.distances.Dtw;
import utilities.*;
import utilities.collections.PrunedMultimap;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import utilities.stopwatch.StopWatch;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;

import static experiments.data.DatasetLoading.sampleGunPoint;
import static tsml.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_FUNCTION_FLAG;

/**
 * k-nearest-neighbour classifier.
 *
 * Change history:
 *  27/2/20 - goastler - refactored to include multiple distance measures
 */
public class Knn extends EnhancedAbstractClassifier implements Checkpointable, GcMemoryWatchable,
                                                               StopWatchTrainTimeable, Trainable, TestSeedable {

    private static String getKFlag() {
        return "k";
    }

    private static String getEarlyAbandonFlag() {
        return "e";
    }

    private static String getRandomTieBreakFlag() {
        return "r";
    }

    protected void setBuilt(final boolean built) {
        this.built = built;
    }

    // serialisation id
    private static final long serialVersionUID = 0;
    // train data (we won't store this in serialisation because it causes mega bloat!)
    private transient Instances trainData;
    // k, the number of neighbours to use
    private int k = 1;
    // whether to early abandon on distance measure calculations
    private boolean earlyAbandon = true;
    // the distance function
    private DistanceFunction distanceFunction = new Dtw(0);
    // track the train time
    private StopWatch trainTimer = new StopWatch();
    // track the memory
    private MemoryWatcher memoryWatcher = new MemoryWatcher();
    // min time between checkpointing
    private transient long minCheckpointIntervalNanos = Checkpointable.DEFAULT_MIN_CHECKPOINT_INTERVAL;
    // last timestamp of checkpoint
    private transient long lastCheckpointTimeStamp = 0;
    // path to save checkpoints to
    private transient String savePath = null;
    // path to load checkpoints from
    private transient String loadPath = null;
    // whether we should skip the final checkpoint
    private transient boolean skipFinalCheckpoint = false;
    // whether we're rebuilding the classifier or not
    private boolean rebuild = true; // shadows super
    // whether the classifier is built
    private boolean built = false;

    @Override
    public boolean isSkipFinalCheckpoint() {
        return skipFinalCheckpoint;
    }

    @Override
    public void setSkipFinalCheckpoint(boolean skipFinalCheckpoint) {
        this.skipFinalCheckpoint = skipFinalCheckpoint;
    }

    @Override
    public String getSavePath() {
        return savePath;
    }

    @Override
    public boolean setSavePath(String path) {
        boolean result = Checkpointable.super.setSavePath(path);
        if(result) {
            savePath = StrUtils.asDirPath(path);
        } else {
            savePath = null;
        }
        return result;
    }

    @Override public String getLoadPath() {
        return loadPath;
    }

    @Override public boolean setLoadPath(final String path) {
        boolean result = Checkpointable.super.setLoadPath(path);
        if(result) {
            loadPath = StrUtils.asDirPath(path);
        } else {
            loadPath = null;
        }
        return result;
    }

    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    public Instances getTrainData() {
        return trainData;
    }

    public long getLastCheckpointTimeStamp() {
        return lastCheckpointTimeStamp;
    }

    public boolean saveToCheckpoint() throws Exception {
        trainTimer.suspend();
        memoryWatcher.suspend();
        boolean result = CheckpointUtils.saveToSingleCheckpoint(this, logger, built && !skipFinalCheckpoint);
        memoryWatcher.unsuspend();
        trainTimer.unsuspend();
        return result;
    }

    public boolean loadFromCheckpoint() {
        trainTimer.suspend();
        memoryWatcher.suspend();
        boolean result = CheckpointUtils.loadFromSingleCheckpoint(this, logger);
        lastCheckpointTimeStamp = System.nanoTime();
        memoryWatcher.unsuspend();
        trainTimer.unsuspend();
        return result;
    }

    public void setMinCheckpointIntervalNanos(final long nanos) {
        minCheckpointIntervalNanos = nanos;
    }

    public long getMinCheckpointIntervalNanos() {
        return minCheckpointIntervalNanos;
    }

    @Override public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }


    @Override public ParamSet getParams() {
        return super.getParams()
                    .add(getEarlyAbandonFlag(), earlyAbandon)
                    .add(getKFlag(), k)
                    .add(DISTANCE_FUNCTION_FLAG, distanceFunction);
    }

    @Override public void setParams(final ParamSet params) {
        ParamHandler.setParam(params, DISTANCE_FUNCTION_FLAG, this::setDistanceFunction, DistanceFunction.class);
        ParamHandler.setParam(params, getKFlag(), this::setK, Integer.class);
        ParamHandler.setParam(params, getEarlyAbandonFlag(), this::setEarlyAbandon, Boolean.class);
    }

    @Override
    public void setRebuild(boolean rebuild) {
        this.rebuild = rebuild;
        super.setRebuild(rebuild);
    }

    @Override public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
    }

    public Knn() {
        super(false);
    }

    public Knn(DistanceFunction df) {
        this();
        setDistanceFunction(df);
    }

    public boolean isEarlyAbandon() {
        return earlyAbandon;
    }

    public void setEarlyAbandon(final boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        boolean loadedFromCheckpoint = loadFromCheckpoint();
        memoryWatcher.enable();
        trainTimer.enable();
        if(rebuild) {
            memoryWatcher.resetAndEnable();
            trainTimer.resetAndEnable();
            built = false;
            rebuild = false;
        }
        super.buildClassifier(trainData);
        distanceFunction.setInstances(trainData);
        this.trainData = trainData;
        built = true;
        trainTimer.disable();
        memoryWatcher.disable();
        if(!loadedFromCheckpoint) {
            saveToCheckpoint();
        } else {
            logger.info("loaded from checkpoint so not overwriting");
        }
    }

    @Override
    public boolean isFullyTrained() { // todo not sure if we need this
        return built;
    }

    // todo fail capabilities + make sure class val at end

    /**
     * NeighbourSearcher class to find the set of nearest neighbours for a given instance.
     */
    public class NeighbourSearcher implements Serializable {
        // map of distance to instances, allowing multiple instances to have the same distance
        private final PrunedMultimap<Double, Instance> prunedMap;
        // the target instance we're trying to find the closest neighbour to
        private final Instance instance;
        // distance limit if we're early abandoning
        private double limit = Double.POSITIVE_INFINITY;
        // timer to record comparison time
        private StopWatch comparisonTimer = new StopWatch();
        // timer to record the prediction time
        private StopWatch predictTimer = new StopWatch();

        public Instance getInstance() {
            return instance;
        }

        public NeighbourSearcher(Instance instance) {
            this.prunedMap =
                new PrunedMultimap<>(((Comparator<Double> & Serializable) Double::compare));
            // set the map to look for the k closest neighbours but keep neighbours which draw (e.g. both have a
            // distance of 2, say
            prunedMap.setSoftLimit(k);
            this.instance = instance;
        }

        // add an instance, finding the distance between the target instance and the given instance
        public double add(Instance neighbour) {
            StopWatch timer = StopWatch.newStopWatchEnabled();
            final double distance = distanceFunction.distance(this.instance, neighbour, limit);
            timer.disable();
            add(neighbour, distance, timer.getTimeNanos());
            return distance;
        }

        // add an instance given a precomputed distance and corresponding time it took to find that distance
        public void add(Instance neighbour, double distance, long distanceMeasurementTime) {
            comparisonTimer.enable();
            prunedMap.put(distance, neighbour);
            if(earlyAbandon) {
                limit = prunedMap.lastKey();
            }
            comparisonTimer.add(distanceMeasurementTime);
            comparisonTimer.disable();
        }

        public double[] predict() {
            predictTimer.resetAndEnable();
            final PrunedMultimap<Double, Instance> nearestNeighbourMap = prunedMap;
            final Random random = getRand();
            final double[] distribution = new double[instance.numClasses()];
            if(nearestNeighbourMap.isEmpty()) {
                logger.info("no neighbours available, random guessing");
                distribution[random.nextInt(distribution.length)]++;
            } else {
                for(final Double key : nearestNeighbourMap.keys()) {
                    for(final Instance nearestNeighbour : nearestNeighbourMap.get(key)) {
                        distribution[(int) nearestNeighbour.classValue()]++; // todo weight by distance
                    }
                }
                ArrayUtilities.normaliseInPlace(distribution);
            }
            predictTimer.disable();
            return distribution;
        }

        public long getTimeInNanos() {
            // the time taken to find the nearest neighbours and make a prediction for the target instance
            return predictTimer.getTimeNanos() + comparisonTimer.getTimeNanos();
        }

        public double getLimit() {
            return limit;
        }
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                     Exception {
        final NeighbourSearcher searcher = new NeighbourSearcher(testInstance);
        for(final Instance trainInstance : trainData) {
            searcher.add(trainInstance);
        }
        return searcher.predict();
    }

    @Override public double classifyInstance(final Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }

    public int getK() {
        return k;
    }

    public void setK(final int k) {
        this.k = k;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public void setDistanceFunction(final DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }

    public static void main(String[] args) throws Exception {
        int seed = 0;
        Instances[] data = sampleGunPoint(seed);
        Instances trainData = data[0];
        Knn classifier = new Knn(new Dtw(trainData.numAttributes() - 1));
        classifier.setSeed(0);
        ClassifierResults results = ClassifierTools.trainAndTest(data, classifier);
        System.out.println(results.writeSummaryResultsToString());
    }

}

