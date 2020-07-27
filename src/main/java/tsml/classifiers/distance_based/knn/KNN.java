package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.*;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrain;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.classifiers.Rebuildable;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import utilities.*;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.logging.Logger;

/**
 * k-nearest-neighbour classifier.
 *
 * Contributors: goastler
 */
public class KNN extends BaseClassifier implements Rebuildable, Checkpointable, WatchedMemory,
    TimedTrain {

    /**
     * flag for k variable. This is used in representing parameters in the form of a string.
     * @return
     */
    public static String getKFlag() {
        return "k";
    }

    /**
     * flag for early abandon variable. This is used in representing parameters in the form of a string.
     * @return
     */
    public static String getEarlyAbandonFlag() {
        return "e";
    }

    /**
     * flag for random tie break variable. This is used in representing parameters in the form of a string.
     * @return
     */
    public static String getRandomTieBreakFlag() {
        return "r";
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
    private DistanceFunction distanceFunction = new DTWDistance();
    // track the train time
    private StopWatch trainTimer = new StopWatch();
    // track the memory
    private MemoryWatcher memoryWatcher = new MemoryWatcher();
    // min time between checkpointing
    private transient long minCheckpointIntervalNanos = 0;//Checkpointable.DEFAULT_MIN_CHECKPOINT_INTERVAL;
    // last timestamp of checkpoint
    private transient long lastCheckpointTimeStamp = 0;
    // path to save checkpoints to
    private transient String savePath = null;
    // path to load checkpoints from
    private transient String loadPath = null;
    // whether we should skip the final checkpoint
    private transient boolean skipFinalCheckpoint = false;
    // whether to random tie break (defaults to true / yes and drawing neighbours are put into a majority vote)
    private boolean randomTieBreak = true;

    public boolean isSkipFinalCheckpoint() {
        return skipFinalCheckpoint;
    }

    public void setSkipFinalCheckpoint(boolean skipFinalCheckpoint) {
        this.skipFinalCheckpoint = skipFinalCheckpoint;
    }

    public String getSavePath() {
        return savePath;
    }

    @Override
    public boolean setCheckpointPath(String path) {
        boolean result = Checkpointable.super.createDirectories(path);
        if(result) {
            savePath = StrUtils.asDirPath(path);
        } else {
            savePath = null;
        }
        return result;
    }

    @Override public void copyFromSerObject(final Object obj) throws Exception {

    }

    public String getLoadPath() {
        return loadPath;
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

    public boolean checkpointIfIntervalExpired() throws Exception {
//        trainTimer.suspend();
//        memoryWatcher.suspend();
//        boolean result = CheckpointUtils.saveToSingleCheckpoint(this, getLogger(),
//            isBuilt()
//            &&
//                                                                !skipFinalCheckpoint);
//        memoryWatcher.unsuspend();
//        trainTimer.unsuspend(); todo fix
//        return result;
        return true;
    }

    public boolean loadCheckpoint() {
//        trainTimer.suspend(); // todo interface for this
//        memoryWatcher.suspend();
//        boolean result = CheckpointUtils.loadFromSingleCheckpoint(this, getLogger());
//        memoryWatcher.unsuspend(); todo fix
//        trainTimer.unsuspend();
//        return result;
        return true;
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
                    .add(getRandomTieBreakFlag(), randomTieBreak)
                    .add(DistanceMeasure.DISTANCE_MEASURE_FLAG, distanceFunction);
    }

    @Override public void setParams(final ParamSet params) throws Exception {
        ParamHandlerUtils
                .setParam(params, DistanceMeasure.DISTANCE_MEASURE_FLAG, this::setDistanceFunction, DistanceFunction.class);
        ParamHandlerUtils.setParam(params, getKFlag(), this::setK, Integer.class);
        ParamHandlerUtils.setParam(params, getEarlyAbandonFlag(), this::setEarlyAbandon, Boolean.class);
        ParamHandlerUtils.setParam(params, getRandomTieBreakFlag(), this::setRandomTieBreak, Boolean.class);
    }

    public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
    }

    public KNN() {
        super(false);
    }

    public KNN(DistanceFunction df) {
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
        // load from a previous checkpoint
        boolean loadedFromCheckpoint = loadCheckpoint();
        // enable resource monitors
        memoryWatcher.start(); // todo can we share emitters in mem watcher?
        trainTimer.start();
        final boolean rebuild = isRebuild();
        if(rebuild) {
            // if we're rebuilding then reset resource monitors
            memoryWatcher.resetAndStart();
            trainTimer.resetAndStart();
        }
        // build parent
        super.buildClassifier(trainData);
        // let the distance function know about the instances
        distanceFunction.setInstances(trainData);
        // save our model data
        this.trainData = trainData;
        // we're fully built now
//        setBuilt(true);
        // disable resource monitors
        trainTimer.stop();
        memoryWatcher.stop();
        // save checkpoint unless we loaded from checkpoint
        if(!loadedFromCheckpoint) {
            checkpointIfIntervalExpired();
        } else {
            // if we've loaded from checkpoint then there's no point in re saving a checkpoint as we haven't done any
            // further work
            getLogger().info("loaded from checkpoint so saving checkpoint");
        }
    }

    public boolean isRandomTieBreak() {
        return randomTieBreak;
    }

    public KNN setRandomTieBreak(boolean randomTieBreak) {
        this.randomTieBreak = randomTieBreak;
        return this;
    }

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
                ArrayUtilities.normalise(distribution);
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

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                     Exception {
        final NeighbourSearcher searcher = new NeighbourSearcher(testInstance);
        for(final Instance trainInstance : trainData) {
            searcher.add(trainInstance);
        }
        return searcher.predict();
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

}

