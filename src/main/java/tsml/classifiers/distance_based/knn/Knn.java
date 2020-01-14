package tsml.classifiers.distance_based.knn;

import tsml.classifiers.Checkpointable;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.IncClassifier;
import tsml.classifiers.RebuildableClassifier;
import tsml.classifiers.distance_based.distances.Dtw;
import utilities.*;
import utilities.collections.PrunedTreeMultiMap;
import utilities.collections.TreeMultiMap;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static tsml.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_FUNCTION_FLAG;

public class Knn extends EnhancedAbstractClassifier
    implements
    Checkpointable, ParamHandler {

    protected transient Instances trainData;
    public static final String K_FLAG = "k";
    public static final String EARLY_ABANDON_FLAG = "e";
    public static final String RANDOM_TIE_BREAK_FLAG = "r";
    protected int k = 1;
    protected boolean earlyAbandon = true;
    protected DistanceFunction distanceFunction = new Dtw();
    protected StopWatch trainTimer = new StopWatch();
    protected MemoryWatcher memoryWatcher = new MemoryWatcher();
    protected boolean randomTieBreak = false;
    protected boolean rebuild = true;
    protected long minCheckpointIntervalNanos = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    protected boolean ignorePreviousCheckpoints = false;
    protected long lastCheckpointTimeStamp = 0;
    protected String checkpointDirPath;
    public static final String checkpointFileName = "checkpoint.ser";
    public static final String tempCheckpointFileName = checkpointFileName + ".tmp";

    @Override
    public boolean setSavePath(String path) {
        if(path == null) {
            return false;
        }
        checkpointDirPath = StrUtils.asDirPath(path);
        return true;
    }

    @Override
    public String getSavePath() {
        return checkpointDirPath;
    }

    public void checkpoint(boolean force) throws Exception {
        trainTimer.suspend();
        memoryWatcher.suspend();
        if(isCheckpointing() && (force || lastCheckpointTimeStamp + minCheckpointIntervalNanos < System.nanoTime())) {
            String path = checkpointDirPath + tempCheckpointFileName;
            logger.log("saving checkpoint to: " + path);
            saveToFile(path);
            boolean success = new File(path).renameTo(new File(checkpointDirPath + checkpointFileName));
            if(!success) {
                throw new IllegalStateException("could not rename checkpoint file");
            }
            lastCheckpointTimeStamp = System.nanoTime();
        }
        trainTimer.unsuspend();
        memoryWatcher.unsuspend();
    }

    public void checkpoint() throws Exception {
        checkpoint(false);
    }

    protected void loadFromCheckpoint() throws Exception {
        trainTimer.suspend();
        memoryWatcher.suspend();
        if(!isIgnorePreviousCheckpoints() && isCheckpointing() && isRebuild()) {
            String path = checkpointDirPath + checkpointFileName;
            logger.log("loading from checkpoint: " + path);
            loadFromFile(path);
        }
        trainTimer.unsuspend();
        memoryWatcher.unsuspend();
    }

    @Override public boolean isIgnorePreviousCheckpoints() {
        return ignorePreviousCheckpoints;
    }

    @Override public void setIgnorePreviousCheckpoints(final boolean ignorePreviousCheckpoints) {
        this.ignorePreviousCheckpoints = ignorePreviousCheckpoints;
    }

    @Override public void setMinCheckpointIntervalNanos(final long nanos) {
        minCheckpointIntervalNanos = nanos;
    }

    @Override public long getMinCheckpointIntervalNanos() {
        return minCheckpointIntervalNanos;
    }

    public boolean isRebuild() {
        return rebuild;
    }

    public void setRebuild(boolean rebuild) {
        this.rebuild = rebuild;
    }

    public boolean isRandomTieBreak() {
        return randomTieBreak;
    }

    public void setRandomTieBreak(boolean randomTieBreak) {
        this.randomTieBreak = randomTieBreak;
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

    @Override public ParamSet getParams() {
        return ParamHandler.super.getParams()
                                 .add(EARLY_ABANDON_FLAG, earlyAbandon)
                                 .add(RANDOM_TIE_BREAK_FLAG, randomTieBreak)
                                 .add(K_FLAG, k)
                                 .add(DISTANCE_FUNCTION_FLAG, distanceFunction);
    }

    @Override public void setParams(final ParamSet params) {
        ParamHandler.setParam(params, DISTANCE_FUNCTION_FLAG, this::setDistanceFunction, DistanceFunction.class);
        ParamHandler.setParam(params, K_FLAG, this::setK, Integer.class);
        ParamHandler.setParam(params, RANDOM_TIE_BREAK_FLAG, this::setRandomTieBreak, Boolean.class);
        ParamHandler.setParam(params, EARLY_ABANDON_FLAG, this::setEarlyAbandon, Boolean.class);
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        if(rebuild) {
            trainTimer.enable();
            memoryWatcher.enable();
            super.buildClassifier(trainData);
            rebuild = false;
            distanceFunction.setInstances(trainData);
            this.trainData = trainData;
            checkpoint();
            trainTimer.disable();
            memoryWatcher.disable();
        }
    }

    // todo fail capabilities

    public class NeighbourSearcher implements Serializable {
        private final PrunedTreeMultiMap<Double, Instance> prunedMap;
        private final Instance instance;
        private double limit = Double.POSITIVE_INFINITY;
        private StopWatch comparisonTimer = new StopWatch();
        private StopWatch predictTimer = new StopWatch();

        public Instance getInstance() {
            return instance;
        }

        public NeighbourSearcher(Instance instance) {
            this.prunedMap = new PrunedTreeMultiMap<>(new TreeMultiMap<>((Comparator<Double> & Serializable) Double::compare));
            prunedMap.setLimit(k);
            this.instance = instance;
        }

        public double add(Instance neighbour) {
            long timeStamp = System.nanoTime();
            double distance = distanceFunction.distance(this.instance, neighbour, limit);
            add(neighbour, distance, System.nanoTime() - timeStamp);
            return distance;
        }

        public void add(Instance neighbour, double distance, long distanceMeasurementTime) {
            comparisonTimer.enable();
            prunedMap.add(distance, neighbour);
            if(earlyAbandon) {
                limit = prunedMap.getMap().lastEntry().getKey();
            }
            comparisonTimer.add(distanceMeasurementTime);
            comparisonTimer.disable();
        }

        public double[] predict() {
            predictTimer.resetAndEnable();
            TreeMultiMap<Double, Instance> nearestNeighbourMap = prunedMap.getMap();
            double[] distribution = new double[instance.numClasses()];
            if(nearestNeighbourMap.isEmpty()) {
                distribution[rand.nextInt(distribution.length)]++;
            } else {
                for(Map.Entry<Double, List<Instance>> entry : nearestNeighbourMap.entrySet()) {
                    for(Instance nearestNeighbour : entry.getValue()) {
                        distribution[(int) nearestNeighbour.classValue()]++;
                        if(!randomTieBreak) { // todo sort this mess out
                            break;
                        }
                    }
                    if(!randomTieBreak) {
                        break;
                    }
                }
                ArrayUtilities.normaliseInPlace(distribution);
            }
            predictTimer.disable();
            return distribution;
        }

        public long getTimeNanos() {
            return predictTimer.getTimeNanos() + comparisonTimer.getTimeNanos();
        }

        public double getLimit() {
            return limit;
        }
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                     Exception {
        NeighbourSearcher searcher = new NeighbourSearcher(testInstance);
        for(Instance trainInstance : trainData) {
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

