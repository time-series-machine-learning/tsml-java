package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.*;
import tsml.classifiers.distance_based.distances.Dtw;
import utilities.*;
import utilities.collections.PrunedMultimap;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static experiments.data.DatasetLoading.sampleGunPoint;
import static tsml.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_FUNCTION_FLAG;

public class Knn extends EnhancedAbstractClassifier implements Checkpointable, MemoryWatchable, TrainTimeable {

    private static final long serialVersionUID = 0;
    protected transient Instances trainData;
    public static final String K_FLAG = "k";
    public static final String EARLY_ABANDON_FLAG = "e";
    public static final String RANDOM_TIE_BREAK_FLAG = "r";
    protected int k = 1;
    protected boolean earlyAbandon = true;
    protected DistanceFunction distanceFunction = new Dtw(0);
    protected StopWatch trainTimer = new StopWatch();
    protected MemoryWatcher memoryWatcher = new MemoryWatcher();
    protected boolean randomTieBreak = false;
    protected transient long minCheckpointIntervalNanos = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    protected transient long lastCheckpointTimeStamp = 0;
    protected transient String savePath = null;
    protected transient String loadPath = null;
    protected transient boolean skipFinalCheckpoint = false;
    private boolean rebuild = true; // shadows super

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
        return super.getParams()
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

    @Override
    public void setRebuild(boolean rebuild) {
        this.rebuild = rebuild;
        super.setRebuild(rebuild);
    }

    @Override public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        boolean loadedFromCheckpoint = loadFromCheckpoint();
        if(rebuild) {
            memoryWatcher.reset();
            trainTimer.reset();
        }
        memoryWatcher.enable();
        trainTimer.enable();
        super.buildClassifier(trainData);
        built = false;
        distanceFunction.setInstances(trainData);
        this.trainData = trainData;
        rebuild = false;
        built = true;
        trainTimer.disable();
        memoryWatcher.cleanup();
        memoryWatcher.disable();
        if(!loadedFromCheckpoint) {
            saveToCheckpoint();
        } else {
            logger.info("loaded from checkpoint so not overwriting");
        }
    }

    // todo fail capabilities

    public class NeighbourSearcher implements Serializable {
        private final PrunedMultimap<Double, Instance> prunedMap;
        private final Instance instance;
        private double limit = Double.POSITIVE_INFINITY;
        private StopWatch comparisonTimer = new StopWatch();
        private StopWatch predictTimer = new StopWatch();

        public Instance getInstance() {
            return instance;
        }

        public NeighbourSearcher(Instance instance) {
            this.prunedMap =
                new PrunedMultimap<>(((Comparator<Double> & Serializable) Double::compare));
            prunedMap.setSoftLimit(k);
            this.instance = instance;
        }

        public double add(Instance neighbour) {
            final long timeStamp = System.nanoTime();
            final double distance = distanceFunction.distance(this.instance, neighbour, limit);
            add(neighbour, distance, System.nanoTime() - timeStamp);
            return distance;
        }

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
            final double[] distribution = new double[instance.numClasses()];
            if(nearestNeighbourMap.isEmpty()) {
                distribution[rand.nextInt(distribution.length)]++;
            } else {
                for(final Double key : nearestNeighbourMap.keys()) {
                    for(final Instance nearestNeighbour : nearestNeighbourMap.get(key)) {
                        distribution[(int) nearestNeighbour.classValue()]++; // todo weight by distance
                        if(!randomTieBreak) {
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

