package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.ProgressiveBuildClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.distance_based.distances.AbstractDistanceMeasure;
import tsml.classifiers.distance_based.knn.neighbour_iteration.LinearNeighbourIterationStrategy;
import tsml.filters.IndexFilter;
import utilities.*;
import utilities.cache.Cache;
import utilities.cache.SymmetricCache;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class KNNCV
    extends KNN implements TrainTimeContractable,
                           Checkpointable,
                           ProgressiveBuildClassifier {
    public KNNCV() {
        setAbleToEstimateOwnPerformance(true);
    }

    public KNNCV(DistanceFunction df) {
        super(df);
        setAbleToEstimateOwnPerformance(true);
    }

    @Override
    public void setTrainTimeLimit(TimeUnit time, long amount) {
        trainTimeLimitNanos = TimeUnit.NANOSECONDS.convert(amount, time);
    }

    @Override
    public boolean setSavePath(String path) {
        if(path == null) {
            return false;
        }
        checkpointDirPath = StrUtils.asDirPath(path);
        return true;
    }

    public void checkpoint(boolean force) {
        trainEstimateTimer.pause();
        if(isCheckpointing() && (force || lastCheckpointTimeStamp + minCheckpointIntervalNanos < System.nanoTime())) {
            try {
                saveToFile(checkpointDirPath + tempCheckpointFileName);
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
            boolean success = new File(checkpointDirPath + tempCheckpointFileName).renameTo(new File(checkpointDirPath + checkpointFileName));
            if(!success) {
                throw new IllegalStateException("could not rename checkpoint file");
            }
            lastCheckpointTimeStamp = System.nanoTime();
        }
        trainEstimateTimer.resume();
    }

    public void checkpoint() {
        checkpoint(false);
    }

    public boolean isCheckpointing() {
        return checkpointDirPath != null;
    }

    public long getMinCheckpointIntervalNanos() {
        return minCheckpointIntervalNanos;
    }

    public void setMinCheckpointIntervalNanos(final long minCheckpointInterval) {
        this.minCheckpointIntervalNanos = minCheckpointInterval;
    }

    public List<NeighbourSearcher> getSearchers() {
        return searchers;
    }

    public static final String NEIGHBOUR_LIMIT_FLAG = "n";
    public static final String NEIGHBOUR_ITERATION_STRATEGY_FLAG = "s";
//    public static final String CACHE_FLAG = "c";
    public static final String checkpointFileName = "checkpoint.ser";
    public static final String tempCheckpointFileName = checkpointFileName + ".tmp";
    protected long lastCheckpointTimeStamp = 0;
    protected long minCheckpointIntervalNanos = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    protected String checkpointDirPath;
    protected long trainTimeLimitNanos = -1;
    protected List<NeighbourSearcher> searchers;
    protected long previousNeighbourBatchTimeNanos = 0;
    protected int neighbourLimit = -1;
    protected int neighbourCount = 0;
    protected StopWatch trainEstimateTimer = new StopWatch();
    protected Cache<Instance, Instance, Double> cache;
    protected Iterator<NeighbourSearcher> iterator;
    protected NeighbourIterationStrategy neighbourIterationStrategy = new LinearNeighbourIterationStrategy();
    protected boolean trainEstimateChange = false;

    @Override
    public String getSavePath() {
        return checkpointDirPath;
    }

    public boolean hasNextTrainTimeLimit() {
        return (trainTimeLimitNanos < 0 || trainEstimateTimer.getTimeNanos() + previousNeighbourBatchTimeNanos < trainTimeLimitNanos);
    }

    public boolean hasNextNeighbour() {
        return iterator.hasNext();
    }

    public boolean hasNextNeighbourLimit() {
        return (neighbourCount < neighbourLimit || neighbourLimit < 0);
    }

    public boolean hasNextUnlimitedTrainTime() {
        return hasNextNeighbour() && hasNextNeighbourLimit();
    }

    public boolean hasNextBuildTick() {
        return hasNextUnlimitedTrainTime() && hasNextTrainTimeLimit();
    }

    public long predictNextTrainTimeNanos() {
        return previousNeighbourBatchTimeNanos;
    }

    public void nextBuildTick() {
        trainEstimateTimer.resume();
        trainEstimateChange = true;
        long timeStamp = System.nanoTime();
        NeighbourSearcher current = iterator.next();
        iterator.remove();
        neighbourCount++;
        for(int i = 0; i < searchers.size(); i++) {
            NeighbourSearcher searcher = searchers.get(i);
            if(!current.getInstance().equals(searcher.getInstance())) {
                // todo loocv issue with cache GO
                long distanceMeasurementTimeStamp = System.nanoTime();
                Double cachedDistance = cache.get(searcher.getInstance(), current.getInstance());
                if(cachedDistance == null) {
                    double distance = searcher.add(current.getInstance());
                    cache.put(searcher.getInstance(), current.getInstance(), distance);
                } else {
                    cache.remove(searcher.getInstance(), current.getInstance());
                    searcher.add(current.getInstance(), cachedDistance, System.nanoTime() - distanceMeasurementTimeStamp);
                }
            }
        }
        previousNeighbourBatchTimeNanos = System.nanoTime() - timeStamp;
        checkpoint();
        trainEstimateTimer.pause();
    }

    public NeighbourIterationStrategy getNeighbourIterationStrategy() {
        return neighbourIterationStrategy;
    }

    public void setNeighbourIterationStrategy(NeighbourIterationStrategy neighbourIterationStrategy) {
        this.neighbourIterationStrategy = neighbourIterationStrategy;
    }

    public interface NeighbourIterationStrategy {
        Iterator<NeighbourSearcher> build(KNNCV knn);
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        super.setOptions(options);
        TrainTimeContractable.super.setOptions(options);
        StrUtils.setOption(options, NEIGHBOUR_LIMIT_FLAG, this::setNeighbourLimit, Integer::parseInt);
        StrUtils.setOption(options, NEIGHBOUR_ITERATION_STRATEGY_FLAG, this::setNeighbourIterationStrategy, NeighbourIterationStrategy.class);
//        StringUtilities.setOption(options, CACHE_FLAG, this::setCache, Cache.class);
    }

    @Override
    public String[] getOptions() {
        ArrayList<String> options = new ArrayList<>();
        StrUtils.addOption(NEIGHBOUR_LIMIT_FLAG, options, neighbourLimit);
        StrUtils.addOption(NEIGHBOUR_ITERATION_STRATEGY_FLAG, options, neighbourIterationStrategy);
//        StringUtilities.addOption(CACHE_FLAG, options, cache);
        Collections.addAll(options, super.getOptions());
        Collections.addAll(options, TrainTimeContractable.super.getOptions());
        return options.toArray(new String[0]);
    }

    protected void loadFromCheckpoint() {
        if(isCheckpointing() && isRebuild()) {
            trainEstimateTimer.pause();
            try {
                loadFromFile(checkpointDirPath + checkpointFileName);
                setRebuild(false);
            } catch (Exception e) {

            }
            trainEstimateTimer.resume();
        }
    }

    public void startBuild(Instances data) throws Exception {
        if(isRebuild()) {
            super.buildClassifier(data);
            if(getEstimateOwnPerformance()) {
                if(isCheckpointing()) {
                    IndexFilter.hashifyInstances(data);
                }
                // build a progressive leave-one-out-cross-validation
                searchers = new ArrayList<>(data.size());
                // build a neighbour searcher for every train instance
                for(int i = 0; i < data.size(); i++) {
                    NeighbourSearcher searcher = new NeighbourSearcher(data.get(i));
                    searchers.add(i, searcher);
                }
                if(distanceFunction instanceof AbstractDistanceMeasure) { // todo cached version of dist meas
                    if(((AbstractDistanceMeasure) distanceFunction).isSymmetric()) {
                        cache = new SymmetricCache<>();
                    } else {
                        cache = new Cache<>();
                    }
                }
                iterator = neighbourIterationStrategy.build(this);
            }
        }
    }

    public void finishBuild() throws Exception {
        if(trainEstimateChange) {
            trainEstimateChange = false;
            if(neighbourLimit >= 0 && neighbourCount < neighbourLimit || neighbourLimit < 0 && neighbourCount < trainData.size()) {
                throw new IllegalStateException("this should not happen");
            }
            // populate train results
            trainResults = new ClassifierResults();
            for(NeighbourSearcher searcher : searchers) {
                double[] distribution = searcher.predict();
                int prediction = ArrayUtilities.argMax(distribution);
                long time = searcher.getTimeNanos();
                double trueClassValue = searcher.getInstance().classValue();
                trainResults.addPrediction(trueClassValue, distribution, prediction, time, null);
            }
            trainEstimateTimer.pause();
            trainResults.setDetails(this, trainData);
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            trainResults.setBuildTime(trainEstimateTimer.getTimeNanos());
            trainResults.setBuildPlusEstimateTime(trainEstimateTimer.getTimeNanos() + buildTimer.getTimeNanos());
        }
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainEstimateTimer.resume();
        loadFromCheckpoint();
        startBuild(data);
        // if we're generating an estimate of the train data performance
        if(getEstimateOwnPerformance()) { // todo make this generic in progressive build classifier
            // while we've got time / neighbours remaining, add neighbours to the searchers
            while(hasNextBuildTick()) {
                nextBuildTick();
            }
            finishBuild();
        }
        checkpoint(true);
        trainEstimateTimer.pause();
    }

    public long getTrainTimeNanos() {
        return trainEstimateTimer.getTimeNanos();
    }

    public long getTrainTimeLimitNanos() {
        return trainTimeLimitNanos;
    }

    @Override
    public void setTrainTimeLimitNanos(long trainTimeLimit) {
        this.trainTimeLimitNanos = trainTimeLimit;
    }

    public int getNeighbourLimit() {
        return neighbourLimit;
    }

    public void setNeighbourLimit(int neighbourLimit) {
        this.neighbourLimit = neighbourLimit;
    }

    public Cache<Instance, Instance, Double> getCache() {
        return cache;
    }

    public void setCache(Cache<Instance, Instance, Double> cache) {
        this.cache = cache;
    }

}
