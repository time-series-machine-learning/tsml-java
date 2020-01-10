package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.IncClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.distance_based.distances.AbstractDistanceMeasure;
import tsml.classifiers.distance_based.knn.neighbour_iteration.LinearNeighbourIteratorBuilder;
import tsml.filters.IndexFilter;
import utilities.*;
import utilities.cache.Cache;
import utilities.cache.SymmetricCache;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.TimeUnit;

public class KnnLoocv
    extends Knn implements TrainTimeContractable,
                           Checkpointable,
        IncClassifier {

    public static final String NEIGHBOUR_LIMIT_FLAG = "n";
    public static final String NEIGHBOUR_ITERATION_STRATEGY_FLAG = "s";
    //    public static final String CACHE_FLAG = "c";
    protected long trainTimeLimitNanos = -1;
    protected List<NeighbourSearcher> searchers;
    protected long previousNeighbourBatchTimeNanos = 0;
    protected int neighbourLimit = -1;
    protected int neighbourCount = 0;
    protected StopWatch trainEstimateTimer = new StopWatch();
    protected Cache<Instance, Instance, Double> cache;
    protected Iterator<NeighbourSearcher> iterator;
    protected NeighbourIteratorBuilder neighbourIteratorBuilder = new LinearNeighbourIteratorBuilder(this);
    protected boolean trainEstimateChange = false;

    public KnnLoocv() {
        setAbleToEstimateOwnPerformance(true);
    }

    public KnnLoocv(DistanceFunction df) {
        super(df);
        setAbleToEstimateOwnPerformance(true);
    }

    @Override
    public void setTrainTimeLimit(long nanos) {
        trainTimeLimitNanos = nanos;
    }

    public List<NeighbourSearcher> getSearchers() {
        return searchers;
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

    public boolean hasNextBuildTick() throws Exception {
        trainTimer.checkPaused();
        trainEstimateTimer.resume();
        memoryWatcher.resume();
        boolean result = estimateOwnPerformance && hasNextUnlimitedTrainTime() && hasNextTrainTimeLimit();
        trainEstimateTimer.pause();
        memoryWatcher.pause();
        return result;
    }

    public long predictNextTrainTimeNanos() {
        return previousNeighbourBatchTimeNanos;
    }

    public void nextBuildTick() throws Exception {
        trainTimer.checkPaused();
        trainEstimateTimer.resume();
        memoryWatcher.resume();
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
        memoryWatcher.pause();
    }

    public NeighbourIteratorBuilder getNeighbourIteratorBuilder() {
        return neighbourIteratorBuilder;
    }

    public void setNeighbourIteratorBuilder(NeighbourIteratorBuilder neighbourIteratorBuilder) {
        this.neighbourIteratorBuilder = neighbourIteratorBuilder;
    }

    public interface NeighbourIteratorBuilder {
        Iterator<NeighbourSearcher> build();
    }

    @Override public ParamSet getParams() {
        return super.getParams().add(NEIGHBOUR_ITERATION_STRATEGY_FLAG, neighbourIteratorBuilder).add(NEIGHBOUR_LIMIT_FLAG, neighbourLimit).addAll(TrainTimeContractable.super.getParams());
    }

    @Override public void setParams(final ParamSet params) {
        super.setParams(params);
        ParamHandler.setParam(params, NEIGHBOUR_LIMIT_FLAG, this::setNeighbourLimit, Integer.class);
        ParamHandler.setParam(params, NEIGHBOUR_ITERATION_STRATEGY_FLAG, this::setNeighbourIteratorBuilder,
                              NeighbourIteratorBuilder.class);
        TrainTimeContractable.super.setParams(params);
    }

    protected void loadFromCheckpoint() throws Exception {
        trainEstimateTimer.pause();
        super.loadFromCheckpoint();
        trainEstimateTimer.resume();
    }

    public void startBuild(Instances data) throws Exception { // todo watch mem
        trainTimer.resume();
        memoryWatcher.resume();
        if(rebuild) {
            loadFromCheckpoint();
            trainTimer.pause();
            memoryWatcher.pause();
            super.buildClassifier(data);
            memoryWatcher.resumeAnyway();
            trainTimer.pauseAnyway();
            trainEstimateTimer.resetAndResume();
            rebuild = false;
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
                iterator = neighbourIteratorBuilder.build();
                trainEstimateChange = true; // build the first train estimate irrelevant of any progress made
            }
        }
        trainTimer.pauseAnyway();
        trainEstimateTimer.pauseAnyway();
        memoryWatcher.pause();
    }

    public void finishBuild() throws Exception {
        trainTimer.checkPaused();
        if(trainEstimateChange) {
            // todo make sure train timer is paused here + other timings checks
            trainEstimateTimer.resume();
            memoryWatcher.resume();
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
            memoryWatcher.pause();
            trainResults.setDetails(this, trainData);
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            trainResults.setBuildTime(trainEstimateTimer.getTimeNanos());
            trainResults.setBuildPlusEstimateTime(trainEstimateTimer.getTimeNanos() + trainTimer.getTimeNanos());
        }
    }

    public long getTrainTimeNanos() {
        return trainEstimateTimer.getTimeNanos() + trainTimer.getTimeNanos();
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
