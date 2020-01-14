package tsml.classifiers.distance_based.knn;

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.IncClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.distance_based.distances.AbstractDistanceMeasure;
import tsml.classifiers.distance_based.knn.neighbour_iteration.LinearNeighbourIteratorBuilder;
import tsml.filters.IndexFilter;
import utilities.ArrayUtilities;
import utilities.MemoryWatcher;
import utilities.StopWatch;
import utilities.cache.Cache;
import utilities.cache.SymmetricCache;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.classifiers.Classifier;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Tmp extends Evaluator implements Checkpointable, TrainTimeContractable {

    public Tmp() {
        super(-1, false, false);
    }

    @Override public ClassifierResults evaluate(final Classifier classifier, final Instances dataset) throws Exception {
        if(classifier instanceof Knn) {
            throw new IllegalArgumentException("Cannot handle anything other than KNN");
        }
        return null; // todo
    }

    public static final String NEIGHBOUR_LIMIT_FLAG = "n";
    public static final String NEIGHBOUR_ITERATION_STRATEGY_FLAG = "s";
    //    public static final String CACHE_FLAG = "c";
    protected long trainTimeLimitNanos = -1;
    protected List<Knn.NeighbourSearcher> searchers;
    protected long previousNeighbourBatchTimeNanos = 0;
    protected int neighbourLimit = -1;
    protected int neighbourCount = 0;
    protected StopWatch trainEstimateTimer = new StopWatch();
    protected Cache<Instance, Instance, Double> cache;
    protected Iterator<Knn.NeighbourSearcher> iterator;
    protected KnnLoocv.NeighbourIteratorBuilder neighbourIteratorBuilder = new LinearNeighbourIteratorBuilder();
    protected boolean trainEstimateChange = false;
    protected StopWatch trainTimer = new StopWatch();
    protected MemoryWatcher memoryWatcher = new MemoryWatcher();
    protected Knn knn;
    protected boolean rebuild = true;
    protected ClassifierResults trainResults = new ClassifierResults();
    protected Instances trainData;

    @Override
    public void setTrainTimeLimit(long nanos) {
        trainTimeLimitNanos = nanos;
    }

    public List<Knn.NeighbourSearcher> getSearchers() {
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
        trainTimer.checkDisabled();
        trainEstimateTimer.enable();
        memoryWatcher.enable();
        boolean result = hasNextUnlimitedTrainTime() && hasNextTrainTimeLimit();
        trainEstimateTimer.disable();
        memoryWatcher.disable();
        return result;
    }

    public long predictNextTrainTimeNanos() {
        return previousNeighbourBatchTimeNanos;
    }

    public void nextBuildTick() throws Exception {
        trainTimer.checkDisabled();
        trainEstimateTimer.enable();
        memoryWatcher.enable();
        trainEstimateChange = true;
        long timeStamp = System.nanoTime();
        Knn.NeighbourSearcher current = iterator.next();
        iterator.remove();
        neighbourCount++;
        for(int i = 0; i < searchers.size(); i++) {
            Knn.NeighbourSearcher searcher = searchers.get(i);
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
        trainEstimateTimer.disable();
        memoryWatcher.disable();
    }

    public KnnLoocv.NeighbourIteratorBuilder getNeighbourIteratorBuilder() {
        return neighbourIteratorBuilder;
    }

    public void setNeighbourIteratorBuilder(KnnLoocv.NeighbourIteratorBuilder neighbourIteratorBuilder) {
        this.neighbourIteratorBuilder = neighbourIteratorBuilder;
    }

    public interface NeighbourIteratorBuilder {
        Iterator<Knn.NeighbourSearcher> build();
    }

    @Override public ParamSet getParams() {
        return knn.getParams().add(NEIGHBOUR_ITERATION_STRATEGY_FLAG, neighbourIteratorBuilder).add(NEIGHBOUR_LIMIT_FLAG, neighbourLimit).addAll(
            TrainTimeContractable.super.getParams());
    }

    @Override public void setParams(final ParamSet params) {
        knn.setParams(params);
        ParamHandler.setParam(params, NEIGHBOUR_LIMIT_FLAG, this::setNeighbourLimit, Integer.class);
        ParamHandler.setParam(params, NEIGHBOUR_ITERATION_STRATEGY_FLAG, this::setNeighbourIteratorBuilder,
                              KnnLoocv.NeighbourIteratorBuilder.class);
        TrainTimeContractable.super.setParams(params);
    }

    protected void loadFromCheckpoint() throws Exception {
        trainEstimateTimer.suspend();
        knn.loadFromCheckpoint();
        trainEstimateTimer.unsuspend();
    }

    public void startBuild(Instances data) throws Exception { // todo watch mem
        trainEstimateTimer.checkDisabled();
        trainTimer.enable();
        memoryWatcher.enable();
        if(rebuild) {
            loadFromCheckpoint();
            trainTimer.disableAnyway();
            memoryWatcher.disableAnyway();
            knn.buildClassifier(data);
            memoryWatcher.enableAnyway();
            trainTimer.disableAnyway();
            trainEstimateTimer.resetAndEnable();
            rebuild = false;
            if(isCheckpointing()) {
                IndexFilter.hashifyInstances(data);
            }
            // build a progressive leave-one-out-cross-validation
            searchers = new ArrayList<>(data.size());
            // build a neighbour searcher for every train instance
            for(int i = 0; i < data.size(); i++) {
                Knn.NeighbourSearcher searcher = knn.new NeighbourSearcher(data.get(i));
                searchers.add(i, searcher);
            }
            if(knn.distanceFunction instanceof AbstractDistanceMeasure) { // todo cached version of dist meas
                if(((AbstractDistanceMeasure) knn.distanceFunction).isSymmetric()) {
                    cache = new SymmetricCache<>();
                } else {
                    cache = new Cache<>();
                }
            }
            iterator = neighbourIteratorBuilder.build();
            trainEstimateChange = true; // build the first train estimate irrelevant of any progress made
        }
        trainTimer.disableAnyway();
        trainEstimateTimer.disableAnyway();
        memoryWatcher.disable();
    }

    public void finishBuild() throws Exception {
        trainTimer.checkDisabled();
        memoryWatcher.checkDisabled();
        trainEstimateTimer.checkDisabled();
        if(trainEstimateChange) {
            // todo make sure train timer is paused here + other timings checks
            trainEstimateTimer.enable();
            memoryWatcher.enable();
            trainEstimateChange = false;
            if(neighbourLimit >= 0 && neighbourCount < neighbourLimit || neighbourLimit < 0 && neighbourCount < trainData.size()) {
                throw new IllegalStateException("this should not happen");
            }
            // populate train results
            trainResults = new ClassifierResults();
            for(Knn.NeighbourSearcher searcher : searchers) {
                double[] distribution = searcher.predict();
                int prediction = ArrayUtilities.argMax(distribution);
                long time = searcher.getTimeNanos();
                double trueClassValue = searcher.getInstance().classValue();
                trainResults.addPrediction(trueClassValue, distribution, prediction, time, null);
            }
            trainEstimateTimer.disable();
            memoryWatcher.disable();
            // todo
//            trainResults.setDetails(this, trainData);
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            trainResults.setBuildTime(trainEstimateTimer.getTimeNanos());
            trainResults.setBuildPlusEstimateTime(trainEstimateTimer.getTimeNanos() + trainTimer.getTimeNanos());
        }
        trainData = null;
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
