package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.distances.ddtw.DDTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.knn.neighbour_iteration.LinearNeighbourIteratorBuilder;
import tsml.classifiers.distance_based.knn.neighbour_iteration.RandomNeighbourIteratorBuilder;
import tsml.classifiers.distance_based.knn.strategies.RLTunedKNNSetup;
import tsml.classifiers.distance_based.tuned.RLTunedClassifier;
import tsml.classifiers.distance_based.utils.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.stopwatch.StopWatch;
import tsml.classifiers.distance_based.utils.classifier_building.CompileTimeClassifierBuilderFactory;
import tsml.classifiers.distance_based.utils.iteration.LinearListIterator;
import tsml.classifiers.distance_based.utils.iteration.RandomListIterator;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.transformers.HashTransformer;
import utilities.*;
import tsml.classifiers.distance_based.utils.cache.Cache;
import tsml.classifiers.distance_based.utils.cache.SymmetricCache;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

public class KNNLOOCV
    extends KNN implements TrainTimeContractable {

    public static final Factory FACTORY = new Factory();
    public static final TunedFactory TUNED_FACTORY = new TunedFactory();

    public static class TunedFactory extends CompileTimeClassifierBuilderFactory<RLTunedClassifier> {
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_MSM_1NN_V1 = add(new SuppliedClassifierBuilder<>("TUNED_MSM_1NN_V1",
            TunedFactory::buildTunedMsm1nnV1));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_ERP_1NN_V1 = add(new SuppliedClassifierBuilder<>("TUNED_ERP_1NN_V1",
            TunedFactory::buildTunedErp1nnV1));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_LCSS_1NN_V1 = add(new SuppliedClassifierBuilder<>("TUNED_LCSS_1NN_V1",
            TunedFactory::buildTunedLcss1nnV1));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_TWED_1NN_V1 = add(new SuppliedClassifierBuilder<>("TUNED_TWED_1NN_V1",
            TunedFactory::buildTunedTwed1nnV1));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_DTW_1NN_V1 = add(new SuppliedClassifierBuilder<>("TUNED_DTW_1NN_V1",
            TunedFactory::buildTunedDtw1nnV1));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_DDTW_1NN_V1 = add(new SuppliedClassifierBuilder<>("TUNED_DDTW_1NN_V1",
            TunedFactory::buildTunedDdtw1nnV1));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_WDTW_1NN_V1 = add(new SuppliedClassifierBuilder<>("TUNED_WDTW_1NN_V1",
            TunedFactory::buildTunedWdtw1nnV1));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_WDDTW_1NN_V1 = add(new SuppliedClassifierBuilder<>("TUNED_WDDTW_1NN_V1",
            TunedFactory::buildTunedWddtw1nnV1));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_MSM_1NN_V2 = add(new SuppliedClassifierBuilder<>("TUNED_MSM_1NN_V2",
            TunedFactory::buildTunedMsm1nnV2));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_ERP_1NN_V2 = add(new SuppliedClassifierBuilder<>("TUNED_ERP_1NN_V2",
            TunedFactory::buildTunedErp1nnV2));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_LCSS_1NN_V2 = add(new SuppliedClassifierBuilder<>("TUNED_LCSS_1NN_V2",
            TunedFactory::buildTunedLcss1nnV2));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_TWED_1NN_V2 = add(new SuppliedClassifierBuilder<>("TUNED_TWED_1NN_V2",
            TunedFactory::buildTunedTwed1nnV2));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_DTW_1NN_V2 = add(new SuppliedClassifierBuilder<>("TUNED_DTW_1NN_V2",
            TunedFactory::buildTunedDtw1nnV2));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_DDTW_1NN_V2 = add(new SuppliedClassifierBuilder<>("TUNED_DDTW_1NN_V2",
            TunedFactory::buildTunedDdtw1nnV2));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_WDTW_1NN_V2 = add(new SuppliedClassifierBuilder<>("TUNED_WDTW_1NN_V2",
            TunedFactory::buildTunedWdtw1nnV2));
        public final ClassifierBuilder<? extends RLTunedClassifier> TUNED_WDDTW_1NN_V2 = add(new SuppliedClassifierBuilder<>("TUNED_WDDTW_1NN_V2",
            TunedFactory::buildTunedWddtw1nnV2));


        public static RLTunedClassifier buildTunedDtw1nnV1() {
            return buildTuned1nnV1(DistanceMeasureConfigs::buildDtwSpaceV1);
        }

        public static RLTunedClassifier buildTunedDdtw1nnV1() {
            return buildTuned1nnV1(DistanceMeasureConfigs::buildDdtwSpaceV1);
        }

        public static RLTunedClassifier buildTunedWdtw1nnV1() {
            return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildWdtwSpaceV1());
        }

        public static RLTunedClassifier buildTunedWddtw1nnV1() {
            return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildWddtwSpaceV1());
        }

        public static RLTunedClassifier buildTunedDtw1nnV2() {
            return buildTuned1nnV2(DistanceMeasureConfigs::buildDtwSpaceV2);
        }

        public static RLTunedClassifier buildTunedDdtw1nnV2() {
            return buildTuned1nnV2(DistanceMeasureConfigs::buildDdtwSpaceV2);
        }

        public static RLTunedClassifier buildTunedWdtw1nnV2() {
            return buildTuned1nnV2(i -> DistanceMeasureConfigs.buildWdtwSpaceV2());
        }

        public static RLTunedClassifier buildTunedWddtw1nnV2() {
            return buildTuned1nnV2(i -> DistanceMeasureConfigs.buildWddtwSpaceV2());
        }

        public static RLTunedClassifier buildTunedMsm1nnV1() {
            return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildMsmSpace());
        }

        public static RLTunedClassifier buildTunedTwed1nnV1() {
            return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildTwedSpace());
        }

        public static RLTunedClassifier buildTunedErp1nnV1() {
            return buildTuned1nnV1(DistanceMeasureConfigs::buildErpSpace);
        }

        public static RLTunedClassifier buildTunedLcss1nnV1() {
            return buildTuned1nnV1(DistanceMeasureConfigs::buildLcssSpace);
        }

        public static RLTunedClassifier buildTunedMsm1nnV2() {
            return buildTuned1nnV2(i -> DistanceMeasureConfigs.buildMsmSpace());
        }

        public static RLTunedClassifier buildTunedTwed1nnV2() {
            return buildTuned1nnV2(i -> DistanceMeasureConfigs.buildTwedSpace());
        }

        public static RLTunedClassifier buildTunedErp1nnV2() {
            return buildTuned1nnV2(DistanceMeasureConfigs::buildErpSpace);
        }

        public static RLTunedClassifier buildTunedLcss1nnV2() {
            return buildTuned1nnV2(DistanceMeasureConfigs::buildLcssSpace);
        }


        public static RLTunedClassifier buildTuned1nnV1(RLTunedKNNSetup.ParamSpaceBuilder paramSpaceFunction) {
            RLTunedClassifier incTunedClassifier = new RLTunedClassifier();
            RLTunedKNNSetup RLTunedKNNSetup = new RLTunedKNNSetup();
            RLTunedKNNSetup
                .setRlTunedClassifier(incTunedClassifier)
                .setParamSpace(paramSpaceFunction)
                .setKnnSupplier(Factory::build1nnV1).setImproveableBenchmarkIteratorBuilder(LinearListIterator::new);
            incTunedClassifier.setTrainSetupFunction(RLTunedKNNSetup);
            return incTunedClassifier;
        }

        public static RLTunedClassifier buildTuned1nnV1(ParamSpace paramSpace) {
            return buildTuned1nnV1(i -> paramSpace);
        }

        public static RLTunedClassifier buildTuned1nnV2(RLTunedKNNSetup.ParamSpaceBuilder paramSpaceFunction) {
            RLTunedClassifier incTunedClassifier = new RLTunedClassifier();
            RLTunedKNNSetup RLTunedKNNSetup = new RLTunedKNNSetup();
            RLTunedKNNSetup
                .setRlTunedClassifier(incTunedClassifier)
                .setParamSpace(paramSpaceFunction)
                .setKnnSupplier(Factory::build1nnV1).setImproveableBenchmarkIteratorBuilder(benchmarks -> new RandomListIterator<>(benchmarks, incTunedClassifier.getSeed()));
            incTunedClassifier.setTrainSetupFunction(RLTunedKNNSetup);
            return incTunedClassifier;
        }

        public static RLTunedClassifier buildTuned1nnV2(ParamSpace paramSpace) {
            return buildTuned1nnV2(i -> paramSpace);
        }
    }

    public static class Factory extends CompileTimeClassifierBuilderFactory<KNNLOOCV> {

        public final ClassifierBuilder<? extends KNNLOOCV> ED_1NN_V1 = add(new SuppliedClassifierBuilder<>(
            "ED_1NN_V1",
            Factory::buildEd1nnV1));
        public final ClassifierBuilder<? extends KNNLOOCV> DTW_1NN_V1 = add(new SuppliedClassifierBuilder<>("DTW_1NN_V1",
            Factory::buildDtw1nnV1));
        public final ClassifierBuilder<? extends KNNLOOCV> DDTW_1NN_V1 = add(new SuppliedClassifierBuilder<>(
            "DDTW_1NN_V1",
            Factory::buildDdtw1nnV1));
        public final ClassifierBuilder<? extends KNNLOOCV> ED_1NN_V2 = add(new SuppliedClassifierBuilder<>("ED_1NN_V2",
            Factory::buildEd1nnV2));
        public final ClassifierBuilder<? extends KNNLOOCV> DTW_1NN_V2 = add(new SuppliedClassifierBuilder<>("DTW_1NN_V2",
            Factory::buildDtw1nnV2));
        public final ClassifierBuilder<? extends KNNLOOCV> DDTW_1NN_V2 = add(new SuppliedClassifierBuilder<>("DDTW_1NN_V2",
            Factory::buildDdtw1nnV2));


        public static KNNLOOCV build1nnV1() {
            KNNLOOCV classifier = new KNNLOOCV();
            classifier.setEarlyAbandon(true);
            classifier.setK(1);
            classifier.setNeighbourLimit(-1);
            classifier.setNeighbourIteratorBuilder(new LinearNeighbourIteratorBuilder(classifier));
            classifier.setCvSearcherIteratorBuilder(new LinearNeighbourIteratorBuilder(classifier));
            classifier.setRandomTieBreak(false);
            return classifier;
        }

        public static KNNLOOCV build1nnV2() {
            KNNLOOCV classifier = build1nnV1();
            classifier.setNeighbourIteratorBuilder(new RandomNeighbourIteratorBuilder(classifier));
            classifier.setCvSearcherIteratorBuilder(new RandomNeighbourIteratorBuilder(classifier));
            classifier.setRandomTieBreak(true);
            return classifier;
        }

        public static KNNLOOCV buildEd1nnV1() {
            KNNLOOCV knn = build1nnV1();
            knn.setDistanceFunction(new DTWDistance(0));
            return knn;
        }

        public static KNNLOOCV buildDtw1nnV1() {
            KNNLOOCV knn = build1nnV1();
            knn.setDistanceFunction(new DTWDistance(-1));
            return knn;
        }

        public static KNNLOOCV buildDdtw1nnV1() {
            KNNLOOCV knn = build1nnV1();
            knn.setDistanceFunction(new DDTWDistance(-1));
            return knn;
        }

        public static KNNLOOCV buildEd1nnV2() {
            KNNLOOCV knn = build1nnV2();
            knn.setDistanceFunction(new DTWDistance(0));
            return knn;
        }

        public static KNNLOOCV buildDtw1nnV2() {
            KNNLOOCV knn = build1nnV2();
            knn.setDistanceFunction(new DTWDistance(-1));
            return knn;
        }

        public static KNNLOOCV buildDdtw1nnV2() {
            KNNLOOCV knn = build1nnV2();
            knn.setDistanceFunction(new DDTWDistance(-1));
            return knn;
        }
        
        public static void main(String[] args) throws Exception {
            int seed = 0;
            Instances[] data = DatasetLoading.sampleGunPoint(seed);
            RLTunedClassifier classifier = TunedFactory.buildTunedWdtw1nnV2();
            classifier.setSeed(seed); // set seed
            classifier.getLogger().setLevel(Level.ALL);
            classifier.setEstimateOwnPerformance(true);
            ClassifierResults results = ClassifierTools.trainAndTest(data, classifier);
            results.setDetails(classifier, data[1]);
            ClassifierResults trainResults = classifier.getTrainResults();
            trainResults.setDetails(classifier, data[0]);
            System.out.println(trainResults.writeSummaryResultsToString());
            System.out.println(results.writeSummaryResultsToString());
        }

    }

    private static final long serialVersionUID = 0;
    public static final String NEIGHBOUR_LIMIT_FLAG = "n";
    public static final String NEIGHBOUR_ITERATION_STRATEGY_FLAG = "s";
    protected transient long trainTimeLimitNanos = -1;
    protected List<NeighbourSearcher> searchers;
    protected long longestNeighbourEvalTimeInNanos;
    protected int neighbourLimit = -1;
    protected int neighbourCount;
    protected int comparisonCount;
    protected StopWatch trainEstimateTimer = new StopWatch();
    protected Cache<Instance, Instance, Double> cache;
    protected NeighbourSearcher leftOutSearcher = null;
    protected Iterator<NeighbourSearcher> leftOutSearcherIterator;
    protected Iterator<NeighbourSearcher> cvSearcherIterator;
    protected NeighbourIteratorBuilder neighbourIteratorBuilder = new RandomNeighbourIteratorBuilder(this);
    protected NeighbourIteratorBuilder cvSearcherIteratorBuilder = new RandomNeighbourIteratorBuilder(this);
    protected boolean customCache = false;
    private boolean regenerateTrainEstimate = true;

    public KNNLOOCV() {
        setAbleToEstimateOwnPerformance(true);
    }

    public KNNLOOCV(DistanceFunction df) {
        super(df);
        setAbleToEstimateOwnPerformance(true);
    }

    public StopWatch getTrainEstimateTimer() {
        return trainEstimateTimer;
    }

    public int getNeighbourCount() {
        return neighbourCount;
    }

    public int getComparisonCount() {
        return comparisonCount;
    }

    @Override
    public void setTrainTimeLimit(long nanos) {
        trainTimeLimitNanos = nanos;
    }

    public List<NeighbourSearcher> getSearchers() {
        return searchers;
    }

    public boolean hasNextNeighbourSearch() {
        return leftOutSearcherIterator.hasNext() || cvSearcherIterator.hasNext();
    }

    public boolean hasNextNeighbourLimit() {
        return (neighbourCount < neighbourLimit || neighbourLimit < 0);
    }

    public boolean hasNextNeighbour() {
        return hasNextNeighbourSearch() && hasNextNeighbourLimit();
    }

    public boolean hasNextBuildTick() throws Exception {
        return estimateOwnPerformance && hasNextNeighbour() && hasRemainingTrainTime();
    }

    public long predictNextTrainTimeNanos() {
        return longestNeighbourEvalTimeInNanos;
    }

    protected void nextBuildTick() throws Exception {
        final Logger logger = getLogger();
        regenerateTrainEstimate = true;
        final long timeStamp = System.nanoTime();
        if(leftOutSearcher == null) {
            leftOutSearcher = leftOutSearcherIterator.next();
            leftOutSearcherIterator.remove();
        }
        comparisonCount++;
        final NeighbourSearcher searcher = cvSearcherIterator.next();
        cvSearcherIterator.remove();
        final Instance instance = searcher.getInstance();
        final Instance leftOutInstance = leftOutSearcher.getInstance();
        if(!leftOutInstance.equals(instance)) {
            boolean seen;
            if(customCache) {
                seen = cache.contains(leftOutInstance, instance);
            } else {
                seen = cache.remove(leftOutInstance, instance);
            }
            if(seen) {
                // we've already seen this instance
                logger.info(() -> comparisonCount + ") " + "already seen i" + instance.hashCode() + " and i" + leftOutInstance.hashCode());
            } else {
                final long distanceMeasurementTimeStamp = System.nanoTime();
                Double distance = customCache ? cache.get(instance, leftOutInstance) : null;
                final long timeTakenInNanos = System.nanoTime() - distanceMeasurementTimeStamp;
                if(distance == null) {
                    distance = searcher.add(leftOutInstance);
                } else {
                    searcher.add(leftOutInstance, distance, timeTakenInNanos);
                }
                leftOutSearcher.add(instance, distance, 0); // we get this for free!
                cache.put(instance, leftOutInstance, distance);
                final Double finalDistance = distance;
                logger.info(() -> comparisonCount + ") i" + instance.hashCode() + " and i" + leftOutInstance.hashCode() +
                                 ": " + finalDistance);
            }
        } else {
            logger.info(() -> comparisonCount + ") i" + instance.hashCode() + " and i" + leftOutInstance.hashCode() +
                            ": left out");
        }
        if(!cvSearcherIterator.hasNext()) {
            if(leftOutSearcherIterator.hasNext()) {
                cvSearcherIterator = cvSearcherIteratorBuilder.build();
                if(logger.isLoggable(Level.WARNING)) {
                    logger.info("---- end of batch ----");
                    if(!cvSearcherIterator.hasNext()) {
                        throw new IllegalStateException("this shouldn't happen!");
                    }
                    if(!leftOutSearcherIterator.hasNext()) {
                        throw new IllegalStateException("this shouldn't happen!");
                    }
                }
            }
            leftOutSearcher = null;
            neighbourCount++;
        }
        longestNeighbourEvalTimeInNanos = System.nanoTime() - timeStamp;
    }

    public NeighbourIteratorBuilder getNeighbourIteratorBuilder() {
        return neighbourIteratorBuilder;
    }

    public void setNeighbourIteratorBuilder(final NeighbourIteratorBuilder neighbourIteratorBuilder) {
        this.neighbourIteratorBuilder = neighbourIteratorBuilder;
    }

    public boolean hasNeighbourLimit() {
        return neighbourLimit >= 0;
    }

    public interface NeighbourIteratorBuilder extends Serializable {
        Iterator<NeighbourSearcher> build();
    }

    @Override public ParamSet getParams() {
        return super.getParams()
                    .add(NEIGHBOUR_ITERATION_STRATEGY_FLAG, neighbourIteratorBuilder)
                    .add(NEIGHBOUR_LIMIT_FLAG, neighbourLimit)
                    .addAll(TrainTimeContractable.super.getParams());
    }

    @Override public void setParams(final ParamSet params) {
        super.setParams(params);
        ParamHandler.setParam(params, NEIGHBOUR_LIMIT_FLAG, this::setNeighbourLimit, Integer.class);
        ParamHandler.setParam(params, NEIGHBOUR_ITERATION_STRATEGY_FLAG, this::setNeighbourIteratorBuilder,
                              NeighbourIteratorBuilder.class);
        TrainTimeContractable.super.setParams(params);
    }

    public boolean loadFromCheckpoint() {
        final StopWatch trainTimer = getTrainTimer();
        final MemoryWatcher memoryWatcher = getMemoryWatcher();
        trainTimer.suspend();
        trainEstimateTimer.suspend();
        memoryWatcher.suspend();
        boolean result = super.loadFromCheckpoint();
        memoryWatcher.unsuspend();
        trainEstimateTimer.unsuspend();
        trainTimer.unsuspend();
        return result;
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        final StopWatch trainTimer = getTrainTimer();
        final MemoryWatcher memoryWatcher = getMemoryWatcher();
        memoryWatcher.enable();
        trainEstimateTimer.checkDisabled();
        trainTimer.enable();
        final DistanceFunction distanceFunction = getDistanceFunction();
        final Logger logger = getLogger();
        final boolean rebuild = isRebuild();
        // stop super knn from checkpointing after its built
        boolean skip = isSkipFinalCheckpoint();
        setSkipFinalCheckpoint(true);
        // must disable train timer and memory watcher as super enables them at start of build
        trainTimer.disable();
        memoryWatcher.disable();
        super.buildClassifier(trainData);
        // re-enable skipping the final checkpoint
        setSkipFinalCheckpoint(skip);
        memoryWatcher.enableAnyway();
        trainEstimateTimer.checkDisabled();
        trainTimer.enableAnyway();
        if(rebuild) {
            trainTimer.disableAnyway();
            memoryWatcher.enableAnyway();
            trainEstimateTimer.resetAndEnable();
            if(getEstimateOwnPerformance()) {
                if(isCheckpointSavingEnabled()) { // was needed for caching
                    HashTransformer.hashInstances(trainData);
                }
                // build a progressive leave-one-out-cross-validation
                searchers = new ArrayList<>(trainData.size());
                // build a neighbour searcher for every train instance
                for(int i = 0; i < trainData.size(); i++) {
                    final NeighbourSearcher searcher = new NeighbourSearcher(trainData.get(i));
                    searchers.add(i, searcher);
                }
                if(distanceFunction instanceof BaseDistanceMeasure) {
                    if(((BaseDistanceMeasure) distanceFunction).isSymmetric()) {
                        cache = new SymmetricCache<>();
                    } else {
                        cache = new Cache<>();
                    }
                }
                leftOutSearcherIterator = neighbourIteratorBuilder.build();
                regenerateTrainEstimate = true; // build the first train estimate irrelevant of any progress made
                cvSearcherIterator = cvSearcherIteratorBuilder.build();
                if(logger.isLoggable(Level.WARNING)) {
                    if(!leftOutSearcherIterator.hasNext()) {
                        throw new IllegalStateException("hasNext false");
                    }
                    if(!cvSearcherIterator.hasNext()) {
                        throw new IllegalStateException("this shouldn't happen!");
                    }
                }
                longestNeighbourEvalTimeInNanos = -1;
                leftOutSearcher = null;
                cvSearcherIterator = cvSearcherIteratorBuilder.build();
                neighbourCount = 0;
                comparisonCount = 0;
            }
        }
        trainTimer.disableAnyway();
        trainEstimateTimer.enableAnyway();
        while(hasNextBuildTick()) {
            nextBuildTick();
            saveToCheckpoint();
        }
        trainTimer.checkDisabled();
        if(regenerateTrainEstimate) {
            if(logger.isLoggable(Level.WARNING)
                && !hasTrainTimeLimit()
                && ((hasNeighbourLimit() && neighbourCount < neighbourLimit) ||
                        (!hasNeighbourLimit() && neighbourCount < trainData.size()))) {
                throw new IllegalStateException("not fully built");
            }
            // populate train results
            trainResults = new ClassifierResults();
            for(final NeighbourSearcher searcher : searchers) {
                final double[] distribution = searcher.predict();
                final int prediction = Utilities.argMax(distribution, rand);
                final long time = searcher.getTimeInNanos();
                final double trueClassValue = searcher.getInstance().classValue();
                trainResults.addPrediction(trueClassValue, distribution, prediction, time, null);
            }
        }
        trainEstimateTimer.disable();
        memoryWatcher.disable();
        if(regenerateTrainEstimate) {
            trainResults.setDetails(this, trainData);
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            trainResults.setBuildTime(trainEstimateTimer.getTimeNanos());
            trainResults.setBuildPlusEstimateTime(trainEstimateTimer.getTimeNanos() + trainTimer.getTimeNanos());
        }
        regenerateTrainEstimate = false;
        setBuilt(true);
        saveToCheckpoint();
    }

    public long getTrainTimeNanos() {
        return trainEstimateTimer.getTimeNanos() + getTrainTimer().getTimeNanos();
    }

    public long getTrainContractTimeNanos() {
        return trainTimeLimitNanos;
    }

    @Override
    public void setTrainTimeLimitNanos(final long trainTimeLimit) {
        this.trainTimeLimitNanos = trainTimeLimit;
    }

    public int getNeighbourLimit() {
        return neighbourLimit;
    }

    public void setNeighbourLimit(final int neighbourLimit) {
        this.neighbourLimit = neighbourLimit;
    }

    public Cache<Instance, Instance, Double> getCache() {
        return cache;
    }

    public void setCache(final Cache<Instance, Instance, Double> cache) {
        this.cache = cache;
        customCache = cache != null;
    }

    public void setDefaultCache() {
        setCache(null);
    }

    public NeighbourIteratorBuilder getCvSearcherIteratorBuilder() {
        return cvSearcherIteratorBuilder;
    }

    public void setCvSearcherIteratorBuilder(final NeighbourIteratorBuilder cvSearcherIteratorBuilder) {
        this.cvSearcherIteratorBuilder = cvSearcherIteratorBuilder;
    }

    public static void main(String[] args) throws Exception {
        int seed = 0;
        Instances[] data = DatasetLoading.sampleGunPoint(seed);
        KNNLOOCV classifier = new KNNLOOCV();
        classifier.setSeed(seed); // set seed
        classifier.setEstimateOwnPerformance(true);
        ClassifierResults results = ClassifierTools.trainAndTest(data, classifier);
        results.setDetails(classifier, data[1]);
        ClassifierResults trainResults = classifier.getTrainResults();
        trainResults.setDetails(classifier, data[0]);
        System.out.println(trainResults.writeSummaryResultsToString());
        System.out.println(results.writeSummaryResultsToString());
    }
}

