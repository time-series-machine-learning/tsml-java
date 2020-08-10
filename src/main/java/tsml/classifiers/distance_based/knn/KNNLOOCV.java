package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceConfigs;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceConfigs;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistanceConfigs;
import tsml.classifiers.distance_based.distances.msm.MSMDistanceConfigs;
import tsml.classifiers.distance_based.distances.twed.TWEDistanceConfigs;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistanceConfigs;
import tsml.classifiers.distance_based.knn.neighbour_iteration.LinearNeighbourIteratorBuilder;
import tsml.classifiers.distance_based.knn.neighbour_iteration.RandomNeighbourIteratorBuilder;
import tsml.classifiers.distance_based.knn.strategies.RLTunedKNNSetup;
import tsml.classifiers.distance_based.tuned.RLTunedClassifier;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import tsml.classifiers.distance_based.utils.collections.params.*;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.classifiers.CompileTimeClassifierBuilderFactory;
import tsml.classifiers.distance_based.utils.collections.iteration.LinearListIterator;
import utilities.*;
import tsml.classifiers.distance_based.utils.collections.cache.BiCache;
import tsml.classifiers.distance_based.utils.collections.cache.SymmetricBiCache;
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
            return buildTuned1nnV1(DTWDistanceConfigs::buildDTWSpace);
        }

        public static RLTunedClassifier buildTunedDdtw1nnV1() {
            return buildTuned1nnV1(DTWDistanceConfigs::buildDDTWSpace);
        }

        public static RLTunedClassifier buildTunedWdtw1nnV1() {
            return buildTuned1nnV1(i -> WDTWDistanceConfigs.buildWDTWSpace());
        }

        public static RLTunedClassifier buildTunedWddtw1nnV1() {
            return buildTuned1nnV1(i -> WDTWDistanceConfigs.buildWDDTWSpace());
        }

        public static RLTunedClassifier buildTunedDtw1nnV2() {
            return buildTuned1nnV2(DTWDistanceConfigs::buildDTWSpace);
        }

        public static RLTunedClassifier buildTunedDdtw1nnV2() {
            return buildTuned1nnV2(DTWDistanceConfigs::buildDDTWSpace);
        }

        public static RLTunedClassifier buildTunedWdtw1nnV2() {
            return buildTuned1nnV2(i -> WDTWDistanceConfigs.buildWDTWSpace());
        }

        public static RLTunedClassifier buildTunedWddtw1nnV2() {
            return buildTuned1nnV2(i -> WDTWDistanceConfigs.buildWDDTWSpace());
        }

        public static RLTunedClassifier buildTunedMsm1nnV1() {
            return buildTuned1nnV1(i -> MSMDistanceConfigs.buildMSMSpace());
        }

        public static RLTunedClassifier buildTunedTwed1nnV1() {
            return buildTuned1nnV1(i -> TWEDistanceConfigs.buildTWEDSpace());
        }

        public static RLTunedClassifier buildTunedErp1nnV1() {
            return buildTuned1nnV1(ERPDistanceConfigs::buildERPSpace);
        }

        public static RLTunedClassifier buildTunedLcss1nnV1() {
            return buildTuned1nnV1(LCSSDistanceConfigs::buildLCSSSpace);
        }

        public static RLTunedClassifier buildTunedMsm1nnV2() {
            return buildTuned1nnV2(i -> MSMDistanceConfigs.buildMSMSpace());
        }

        public static RLTunedClassifier buildTunedTwed1nnV2() {
            return buildTuned1nnV2(i -> TWEDistanceConfigs.buildTWEDSpace());
        }

        public static RLTunedClassifier buildTunedErp1nnV2() {
            return buildTuned1nnV2(ERPDistanceConfigs::buildERPSpace);
        }

        public static RLTunedClassifier buildTunedLcss1nnV2() {
            return buildTuned1nnV2(LCSSDistanceConfigs::buildLCSSSpace);
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
                .setKnnSupplier(Factory::build1nnV1).setImproveableBenchmarkIteratorBuilder(benchmarks -> new RandomIterator<>(
                new Random(incTunedClassifier.getSeed()), benchmarks));
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
            knn.setDistanceFunction(new DTWDistance()); // todo ed
            return knn;
        }

        public static KNNLOOCV buildDtw1nnV1() {
            KNNLOOCV knn = build1nnV1();
            knn.setDistanceFunction(new DTWDistance()); // todo full
            return knn;
        }

        public static KNNLOOCV buildDdtw1nnV1() {
            KNNLOOCV knn = build1nnV1();
            knn.setDistanceFunction(DTWDistanceConfigs.newDDTWDistance()); // todo full
            return knn;
        }

        public static KNNLOOCV buildEd1nnV2() {
            KNNLOOCV knn = build1nnV2();
            knn.setDistanceFunction(new DTWDistance()); // todo ed
            return knn;
        }

        public static KNNLOOCV buildDtw1nnV2() {
            KNNLOOCV knn = build1nnV2();
            knn.setDistanceFunction(new DTWDistance()); // todo full
            return knn;
        }

        public static KNNLOOCV buildDdtw1nnV2() {
            KNNLOOCV knn = build1nnV2();
            knn.setDistanceFunction(DTWDistanceConfigs.newDDTWDistance()); // todo full
            return knn;
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
    protected BiCache<Instance, Instance, Double> biCache;
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
        return estimateOwnPerformance && hasNextNeighbour() ;//&& hasRemainingTrainTime();
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
                seen = biCache.contains(leftOutInstance, instance);
            } else {
                seen = biCache.remove(leftOutInstance, instance);
            }
            if(seen) {
                // we've already seen this instance
                logger.info(() -> comparisonCount + ") " + "already seen i" + instance.hashCode() + " and i" + leftOutInstance.hashCode());
            } else {
                final long distanceMeasurementTimeStamp = System.nanoTime();
                Double distance = customCache ? biCache.get(instance, leftOutInstance) : null;
                final long timeTakenInNanos = System.nanoTime() - distanceMeasurementTimeStamp;
                if(distance == null) {
                    distance = searcher.add(leftOutInstance);
                } else {
                    searcher.add(leftOutInstance, distance, timeTakenInNanos);
                }
                leftOutSearcher.add(instance, distance, 0); // we get this for free!
                biCache.put(instance, leftOutInstance, distance);
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
                    .add(NEIGHBOUR_LIMIT_FLAG, neighbourLimit);
//                    .addAll(TrainTimeContractable.super.getParams());
    }

    @Override public void setParams(final ParamSet params) throws Exception {
        super.setParams(params);
        ParamHandlerUtils.setParam(params, NEIGHBOUR_LIMIT_FLAG, this::setNeighbourLimit, Integer.class);
        ParamHandlerUtils.setParam(params, NEIGHBOUR_ITERATION_STRATEGY_FLAG, this::setNeighbourIteratorBuilder,
                              NeighbourIteratorBuilder.class);
//        TrainTimeContractable.super.setParams(params);
    }

    public boolean loadCheckpoint() {
        final StopWatch trainTimer = getTrainTimer();
        final MemoryWatcher memoryWatcher = getMemoryWatcher();
//        trainTimer.suspend();
//        trainEstimateTimer.suspend() todo fix;
//        memoryWatcher.suspend();
        boolean result = super.loadCheckpoint();
//        memoryWatcher.unsuspend();
//        trainEstimateTimer.unsuspend();
//        trainTimer.unsuspend();
        return result;
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        final StopWatch trainTimer = getTrainTimer();
        final MemoryWatcher memoryWatcher = getMemoryWatcher();
        memoryWatcher.start();
        trainEstimateTimer.checkStopped();
        trainTimer.start();
        final DistanceFunction distanceFunction = getDistanceFunction();
        final Logger logger = getLogger();
        final boolean rebuild = isRebuild();
        // stop super knn from checkpointing after its built
        boolean skip = isSkipFinalCheckpoint();
        setSkipFinalCheckpoint(true);
        // must disable train timer and memory watcher as super enables them at start of build
        trainTimer.stop();
        memoryWatcher.stop();
        super.buildClassifier(trainData);
        // re-enable skipping the final checkpoint
        setSkipFinalCheckpoint(skip);
        memoryWatcher.start(false);
        trainEstimateTimer.checkStopped();
        trainTimer.start(false);
        if(rebuild) {
            trainTimer.stop(false);
            memoryWatcher.start(false);
            trainEstimateTimer.resetAndStart();
            if(getEstimateOwnPerformance()) {
//                if(isCheckpointSavingEnabled()) { // was needed for caching
//                    HashTransformer.hashInstances(trainData);
//                }
                // build a progressive leave-one-out-cross-validation
                searchers = new ArrayList<>(trainData.size());
                // build a neighbour searcher for every train instance
                for(int i = 0; i < trainData.size(); i++) {
                    final NeighbourSearcher searcher = new NeighbourSearcher(trainData.get(i));
                    searchers.add(i, searcher);
                }
                if(distanceFunction instanceof BaseDistanceMeasure) {
                    if(((BaseDistanceMeasure) distanceFunction).isSymmetric()) {
                        biCache = new SymmetricBiCache<>();
                    } else {
                        biCache = new BiCache<>();
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
        trainTimer.stop(false);
        trainEstimateTimer.start(false);
        while(hasNextBuildTick()) {
            nextBuildTick();
            checkpointIfIntervalExpired();
        }
        trainTimer.checkStopped();
        if(regenerateTrainEstimate) {
            if(logger.isLoggable(Level.WARNING)
//                && !hasTrainTimeLimit()
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
        trainEstimateTimer.stop();
        memoryWatcher.stop();
        if(regenerateTrainEstimate) {
//            trainResults.setDetails(this, trainData);
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            trainResults.setBuildTime(trainEstimateTimer.getTime());
            trainResults.setBuildPlusEstimateTime(trainEstimateTimer.getTime() + trainTimer.getTime());
        }
        regenerateTrainEstimate = false;
//        setBuilt(true);
        checkpointIfIntervalExpired();
    }

    public long getTrainTime() {
        return trainEstimateTimer.getTime() + getTrainTimer().getTime();
    }

    public long getTrainTimeLimit() {
        return trainTimeLimitNanos;
    }

    public void setTrainTimeLimitNanos(final long trainTimeLimit) {
        this.trainTimeLimitNanos = trainTimeLimit;
    }

    public int getNeighbourLimit() {
        return neighbourLimit;
    }

    public void setNeighbourLimit(final int neighbourLimit) {
        this.neighbourLimit = neighbourLimit;
    }

    public BiCache<Instance, Instance, Double> getBiCache() {
        return biCache;
    }

    public void setBiCache(final BiCache<Instance, Instance, Double> biCache) {
        this.biCache = biCache;
        customCache = biCache != null;
    }

    public void setDefaultCache() {
        setBiCache(null);
    }

    public NeighbourIteratorBuilder getCvSearcherIteratorBuilder() {
        return cvSearcherIteratorBuilder;
    }

    public void setCvSearcherIteratorBuilder(final NeighbourIteratorBuilder cvSearcherIteratorBuilder) {
        this.cvSearcherIteratorBuilder = cvSearcherIteratorBuilder;
    }

}

