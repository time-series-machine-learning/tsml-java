package tsml.classifiers.distance_based.elastic_ensemble;

import com.google.common.collect.ImmutableList;
import evaluation.storage.ClassifierResults;
import java.util.function.Consumer;

import experiments.data.DatasetLoading;
import machine_learning.classifiers.ensembles.AbstractEnsemble;
import machine_learning.classifiers.ensembles.voting.MajorityVote;
import machine_learning.classifiers.ensembles.voting.ModuleVotingScheme;
import machine_learning.classifiers.ensembles.weightings.ModuleWeightingScheme;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.*;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import tsml.classifiers.distance_based.knn.strategies.RLTunedKNNSetup;
import tsml.classifiers.distance_based.tuned.RLTunedClassifier;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrain;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrainEstimate;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatchable;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.TrainEstimateable;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.classifiers.distance_based.utils.classifiers.CompileTimeClassifierBuilderFactory;
import utilities.*;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

// todo this has likeness to RLTuner, perhaps need to unify somewhere / make this a RLTuner?

public class ElasticEnsemble extends BaseClassifier implements TrainTimeContractable, Checkpointable,
    WatchedMemory, TimedTrain, TimedTrainEstimate {

    public static final Factory FACTORY = new Factory();

    /**
     * get whether the train estimate will be regenerated
     * @return
     */
    public boolean isRebuildTrainEstimateResults() {
        return regenerateTrainEstimate;
    }

    /**
     * set whether the train estimate will be regenerated
     * @param rebuildTrainEstimateResults
     * @return
     */
    public void setRebuildTrainEstimateResults(boolean rebuildTrainEstimateResults) {
        this.regenerateTrainEstimate = rebuildTrainEstimateResults;
    }

    public static class Factory extends CompileTimeClassifierBuilderFactory<ElasticEnsemble> {
        public final ClassifierBuilder<? extends ElasticEnsemble> EE_V1 = add(new SuppliedClassifierBuilder<>("EE_V1",
            Factory::buildEeV1));
        public final ClassifierBuilder<? extends ElasticEnsemble> EE_V2 = add(new SuppliedClassifierBuilder<>("EE_V2",
            Factory::buildEeV2));
        public final ClassifierBuilder<? extends ElasticEnsemble> LEE = add(new SuppliedClassifierBuilder<>("LEE",
            Factory::buildLee));


        public static ImmutableList<Classifier> buildV1Constituents() {
            return ImmutableList.of(
                KNNLOOCV.FACTORY.ED_1NN_V1.build(),
                KNNLOOCV.FACTORY.DTW_1NN_V1.build(),
                KNNLOOCV.FACTORY.DDTW_1NN_V1.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_DTW_1NN_V1.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_DDTW_1NN_V1.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_WDTW_1NN_V1.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_WDDTW_1NN_V1.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_ERP_1NN_V1.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_MSM_1NN_V1.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_LCSS_1NN_V1.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_TWED_1NN_V1.build()
            );
        }

        public static ImmutableList<Classifier> buildV2Constituents() {
            return ImmutableList.of(
                KNNLOOCV.FACTORY.ED_1NN_V2.build(),
                KNNLOOCV.FACTORY.DTW_1NN_V2.build(),
                KNNLOOCV.FACTORY.DDTW_1NN_V2.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_DTW_1NN_V2.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_DDTW_1NN_V2.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_WDTW_1NN_V2.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_WDDTW_1NN_V2.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_ERP_1NN_V2.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_MSM_1NN_V2.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_LCSS_1NN_V2.build(),
                KNNLOOCV.TUNED_FACTORY.TUNED_TWED_1NN_V2.build()
            );
        }

        public static ElasticEnsemble buildEeV1() {
            ElasticEnsemble elasticEnsemble = new ElasticEnsemble();
            elasticEnsemble.setConstituents(buildV1Constituents());
            setTrainSelectedBenchmarksFully(elasticEnsemble,false);
            return elasticEnsemble;
        }

        public static ElasticEnsemble buildEeV2() {
            ElasticEnsemble elasticEnsemble = new ElasticEnsemble();
            elasticEnsemble.setConstituents(buildV2Constituents());
            setTrainSelectedBenchmarksFully(elasticEnsemble,false);
            return elasticEnsemble;
        }

        private static ElasticEnsemble forEachTunedConstituent(ElasticEnsemble elasticEnsemble, Consumer<RLTunedKNNSetup> consumer) {
            for(Classifier classifier : elasticEnsemble.getConstituents()) {
                if(!(classifier instanceof RLTunedClassifier)) {
                    continue;
                }
                RLTunedClassifier tuner = (RLTunedClassifier) classifier;
                RLTunedKNNSetup config = (RLTunedKNNSetup) tuner.getTrainSetupFunction();
                consumer.accept(config);
            }
            return elasticEnsemble;
        }

        public static ElasticEnsemble setLimitedParameters(ElasticEnsemble elasticEnsemble, int limit) {
            return forEachTunedConstituent(elasticEnsemble, RLTunedKNNSetup -> RLTunedKNNSetup.setParamSpaceSizeLimit(limit));
        }

        public static ElasticEnsemble setLimitedParametersPercentage(ElasticEnsemble elasticEnsemble, double limit) {
            return forEachTunedConstituent(elasticEnsemble, RLTunedKNNSetup -> RLTunedKNNSetup.setParamSpaceSizeLimitPercentage(limit));
        }

        public static ElasticEnsemble setLimitedNeighbours(ElasticEnsemble elasticEnsemble, int limit) {
            return forEachTunedConstituent(elasticEnsemble, RLTunedKNNSetup -> RLTunedKNNSetup.setNeighbourhoodSizeLimit(limit));
        }

        public static ElasticEnsemble setLimitedNeighboursPercentage(ElasticEnsemble elasticEnsemble, double limit) { // todo params from cmdline in experiment + append to cls name
            return forEachTunedConstituent(elasticEnsemble, RLTunedKNNSetup -> RLTunedKNNSetup.setParamSpaceSizeLimitPercentage(limit));
        }

        public static ElasticEnsemble setTrainSelectedBenchmarksFully(ElasticEnsemble elasticEnsemble, boolean state) { // todo params from cmdline in experiment + append to cls name
            return forEachTunedConstituent(elasticEnsemble, RLTunedKNNSetup -> RLTunedKNNSetup.setTrainSelectedBenchmarksFully(state));
        }

        private static ElasticEnsemble buildLee() {
            ElasticEnsemble elasticEnsemble = new ElasticEnsemble();
            ImmutableList<Classifier> constituents = buildV2Constituents();
            elasticEnsemble.setConstituents(constituents);
            setLimitedNeighboursPercentage(elasticEnsemble, 0.1);
            setLimitedParametersPercentage(elasticEnsemble, 0.5);
            setTrainSelectedBenchmarksFully(elasticEnsemble,true);
            return elasticEnsemble;
        }
    }

    /**
     * get the constituents in this ensemble
     * @return
     */
    public ImmutableList<EnhancedAbstractClassifier> getConstituents() {
        return constituents;
    }

    /**
     * set the constituents
     * @param constituents
     */
    public void setConstituents(final Iterable<? extends Classifier> constituents) {
        List<EnhancedAbstractClassifier> list = new ArrayList<>();
        for(Classifier constituent : constituents) {
            if(constituent instanceof EnhancedAbstractClassifier) {
                list.add((EnhancedAbstractClassifier) constituent);
            } else {
                throw new IllegalArgumentException("constituents have to be EAC"); // todo some kind of wrapper
                // around ones which aren't EAC for generic'ness, not important right now
            }
        }
        this.constituents = ImmutableList.copyOf(list);
    }

    /**
     * build default v1 EE (the traditional version)
     */
    public ElasticEnsemble() {
        super(true);
        setConstituents(Factory.buildV1Constituents());
    }

    // the constituents
    private ImmutableList<EnhancedAbstractClassifier> constituents = ImmutableList.of();
    // the constituents which we're currently looking at as part of a batch of training. For example, if we chose to
    // train each constituent for 5 mins then this list contains all of the constituents which have *NOT* had their 5
    // mins of training yet
    private List<EnhancedAbstractClassifier> constituentsBatch = new ArrayList<>();
    // constituents which still have work remaining. For example, this would be any constituent which is not done after
    // the 5 mins above
    private List<EnhancedAbstractClassifier> nextConstituentsBatch = new ArrayList<>();
    // constituents which have been fully built
    private List<EnhancedAbstractClassifier> trainedConstituents = new ArrayList<>();
    // track the train time
    private StopWatch trainTimer = new StopWatch();
    // track the train estimate time
    private StopWatch trainEstimateTimer = new StopWatch();
    // how to combine the constituent votes
    private ModuleVotingScheme votingScheme = new MajorityVote();
    // how to weight each constituent
    private ModuleWeightingScheme weightingScheme = new TrainAcc();
    // final module array of constituents
    private AbstractEnsemble.EnsembleModule[] modules;
    // the amount of train time remaining for each constituent. In our above example this would be 5 mins
    private long remainingTrainTimeNanosPerConstituent;
    // whether we've done a first pass of all constituents
    private boolean firstBatchDone;
    // record the memory usage
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    // store the train data
    private transient Instances trainData;
    // switch to regenerate the train estimate
    private boolean regenerateTrainEstimate = true;
    // train time limit
    private transient long trainTimeLimitNanos = -1;
    // version id for serialisation
    protected transient long trainContractTimeNanos = -1;
    //TODO George to integrate the boolean into the classifier logic
    private boolean trainTimeContract = false;

    private static final long serialVersionUID = 0;
    // minimum checkpoint interval
    private transient long minCheckpointIntervalNanos = 0;//Checkpointable.DEFAULT_MIN_CHECKPOINT_INTERVAL;
    // timestamp of last checkpoint
    private transient long lastCheckpointTimeStamp = 0;
    // save path for checkpoints
    private transient String savePath = null;
    // load path for checkpoints
    private transient String loadPath = null;
    // whether to skip the final checkpoint
    private transient boolean skipFinalCheckpoint = false;

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

    public boolean setLoadPath(final String path) {
//        boolean result = Checkpointable.super.setLoadPath(path);
//        if(result) {
//            loadPath = StrUtils.asDirPath(path);
//        } else {
//            loadPath = null;
//        }
//        return result;
        return true;
    }

    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    public Instances getTrainData() { // todo is this needed?
        return trainData;
    }

    public long getLastCheckpointTimeStamp() {
        return lastCheckpointTimeStamp;
    }

    public boolean checkpointIfIntervalExpired() throws Exception {
//        trainTimer.stop();
//        trainEstimateTimer.stop();
//        memoryWatcher.stop();
////        boolean result = CheckpointUtils.saveToSingleCheckpoint(this, getLogger(), isBuilt() &&
//        //        !skipFinalCheckpoint); // todo fix
//        memoryWatcher.unsuspend();
//        trainEstimateTimer.unsuspend();
//        trainTimer.unsuspend();
//        return result;
        return false;
    }

    public boolean loadCheckpoint() {
//        trainTimer.stop(); // todo fix
//        trainEstimateTimer.stop();
//        memoryWatcher.stop();
//        boolean result = CheckpointUtils.loadFromSingleCheckpoint(this, getLogger());
//        lastCheckpointTimeStamp = System.nanoTime();
//        memoryWatcher.unsuspend();
//        trainEstimateTimer.unsuspend();
//        trainTimer.unsuspend();
//        return result;
        return false;
    }

    public void setMinCheckpointIntervalNanos(final long nanos) {
        if(minCheckpointIntervalNanos < 0) {
            throw new IllegalArgumentException("cannot be less than 0: " + nanos);
        }
        minCheckpointIntervalNanos = nanos;
    }

    public long getMinCheckpointIntervalNanos() {
        return minCheckpointIntervalNanos;
    }

    @Override public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
    }

    public StopWatch getTrainEstimateTimer() {
        return trainEstimateTimer;
    }

    @Override
    public void setTrainTimeLimit(long nanos) {
        trainTimeLimitNanos = nanos;
    }

    public long predictNextTrainTimeNanos() { // todo this may be better in its own interface
        long result = 0;
        // if we've got no more constituents to look at then we're done
        if(!nextConstituentsBatch.isEmpty()) {
            // otherwise get the next constituent
            EnhancedAbstractClassifier classifier = nextConstituentsBatch.get(0);
            // if it's able to predict its next amount of time then use that
            if(classifier instanceof TrainTimeContractable) {
//                result = ((TrainTimeContractable) classifier).predictNextTrainTimeNanos();
            }
        }
        return result;
    }

    public long getTrainContractTimeNanos() {
        return trainContractTimeNanos;
    }

    private void setRemainingTrainTimeNanosPerConstituent() {
        // if we've got no train time limit then the constituents can take as long as they like
        // if we've got no constituents in the batch then there's no remaining time
//        if(!hasTrainTimeLimit() || constituentsBatch.isEmpty()) {
//            remainingTrainTimeNanosPerConstituent = -1;
//        } else {
//            remainingTrainTimeNanosPerConstituent = getRemainingTrainTimeNanos() / constituentsBatch.size();
//        }
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        // first lets load from a checkpoint if there is one
        loadCheckpoint();
        // enable the resource trackers
        trainTimer.start();
        memoryWatcher.start();
        trainEstimateTimer.checkStopped();
        final Logger logger = getLogger();
        // find whether we're rebuilding
        final boolean rebuild = isRebuild();
        // if we're rebuilding
        if(rebuild) {
            // reset the resource trackers
            trainTimer.resetAndStart();
            memoryWatcher.resetAndStart();
            trainEstimateTimer.resetAndStop();
        }
        // let super build
        super.buildClassifier(trainData);
        // hold the train data
        this.trainData = trainData;
        if(rebuild) {
            // if we're rebuilding then setup
            if(constituents == null || constituents.isEmpty()) {
                throw new IllegalStateException("empty constituents");
            }
            // initialise
            firstBatchDone = false;
            constituentsBatch = new ArrayList<>(constituents);
            trainedConstituents = new ArrayList<>();
            // for each constituent
            for(EnhancedAbstractClassifier constituent : constituents) {
                // set their seed to match ours - better reproducibility if we run a constituent individually with
                // same seed
                constituent.setSeed(seed);
                // tell them to find a train estimate for weighting
                constituent.setEstimateOwnPerformance(true);
                // if the constituent can do checkpointing
                if(constituent instanceof Checkpointable) {
                    // setup all the checkpointing details
//                    if(isCheckpointLoadingEnabled()) { // todo paths need to be appended with constituent name
//                        ((Checkpointable) constituent).setLoadPath(loadPath);
//                    }
//                    if(isCheckpointSavingEnabled()) {
//                        ((Checkpointable) constituent).setCheckpointPath(savePath);
//                    }
//                    ((Checkpointable) constituent).setMinCheckpointIntervalNanos(minCheckpointIntervalNanos);
//                    ((Checkpointable) constituent).setSkipFinalCheckpoint(skipFinalCheckpoint);
                }
            }
            nextConstituentsBatch = new ArrayList<>();
            // find how much train time remains and split between the constituents
            trainTimer.lap();
            setRemainingTrainTimeNanosPerConstituent();
        }
        // switch resource monitors if not already
        trainTimer.start(false);
        trainEstimateTimer.stop(false);
        // while there's constituents to process and time left
        while(hasNextBuildTick()) {
            // process another constituent
            nextBuildTick();
            // save this to checkpoint
            checkpointIfIntervalExpired();
        }
        // if we're estimating our train
        if(regenerateTrainEstimate) {
            logger.fine("generating train estimate");
            modules = new AbstractEnsemble.EnsembleModule[constituents.size()];
            int i = 0;
            // translate constituents to modules
            for(EnhancedAbstractClassifier constituent : constituents) {
                trainEstimateTimer.add(constituent.getTrainResults().getBuildPlusEstimateTime());
                modules[i] = new AbstractEnsemble.EnsembleModule();
                modules[i].setClassifier(constituent);
                modules[i].trainResults = constituent.getTrainResults();
                i++;
            }
            // weight constituents
            logger.fine("weighting constituents");
            weightingScheme.defineWeightings(modules, trainData.numClasses());
            votingScheme.trainVotingScheme(modules, trainData.numClasses());
            trainResults = new ClassifierResults();
            // vote constituents
            for(i = 0; i < trainData.size(); i++) {
                StopWatch predictionTimer = new StopWatch();
                predictionTimer.start();
                double[] distribution = votingScheme.distributionForTrainInstance(modules, i);
                int prediction = Utilities.argMax(distribution, rand);
                predictionTimer.stop();
                double trueClassValue = trainData.get(i).classValue();
                trainResults.addPrediction(trueClassValue, distribution, prediction, predictionTimer.getTime(), null);
            }
        }
        // have regenerated train estimate so disable
        regenerateTrainEstimate = false;
        // disable resource monitors
        memoryWatcher.stop(false);
        trainEstimateTimer.stop(false);
        trainTimer.stop(false);
        // set train results details
//        trainResults.setDetails(this, trainData);
        // free up train data
        this.trainData = null;
        // we're built by here
//        setBuilt(true);
        logger.info("build finished");
        checkpointIfIntervalExpired();
    }

    private boolean hasTimeRemainingPerConstituent() {
        return remainingTrainTimeNanosPerConstituent >= 0;
    }

    /**
     * further iterations training a singular constituent by the remaining train time per constituent. Updates the
     * constituent records afterwards reflecting whether the constituent is done or has training remaining. If this
     * handles the last untrained constituent for this batch then we repopulate the batch from the constituents which
     * are not finished training, distributing the train time between them again. E.g. if we have 3 classifiers, A, B
     * and C. If we have a train time of 15 mins we would do 3 executions of this function with 5 mins for each
     * classifier. Suppose classifier B and C finished in the full 5 mins and classifier A finished in 3 mins then
     * there is a remaining 2 mins left of the 15 mins total train contract. We then repopulate the batch of
     * classifiers with B and C (the unfinished classifiers) and split the remaining train time between them (2 mins
     * --> 1 min each). Repeat until all classifiers are trained or train time is depleted to zero.
     * @throws Exception
     */
    private void nextBuildTick() throws Exception {
        final Logger logger = getLogger();
        // get the next constituent
        EnhancedAbstractClassifier constituent = constituentsBatch.remove(0);
        if(constituent == null) {
            throw new IllegalStateException("something has gone wrong, constituent should not be null");
        }
        // set the train time limit if possible
        if(constituent instanceof TrainTimeContractable && hasTimeRemainingPerConstituent()) {
//            ((TrainTimeContractable) constituent).setTrainTimeLimitNanos(remainingTrainTimeNanosPerConstituent);
        }
        // track the train time of the constituent
        StopWatch constituentTrainTimer = new StopWatch();
        // disable our train timer as the constituent train timer will take it from here
        trainTimer.stop();
        if(constituent instanceof TrainTimeable) {
            constituentTrainTimer.stop(false);
        } else {
            constituentTrainTimer.start(false);
        }
        // track the memory of the constituent
        MemoryWatcher constituentMemoryWatcher = new MemoryWatcher();
        // disable our memory watcher as the constituent memory watcher will take it from here
        memoryWatcher.stop();
        if(constituent instanceof MemoryWatchable) {
            constituentMemoryWatcher.stop(false);
        } else {
            constituentMemoryWatcher.start(false);
        }
        logger.fine(() -> "running constituent {id: "+
                   (constituents.size() - constituentsBatch.size())+
                   " " +
                   constituent.getClassifierName()+
                   " }");
        constituent.buildClassifier(trainData);
        logger.fine(() -> "ran constituent {id: "+
                   (constituents.size() - constituentsBatch.size())+
                   " acc: "+
                   constituent.getTrainResults().getAcc()+
                   " "+
                   constituent.getClassifierName()+
                   " }");
        // disable resource monitors for the constituent and re-enable ours
        constituentTrainTimer.stop(false);
        constituentMemoryWatcher.stop(false);
        memoryWatcher.start();
        trainTimer.start();
        // sanity check the train estimate timer is disabled
        trainEstimateTimer.checkStopped();
        // add the constituent's train time onto ours
        if(constituent instanceof TrainTimeable) { // todo these can probs be a util method as similar elsewhere
            // (RLTune)
            trainTimer.add(((TrainTimeable) constituent).getTrainTime());
        } else {
            trainTimer.add(constituentTrainTimer);
        }
        // add the constituent's train estimate time onto ours
        if(constituent instanceof TrainEstimateTimeable) {
            // the classifier tracked its time internally
            this.trainEstimateTimer.add(((TrainTimeable) constituent).getTrainTime());
        } else {
            // we already tracked this as part of the train time
        }
        // add the constituents memory usage onto ours
        if(constituent instanceof MemoryWatchable) {
            memoryWatcher.add((MemoryWatchable) constituent);
        } else {
            memoryWatcher.add(constituentMemoryWatcher);
        }
        // if the constituent is contracting train time AND there's time remaining for each constituent AND the
        // constituent has remaining work to do
        if(constituent instanceof TrainTimeContractable && hasTimeRemainingPerConstituent()
//                   &&
//                ((TrainTimeContractable) constituent).hasRemainingTraining()
        ) {
            // add it to the next batch of constituents
            nextConstituentsBatch.add(constituent);
        }
        // if there's no more constituents to process
        if(constituentsBatch.isEmpty()) {
            // we have definitely seen all constituents here
            firstBatchDone = true;
            // add all of the next constituent batch to the current batch
            constituentsBatch.addAll(nextConstituentsBatch);
            // clear out the next constituent batch as they've all been added to the current batch
            nextConstituentsBatch.clear();
            // recalculate the remaining time for each constituent
            setRemainingTrainTimeNanosPerConstituent();
        }
        // we've adjusted one of the constituents therefore we need to regenerate the train estimate
        setRebuildTrainEstimateResults(true);
    }

    /**
     * whether further build steps remain
     * @return
     * @throws Exception
     */
    public boolean hasNextBuildTick() throws Exception {
        // must do a first pass of all constituents, therefore if the first batch hasn't been completed this should
        // always return true
        // otherwise, it's dependent on whether there's further training remaining
        return !firstBatchDone || (
//                hasRemainingTrainTime() &&
                        !constituentsBatch.isEmpty());
    }

    @Override public double[] distributionForInstance(final Instance instance) throws Exception {
        return votingScheme.distributionForInstance(modules, instance);
    }

    public ModuleVotingScheme getVotingScheme() {
        return votingScheme;
    }

    public void setVotingScheme(final ModuleVotingScheme votingScheme) {
        this.votingScheme = votingScheme;
    }

    public ModuleWeightingScheme getWeightingScheme() {
        return weightingScheme;
    }

    public void setWeightingScheme(final ModuleWeightingScheme weightingScheme) {
        this.weightingScheme = weightingScheme;
    }
}
