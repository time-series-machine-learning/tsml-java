package tsml.classifiers.distance_based.elastic_ensemble;

import static tsml.classifiers.distance_based.knn.KNNLOOCV.FACTORY;

import com.google.common.collect.ImmutableList;
import evaluation.storage.ClassifierResults;
import java.util.function.Consumer;
import machine_learning.classifiers.ensembles.AbstractEnsemble;
import machine_learning.classifiers.ensembles.voting.MajorityVote;
import machine_learning.classifiers.ensembles.voting.ModuleVotingScheme;
import machine_learning.classifiers.ensembles.weightings.ModuleWeightingScheme;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.*;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import tsml.classifiers.distance_based.knn.strategies.RLTunedKNNSetup;
import tsml.classifiers.distance_based.tuned.RLTunedClassifier;
import tsml.classifiers.distance_based.utils.CheckpointUtils;
import tsml.classifiers.distance_based.utils.MemoryWatcher;
import tsml.classifiers.distance_based.utils.Stated;
import tsml.classifiers.distance_based.utils.StopWatch;
import tsml.classifiers.distance_based.utils.StrUtils;
import tsml.classifiers.distance_based.utils.classifier_building.CompileTimeClassifierBuilderFactory;
import utilities.*;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ElasticEnsemble extends EnhancedAbstractClassifier implements TrainTimeContractable, Checkpointable,
        GcMemoryWatchable, StopWatchTrainTimeable {

    public static final ClassifierBuilderFactory FACTORY = new ClassifierBuilderFactory();

    public static class ClassifierBuilderFactory extends CompileTimeClassifierBuilderFactory {
        public final ClassifierBuilder EE_V1 = add(new SuppliedClassifierBuilder("EE_V1",
            ClassifierBuilderFactory::buildEeV1));
        public final ClassifierBuilder EE_V2 = add(new SuppliedClassifierBuilder("EE_V2",
            ClassifierBuilderFactory::buildEeV2));
        public final ClassifierBuilder CEE_V1 = add(new SuppliedClassifierBuilder("CEE_V1",
            ClassifierBuilderFactory::buildCeeV1));
        public final ClassifierBuilder CEE_V2 = add(new SuppliedClassifierBuilder("CEE_V2",
            ClassifierBuilderFactory::buildCeeV2));
        public final ClassifierBuilder LEE = add(new SuppliedClassifierBuilder("LEE",
            ClassifierBuilderFactory::buildLee));


        public static ImmutableList<Classifier> buildV1Constituents() {
            return ImmutableList.of(
                KNNLOOCV.FACTORY.ED_1NN_V1.build(),
                KNNLOOCV.FACTORY.DTW_1NN_V1.build(),
                KNNLOOCV.FACTORY.DDTW_1NN_V1.build(),
                KNNLOOCV.FACTORY.TUNED_DTW_1NN_V1.build(),
                KNNLOOCV.FACTORY.TUNED_DDTW_1NN_V1.build(),
                KNNLOOCV.FACTORY.TUNED_WDTW_1NN_V1.build(),
                KNNLOOCV.FACTORY.TUNED_WDDTW_1NN_V1.build(),
                KNNLOOCV.FACTORY.TUNED_ERP_1NN_V1.build(),
                KNNLOOCV.FACTORY.TUNED_MSM_1NN_V1.build(),
                KNNLOOCV.FACTORY.TUNED_LCSS_1NN_V1.build(),
                KNNLOOCV.FACTORY.TUNED_TWED_1NN_V1.build()
            );
        }

        public static ImmutableList<Classifier> buildV2Constituents() {
            return ImmutableList.of(
                KNNLOOCV.FACTORY.ED_1NN_V2.build(),
                KNNLOOCV.FACTORY.DTW_1NN_V2.build(),
                KNNLOOCV.FACTORY.DDTW_1NN_V2.build(),
                KNNLOOCV.FACTORY.TUNED_DTW_1NN_V2.build(),
                KNNLOOCV.FACTORY.TUNED_DDTW_1NN_V2.build(),
                KNNLOOCV.FACTORY.TUNED_WDTW_1NN_V2.build(),
                KNNLOOCV.FACTORY.TUNED_WDDTW_1NN_V2.build(),
                KNNLOOCV.FACTORY.TUNED_ERP_1NN_V2.build(),
                KNNLOOCV.FACTORY.TUNED_MSM_1NN_V2.build(),
                KNNLOOCV.FACTORY.TUNED_LCSS_1NN_V2.build(),
                KNNLOOCV.FACTORY.TUNED_TWED_1NN_V2.build()
            );
        }

        public static ElasticEnsemble buildEeV1() {
            ElasticEnsemble elasticEnsemble = new ElasticEnsemble();
            elasticEnsemble.setConstituents(buildV1Constituents());
            setTrainSelectedBenchmarksFully(elasticEnsemble,false);
            return elasticEnsemble; // todo set full ee?
        }

        public static ElasticEnsemble buildEeV2() {
            ElasticEnsemble elasticEnsemble = new ElasticEnsemble();
            elasticEnsemble.setConstituents(buildV2Constituents());
            setTrainSelectedBenchmarksFully(elasticEnsemble,false);
            return elasticEnsemble; // todo set full ee?
        }

        public static ElasticEnsemble buildCeeV1() {
            return buildEeV1(); // todo turn off full ee?
        }

        public static ElasticEnsemble buildCeeV2() {
            return buildEeV2(); // todo turn off full ee?
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

    public ImmutableList<EnhancedAbstractClassifier> getConstituents() {
        return constituents;
    }

    public void setConstituents(final List<? extends Classifier> constituents) {
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

    public ElasticEnsemble() {
        super(true);
        setConstituents(ClassifierBuilderFactory.buildV1Constituents());
    }

    public boolean isLimitedVersion() {
        return limitedVersion;
    }

    public void setLimitedVersion(boolean limitedVersion) {
        this.limitedVersion = limitedVersion;
    }

    protected boolean limitedVersion = false;
    protected ImmutableList<EnhancedAbstractClassifier> constituents = ImmutableList.of();
    protected List<EnhancedAbstractClassifier> partialConstituentsBatch = new ArrayList<>(); //
    // constituents which still have work remaining
    protected List<EnhancedAbstractClassifier> nextPartialConstituentsBatch = new ArrayList<>(); //
    // constituents which still have work remaining - tentative version
    protected List<EnhancedAbstractClassifier> trainedConstituents = new ArrayList<>(); // fully trained constituents
    protected StopWatch trainTimer = new StopWatch();
    protected StopWatch trainEstimateTimer = new StopWatch();
    protected ModuleVotingScheme votingScheme = new MajorityVote();
    protected ModuleWeightingScheme weightingScheme = new TrainAcc();
    protected AbstractEnsemble.EnsembleModule[] modules;
    protected long remainingTrainTimeNanosPerConstituent;
    protected boolean firstBatchDone;
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private transient Instances trainData;
    private transient boolean debugConstituents = false;
    private transient boolean logConstituents = false;
    protected boolean built = false;

    // start boiler plate ----------------------------------------------------------------------------------------------

    private boolean rebuild = true; // shadows super
    protected transient long trainTimeLimitNanos = -1;
    private static final long serialVersionUID = 0;
    protected transient long minCheckpointIntervalNanos = Checkpointable.DEFAULT_MIN_CHECKPOINT_INTERVAL;
    protected transient long lastCheckpointTimeStamp = 0;
    protected transient String savePath = null;
    protected transient String loadPath = null;
    protected transient boolean skipFinalCheckpoint = false;

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
        boolean result = CheckpointUtils.saveToSingleCheckpoint(this, getLogger(), built && !skipFinalCheckpoint);
        memoryWatcher.unsuspend();
        trainTimer.unsuspend();
        return result;
    }

    public boolean loadFromCheckpoint() {
        trainTimer.suspend();
        memoryWatcher.suspend();
        boolean result = CheckpointUtils.loadFromSingleCheckpoint(this, getLogger());
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

    @Override
    public void setRebuild(boolean rebuild) {
        this.rebuild = rebuild;
        super.setRebuild(rebuild);
    }

    @Override public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
    }

    public StopWatch getTrainEstimateTimer() {
        return trainEstimateTimer;
    }


    @Override public void setTrainTimeLimitNanos(final long nanos) {
        trainTimeLimitNanos = nanos;
    }

    @Override public long predictNextTrainTimeNanos() {
        long result = -1;
        if(!nextPartialConstituentsBatch.isEmpty()) {
            EnhancedAbstractClassifier classifier = nextPartialConstituentsBatch.get(0);
            if(classifier instanceof TrainTimeContractable) {
                result = ((TrainTimeContractable) classifier).predictNextTrainTimeNanos();
            }
        }
        return result;
    }

    @Override public boolean isBuilt() {
        return built;
    }

    @Override public long getTrainTimeLimitNanos() {
        return trainTimeLimitNanos;
    }

    // end boiler plate ------------------------------------------------------------------------------------------------

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        final Logger logger = getLogger();
        loadFromCheckpoint();
        trainTimer.enable();
        memoryWatcher.enable();
        trainEstimateTimer.checkDisabled();
        if(rebuild) {
            trainTimer.resetAndEnable();
            memoryWatcher.resetAndEnable();
            trainEstimateTimer.resetAndDisable();
        }
        super.buildClassifier(trainData);
        built = false;
        this.trainData = trainData;
        if(rebuild) {
            if(constituents == null || constituents.isEmpty()) {
                throw new IllegalStateException("empty constituents");
            }
            if(isLimitedVersion() && hasTrainTimeLimit()) {
                throw new IllegalStateException("cannot run limited version under a train contract");
            }
            firstBatchDone = false;
            partialConstituentsBatch = new ArrayList<>(constituents);
            trainedConstituents = new ArrayList<>();
            for(EnhancedAbstractClassifier constituent : constituents) {
                constituent.setDebug(debugConstituents);
                constituent.setSeed(rand.nextInt());
                constituent.setEstimateOwnPerformance(true);
                if(logConstituents) {
                    constituent.getLogger().setLevel(getLogger().getLevel());
                } else {
                    constituent.getLogger().setLevel(Level.OFF);
                }
                if(constituent instanceof Checkpointable) {
                    if(isCheckpointLoadingEnabled()) {
                        ((Checkpointable) constituent).setLoadPath(loadPath);
                    }
                    if(isCheckpointSavingEnabled()) {
                        ((Checkpointable) constituent).setSavePath(savePath);
                    }
                    ((Checkpointable) constituent).setMinCheckpointIntervalNanos(minCheckpointIntervalNanos);
                    ((Checkpointable) constituent).setSkipFinalCheckpoint(skipFinalCheckpoint);
                }
            }
            nextPartialConstituentsBatch = new ArrayList<>();
            trainTimer.lap();
            if(!hasTrainTimeLimit()) {
                remainingTrainTimeNanosPerConstituent = -1;
            } else {
                remainingTrainTimeNanosPerConstituent = getRemainingTrainTimeNanos() / partialConstituentsBatch.size();
            }
            rebuild = false;
        }
        trainTimer.enableAnyway();
        trainEstimateTimer.disableAnyway();
        while(hasNextBuildTick()) {
            nextBuildTick();
            saveToCheckpoint();
        }
        if(isRegenerateTrainEstimate() && getEstimateOwnPerformance()) {
            logger.fine("generating train estimate");
            setRegenerateTrainEstimate(false);
            modules = new AbstractEnsemble.EnsembleModule[constituents.size()];
            int i = 0;
            for(EnhancedAbstractClassifier constituent : constituents) {
                trainEstimateTimer.add(constituent.getTrainResults().getBuildPlusEstimateTime());
                modules[i] = new AbstractEnsemble.EnsembleModule();
                modules[i].setClassifier(constituent);
                modules[i].trainResults = constituent.getTrainResults();
                i++;
            }
            logger.fine("weighting constituents");
            weightingScheme.defineWeightings(modules, trainData.numClasses());
            votingScheme.trainVotingScheme(modules, trainData.numClasses());
            trainResults = new ClassifierResults();
            for(i = 0; i < trainData.size(); i++) {
                StopWatch predictionTimer = new StopWatch(Stated.State.ENABLED);
                double[] distribution = votingScheme.distributionForTrainInstance(modules, i);
                int prediction = ArrayUtilities.argMax(distribution);
                predictionTimer.disable();
                double trueClassValue = trainData.get(i).classValue();
                trainResults.addPrediction(trueClassValue, distribution, prediction, predictionTimer.getTimeNanos(), null);
            }
        }
        memoryWatcher.disableAnyway();
        trainEstimateTimer.disableAnyway();
        trainTimer.disableAnyway();
        trainResults.setDetails(this, trainData);
        this.trainData = null;
        built = true;
        logger.info("build finished");
    }

    private boolean timeRemainingPerConstituent() {
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
    private void improveNextConstituent() throws Exception {
        final Logger logger = getLogger();
        EnhancedAbstractClassifier constituent = partialConstituentsBatch.remove(0);
        if(constituent == null) {
            throw new IllegalStateException("something has gone wrong, constituent should not be null");
        }
        if(constituent instanceof TrainTimeContractable && timeRemainingPerConstituent()) {
            ((TrainTimeContractable) constituent).setTrainTimeLimitNanos(remainingTrainTimeNanosPerConstituent);
        }
        StopWatch constituentTrainTimer = new StopWatch();
        trainTimer.disable();
        if(constituent instanceof TrainTimeable) {
            constituentTrainTimer.disableAnyway();
        } else {
            constituentTrainTimer.enableAnyway();
        }
        MemoryWatcher constituentMemoryWatcher = new MemoryWatcher();
        memoryWatcher.disable();
        if(constituent instanceof MemoryWatchable) {
            constituentMemoryWatcher.disableAnyway();
        } else {
            constituentMemoryWatcher.enableAnyway();
        }
        logger.fine(() -> "running constituent {id: "+
                   (constituents.size() - partialConstituentsBatch.size())+
                   " " +
                   constituent.getClassifierName()+
                   " }");
        constituent.buildClassifier(trainData); // todo add train time onto train estimate + mem
        logger.fine(() -> "ran constituent {id: "+
                   (constituents.size() - partialConstituentsBatch.size())+
                   " acc: "+
                   constituent.getTrainResults().getAcc()+
                   " "+
                   constituent.getClassifierName()+
                   " }");
        constituentTrainTimer.disableAnyway();
        constituentMemoryWatcher.disableAnyway();
        memoryWatcher.enable();
        trainTimer.enable();
        if(constituent instanceof TrainTimeable) {
            trainTimer.add(((TrainTimeable) constituent).getTrainTimeNanos());
        } else {
            trainTimer.add(constituentTrainTimer);
        }
        if(constituent instanceof TrainEstimateTimeable) {
            // the classifier tracked its time internally
            this.trainEstimateTimer.add(((TrainTimeable) constituent).getTrainTimeNanos());
        } else {
            // we already tracked this as part of the train time
        }
        if(constituent instanceof MemoryWatchable) {
            memoryWatcher.add((MemoryWatchable) constituent);
        } else {
            memoryWatcher.add(constituentMemoryWatcher);
        }
        if(constituent instanceof TrainTimeContractable && timeRemainingPerConstituent() &&
                ((TrainTimeContractable) constituent).hasRemainingTraining()) {
            nextPartialConstituentsBatch.add(constituent);
        }
        if(partialConstituentsBatch.isEmpty()) {
            firstBatchDone = true;
            if(!nextPartialConstituentsBatch.isEmpty()) {
                partialConstituentsBatch.addAll(nextPartialConstituentsBatch);
                nextPartialConstituentsBatch.clear();
                remainingTrainTimeNanosPerConstituent = getRemainingTrainTimeNanos() / partialConstituentsBatch.size();
            }
        }
    }


    public boolean hasNextBuildTick() throws Exception {
        return !firstBatchDone || hasRemainingTraining();
    }

    public void nextBuildTick() throws Exception {
        improveNextConstituent();
        setRegenerateTrainEstimate(true);
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

    public boolean isDebugConstituents() {
        return debugConstituents;
    }

    public void setDebugConstituents(final boolean debugConstituents) {
        this.debugConstituents = debugConstituents;
    }

    public boolean isLogConstituents() {
        return logConstituents;
    }

    public void setLogConstituents(boolean logConstituents) {
        this.logConstituents = logConstituents;
    }
}
