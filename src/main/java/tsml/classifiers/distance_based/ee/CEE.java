package tsml.classifiers.distance_based.ee;

import com.google.common.collect.ImmutableList;
import evaluation.storage.ClassifierResults;
import machine_learning.classifiers.ensembles.AbstractEnsemble;
import machine_learning.classifiers.ensembles.voting.MajorityVote;
import machine_learning.classifiers.ensembles.voting.ModuleVotingScheme;
import machine_learning.classifiers.ensembles.weightings.ModuleWeightingScheme;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import machine_learning.classifiers.tuned.incremental.configs.IncKnnTunerBuilder;
import tsml.classifiers.*;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.knn.KnnConfigs;
import utilities.*;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static machine_learning.classifiers.tuned.incremental.configs.Configs.*;
import static utilities.collections.Utils.replace;

public class CEE extends EnhancedAbstractClassifier implements TrainTimeContractable, Checkpointable,
        IncClassifier, MemoryWatchable {

    public static CEE buildV1() {
        CEE cee = new CEE();
        return cee;
    }

    public static CEE buildV2() {
        CEE cee = new CEE();
        cee.setConstituents(ImmutableList.of(
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildDtwSpaceV2).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildDdtwSpaceV2).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildErpSpace).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildLcssSpace).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildMsmSpace).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildTwedSpace).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildWdtwSpaceV2).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildWddtwSpaceV2).build()
                                            ));
        return cee;
    }

    public static CEE buildLEE() {
        CEE cee = new CEE();
        cee.setConstituents(ImmutableList.of(
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildDtwSpaceV1).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildDdtwSpaceV1).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildErpSpace).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildLcssSpace).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildMsmSpace).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildTwedSpace).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildWdtwSpaceV1).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
            new IncKnnTunerBuilder().setKnnSupplier(KnnConfigs::build1nnV2).setParamSpace(
                DistanceMeasureConfigs::buildWddtwSpaceV1).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build()
                                            ));
        return cee;
    }

    public static CEE buildLimitedEE() {
        CEE cee = new CEE();
        cee.setConstituents(ImmutableList.of(
            buildTunedDdtw1nnV1(),
            buildTunedErp1nnV1(),
            buildTunedLcss1nnV1(),
            buildTunedMsm1nnV1(),
            buildTunedWdtw1nnV1(),
            buildTunedWddtw1nnV1(),
            buildTunedTwed1nnV1()
                                            ));
        return cee;
    }

    public CEE() {
        super(true);
    }

    protected static final ImmutableList<EnhancedAbstractClassifier> DEFAULT_CONSTITUENTS = ImmutableList.of(
        buildTunedDtw1nnV1(),
        buildTunedDdtw1nnV1(),
        buildTunedErp1nnV1(),
        buildTunedLcss1nnV1(),
        buildTunedMsm1nnV1(),
        buildTunedWdtw1nnV1(),
        buildTunedWddtw1nnV1(),
        buildTunedTwed1nnV1()
                                                                                                            );
    protected ImmutableList<EnhancedAbstractClassifier> constituents = ImmutableList.copyOf(DEFAULT_CONSTITUENTS);
    protected List<EnhancedAbstractClassifier> partialConstituentsBatch = new ArrayList<>(); //
    // constituents which
    // still have work remaining
    protected List<EnhancedAbstractClassifier> nextPartialConstituentsBatch = new ArrayList<>(); //
    // constituents which still have work remaining - tentative version
    protected List<EnhancedAbstractClassifier> trainedConstituents = new ArrayList<>(); // fully trained constituents
    protected long trainTimeLimitNanos = -1;
    protected StopWatch trainTimer = new StopWatch();
    protected StopWatch estimateTrainTimer = new StopWatch();
    protected boolean rebuild = true;
    protected boolean regenerateTrainEstimate = true;
    protected ModuleVotingScheme votingScheme = new MajorityVote();
    protected ModuleWeightingScheme weightingScheme = new TrainAcc();
    protected AbstractEnsemble.EnsembleModule[] modules;
    protected boolean hasNext = false;
    protected long remainingTrainTimeNanosPerConstituent;
    protected boolean firstBatchDone;
    private String checkpointDirPath;
    private long lastCheckpointTimeStamp = 0;
    private long minCheckpointIntervalNanos = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    private static final String checkpointFileName = "checkpoint.ser";
    private static final String tempCheckpointFileName = checkpointFileName + ".tmp";
    private boolean ignorePreviousCheckpoints = false;
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private transient Instances trainData;
    private boolean debugConstituents = false;

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

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        IncClassifier.super.buildClassifier(trainData);
    }

    @Override public void startBuild(final Instances trainData) throws Exception {
        trainTimer.resume();
        memoryWatcher.resume();
        super.buildClassifier(trainData);
        this.trainData = trainData;
        if(rebuild) {
            logger.log("rebuilding");
            trainTimer.resetAndResume();
            estimateTrainTimer.resetAndResume();
            rebuild = false;
            if(constituents == null || constituents.isEmpty()) {
                throw new IllegalStateException("empty constituents");
            }
            firstBatchDone = false;
            partialConstituentsBatch = new ArrayList<>(constituents);
            trainedConstituents = new ArrayList<>();
            for(EnhancedAbstractClassifier constituent : constituents) {
                constituent.setDebug(debugConstituents);
                constituent.setSeed(rand.nextInt());
                constituent.setEstimateOwnPerformance(true);
            }
            nextPartialConstituentsBatch = new ArrayList<>();
            trainTimer.lap();
            if(!hasTrainTimeLimit()) {
                remainingTrainTimeNanosPerConstituent = -1;
            } else {
                remainingTrainTimeNanosPerConstituent = getRemainingTrainTimeNanos() / partialConstituentsBatch.size();
            }
        }
        memoryWatcher.pause();
        trainTimer.pause();
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
        EnhancedAbstractClassifier constituent = partialConstituentsBatch.remove(0);
        if(constituent == null) {
            throw new IllegalStateException("something has gone wrong, constituent should not be null");
        }
        if(constituent instanceof TrainTimeContractable) {
            ((TrainTimeContractable) constituent).setTrainTimeLimitNanos(remainingTrainTimeNanosPerConstituent);
        }
        trainTimer.pause();
        memoryWatcher.pause();
        logger.log("running constituent {id: ",
                   (constituents.size() - partialConstituentsBatch.size()),
                   ", ",
                   constituent.getClassifierName(),
                   " }");
        constituent.buildClassifier(trainData); // todo add train time onto train estimate + mem
        logger.log("ran constituent {id: ",
                   (constituents.size() - partialConstituentsBatch.size()),
                   ", acc: ",
                   constituent.getTrainResults().getAcc(),
                   ", ",
                   constituent.getClassifierName(),
                   " }");
        memoryWatcher.resume();
        trainTimer.resume();
        if(constituent instanceof TrainTimeContractable && ((TrainTimeContractable) constituent).hasRemainingTraining()) {
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

    @Override public boolean isDone() {
         return partialConstituentsBatch.isEmpty();
    }

    @Override
    public boolean hasNextBuildTick() throws Exception {
        trainTimer.resume();
        memoryWatcher.resume();
        hasNext = !firstBatchDone || hasRemainingTraining();
        memoryWatcher.pause();
        trainTimer.pause();
        return hasNext;
    }

    @Override
    public void nextBuildTick() throws Exception {
        trainTimer.resume();
        memoryWatcher.resume();
        if(!hasNext) {
            throw new IllegalStateException("cannot run next, hasNext is false");
        }
        improveNextConstituent();
        regenerateTrainEstimate = true;
        checkpoint();
        memoryWatcher.pause();
        trainTimer.pause();
    }

    @Override
    public void finishBuild() throws Exception {
        trainTimer.resume();
        memoryWatcher.resume();
        if(regenerateTrainEstimate && getEstimateOwnPerformance()) {
            logger.log("generating train estimate");
            regenerateTrainEstimate = false;
            modules = new AbstractEnsemble.EnsembleModule[constituents.size()];
            int i = 0;
            for(EnhancedAbstractClassifier constituent : constituents) {
                estimateTrainTimer.add(constituent.getTrainResults().getBuildPlusEstimateTime());
                modules[i] = new AbstractEnsemble.EnsembleModule();
                modules[i].setClassifier(constituent);
                modules[i].trainResults = constituent.getTrainResults();
                i++;
            }
            logger.log("weighting constituents");
            weightingScheme.defineWeightings(modules, trainData.numClasses());
            votingScheme.trainVotingScheme(modules, trainData.numClasses());
            trainResults = new ClassifierResults();
            for(i = 0; i < trainData.size(); i++) {
                StopWatch predictionTimer = new StopWatch(true);
                double[] distribution = votingScheme.distributionForTrainInstance(modules, i);
                int prediction = ArrayUtilities.argMax(distribution);
                predictionTimer.pause();
                double trueClassValue = trainData.get(i).classValue();
                trainResults.addPrediction(trueClassValue, distribution, prediction, predictionTimer.getTimeNanos(), null);
            }
        }
        memoryWatcher.pause();
        trainTimer.pause();
        trainResults.setSplit("train");
        trainResults.setMemory(getMaxMemoryUsageInBytes()); // todo other fields
        trainResults.setBuildTime(getTrainTimeNanos()); // todo break down to estimate time also
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setFoldID(seed); // todo set other details
        trainResults.setDetails(this, trainData);
        // todo combine memory watcher + train times + test times
        trainData = null;
        checkpoint(true);
        logger.log("build finished");
    }

    @Override public double[] distributionForInstance(final Instance instance) throws Exception {
        return votingScheme.distributionForInstance(modules, instance);
    }

    public ImmutableList<EnhancedAbstractClassifier> getConstituents() {
        return constituents;
    }

    public void setConstituents(final List<EnhancedAbstractClassifier> constituents) {
        this.constituents = ImmutableList.copyOf(constituents);
    }

    @Override
    public boolean setSavePath(final String path) {
        checkpointDirPath = StrUtils.asDirPath(path);
        return true;
    }

    @Override
    public String getSavePath() {
        return checkpointDirPath;
    }

    public void checkpoint(boolean force) throws
                                          Exception {
        trainTimer.pause();
        memoryWatcher.pause();
        if(isCheckpointing() && (force || lastCheckpointTimeStamp + minCheckpointIntervalNanos < System.nanoTime())) {
            logger.log("checkpointing");
            saveToFile(checkpointDirPath + tempCheckpointFileName);
            boolean success = new File(checkpointDirPath + tempCheckpointFileName).renameTo(new File(checkpointDirPath + checkpointDirPath));
            if(!success) {
                throw new IllegalStateException("could not rename checkpoint file");
            }
            lastCheckpointTimeStamp = System.nanoTime();
        }
        memoryWatcher.resume();
        trainTimer.resume();
    }

    public boolean isRebuild() {
        return rebuild;
    }

    public void setRebuild(final boolean rebuild) {
        this.rebuild = rebuild;
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

    @Override public long getMinCheckpointIntervalNanos() {
        return minCheckpointIntervalNanos;
    }

    @Override public void setMinCheckpointIntervalNanos(final long minCheckpointInterval) {
        this.minCheckpointIntervalNanos = minCheckpointInterval;
    }

    @Override public boolean isIgnorePreviousCheckpoints() {
        return ignorePreviousCheckpoints;
    }

    @Override public void setIgnorePreviousCheckpoints(final boolean state) {
        this.ignorePreviousCheckpoints = state;
    }

    @Override public long getTrainTimeLimitNanos() {
        return trainTimeLimitNanos;
    }

    @Override public void setTrainTimeLimitNanos(final long nanos) {
        this.trainTimeLimitNanos = nanos;
    }

    @Override public long getTrainTimeNanos() {
        return trainTimer.getTimeNanos();
    }

    @Override public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    public boolean isDebugConstituents() {
        return debugConstituents;
    }

    public void setDebugConstituents(final boolean debugConstituents) {
        this.debugConstituents = debugConstituents;
    }
}
