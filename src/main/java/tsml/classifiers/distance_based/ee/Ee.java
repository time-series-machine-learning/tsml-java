package tsml.classifiers.distance_based.ee;

import com.google.common.collect.ImmutableList;
import evaluation.storage.ClassifierResults;
import machine_learning.classifiers.ensembles.AbstractEnsemble;
import machine_learning.classifiers.ensembles.voting.MajorityVote;
import machine_learning.classifiers.ensembles.voting.ModuleVotingScheme;
import machine_learning.classifiers.ensembles.weightings.ModuleWeightingScheme;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.*;
import utilities.*;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

import static tsml.classifiers.distance_based.ee.EeConfig.buildV1Constituents;
import static tsml.classifiers.distance_based.knn.configs.KnnConfig.*;

public class Ee extends EnhancedAbstractClassifier implements TrainTimeContractable, Checkpointable,
        MemoryWatchable {

    public ImmutableList<EnhancedAbstractClassifier> getConstituents() {
        return constituents;
    }

    public void setConstituents(final List<? extends Classifier> constituents) {
        List<EnhancedAbstractClassifier> list = new ArrayList<>();
        for(Classifier constituent : constituents) {
            if(constituent instanceof EnhancedAbstractClassifier) {
                list.add((EnhancedAbstractClassifier) constituent);
            } else {
                throw new IllegalArgumentException("constituents have to be EAC");
            }
        }
        this.constituents = ImmutableList.copyOf(list);
    }

    public Ee() {
        super(true);
        setConstituents(buildV1Constituents());
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
    protected long trainTimeLimitNanos = -1;
    protected StopWatch trainTimer = new StopWatch();
    protected StopWatch trainEstimateTimer = new StopWatch();
    protected ModuleVotingScheme votingScheme = new MajorityVote();
    protected ModuleWeightingScheme weightingScheme = new TrainAcc();
    protected AbstractEnsemble.EnsembleModule[] modules;
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
    private boolean logConstituents = false;

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
        trainTimer.enable();
        memoryWatcher.enable();
        trainEstimateTimer.checkDisabled();
        this.trainData = trainData;
        if(rebuild) {
            trainTimer.resetAndEnable();
            memoryWatcher.resetAndEnable();
            super.buildClassifier(trainData);
            trainEstimateTimer.resetAndDisable();
            rebuild = false;
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
                    constituent.getLogger().setLevel(logger.getLevel());
                } else {
                    constituent.getLogger().setLevel(Level.OFF);
                }
            }
            nextPartialConstituentsBatch = new ArrayList<>();
            trainTimer.lap();
            if(!hasTrainTimeLimit()) {
                remainingTrainTimeNanosPerConstituent = -1;
            } else {
                remainingTrainTimeNanosPerConstituent = getRemainingTrainTimeNanos() / partialConstituentsBatch.size();
            }
        }
        trainTimer.enableAnyway();
        trainEstimateTimer.disableAnyway();
        while(hasNextBuildTick()) {
            nextBuildTick();
            checkpoint();
        }
        if(regenerateTrainEstimate && getEstimateOwnPerformance()) {
            logger.fine("generating train estimate");
            regenerateTrainEstimate = false;
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
        trainResults.setSplit("train");
        trainResults.setMemory(getMaxMemoryUsageInBytes()); // todo other fields
        trainResults.setBuildTime(getTrainTimeNanos()); // todo break down to estimate time also
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setFoldID(seed); // todo set other details
        trainResults.setDetails(this, trainData);
        // todo combine memory watcher + train times + test times
        this.trainData = null;
        built = true;
        logger.info("build finished");
        checkpoint();
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
        EnhancedAbstractClassifier constituent = partialConstituentsBatch.remove(0);
        if(constituent == null) {
            throw new IllegalStateException("something has gone wrong, constituent should not be null");
        }
        if(constituent instanceof TrainTimeContractable && timeRemainingPerConstituent()) {
            ((TrainTimeContractable) constituent).setTrainTimeLimitNanos(remainingTrainTimeNanosPerConstituent);
        }
        trainTimer.disable();
        memoryWatcher.disable();
        logger.fine(() -> "running constituent {id: "+
                   (constituents.size() - partialConstituentsBatch.size())+
                   "+  "+
                   constituent.getClassifierName()+
                   " }");
        constituent.buildClassifier(trainData); // todo add train time onto train estimate + mem
        logger.fine(() -> "ran constituent {id: "+
                   (constituents.size() - partialConstituentsBatch.size())+
                   "+  acc: "+
                   constituent.getTrainResults().getAcc()+
                   "+  "+
                   constituent.getClassifierName()+
                   " }");
        memoryWatcher.enable();
        trainTimer.enable();
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

    @Override public boolean isBuilt() {
         return partialConstituentsBatch.isEmpty();
    }

    public boolean hasNextBuildTick() throws Exception {
        return !firstBatchDone || hasRemainingTraining();
    }

    public void nextBuildTick() throws Exception {
        improveNextConstituent();
        regenerateTrainEstimate = true;
        checkpoint();
    }

    @Override public void loadFromFile(final String filename) throws Exception {
        throw new UnsupportedOperationException(); // todo
    }

    @Override public double[] distributionForInstance(final Instance instance) throws Exception {
        return votingScheme.distributionForInstance(modules, instance);
    }

    @Override
    public boolean setSavePath(final String path) {
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

    public void checkpoint() throws
                                          Exception {
        trainTimer.suspend();
        trainEstimateTimer.suspend();
        memoryWatcher.suspend();
        if(isCheckpointing() && (built || lastCheckpointTimeStamp + minCheckpointIntervalNanos < System.nanoTime())) {
            logger.fine("checkpointing");
            saveToFile(checkpointDirPath + tempCheckpointFileName);
            boolean success = new File(checkpointDirPath + tempCheckpointFileName).renameTo(new File(checkpointDirPath + checkpointDirPath));
            if(!success) {
                throw new IllegalStateException("could not rename checkpoint file");
            }
            lastCheckpointTimeStamp = System.nanoTime();
        }
        memoryWatcher.unsuspend();
        trainTimer.unsuspend();
        trainEstimateTimer.unsuspend();
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

    public boolean isLogConstituents() {
        return logConstituents;
    }

    public void setLogConstituents(boolean logConstituents) {
        this.logConstituents = logConstituents;
    }
}
