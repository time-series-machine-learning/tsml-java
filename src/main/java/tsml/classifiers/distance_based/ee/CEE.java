package tsml.classifiers.distance_based.ee;

import evaluation.storage.ClassifierResults;
import machine_learning.classifiers.ensembles.AbstractEnsemble;
import machine_learning.classifiers.ensembles.voting.MajorityVote;
import machine_learning.classifiers.ensembles.voting.ModuleVotingScheme;
import machine_learning.classifiers.ensembles.weightings.ModuleWeightingScheme;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.*;
import utilities.ArrayUtilities;
import utilities.MemoryWatcher;
import utilities.StopWatch;
import utilities.StrUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static utilities.collections.Utils.replace;

public class CEE extends EnhancedAbstractClassifier implements TrainTimeContractable, Checkpointable,
                                                               ProgressiveBuildClassifier, MemoryWatchable {

    public CEE() {
        super(true);
    }

    protected List<EnhancedAbstractClassifier> constituents = new ArrayList<>(); // todo populate
    protected TreeSet<EnhancedAbstractClassifier> partialConstituentsBatch = new TreeSet<>(LEAST_TRAINED_FIRST); // constituents which
    // still have work remaining
    protected TreeSet<EnhancedAbstractClassifier> nextPartialConstituentsBatch = new TreeSet<>(LEAST_TRAINED_FIRST); //
    // constituents which still have work remaining - tentative version
    protected List<EnhancedAbstractClassifier> trainedConstituents = new ArrayList<>(); // fully trained constituents
    protected long trainTimeLimitNanos = -1;
    protected StopWatch trainTimer = new StopWatch();
    protected boolean rebuild = true;
    protected boolean regenerateTrainEstimate = true;
    protected ModuleVotingScheme votingScheme = new MajorityVote();
    protected ModuleWeightingScheme weightingScheme = new TrainAcc();
    protected AbstractEnsemble.EnsembleModule[] modules;
    protected boolean hasNext = false;
    protected transient Instances data;
    protected long remainingTrainTimeNanosPerConstituent;
    protected boolean firstIteration = true;
    private String checkpointDirPath;
    private long lastCheckpointTimeStamp = 0;
    private long minCheckpointIntervalNanos = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    private static final String checkpointFileName = "checkpoint.ser";
    private static final String tempCheckpointFileName = checkpointFileName + ".tmp";
    private boolean ignorePreviousCheckpoints = false;
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private Instances trainData;
    private final static Comparator<EnhancedAbstractClassifier> LEAST_TRAINED_FIRST = Comparator.comparing(eac -> {
        if(eac instanceof TrainTimeContractable) {
            return ((TrainTimeContractable) eac).predictNextTrainTimeNanos() + ((TrainTimeContractable) eac).getTrainTimeNanos();
        } else {
            return -1L;
        }
    });

    @Override public long predictNextTrainTimeNanos() {
        long result = -1;
        if(!nextPartialConstituentsBatch.isEmpty()) {
            EnhancedAbstractClassifier classifier = nextPartialConstituentsBatch.first();
            if(classifier instanceof TrainTimeContractable) {
                result = ((TrainTimeContractable) classifier).predictNextTrainTimeNanos();
            }
        }
        return result;
    }

    @Override public void buildClassifier(final Instances data) throws Exception {
        ProgressiveBuildClassifier.super.buildClassifier(data);
    }

    @Override public void startBuild(final Instances data) throws Exception {
        trainTimer.resume();
        memoryWatcher.resume();
        super.buildClassifier(data);
        this.data = data;
        if(rebuild) {
            rebuild = false;
            if(constituents == null || constituents.isEmpty()) {
                throw new IllegalStateException("empty constituents");
            }
            firstIteration = true;
            partialConstituentsBatch = new TreeSet<>(LEAST_TRAINED_FIRST);
            trainedConstituents = new ArrayList<>();
            partialConstituentsBatch.addAll(constituents);
            for(EnhancedAbstractClassifier constituent : constituents) {
                constituent.setDebug(debug);
                constituent.setSeed(seed);
                constituent.setEstimateOwnPerformance(true);
            }
        }
        memoryWatcher.pause();
        trainTimer.pause();
    }

    private void improveConstituent(EnhancedAbstractClassifier constituent, long trainTimeLimitNanos, Collection<EnhancedAbstractClassifier> next)
        throws Exception {
        if(constituent instanceof TrainTimeContractable) {
            ((TrainTimeContractable) constituent).setTrainTimeLimitNanos(trainTimeLimitNanos);
        }
        trainTimer.pause();
        memoryWatcher.pause();
        constituent.buildClassifier(data);
        memoryWatcher.resume();
        trainTimer.resume();
        if(constituent instanceof TrainTimeContractable && !((TrainTimeContractable) constituent).isDone()) {
            next.add(constituent);
        }
    }

    /**
     * train all of the constituents equally. If there's a contract then distribute it equally between constituents.
     * Else just sequentially build each constituent.
     * @throws Exception
     */
    private void firstIteration() throws Exception {
        if(hasTrainTimeLimit()) {
            remainingTrainTimeNanosPerConstituent = getRemainingTrainTimeNanos() / constituents.size();
        } else {
            remainingTrainTimeNanosPerConstituent = -1;
        }
        for(EnhancedAbstractClassifier constituent : constituents) {
            improveConstituent(constituent, remainingTrainTimeNanosPerConstituent, partialConstituentsBatch);
        }
        if(!partialConstituentsBatch.isEmpty()) {
            remainingTrainTimeNanosPerConstituent = getRemainingTrainTimeNanos() / partialConstituentsBatch.size();
        }
        firstIteration = false;
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
    private void furtherIteration() throws Exception {
        if(!hasTrainTimeLimit()) throw new IllegalStateException("the code shouldn't hit here!");
        EnhancedAbstractClassifier constituent = partialConstituentsBatch.pollFirst();
        if(constituent == null) {
            throw new IllegalStateException("something has gone wrong, remaining constituents should not be empty");
        }
        improveConstituent(constituent, remainingTrainTimeNanosPerConstituent, nextPartialConstituentsBatch);
        if(partialConstituentsBatch.isEmpty()) {
            partialConstituentsBatch = nextPartialConstituentsBatch;
            nextPartialConstituentsBatch = new TreeSet<>(LEAST_TRAINED_FIRST);
            remainingTrainTimeNanosPerConstituent = getRemainingTrainTimeNanos() / partialConstituentsBatch.size();
        }
    }

    @Override
    public boolean hasNextBuildTick() throws Exception {
        trainTimer.resume();
        memoryWatcher.resume();
        boolean hasNext = firstIteration || (hasRemainingTrainTime() && !partialConstituentsBatch.isEmpty());
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
        if(firstIteration) {
            firstIteration();
        } else {
            furtherIteration();
        }
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
            regenerateTrainEstimate = false;
            modules = new AbstractEnsemble.EnsembleModule[constituents.size()];
            int i = 0;
            for(EnhancedAbstractClassifier constituent : constituents) {
                modules[i] = new AbstractEnsemble.EnsembleModule();
                modules[i].setClassifier(constituent);
                modules[i].trainResults = constituent.getTrainResults();
                i++;
            }
            weightingScheme.defineWeightings(modules, data.numClasses());
            votingScheme.trainVotingScheme(modules, data.numClasses());
            trainResults = new ClassifierResults();
            for(i = 0; i < data.size(); i++) {
                StopWatch predictionTimer = new StopWatch(true);
                double[] distribution = votingScheme.distributionForTrainInstance(modules, i);
                int prediction = ArrayUtilities.argMax(distribution);
                predictionTimer.pause();
                double trueClassValue = data.get(i).classValue();
                trainResults.addPrediction(trueClassValue, distribution, prediction, predictionTimer.getTimeNanos(), null);
            }
        }
        checkpoint(true);
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
    }

    @Override public double[] distributionForInstance(final Instance instance) throws Exception {
        return votingScheme.distributionForInstance(modules, instance);
    }

    public List<EnhancedAbstractClassifier> getConstituents() {
        return constituents;
    }

    public void setConstituents(final List<EnhancedAbstractClassifier> constituents) {
        this.constituents = constituents;
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

}
