package machine_learning.classifiers.tuned.incremental;

import com.google.common.primitives.Doubles;
import tsml.classifiers.*;
import utilities.*;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;

import static utilities.collections.Utils.replace;

public class IncTuner extends EnhancedAbstractClassifier implements TrainTimeContractable, MemoryWatchable,
                                                                    Checkpointable {

    /*
        how do we do tuning?

        general idea:
            we have benchmarks which contain a classifier and a score (we don't care whether that's acc or
            auroc or anything, as long as it's comparable).

            we iterate over benchmarks, each iteration returning a set of benchmarks which have been adjusted. we must
            therefore have an id field in benchmarks to differentiate them and recognise when we've already seen them.

            when there are no more iterations then we've collected all benchmarks possible. we must next whittle them
            down to a subset / filter them (i.e. choose the best 10, say).

            the best benchmarks are then ensembled together to represent a classifier. if this is only 1 benchmark then
            obviously ensembling has no effect, we're just wrapping that benchmark at that point.

        considerations:
            every benchmark id represents a new benchmark. if a benchmark is improved then either the benchmark object
            must be kept the same or the id of the previous benchmark copied over to the next.

            contracting becomes complex if we implement it here, therefore it is best left to the iteration strategy to
            handle contracting. we can have a naive version which simply iterates over benchmarks until hitting the
            contract.

            it's a similar deal with checkpointing. the classifier in each benchmark may checkpoint themselves. we don't
            want this behaviour as this may cause too frequent or infrequent checkpointing. therefore, we must do the
            checkpointing here manually.

            we may also be running this in parallel, therefore we need to detect that and run a single benchmark only.
            that way this can be parallelised. we cannot operate in parallel with a contract though, can we? perhaps
            we could divide the contract by how many benchmarks we receive? that would require knowing the number of
            benchmarks to come, which could only be done by draining the iterator fully, which may not work with certain
            iteration patterns. we would have to require the iterator to know this and only provide however many
            benchmarks we're looking at without doing any improvement or anything.

        constraints:
            benchmark explorer *MUST* handle the resource monitors!!

     */

    public IncTuner() {
        super(true);
    }

    public boolean isLogBenchmarks() {
        return logBenchmarks;
    }

    public void setLogBenchmarks(boolean logBenchmarks) {
        this.logBenchmarks = logBenchmarks;
    }

    public boolean isDebugBenchmarks() {
        return debugBenchmarks;
    }

    public void setDebugBenchmarks(boolean debugBenchmarks) {
        this.debugBenchmarks = debugBenchmarks;
    }

    public interface InitFunction extends Serializable, ParamHandler {
        void init(Instances trainData);
    }

    protected Agent agent = new Agent() {
        @Override
        public Set<EnhancedAbstractClassifier> findFinalClassifiers() {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean hasNext() {
            throw new UnsupportedOperationException();
        }

        @Override public EnhancedAbstractClassifier next() {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean feedback(EnhancedAbstractClassifier classifier) {
            throw new UnsupportedOperationException();
        }
    };
    protected transient Set<EnhancedAbstractClassifier> benchmarks = new HashSet<>();
    protected Ensembler ensembler = Ensembler.byScore(benchmark -> benchmark.getTrainResults().getAcc());
    protected List<Double> ensembleWeights = new ArrayList<>();
    protected InitFunction initFunction = instances -> {};
    protected MemoryWatcher memoryWatcher = new MemoryWatcher();
    protected StopWatch trainTimer = new StopWatch();
    protected StopWatch trainEstimateTimer = new StopWatch();
    protected long trainTimeLimitNanos = -1;
    public static final String BENCHMARK_ITERATOR_FLAG = "b";
    public static final String INIT_FUNCTION_FLAG = "i";
    private boolean debugBenchmarks = false;
    private boolean logBenchmarks = false;
    private boolean rebuild = true; // shadows super
    private transient Instances trainData;
    private boolean incrementalMode = true;
    protected String savePath;
    protected String loadPath;

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

    public boolean isIncrementalMode() {
        return incrementalMode;
    }

    public void setIncrementalMode(boolean incrementalMode) {
        this.incrementalMode = incrementalMode;
    }

    public StopWatch getTrainEstimateTimer() {
        return trainEstimateTimer;
    }

    @Override public ParamSet getParams() {
        return TrainTimeContractable.super.getParams()
                                    .add(BENCHMARK_ITERATOR_FLAG, agent)
                                    .add(INIT_FUNCTION_FLAG, initFunction);
    }

    @Override public void setParams(final ParamSet params) {
        TrainTimeContractable.super.setParams(params);
        ParamHandler.setParam(params, BENCHMARK_ITERATOR_FLAG, this::setAgent, Agent.class);
        ParamHandler.setParam(params, INIT_FUNCTION_FLAG, this::setInitFunction, InitFunction.class); // todo finish params
    }

    @Override
    public boolean saveToCheckpoint() throws Exception {
        return false;
    }

    @Override
    public boolean loadFromCheckpoint() throws Exception {
        return false;
    }

    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    @Override
    public void setRebuild(boolean rebuild) {
        this.rebuild = rebuild;
        super.setRebuild(rebuild);
    }



    @Override public void buildClassifier(final Instances trainData) throws Exception {
        // setup parent
        super.buildClassifier(trainData);
        // enable resource monitors
        memoryWatcher.enableAnyway();
        trainEstimateTimer.checkDisabled();
        trainTimer.enableAnyway();
        // we're not built at the moment
        built = false;
        this.trainData = trainData;
        // if we're rebuilding
        if(rebuild) {
            // reset resource monitors
            trainTimer.resetAndEnable();
            memoryWatcher.resetAndEnable();
            initFunction.init(trainData); // todo do the build/rebuild pattern on this with set instances
            // reset this switch so we don't reset again next time (unless someone calls the setter)
            rebuild = false;
        }
        // going into train estimate phase
        trainTimer.disable();
        trainEstimateTimer.enable(); // todo should this be train estimate or just train time?
        // while we've got more benchmarks to examine
        while(hasNextBuildTick()) {
            // get those benchmarks
            nextBuildTick();
        }
        // sanity check resource monitors (as the benchmark iterator *should* have been using them)
        trainTimer.checkDisabled();
        trainEstimateTimer.checkEnabled();
        memoryWatcher.checkEnabled();
        // find the final benchmarks
        benchmarks = agent.findFinalClassifiers();
        // sanity check and ensemble
        if(benchmarks.isEmpty()) {
            logger.info(() -> "no benchmarks collected");
            ensembleWeights = new ArrayList<>();
            throw new UnsupportedOperationException("todo implement random guess here?");
        } else if(benchmarks.size() == 1) {
            logger.info(() -> "single benchmarks collected");
            ensembleWeights = new ArrayList<>(Collections.singletonList(1d));
            trainResults = benchmarks.iterator().next().getTrainResults();
        } else {
            logger.info(() -> benchmarks.size() + " benchmarks collected");
            ensembleWeights = ensembler.weightVotes(benchmarks);
            throw new UnsupportedOperationException("todo apply ensemble weights to train results"); // todo
        }
        // cleanup
        memoryWatcher.disable();
        trainEstimateTimer.disable();
        trainResults.setMemory(getMaxMemoryUsageInBytes());
        trainResults.setBuildTime(trainTimer.getTimeNanos());
        trainResults.setBuildPlusEstimateTime(getTrainTimeNanos());
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setFoldID(seed);
        trainResults.setDetails(this, trainData);
        built = true;
        this.trainData = null;
        memoryWatcher.disableAnyway();
        trainEstimateTimer.disableAnyway();
        trainTimer.disableAnyway();
    }

    protected boolean hasNextBuildTick() throws Exception {
        return hasRemainingTraining(); // todo sure this is what we want? A bit round about the houses
    }

    protected void nextBuildTick() throws Exception {
        // get the next set of benchmarks
        final boolean isExplore = agent.isExploringOrExploiting();
        EnhancedAbstractClassifier classifier = agent.next();
        boolean evaluate;
        final String classifierSavePath = savePath + classifier.getClassifierName() + File.separator;
        if(!incrementalMode && isCheckpointSavingEnabled()) {
            FileUtils.FileLock lock = new FileUtils.FileLock(classifierSavePath);
            evaluate = lock.isLocked();
        } else {
            evaluate = true;
        }
        if(evaluate) {
            if(isExplore) {
                final String classifierLoadPath = loadPath + classifier.getClassifierName() + File.separator;
                classifier = loadClassifier(classifier, classifierLoadPath, classifierSavePath); // todo add the resource monitor stats to current
                classifier.setSeed(seed);
                classifier.setDebug(isDebugBenchmarks());
                if(isLogBenchmarks()) {
                    classifier.getLogger().setLevel(getLogger().getLevel());
                } else {
                    classifier.getLogger().setLevel(Level.SEVERE);
                }
            }
            EnhancedAbstractClassifier finalClassifier = classifier;
            logger.info(() -> "evaluating " + StrUtils.toOptionValue(finalClassifier)); // todo better logs here
            classifier.setEstimateOwnPerformance(true);
            Utilities.listenToMemoryWatcher(classifier, memoryWatcher); // todo alt version for non time tracked classifiers
            Utilities.listenToTrainTimer(classifier, trainTimer);
            Utilities.listenToTrainEstimateTimer(classifier, trainEstimateTimer);
            classifier.buildClassifier(trainData);
            Utilities.unListenFromMemoryWatcher(classifier, memoryWatcher);
            Utilities.unListenFromTrainTimer(classifier, trainTimer);
            Utilities.unListenFromTrainEstimateTimer(classifier, trainEstimateTimer);
            trainEstimateTimer.enable();
            if (!agent.feedback(classifier)) {
                // no more exploitations will be made to this classifier, therefore let's save to disk
                saveClassifier(classifier, classifierSavePath);
            }
        } else {
            EnhancedAbstractClassifier finalClassifier1 = classifier;
            logger.info(() -> "lock skip evaluating " + StrUtils.toOptionValue(finalClassifier1));
        }
    }

    protected EnhancedAbstractClassifier loadClassifier(EnhancedAbstractClassifier classifier, String classifierLoadPath, String classifierSavePath) throws Exception {
        trainTimer.suspend();
        trainEstimateTimer.suspend();
        memoryWatcher.suspend();
        if(classifier instanceof Checkpointable) {
            ((Checkpointable) classifier).setSavePath(classifierSavePath);
            ((Checkpointable) classifier).setLoadPath(classifierLoadPath);
            setSkipFinalCheckpoint(true);
            ((Checkpointable) classifier).setMinCheckpointIntervalNanos(getMinCheckpointIntervalNanos());
        } else {
            // load classifier manually
            classifier = (EnhancedAbstractClassifier) CheckpointUtils.deserialise(classifierSavePath + CheckpointUtils.checkpointFileName);
        }
        memoryWatcher.unsuspend();
        trainEstimateTimer.unsuspend();
        trainTimer.unsuspend();
        return classifier;
    }

    protected void saveClassifier(EnhancedAbstractClassifier classifier, String classifierSavePath) throws Exception {
        trainTimer.suspend();
        trainEstimateTimer.suspend();
        memoryWatcher.suspend();
        if(classifier instanceof Checkpointable) {
            ((Checkpointable) classifier).setSavePath(classifierSavePath);
            ((Checkpointable) classifier).setSkipFinalCheckpoint(false);
            ((Checkpointable) classifier).saveToCheckpoint();
        } else {
            // save classifier manually
            CheckpointUtils.serialise(classifier, classifierSavePath + CheckpointUtils.checkpointFileName);
        }
        memoryWatcher.unsuspend();
        trainEstimateTimer.unsuspend();
        trainTimer.unsuspend();
    }

    public void setAgent(Agent agent) {
        this.agent = agent;
    }

    public Set<EnhancedAbstractClassifier> getBenchmarks() {
        return benchmarks;
    }

    public Ensembler getEnsembler() {
        return ensembler;
    }

    public void setEnsembler(Ensembler ensembler) {
        this.ensembler = ensembler;
    }

    public List<Double> getEnsembleWeights() {
        return ensembleWeights;
    }

    @Override
    public double[] distributionForInstance(Instance testCase) throws Exception {
        Iterator<EnhancedAbstractClassifier> benchmarkIterator = benchmarks.iterator();
        if(benchmarks.size() == 1) {
            return benchmarkIterator.next().distributionForInstance(testCase);
        }
        double[] distribution = new double[numClasses];
        for(int i = 0; i < benchmarks.size(); i++) {
            if(!benchmarkIterator.hasNext()) {
                throw new IllegalStateException("iterator incorrect");
            }
            EnhancedAbstractClassifier benchmark = benchmarkIterator.next();
            double[] constituentDistribution = benchmark.distributionForInstance(testCase);
            ArrayUtilities.multiplyInPlace(constituentDistribution, ensembleWeights.get(i));
            ArrayUtilities.addInPlace(distribution, constituentDistribution);
        }
        ArrayUtilities.normaliseInPlace(distribution);
        return distribution;
    }

    @Override
    public double classifyInstance(Instance testCase) throws Exception {
        return ArrayUtilities.bestIndex(Doubles.asList(distributionForInstance(testCase)), rand);
    }

    public InitFunction getInitFunction() {
        return initFunction;
    }

    public void setInitFunction(final InitFunction initFunction) {
        this.initFunction = initFunction;
    }

    @Override public void setTrainTimeLimitNanos(final long nanos) {
        trainTimeLimitNanos = nanos;
    }

    @Override public long predictNextTrainTimeNanos() {
        return agent.predictNextTimeNanos();
    }

    @Override public boolean isBuilt() {
        return !agent.hasNext();
    }

    @Override public long getTrainTimeNanos() {
        return trainTimer.getTimeNanos() + trainEstimateTimer.getTimeNanos();
    }

    @Override public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    @Override public long getTrainTimeLimitNanos() {
        return trainTimeLimitNanos;
    }

    public Agent getAgent() {
        return agent;
    }
}
