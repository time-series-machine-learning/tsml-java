package tsml.classifiers.distance_based.optimised;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.TestTimeContractable;
import tsml.classifiers.TrainEstimateTimeable;
import tsml.classifiers.distance_based.distances.dtw.spaces.DTWDistanceSpace;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.TimedTest;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.CheckpointConfig;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ProgressiveBuild;
import tsml.classifiers.distance_based.utils.classifiers.contracting.TimedTrain;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearch;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatchable;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.ClassifierTools;
import weka.core.Instances;

import java.util.List;
import java.util.Objects;

public class OptimisedClassifier extends BaseClassifier implements Checkpointed, ProgressiveBuild, TimedTrain, TimedTest,
                                                                     ContractedTrain, MemoryWatchable,
                                                                           TrainEstimateTimeable, IterableBuild {

    public static void main(String[] args) throws Exception {
        final int seed = 0;
        final OptimisedClassifier classifier = new OptimisedClassifier();
        final KnnAgent agent = new KnnAgent();
//        agent.setParamSpaceBuilder(new EDistanceSpace());
        agent.setParamSpaceBuilder(new DTWDistanceSpace());
        agent.setSearch(new RandomSearch());
        agent.setUsePatience(true);
        classifier.setAgent(agent);
        final Instances[] instances = DatasetLoading.sampleGunPoint(seed);
//        while(instances[0].size() > 5) {
//            instances[0].remove(instances[0].size() - 1);
//        }
        ClassifierTools.trainTestPrint(classifier, instances, seed);
    }
    
    private Agent agent;
    private Evaluation bestEvaluation;
    private final CheckpointConfig checkpointConfig = new CheckpointConfig();
    private final StopWatch runTimer = new StopWatch();
    private final StopWatch testTimer = new StopWatch();
    private final StopWatch trainEstimateTimer = new StopWatch();
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private long trainTimeLimit = -1;
    private long testTimeLimit = -1;
    private long longestEvaluationTime;
    private TimeSeriesInstances trainData;
    private boolean explore;
    
    public void setTrainData(TimeSeriesInstances trainData) {
        this.trainData = Objects.requireNonNull(trainData);
    }

    @Override public boolean isFullyBuilt() {
        return IterableBuild.super.isFullyBuilt();
    }

    @Override public void beforeBuild() throws Exception {
        long timeStamp = System.nanoTime();
        memoryWatcher.start();
        checkpointConfig.setLogger(getLogger());

        if(isRebuild()) {
            // attempt to load from a checkpoint
            if(loadCheckpoint()) {
                memoryWatcher.start();
                checkpointConfig.setLogger(getLogger());
            } else {
                super.buildClassifier(trainData);
                if(agent == null) {
                    throw new IllegalStateException("agent must be set");
                }
                checkRandom();
                copyRandomTo(agent);
                agent.buildAgent(trainData);
                longestEvaluationTime = 0;
                runTimer.reset();
                trainEstimateTimer.reset();
            }

        }
        runTimer.start(timeStamp);
        
        memoryWatcher.stop();
        runTimer.stop();
    }

    @Override public boolean hasNextBuildStep() throws Exception {

        // x2 on the longest eval time because we need an extra slot of eval time to recompute the results
        // granted this means we end up re-evaluating the best classifier again, but a) some classifiers like knn bare
        // little cost in doing this and b) it's better to do this than store all the results for every single
        // evaluation and eat a million gigs of ram
        return agent.hasNext() && insideTrainTimeLimit(getRunTime() + longestEvaluationTime * 2);
    }

    @Override public void nextBuildStep() throws Exception {
        runTimer.start();
        memoryWatcher.stop();
        
        final long timeStamp = System.nanoTime();
        final Evaluation evaluation = agent.next();
        if(explore != evaluation.isExplore()) {
            explore = !explore;
            getLogger().info("----");
        }
        execute(evaluation, trainData);
        agent.feedback(evaluation);
        longestEvaluationTime = Math.max(longestEvaluationTime, System.nanoTime() - timeStamp);
        
        memoryWatcher.stop();
        runTimer.stop();
    }

    @Override public void afterBuild() throws Exception {
        runTimer.start();
        memoryWatcher.start();
        
        final List<Evaluation> bestEvaluations = agent.getBestEvaluations();
        bestEvaluation = RandomUtils.choice(bestEvaluations, getRandom());
        // best evaluation may need results recalculating if they're dropped because of memory requirements
        if(bestEvaluation.getResults() == null) {
            getLogger().info("----");
            getLogger().info("recomputing evaluation for best");
            execute(bestEvaluation, trainData);
        }
        trainResults = bestEvaluation.getResults();

        memoryWatcher.stop();
        runTimer.stop();
        // we do this after the timers have been stopped, etc, otherwise times are inaccurate
        // this sets the classifier name / dataset name / timings / meta in the results
        ResultUtils.setInfo(trainResults, this, trainData);
    }
    
    private void execute(Evaluation evaluation, TimeSeriesInstances trainData) throws Exception {
        trainEstimateTimer.start();
        final TSClassifier classifier = evaluation.getClassifier();
        final ClassifierResults results = evaluation.getEvaluator().evaluate(classifier, trainData);
        evaluation.setResults(results);
        getLogger().info(evaluation::toStringVerbose);
        trainEstimateTimer.stop();
    }

    @Override public double[] distributionForInstance(final TimeSeriesInstance inst) throws Exception {
        testTimer.resetAndStart();
        final TSClassifier classifier = bestEvaluation.getClassifier();
        if(classifier instanceof TestTimeContractable) {
            ((TestTimeContractable) classifier).setTestTimeLimit(testTimeLimit);
        }
        final double[] distribution = classifier.distributionForInstance(inst);
        testTimer.stop();
        return distribution;
    }

    public Agent getAgent() {
        return agent;
    }

    public void setAgent(final Agent agent) {
        this.agent = agent;
    }

    public CheckpointConfig getCheckpointConfig() {
        return checkpointConfig;
    }

    @Override public long getTrainTime() {
        return getRunTime() - getCheckpointingTime() - getTrainEstimateTime();
    }

    @Override public long getTrainEstimateTime() {
        return trainEstimateTimer.elapsedTime();
    }

    @Override public long getRunTime() {
        return runTimer.elapsedTime();
    }

    @Override public long getTestTime() {
        return testTimer.elapsedTime();
    }

    public long getTestTimeLimit() {
        return testTimeLimit;
    }

    public void setTestTimeLimit(final long testTimeLimit) {
        this.testTimeLimit = testTimeLimit;
    }

    @Override public long getTrainTimeLimit() {
        return trainTimeLimit;
    }

    @Override public void setTrainTimeLimit(final long trainTimeLimit) {
        this.trainTimeLimit = trainTimeLimit;
    }

    @Override public long getMaxMemoryUsage() {
        return memoryWatcher.getMaxMemoryUsage();
    }
}
