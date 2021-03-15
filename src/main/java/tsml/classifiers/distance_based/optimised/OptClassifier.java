package tsml.classifiers.distance_based.optimised;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.TestTimeContractable;
import tsml.classifiers.distance_based.distances.dtw.spaces.DTWDistanceSpace;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.TimedTest;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.CheckpointConfig;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Chkpt;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ProgressiveBuild;
import tsml.classifiers.distance_based.utils.classifiers.contracting.TimedTrain;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearch;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.ClassifierTools;
import weka.core.Instances;

import java.util.List;

public class OptClassifier extends BaseClassifier implements Chkpt, ProgressiveBuild, TimedTrain, TimedTest,
                                                                     ContractedTrain {

    public static void main(String[] args) throws Exception {
        final int seed = 0;
        final OptClassifier classifier = new OptClassifier();
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
    private StopWatch runTimer = new StopWatch();
    private StopWatch testTimer = new StopWatch();
    private long trainTimeLimit = -1;
    private long testTimeLimit = -1;
    private long longestEvaluationTime;

    @Override public boolean isFullyBuilt() {
        return agent.hasNext();
    }

    @Override public void buildClassifier(final TimeSeriesInstances trainData) throws Exception {
        long timeStamp = System.nanoTime();
        runTimer.start(timeStamp);
        
        if(isRebuild()) {
            // attempt to load from a checkpoint
            if(loadCheckpoint()) {
                // internals of this object have been changed, so the run timer needs restarting
                runTimer.start(timeStamp); // start from same time point though, no time missed while dealing with chkp
            } else {
                super.buildClassifier(trainData);
                if(agent == null) {
                    throw new IllegalStateException("agent must be set");
                }
                checkRandom();
                copyRandomTo(agent);
                agent.buildAgent(trainData);
                longestEvaluationTime = 0;
            }
            
        }
        
        boolean workDone = false;
        // x2 on the longest eval time because we need an extra slot of eval time to recompute the results
        // granted this means we end up re-evaluating the best classifier again, but a) some classifiers like knn bare
        // little cost in doing this and b) it's better to do this than store all the results for every single
        // evaluation and eat a million gigs of ram
        boolean explore = false;
        while(agent.hasNext() && insideTrainTimeLimit(getRunTime() + longestEvaluationTime * 2)) {
            timeStamp = System.nanoTime();
            final Evaluation evaluation = agent.next();
            if(explore != evaluation.isExplore()) {
                explore = !explore;
                getLogger().info("----");
            }
            execute(evaluation, trainData);
            agent.feedback(evaluation);
            workDone = true;
            longestEvaluationTime = Math.max(longestEvaluationTime, System.nanoTime() - timeStamp);
        }
        
        if(workDone || trainResults.getPredClassVals() == null) {
            final List<Evaluation> bestEvaluations = agent.getBestEvaluations();
            bestEvaluation = RandomUtils.choice(bestEvaluations, getRandom());
            // best evaluation may need results recalculating if they're dropped because of memory requirements
            if(bestEvaluation.getResults() == null) {
                getLogger().info("----");
                getLogger().info("recomputing evaluation for best");
                execute( bestEvaluation, trainData);
            }
            trainResults = bestEvaluation.getResults();
        }
        
        runTimer.stop();

        // we do this after the timers have been stopped, etc, otherwise times are inaccurate
        // this sets the classifier name / dataset name / timings / meta in the results
        ResultUtils.setInfo(trainResults, this, trainData);
    }
    
    private void execute(Evaluation evaluation, TimeSeriesInstances trainData) throws Exception {
        final TSClassifier classifier = evaluation.getClassifier();
        final ClassifierResults results = evaluation.getEvaluator().evaluate(classifier, trainData);
        evaluation.setResults(results);
        getLogger().info(evaluation::toStringVerbose);
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
        return getRunTime() - getCheckpointingTime();
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
}
