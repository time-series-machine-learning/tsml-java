/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.proximity;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.evaluators.OutOfBagEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.TrainEstimateTimeable;
import tsml.classifiers.distance_based.utils.classifiers.*;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.CheckpointConfig;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.classifiers.configs.Builder;
import tsml.classifiers.distance_based.utils.classifiers.configs.Config;
import tsml.classifiers.distance_based.utils.classifiers.configs.Configs;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatchable;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.ClassifierTools;

import java.util.*;
import java.util.concurrent.TimeUnit;

import static utilities.ArrayUtilities.*;
import static utilities.Utilities.argMax;

/**
 * Proximity Forest
 * <p>
 * Contributors: goastler
 */
public class ProximityForest extends BaseClassifier implements ContractedTrain, ContractedTest, TrainEstimateTimeable,
                                                                       Checkpointed, MemoryWatchable {

    public static void main(String[] args) throws Exception {
////        Thread.sleep(10000);
        for(int i = 0; i < 1; i++) {
            int seed = i;
            ProximityForest classifier = CONFIGS.get("PF_R5").build();
//            classifier.setEstimateOwnPerformance(true);
//            classifier.setEstimatorMethod("oob");
            classifier.setSeed(seed);
//            classifier.setNumTreeLimit(3);
//            classifier.setCheckpointPath("checkpoints");
//            classifier.setNumTreeLimit(14);
//            classifier.setCheckpointPath("checkpoints/PF");
//            classifier.setTrainTimeLimit(1, TimeUnit.SECONDS);
//            classifier.setTrainTimeLimit(30, TimeUnit.SECONDS);
            classifier.setLogLevel("all");
            classifier.setEstimateOwnPerformance(true);
            classifier.setEstimatorMethod("oob");
            classifier.setCheckpointPath("checkpoints");
            classifier.setCheckpointInterval(5, TimeUnit.SECONDS);
            ClassifierTools
                    .trainTestPrint(classifier, DatasetLoading
                                                        .sampleDataset("/bench/phd/data/all_2019", "ItalyPowerDemand", seed), seed);
        }
        //        Thread.sleep(10000);

//        String root = "/bench/phd/experiments/";
//        String expName = "pf_correctness";
//        String expDir = root + "/" + expName + "/";
//        String analysisName = "analysis_pf_v2_vs_pf_wrapped";
//        
//        
//        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(expDir, analysisName, 30);
//        
//        mce.setDatasets("/bench/phd/data/lists/uni_2015_pigless.txt");
//        mce.readInClassifier("PF_R5", "mine", expDir + "v1/results/");
//        mce.readInClassifier("ORIG_PF", "theirs", expDir + "wrapped/results/");
//        mce.setTestResultsOnly(true);
//        mce.setUseAllStatistics();
//        mce.setIgnoreMissingResults(true);
//        mce.setBuildMatlabDiagrams(true, true);
//        mce.runComparison();
    }
    
    public static final Configs<ProximityForest> CONFIGS = buildConfigs().immutable();
    
    public static Configs<ProximityForest> buildConfigs() {
        final Configs<ProximityForest> configs = new Configs<>();
        
        configs.add("PF_R1", "PF with 1 split per node", ProximityForest::new,
            pf -> {
                pf.setTrainTimeLimit(-1);
                pf.setTestTimeLimit(-1);
                pf.setEstimatorMethod("none");
                pf.setEstimateOwnPerformance(false);
                pf.setNumTreeLimit(100);
                pf.setProximityTreeFactory(ProximityTree.CONFIGS.get("PT_R1"));
        });
        configs.add("PF_R5", "PF with 5 splits per node", "PF_R1", pf -> pf.setProximityTreeFactory(ProximityTree.CONFIGS.get("PT_R5")));
        configs.add("PF_R10", "PF with 10 splits per node", "PF_R1", pf -> pf.setProximityTreeFactory(ProximityTree.CONFIGS.get("PT_R10")));
        
        for(String method : Arrays.asList("OOB", "CV")) {
            for(String name : Arrays.asList("PF_R1", "PF_R5", "PF_R10")) {
                final Config<ProximityForest> conf = configs.get(name);
                configs.add(name + "_" + method, method, conf, pf -> pf.setEstimatorMethod(method));
            }
        }
        
        return configs;
    }

    public ProximityForest() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        CONFIGS.get("PF_R5").configure(this);
    }
    
    private static final long serialVersionUID = 1;
    // the list of trees in this forest
    private List<ProximityTree> trees;
    private List<Evaluator> treeEvaluators;
    private List<ClassifierResults> treeTrainResults;
    // the number of trees
    private int numTreeLimit;
    // the train time limit / contract
    private long trainTimeLimit;
    // the test time limit / contract
    private long testTimeLimit;
    // how long this took to build. THIS INCLUDES THE TRAIN ESTIMATE!
    private final StopWatch runTimer = new StopWatch();
    // how long testing took
    private final StopWatch testTimer = new StopWatch();
    // watch for memory usage
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    // the longest tree build time for predicting train time requirements
    private long longestTrainStageTime;
    // the method of setting the config of the trees
    private Builder<ProximityTree> proximityTreeBuilder;
    // checkpoint config
    private final CheckpointConfig checkpointConfig = new CheckpointConfig();
    // train estimate variables
    private double[][] trainEstimateDistributions;
    private final StopWatch evaluationTimer = new StopWatch();
    private long[] trainEstimatePredictionTimes;

    @Override public long getMaxMemoryUsage() {
        return memoryWatcher.getMaxMemoryUsage();
    }

    @Override
    public void buildClassifier(TimeSeriesInstances trainData) throws Exception {
        // timings:
            // train time tracks the time spent processing the algorithm. This should not be used for contracting.
            // run time tracks the entire time spent processing, whether this is work towards the algorithm or otherwise (e.g. saving checkpoints to disk). This should be used for contracting.
            // evaluation time tracks the time spent evaluating the quality of the classifier, i.e. producing an estimate of the train data error.
            // checkpoint time tracks the time spent loading / saving the classifier to disk.
        // record the start time
        final long timeStamp = System.nanoTime();
        memoryWatcher.start();
        checkpointConfig.setLogger(getLogger());
        // several scenarios for entering this method:
            // 1) from scratch: isRebuild() is true
                // 1a) checkpoint found and loaded, resume from wherever left off
                // 1b) checkpoint not found, therefore initialise classifier and build from scratch
            // 2) rebuild off, i.e. buildClassifier has been called before and already handled 1a or 1b. We can safely continue building from current state. This is often the case if a smaller contract has been executed (e.g. 1h), then the contract is extended (e.g. to 2h) and we enter into this method again. There is no need to reinitialise / discard current progress - simply continue on under new constraints.
        if(isRebuild()) {
            // case (1)
            if(loadCheckpoint()) {
                memoryWatcher.start();
                checkpointConfig.setLogger(getLogger());
            } else {
                // case (1b)
                memoryWatcher.reset();
                // let super build anything necessary (will handle isRebuild accordingly in super class)
                super.buildClassifier(trainData);
                // if rebuilding
                // then init vars
                // build timer is already started so just clear any time already accrued from previous builds. I.e. keep the time stamp of when the timer was started, but clear any record of accumulated time
                runTimer.reset();
                // clear other timers entirely
                evaluationTimer.reset();
                checkpointConfig.resetCheckpointingTime();
                // no constituents to start with
                trees = new ArrayList<>();
                treeEvaluators = new ArrayList<>();
                treeTrainResults = new ArrayList<>();
                // zero tree build time so the first tree build will always set the bar
                longestTrainStageTime = 0;
                // init the running train estimate variables if using OOB
                if(estimateOwnPerformance && estimator.equals(EstimatorMethod.OOB)) {
                    trainEstimatePredictionTimes = new long[trainData.numInstances()];
                    trainEstimateDistributions = new double[trainData.numInstances()][trainData.numClasses()];
                    treeEvaluators = new ArrayList<>();
                    treeTrainResults = new ArrayList<>();
                }
            }  // case (1a)

        } // else case (2)
        
        // update the run timer with the start time of this session 
        // as the runtimer has been overwritten with the one from the checkpoint (if loaded)
        // or the classifier has been initialised from scratch / resumed and can just start from the timestamp
        runTimer.start(timeStamp);
        evaluationTimer.checkStopped();
        
        LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLogger(), "train");
        // whether work has been done in this call to buildClassifier
        boolean workDone = false;
        // maintain a timer for how long trees take to build
        final StopWatch trainStageTimer = new StopWatch();
        // while remaining time / more trees need to be built
        if(estimateOwnPerformance && estimator.equals(EstimatorMethod.CV)) {
            // if there's a train contract then need to spend half the time CV'ing
            evaluationTimer.start();
            getLogger().info("cross validating");
            // copy over the same configuration as this instance
            final ProximityForest pf = deepCopy();
            // turn off train estimation otherwise infinite recursion!
            pf.setEstimateOwnPerformance(false);
            // reset the state of pf to build from scratch
            pf.setRebuild(true);
            pf.setLogger(getLogger());
            // disable checkpointing on pf
            pf.setCheckpointPath(null);
            // evaluate pf using cross validation
            final CrossValidationEvaluator cv = new CrossValidationEvaluator();
            final int numFolds = 10;
            cv.setNumFolds(numFolds);
            cv.setCloneData(false);
            cv.setSeed(getSeed());
            cv.setSetClassMissing(false);
            if(hasTrainTimeLimit()) {
                // must set PF train contract. The total CV time should be half of the train contract. This must be divided by numFolds+1 for a per-fold train contract. Note the +1 is to account for testing, as numFold test batches will be used which is the same size as the data, i.e. equivalent to another fold. Adds a single nano on in case the contract divides up to zero time.
                pf.setTrainTimeLimit(findRemainingTrainTime(runTimer.elapsedTime()) / 2 / (numFolds + 1) + 1);
            }
            // evaluate
            trainResults = cv.evaluate(pf, trainData);
            // stop timer and set meta info in results
            evaluationTimer.stop();
            LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLogger(), "train");
            getLogger().info("cross validation finished, acc " + trainResults.getAcc());
        }
        while(
                // there's remaining trees to be built
                insideNumTreeLimit()
                &&
                // and there's remaining time left to build more trees
                insideTrainTimeLimit(runTimer.elapsedTime() + longestTrainStageTime)
        ) {
            // reset the tree build timer
            trainStageTimer.resetAndStart();
            // setup a new tree
            final int treeIndex = trees.size();
            final ProximityTree tree = proximityTreeBuilder.build();
            final int treeSeed = rand.nextInt();
            tree.setSeed(treeSeed);
            // setup the constituent
            trees.add(tree);
            // estimate the performance of the tree
            if(estimateOwnPerformance && estimator.equals(EstimatorMethod.OOB)) {
                // the timer for contracting the estimate of train error
                evaluationTimer.start();
                // build train estimate based on method
                final OutOfBagEvaluator oobe = new OutOfBagEvaluator();
                oobe.setCloneClassifier(false);
                oobe.setSeed(treeSeed);
                treeEvaluators.add(oobe);
                getLogger().info(() -> "oob evaluating tree " + treeIndex);
                // evaluate the tree
                final ClassifierResults treeEvaluationResults = oobe.evaluate(tree, trainData);
                treeTrainResults.add(treeEvaluationResults);
                // for each index in the test data of the oobe
                final List<Integer> outOfBagTestDataIndices = oobe.getOutOfBagTestDataIndices();
                // for each instance in the oobe test data, add the distribution and prediction time to the corresponding instance predictions in the train estimate results
                for(int oobeIndex = 0; oobeIndex < outOfBagTestDataIndices.size(); oobeIndex++) {
                    final int trainDataIndex = outOfBagTestDataIndices.get(oobeIndex);
                    // get the corresponding distribution from the oobe results
                    double[] distribution = treeEvaluationResults.getProbabilityDistribution(oobeIndex);
                    distribution = vote(treeIndex, distribution);
                    // get the corresponding distribution from the train estimate distribution
                    // add tree's distribution for this instance onto the overall train estimate distribution for this instance
                    add(trainEstimateDistributions[trainDataIndex], distribution);
                    // add the prediction time from the oobe to the time for this instance in the train estimate
                    trainEstimatePredictionTimes[trainDataIndex] += treeEvaluationResults.getPredictionTime(oobeIndex);
                }
                treeEvaluationResults.setErrorEstimateMethod(getEstimatorMethod());
                evaluationTimer.stop();
            }
            // build the tree if not producing train estimate OR rebuild after evaluation
            getLogger().info(() -> "building tree " + treeIndex);
            tree.setRebuild(true);
            tree.buildClassifier(trainData);
            // tree fully built
            trainStageTimer.stop();
            workDone = true;
            // optional checkpoint
            saveCheckpoint();
            // update train timer
            LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLogger(), "train");
            // update longest tree build time
            longestTrainStageTime = Math.max(longestTrainStageTime, trainStageTimer.elapsedTime());
        }
        // if work has been done towards estimating the train error via OOB
        if(estimateOwnPerformance && workDone && estimator.equals(EstimatorMethod.OOB)) {
            // must format the OOB errors into classifier results
            evaluationTimer.start();
            getLogger().info("finalising train estimate");
            // add the final predictions into the results
            for(int i = 0; i < trainData.numInstances(); i++) {
                double[] distribution = trainEstimateDistributions[i];
                // i.e. [71, 29] --> [0.71, 0.29]
                // copies as to not alter the original distribution. This is helpful if more trees are added in the future to avoid having to denormalise
                // set ignoreZeroSum to true to produce a uniform distribution if there is no evaluation of the instance (i.e. it never ended up in an out-of-bag test data set because too few trees were built, say)
                distribution = normalise(copy(distribution), true);
                // get the prediction, rand tie breaking if necessary
                final double prediction = argMax(distribution, rand);
                final double classValue = trainData.get(i).getLabelIndex();
                trainResults.addPrediction(classValue, distribution, prediction, trainEstimatePredictionTimes[i], null);
            }
            evaluationTimer.stop();
        }
        
        // finish up
        LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLogger(), "train");
        // sanity check that all timers have been stopped
        evaluationTimer.checkStopped();
        testTimer.checkStopped();
        memoryWatcher.stop();
        runTimer.stop();
        
        // save the final checkpoint
        if(workDone) {
            ResultUtils.setInfo(trainResults, this, trainData);
            forceSaveCheckpoint();
        }
    }

    @Override
    public double[] distributionForInstance(final TimeSeriesInstance instance) throws Exception {
        // start timer
        testTimer.resetAndStart();
        // track how long every stage (i.e. every tree prediction) takes, recording the longest
        long longestTestStageTimeNanos = 0;
        // time each stage of the prediction
        final StopWatch testStageTimer = new StopWatch();
        final double[] finalDistribution = new double[getNumClasses()];
        // while there's remaining constituents to be examined and remaining test time
        for(int i = 0;
            i < trees.size()
            &&
            (testTimeLimit <= 0 || testTimer.elapsedTime() + longestTestStageTimeNanos < testTimeLimit)
                ; i++) {
            testStageTimer.resetAndStart();
            // let the constituent vote
            final double[] distribution = vote(i, instance);
            // add the vote to the total votes
            add(finalDistribution, distribution);
            // update timings
            testStageTimer.stop();
            longestTestStageTimeNanos = Math.max(longestTestStageTimeNanos, testStageTimer.elapsedTime());
        }
        // normalise the final vote, i.e. [71,29] --> [.71,.29]
        normalise(finalDistribution);
        // finished prediction so stop timer
        testTimer.stop();
        return finalDistribution;
    }
    
    private double[] vote(int constituentIndex, TimeSeriesInstance instance) throws Exception {
        final ProximityTree tree = trees.get(constituentIndex);
        final double[] distribution = tree.distributionForInstance(instance);
        return vote(constituentIndex, distribution);
    }
    
    private double[] vote(int constituentIndex, double[] distribution) {
        // vote for the highest probability class
        final int index = argMax(distribution, getRandom());
        return oneHot(distribution.length, index);
    }

    @Override public boolean isFullyBuilt() {
        return trees != null && trees.size() == numTreeLimit;
    }

    public boolean insideNumTreeLimit() {
        return !hasNumTreeLimit() || trees.size() < numTreeLimit;
    }

    public boolean hasNumTreeLimit() {
        return numTreeLimit > 0;
    }

    public int getNumTreeLimit() {
        return numTreeLimit;
    }

    public void setNumTreeLimit(final int numTreeLimit) {
        this.numTreeLimit = numTreeLimit;
    }

    public Builder<ProximityTree> getProximityTreeFactory() {
        return proximityTreeBuilder;
    }

    public void setProximityTreeFactory(final Builder<ProximityTree> proximityTreeBuilder) {
        this.proximityTreeBuilder = Objects.requireNonNull(proximityTreeBuilder);
    }

    @Override public long getRunTime() {
        return runTimer.elapsedTime();
    }

    @Override
    public long getTrainTimeLimit() {
        return trainTimeLimit;
    }

    /**
     * Overriding TrainTimeContract methods
     * @param nanos
     */
    @Override
    public void setTrainTimeLimit(final long nanos) {
        trainTimeLimit = nanos;
    }
    @Override
    public boolean withinTrainContract(long start) {
        return start<trainContractTimeNanos;
    }

    @Override
    public long getTestTimeLimit() {
        return testTimeLimit;
    }

    @Override
    public void setTestTimeLimit(final long nanos) {
        testTimeLimit = nanos;
    }

    @Override public long getTrainTime() {
        // train time is the overall run time minus any time spent estimating the train error and minus any time spent checkpointing
        return getRunTime() - getTrainEstimateTime() - getCheckpointingTime();
    }

    @Override public long getTrainEstimateTime() {
        return evaluationTimer.elapsedTime();
    }

    @Override public long getTestTime() {
        return testTimer.elapsedTime();
    }

    @Override public CheckpointConfig getCheckpointConfig() {
        return checkpointConfig;
    }
}
