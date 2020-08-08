package tsml.classifiers.distance_based.proximity;

import evaluation.MultipleClassifierEvaluation;
import evaluation.evaluators.Evaluator;
import evaluation.evaluators.OutOfBagEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.Configurer;
import tsml.classifiers.distance_based.utils.classifiers.EnumBasedConfigurer;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.BaseCheckpointer;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointer;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.system.timing.TimedTest;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrain;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrainEstimate;
import utilities.ArrayUtilities;
import utilities.ClassifierTools;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.logging.Logger;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ProximityForest extends BaseClassifier implements ContractedTrain, ContractedTest, TimedTrain, TimedTrainEstimate, TimedTest, WatchedMemory, Checkpointed {

    public static void main(String[] args) throws Exception {
        for(int i = 1; i < 2; i++) {
            int seed = i;
            ProximityForest classifier = new ProximityForest();
            Config.PF_R5.configure(classifier);
//            classifier.setEstimateOwnPerformance(true);
            classifier.setSeed(seed);
            classifier.setRebuildConstituentAfterEvaluation(true);
//            classifier.setNumTreeLimit(2);
//            classifier.setCheckpointPath("checkpoints/PF");
//            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            ClassifierTools
                    .trainTestPrint(classifier, DatasetLoading.sampleDataset("/bench/phd/datasets/all/", "MiddlePhalanxOutlineCorrect", seed), seed);
        }
        //        Thread.sleep(10000);

//        String root = "/bench/phd/experiments";
//        String expName = "pf_correctness";
//        String expDir = root + "/" + expName;
//        String analysisName = "analysis_v3";
//        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(expDir + "/", analysisName, 10);
//        mce.setDatasets("/bench/phd/datasets/lists/2015.txt");
//        mce.readInClassifier("ORIG_PF", "orig", expDir + "/v1/results/");
//        mce.readInClassifier("PF_R5", "v3", expDir + "/v3/results/");
//        mce.readInClassifier("PF_WRAPPED", "wrap", expDir + "/v3/results/");
//        mce.readInClassifier("PF_R5", "v2", expDir + "/v2/results/");
//        mce.setTestResultsOnly(true);
//        mce.setUseAllStatistics();
//        mce.setIgnoreMissingResults(true);
//        mce.setBuildMatlabDiagrams(true, true);
//        mce.runComparison();
    }

    public enum Config implements EnumBasedConfigurer<ProximityForest> {
        PF_R1() {
            @Override
            public <B extends ProximityForest> B configureFromEnum(B proximityForest) {
                proximityForest.setTrainTimeLimit(0);
                proximityForest.setTestTimeLimit(0);
                proximityForest.setRebuildConstituentAfterEvaluation(true);
                proximityForest.setEstimatorMethod("none");
                proximityForest.setNumTreeLimit(100);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.PT_R5);
                proximityForest.setUseDistributionInVoting(false);
                proximityForest.setWeightTreesByTrainEstimate(false);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.PT_R1);
                return proximityForest;
            }
        },
        PF_R5() {
            @Override
            public <B extends ProximityForest> B configureFromEnum(B proximityForest) {
                proximityForest = PF_R1.configure(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.PT_R5);
                return proximityForest;
            }
        },
        PF_R10() {
            @Override
            public <B extends ProximityForest> B configureFromEnum(B proximityForest) {
                proximityForest = PF_R1.configure(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.PT_R10);
                return proximityForest;
            }
        },
        PF_R5_OOB() {
            @Override
            public <B extends ProximityForest> B configureFromEnum(B proximityForest) {
                proximityForest = PF_R5.configure(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(false);
                return proximityForest;
            }
        },
        PF_R5_OOB_R() {
            @Override
            public <B extends ProximityForest> B configureFromEnum(B proximityForest) {
                proximityForest = PF_R5.configure(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(true);
                return proximityForest;
            }
        },
        PF_R5_OOB_W() {
            @Override
            public <B extends ProximityForest> B configureFromEnum(B proximityForest) {
                proximityForest = PF_R5.configure(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(false);
                proximityForest.setWeightTreesByTrainEstimate(true);
                return proximityForest;
            }
        },
        PF_R5_OOB_R_W() {
            @Override
            public <B extends ProximityForest> B configureFromEnum(B proximityForest) {
                proximityForest = PF_R5.configure(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(true);
                proximityForest.setWeightTreesByTrainEstimate(true);
                return proximityForest;
            }
        },
        PF_R5_CV() {
            @Override
            public <B extends ProximityForest> B configureFromEnum(B proximityForest) {
                proximityForest = PF_R5.configure(proximityForest);
                proximityForest.setEstimatorMethod("CV");
                return proximityForest;
            }
        },
        PF_R5_CV_W() {
            @Override
            public <B extends ProximityForest> B configureFromEnum(B proximityForest) {
                proximityForest = PF_R5.configure(proximityForest);
                proximityForest.setEstimatorMethod("CV");
                proximityForest.setWeightTreesByTrainEstimate(true);
                return proximityForest;
            }
        },
        ;
    }

    public ProximityForest() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        Config.PF_R1.configure(this);
    }

    // the timer for contracting the estimate of train error
    private final StopWatch trainEstimateTimer = new StopWatch();
    // train timer for contracting train
    private final StopWatch trainTimer = new StopWatch();
    // memory watcher for monitoring memory
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    // test timer for contracting predictions
    private final StopWatch testTimer = new StopWatch();
    // the train stage timer used to predict how long the next tree will take and whether there is enough contract
    // remaining
    private final StopWatch trainStageTimer = new StopWatch();
    // the test stage timer for checking whether there is enough time to do more prediction work
    private final StopWatch testStageTimer = new StopWatch();
    // the list of trees in this forest
    private List<Constituent> constituents;
    // the number of trees
    private int numTreeLimit;
    // the train time limit / contract
    private transient long trainTimeLimitNanos;
    // the test time limit / contract
    private transient long testTimeLimitNanos;
    // the longest tree build time for predicting train time requirements
    private transient long longestTrainStageTimeNanos;
    // the method of setting the config of the trees
    private Configurer<ProximityTree> proximityTreeConfig;
    // whether to use distributions in voting or predictions
    private boolean useDistributionInVoting;
    // whether to weight trees by their train estimate (if enabled)
    private boolean weightTreesByTrainEstimate;
    // checkpoint config
    private transient final Checkpointer checkpointer = new BaseCheckpointer(this);
    // whether to rebuild the tree after a train estimate has been produced. This is for evaluation methods like OOB where the evaluated tree may not need rebuilding
    private boolean rebuildConstituentAfterEvaluation;

    @Override public Checkpointer getCheckpointer() {
        return checkpointer;
    }

    private static class Constituent implements Serializable {
        private ProximityTree proximityTree;
        private Evaluator evaluator;
        private ClassifierResults evaluationResults;

        public Evaluator getEvaluator() {
            return evaluator;
        }

        public void setEvaluator(final Evaluator evaluator) {
            this.evaluator = evaluator;
        }

        public ProximityTree getProximityTree() {
            return proximityTree;
        }

        public void setProximityTree(final ProximityTree proximityTree) {
            this.proximityTree = proximityTree;
        }

        public ClassifierResults getEvaluationResults() {
            return evaluationResults;
        }

        public void setEvaluationResults(final ClassifierResults evaluationResults) {
            this.evaluationResults = evaluationResults;
        }
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        final Logger logger = getLogger();
        // load from checkpoint
        loadCheckpoint();
        // kick off resource monitors
        memoryWatcher.start();
        trainTimer.start();
        trainEstimateTimer.checkStopped();
        super.buildClassifier(trainData);
        // rebuild if set
        if(isRebuild()) {
            // reset resouce monitors
            memoryWatcher.resetAndStart();
            trainEstimateTimer.resetAndStop();
            trainTimer.resetAndStart();
            // no constituents to start with
            constituents = new ArrayList<>();
            // zero tree build time so the first tree build will always set the bar
            longestTrainStageTimeNanos = 0;
            LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimitNanos, logger, "train");
        }
        // lap train timer
        trainTimer.lap();
        // while remaining time / more trees need to be built
        while(
                insideNumTreeLimit()
                &&
                insideTrainTimeLimit(trainTimer.getTime() + longestTrainStageTimeNanos)
        ) {
            LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimitNanos, logger, "train");
            // reset the tree build timer
            trainStageTimer.resetAndStart();
            final int treeIndex = constituents.size();
            // setup a new tree
            final ProximityTree tree = new ProximityTree();
//            tree.setLogger(logger);
            final Constituent constituent = new Constituent();
            constituent.setProximityTree(tree);
            constituents.add(constituent);
            proximityTreeConfig.configure(tree);
            tree.setSeed(rand.nextInt());
            // estimate the performance of the tree
            if(!estimator.equals(EstimatorMethod.NONE)) {
                trainEstimateTimer.start();
                // build train estimate based on method
                final Evaluator evaluator = buildEvaluator();
                evaluator.setSeed(rand.nextInt());
                constituent.setEvaluator(evaluator);
                logger.info(() -> "evaluating tree " + treeIndex);
                // evaluate the tree
                final ClassifierResults results = evaluator.evaluate(tree, trainData);
                constituent.setEvaluationResults(results);
                // rebuild the train results as the train estimate has been changed
                setRebuildTrainEstimateResults(true);
                // set meta data
                ResultUtils.setInfo(results, tree, trainData);
                results.setErrorEstimateMethod(getEstimatorMethod());
                trainEstimateTimer.stop();
                results.setErrorEstimateTime(trainStageTimer.getTime());
            }
            // build the tree if not producing train estimate OR rebuild after evaluation
            if(estimator.equals(EstimatorMethod.NONE) || rebuildConstituentAfterEvaluation) {
                logger.info(() -> "building tree " + treeIndex);
                tree.setRebuild(true);
                tree.buildClassifier(trainData);
            }
            // tree fully built
            trainStageTimer.stop();
            // update longest tree build time
            longestTrainStageTimeNanos = Math.max(longestTrainStageTimeNanos, trainStageTimer.getTime());
            // optional checkpoint
            checkpointIfIntervalExpired();
            // update train timer
            trainTimer.lap();
        }
        LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimitNanos, logger, "train");
        // if work has been done towards estimating the train error
        if(estimateOwnPerformance && isRebuildTrainEstimateResults()) {
            trainEstimateTimer.start();
            // disable until further work has been done (e.g. buildClassifier called again)
            setRebuildTrainEstimateResults(false);
            logger.info("finalising train estimate");
            // init final distributions for each train instance
            final double[][] finalDistributions = new double[trainData.size()][getNumClasses()];
            // time for each train instance prediction
            final long[] times = new long[trainData.size()];
            // go through every constituent
            for(int j = 0; j < constituents.size(); j++) {
                // add the output of that constituent to the train estimate
                final Constituent constituent = constituents.get(j);
                final Evaluator evaluator = constituent.getEvaluator();
                final ProximityTree tree = constituent.getProximityTree();
                // the indices of dataInTrainEstimate to trainData. I.e. the 0th instance in dataInTrainEstimate is the trainDataIndices.get(0) 'th instance in the train data.
                final List<Integer> trainDataIndices;
                final Instances dataInTrainEstimate;
                // the train estimate data may be different depending on the evaluation method
                if(estimator.equals(EstimatorMethod.OOB)) {
                    dataInTrainEstimate = ((OutOfBagEvaluator) evaluator).getOutOfBagTestData();
                    trainDataIndices = ((OutOfBagEvaluator) evaluator).getOutOfBagTestDataIndices();
                } else if(estimator.equals(EstimatorMethod.CV)) {
                    dataInTrainEstimate = trainData;
                    trainDataIndices = ArrayUtilities.sequence(trainData.size());
                } else {
                    throw new UnsupportedOperationException("cannot get train data from evaluator: " + evaluator);
                }
                final ClassifierResults constituentTrainResults = constituent.getEvaluationResults();
                // add each prediction to the results weighted by the evaluation of the constituent
                for(int i = 0; i < dataInTrainEstimate.size(); i++) {
                    long time = System.nanoTime();
                    final Instance instance = dataInTrainEstimate.get(i);
                    final int instanceIndexInTrainData = trainDataIndices.get(i);
                    double[] distribution = constituentTrainResults.getProbabilityDistribution(i);
                    // weight the vote of this constituent
                    distribution = vote(constituent, instance);
                    ArrayUtilities.add(finalDistributions[instanceIndexInTrainData], distribution);
                    // add onto the prediction time for this instance
                    time = System.nanoTime() - time;
                    time += constituentTrainResults.getPredictionTime(i);
                    times[instanceIndexInTrainData] = time;
                }
            }
            // add the final predictions into the results
            for(int i = 0; i < trainData.size(); i++) {
                long time = System.nanoTime();
                double[] distribution = finalDistributions[i];
                // normalise the distribution as sum of votes has likely pushed sum of distribution >1
                ArrayUtilities.normalise(distribution, true);
                double prediction = Utilities.argMax(distribution, rand);
                double classValue = trainData.get(i).classValue();
                time = System.nanoTime() - time;
                times[i] += time;
                trainResults.addPrediction(classValue, distribution, prediction, times[i], null);
            }
            trainEstimateTimer.stop();
        }
        trainTimer.stop();
        memoryWatcher.stop();
        ResultUtils.setInfo(trainResults, this, trainData);
        checkpointIfWorkDone();
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        testTimer.resetAndStart();
        long longestTestStageTimeNanos = 0;
        final double[] finalDistribution = new double[getNumClasses()];
        for(int i = 0;
            i < constituents.size()
            &&
            (testTimeLimitNanos <= 0 || testTimer.lap() + longestTestStageTimeNanos < testTimeLimitNanos)
                ; i++) {
            testStageTimer.resetAndStart();
            final double[] distribution = vote(constituents.get(i), instance);
            ArrayUtilities.add(finalDistribution, distribution);
            testStageTimer.stop();
            longestTestStageTimeNanos = Math.max(longestTestStageTimeNanos, testStageTimer.getTime());
        }
        ArrayUtilities.normalise(finalDistribution);
        testTimer.stop();
        return finalDistribution;
    }

    /**
     * make a consistuent vote given distributionForInstance (can be precomputed elsewhere, therefore). This is mostly required for processing the train estimate.
     * @param constituent
     * @param distribution
     * @return
     */
    private double[] vote(Constituent constituent, double[] distribution) {
        if(!useDistributionInVoting) {
            // take majority vote from constituent
            int index = Utilities.argMax(distribution, rand);
            distribution = ArrayUtilities.oneHot(distribution.length, index);
        }
        if(weightTreesByTrainEstimate) {
            ClassifierResults treeTrainResult = constituent.getEvaluationResults();
            if(treeTrainResult != null) {
                double weight = treeTrainResult.getAcc();
                ArrayUtilities.multiply(distribution, weight);
            }
        }
        return distribution;
    }

    /**
     * make a constituent vote given instance
     * @param constituent
     * @param instance
     * @return
     * @throws Exception
     */
    private double[] vote(Constituent constituent, Instance instance) throws Exception {
        final ProximityTree tree = constituent.getProximityTree();
        double[] distribution = tree.distributionForInstance(instance);
        return vote(constituent, distribution);
    }

    public boolean insideNumTreeLimit() {
        return !hasNumTreeLimit() || constituents.size() < numTreeLimit;
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

    public Configurer<ProximityTree> getProximityTreeConfig() {
        return proximityTreeConfig;
    }

    public void setProximityTreeConfig(
            final Configurer<ProximityTree> proximityTreeConfig) {
        Assert.assertNotNull(proximityTreeConfig);
        this.proximityTreeConfig = proximityTreeConfig;
    }

    public boolean isUseDistributionInVoting() {
        return useDistributionInVoting;
    }

    public void setUseDistributionInVoting(final boolean useDistributionInVoting) {
        this.useDistributionInVoting = useDistributionInVoting;
    }

    @Override
    public long getTrainTimeLimit() {
        return trainTimeLimitNanos;
    }

    @Override
    public void setTrainTimeLimit(final long nanos) {
        trainTimeLimitNanos = nanos;
    }

    @Override
    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    @Override
    public StopWatch getTrainEstimateTimer() {
        return trainEstimateTimer;
    }

    @Override
    public long getTestTimeLimit() {
        return testTimeLimitNanos;
    }

    @Override
    public void setTestTimeLimit(final long nanos) {
        testTimeLimitNanos = nanos;
    }

    @Override
    public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    @Override
    public StopWatch getTestTimer() {
        return testTimer;
    }

    public boolean isWeightTreesByTrainEstimate() {
        return weightTreesByTrainEstimate;
    }

    public void setWeightTreesByTrainEstimate(final boolean weightTreesByTrainEstimate) {
        this.weightTreesByTrainEstimate = weightTreesByTrainEstimate;
    }

    public boolean isRebuildConstituentAfterEvaluation() {
        return rebuildConstituentAfterEvaluation;
    }

    public void setRebuildConstituentAfterEvaluation(final boolean rebuildConstituentAfterEvaluation) {
        this.rebuildConstituentAfterEvaluation = rebuildConstituentAfterEvaluation;
    }
}
