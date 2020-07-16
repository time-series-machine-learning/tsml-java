package tsml.classifiers.distance_based.proximity;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.evaluators.OutOfBagEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.Configurer;
import tsml.classifiers.distance_based.utils.classifiers.EnumBasedClassifierConfigurer;
import tsml.classifiers.distance_based.utils.classifiers.Utils;
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
import tsml.transformers.Indexer;
import utilities.ArrayUtilities;
import utilities.InstanceTools;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ProximityForest extends BaseClassifier implements ContractedTrain, ContractedTest, TimedTrain, TimedTrainEstimate, TimedTest, WatchedMemory, Checkpointed {

    public static void main(String[] args) throws Exception {
        for(int i = 0; i < 1; i++) {
            int seed = i;
            ProximityForest classifier = new ProximityForest();
            Config.R5_OOB.applyConfigTo(classifier);
            classifier.setEstimateOwnPerformance(true);
            classifier.setSeed(seed);
            classifier.setEstimatorMethod("OOB");
            classifier.setRebuildConstituentAfterEvaluation(true);
//            classifier.setCheckpointPath("checkpoints/PF");
//            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            Utils.trainTestPrint(classifier, DatasetLoading.sampleDataset("/bench/phd/datasets/uni2018/", "GunPoint", seed));
        }
        //        Thread.sleep(10000);
    }

    public enum Config implements EnumBasedClassifierConfigurer<ProximityForest> {
        R1() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainTimeLimit(0);
                proximityForest.setTestTimeLimit(0);
                proximityForest.setRebuildConstituentAfterEvaluation(true);
                proximityForest.setEstimatorMethod("none");
                proximityForest.setNumTreeLimit(100);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.R5);
                proximityForest.setUseDistributionInVoting(false);
                proximityForest.setWeightTreesByTrainEstimate(false);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.R1);
                return proximityForest;
            }
        },
        R5() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R1.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.R5);
                return proximityForest;
            }
        },
        R5_I() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R1.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.R5_I);
                return proximityForest;
            }
        },
        R10_I() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R10.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.R10_I);
                return proximityForest;
            }
        },
        R1_I() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R1.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.R1_I);
                return proximityForest;
            }
        },
        RR5_I() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R1.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.RR5_I);
                return proximityForest;
            }
        },
        RR10_I() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R10.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.RR10_I);
                return proximityForest;
            }
        },
        R10() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R1.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.R10);
                return proximityForest;
            }
        },
        RR5() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.RR5);
                return proximityForest;
            }
        },
        RR10() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R10.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.RR10);
                return proximityForest;
            }
        },
        R5_T10() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setNumTreeLimit(10);
                return proximityForest;
            }
        },
        R5_T20() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setNumTreeLimit(20);
                return proximityForest;
            }
        },
        R5_T50() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setNumTreeLimit(50);
                return proximityForest;
            }
        },
        R5_T200() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setNumTreeLimit(200);
                return proximityForest;
            }
        },
        R5_T500() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setNumTreeLimit(500);
                return proximityForest;
            }
        },
        R5_T1000() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setNumTreeLimit(1000);
                return proximityForest;
            }
        },
        R5_OOB() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(false);
                return proximityForest;
            }
        },
        R5_OOB_R() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(true);
                return proximityForest;
            }
        },
        R5_OOB_D() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(false);
                proximityForest.setUseDistributionInVoting(true);
                return proximityForest;
            }
        },
        R5_OOB_R_D() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(true);
                proximityForest.setUseDistributionInVoting(true);
                return proximityForest;
            }
        },
        R5_OOB_W() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(false);
                proximityForest.setWeightTreesByTrainEstimate(true);
                return proximityForest;
            }
        },
        R5_OOB_R_W() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(true);
                proximityForest.setWeightTreesByTrainEstimate(true);
                return proximityForest;
            }
        },
        R5_OOB_WD() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(false);
                proximityForest.setWeightTreesByTrainEstimate(true);
                proximityForest.setUseDistributionInVoting(true);
                return proximityForest;
            }
        },
        R5_OOB_R_WD() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("OOB");
                proximityForest.setRebuildConstituentAfterEvaluation(true);
                proximityForest.setWeightTreesByTrainEstimate(true);
                proximityForest.setUseDistributionInVoting(true);
                return proximityForest;
            }
        },
        R5_CV() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("CV");
                return proximityForest;
            }
        },
        R5_CV_D() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("CV");
                proximityForest.setUseDistributionInVoting(true);
                return proximityForest;
            }
        },
        R5_CV_W() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("CV");
                proximityForest.setWeightTreesByTrainEstimate(true);
                return proximityForest;
            }
        },
        R5_CV_WD() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setEstimatorMethod("CV");
                proximityForest.setWeightTreesByTrainEstimate(true);
                proximityForest.setUseDistributionInVoting(true);
                return proximityForest;
            }
        },
        ;
    }

    public ProximityForest() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        Config.R1.applyConfigTo(this);
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
    private long trainTimeLimitNanos;
    // the test time limit / contract
    private long testTimeLimitNanos;
    // the longest tree build time for predicting train time requirements
    private long longestTrainStageTimeNanos;
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
    // the train data
    private Instances trainData;
    // map of train data to indices for constructing train estimates


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
        final boolean rebuild = isRebuild();
        super.buildClassifier(trainData);
        // rebuild if set
        if(rebuild) {
            // reset resouce monitors
            memoryWatcher.resetAndStart();
            trainEstimateTimer.resetAndStop();
            trainTimer.resetAndStart();
            // no constituents to start with
            constituents = new ArrayList<>();
            // first run so set the train estimate to be regenerated
            setRebuildTrainEstimateResults(true);
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
            proximityTreeConfig.applyConfigTo(tree);
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
                trainEstimateTimer.stop();
                results.setErrorEstimateTime(trainStageTimer.getTime());
            }
            // build the tree if not producing train estimate OR rebuild after evaluation
            if(estimator.equals(EstimatorMethod.NONE) || rebuildConstituentAfterEvaluation) {
                logger.info(() -> "building tree " + treeIndex);
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
                    ArrayUtilities.addInPlace(finalDistributions[instanceIndexInTrainData], distribution);
                    // add onto the prediction time for this instance
                    time = System.nanoTime() - time;
                    time += constituentTrainResults.getPredictionTime(i);
                    times[instanceIndexInTrainData] = time;
                }
            }
            // consolidate / normalise all of the predictions
            for(int i = 0; i < finalDistributions.length; i++) {
                long time = System.nanoTime();
                // else normalise the sum of distributions from various trees
                ArrayUtilities.normaliseInPlace(finalDistributions[i]);
                time = System.nanoTime() - time;
                times[i] += time;
            }
            // add the final predictions into the results
            for(int i = 0; i < trainData.size(); i++) {
                long time = System.nanoTime();
                double[] distribution = finalDistributions[i];
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
        logger.info("build complete");
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
            ArrayUtilities.addInPlace(finalDistribution, distribution);
            testStageTimer.stop();
            longestTestStageTimeNanos = Math.max(longestTestStageTimeNanos, testStageTimer.getTime());
        }
        ArrayUtilities.normaliseInPlace(finalDistribution);
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
        ClassifierResults treeTrainResult = constituent.getEvaluationResults();
        double weight = 1;
        if(weightTreesByTrainEstimate && treeTrainResult != null) {
            weight = treeTrainResult.getAcc();
        }
        ArrayUtilities.multiplyInPlace(distribution, weight);
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
