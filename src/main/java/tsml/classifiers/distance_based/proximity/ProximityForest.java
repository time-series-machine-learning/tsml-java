package tsml.classifiers.distance_based.proximity;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.Checkpointable;
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
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
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
            classifier.setEstimateOwnPerformance(false);
            classifier.setSeed(seed);
            Config.R5_CV.applyConfigTo(classifier);
            //            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            Utils.trainTestPrint(classifier, DatasetLoading.sampleDataset("/bench/phd/datasets/uni2018/",
                                                                          "GunPoint", seed));
            //            Utils.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed));
        }
        //        Thread.sleep(10000);
    }

    public enum Config implements EnumBasedClassifierConfigurer<ProximityForest> {
        R1() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainEstimateMethod(new OutOfBag(true));
                proximityForest.setTrainTimeLimit(0);
                proximityForest.setTestTimeLimit(0);
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
                proximityForest.setTrainEstimateMethod(new OutOfBag(false));
                return proximityForest;
            }
        },
        R5_OOB_R() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainEstimateMethod(new OutOfBag(true));
                return proximityForest;
            }
        },
        R5_OOB_D() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainEstimateMethod(new OutOfBag(false));
                proximityForest.setUseDistributionInVoting(true);
                return proximityForest;
            }
        },
        R5_OOB_R_D() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainEstimateMethod(new OutOfBag(true));
                proximityForest.setUseDistributionInVoting(true);
                return proximityForest;
            }
        },
        R5_OOB_W() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainEstimateMethod(new OutOfBag(false));
                proximityForest.setWeightTreesByTrainEstimate(true);
                return proximityForest;
            }
        },
        R5_OOB_R_W() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainEstimateMethod(new OutOfBag(true));
                proximityForest.setWeightTreesByTrainEstimate(true);
                return proximityForest;
            }
        },
        R5_OOB_WD() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainEstimateMethod(new OutOfBag(false));
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
                proximityForest.setTrainEstimateMethod(new OutOfBag(true));
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
                proximityForest.setTrainEstimateMethod(new CrossValidation(10));
                return proximityForest;
            }
        },
        R5_CV_D() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainEstimateMethod(new CrossValidation(10));
                proximityForest.setUseDistributionInVoting(true);
                return proximityForest;
            }
        },
        R5_CV_W() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainEstimateMethod(new CrossValidation(10));
                proximityForest.setWeightTreesByTrainEstimate(true);
                return proximityForest;
            }
        },
        R5_CV_WD() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(B proximityForest) {
                proximityForest = R5.applyConfigTo(proximityForest);
                proximityForest = super.applyConfigTo(proximityForest);
                proximityForest.setTrainEstimateMethod(new CrossValidation(10));
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
    private List<ProximityTree> trees = new ArrayList<>();
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
    // the train results for each tree
    private List<ClassifierResults> treeTrainResults;
    // type of train estimate
    private TrainEstimateMethod trainEstimateMethod;
    // todo which stat to weight trees by
    // test set indices for each tree
    private List<Instances> treeOobTestDatas;
    // whether to weight trees by their train estimate (if enabled)
    private boolean weightTreesByTrainEstimate;
    // checkpoint config
    private final Checkpointer checkpointer = new BaseCheckpointer(this);

    @Override public Checkpointer getCheckpointer() {
        return checkpointer;
    }

    public TrainEstimateMethod getTrainEstimateMethod() {
        return trainEstimateMethod;
    }

    public void setTrainEstimateMethod(
            final TrainEstimateMethod trainEstimateMethod) {
        this.trainEstimateMethod = trainEstimateMethod;
    }

    @Override
    public void buildClassifier(final Instances trainData) throws Exception {
        // kick off resource monitors
        memoryWatcher.start();
        trainTimer.start();
        trainEstimateTimer.checkStopped();
        final Logger logger = getLogger();
        checkpointer.setLogger(logger);
        checkpointer.loadCheckpoint();
        LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimitNanos, logger, "train");
        if(isRebuild()) {
            // reset variables
            trainEstimateTimer.resetAndStop();
            memoryWatcher.resetAndStart();
            trainTimer.resetAndStart();
            super.buildClassifier(trainData);
            trees = new ArrayList<>(numTreeLimit);
            longestTrainStageTimeNanos = 0;
            if(estimateOwnPerformance) {
                treeTrainResults = new ArrayList<>(numTreeLimit);
                if(trainEstimateMethod instanceof OutOfBag) {
                    treeOobTestDatas = new ArrayList<>(numTreeLimit);
                }
            } else {
                treeTrainResults = null;
                treeOobTestDatas = null;
            }
        }
        Indexer.index(trainData);
        while(
                insideNumTreeLimit()
                &&
                insideTrainTimeLimit(trainTimer.lap() + longestTrainStageTimeNanos)
        ) {
            LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimitNanos, logger, "train");
            trainStageTimer.resetAndStart();
            int treeIndex = trees.size();
            ProximityTree tree = new ProximityTree();
            tree.setLogger(logger);
            trees.add(tree);
            proximityTreeConfig.applyConfigTo(tree);
            tree.setRandom(rand);
            if(estimateOwnPerformance) {
                trainEstimateTimer.start();
                trainTimer.stop();
                ClassifierResults results;
                if(trainEstimateMethod instanceof OutOfBag) {
                    logger.fine(() -> "out-of-bagging tree " + treeIndex);
                    // build a new oob train / test data
                    final Instances oobTrain = new Instances(trainData, 0);
                    final Set<Instance> oobTestSet = new HashSet<>(trainData.size());
                    oobTestSet.addAll(trainData);
                    for(int i = 0; i < trainData.size(); i++) {
                        int index = rand.nextInt(trainData.size());
                        Instance instance = trainData.get(index);
                        oobTrain.add(instance);
                        oobTestSet.remove(instance);
                    }
                    // quick check that oob test / train are independent
                    for(Instance i : oobTrain) {
                        Assert.assertFalse(oobTestSet.contains(i));
                    }
                    final Instances oobTest = new Instances(trainData, 0);
                    oobTest.addAll(oobTestSet);
                    treeOobTestDatas.add(oobTest);
                    tree.buildClassifier(oobTrain);
                    results = new ClassifierResults();
                    Utils.addPredictions(tree, oobTest, results);
                    if(((OutOfBag) trainEstimateMethod).isRebuildAfterBagging()) {
                        logger.fine(() -> "rebuilding tree " + treeIndex);
                        tree.buildClassifier(trainData);
                    }
                } else if(trainEstimateMethod instanceof CrossValidation) {
                    logger.fine(() -> "cross validating tree " + treeIndex);
                    CrossValidationEvaluator evaluator = new CrossValidationEvaluator();
                    evaluator.setSeed(seed);
                    evaluator.setCloneData(true);
                    evaluator.setSetClassMissing(false);
                    evaluator.setNumFolds(((CrossValidation) trainEstimateMethod).getNumFolds());
                    results = evaluator.evaluate(tree, trainData);
                    logger.fine(() -> "building tree " + treeIndex);
                    tree.buildClassifier(trainData);
                } else {
                    throw new UnsupportedOperationException("unknown trainEstimateMethod: " + trainEstimateMethod);
                }
                treeTrainResults.add(results);
                trainTimer.start();
                trainEstimateTimer.stop();
                ResultUtils.setInfo(results, tree, trainData);
                results.setErrorEstimateMethod(trainEstimateMethod.toString());
                results.setErrorEstimateTime(trainStageTimer.lap());
            } else {
                logger.fine(() -> "building tree " + treeIndex);
                tree.buildClassifier(trainData);
            }
            trainStageTimer.stop();
            longestTrainStageTimeNanos = Math.max(longestTrainStageTimeNanos, trainStageTimer.getTime());
            checkpointer.saveCheckpoint();
        }
        logger.fine("finished building trees");
        LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimitNanos, logger, "train");
        if(estimateOwnPerformance) {
            logger.fine("finalising train estimate");
            trainEstimateTimer.start();
            final double[][] finalDistributions = new double[trainData.size()][];
            final long[] times = new long[trainData.size()];
            for(int j = 0; j < trees.size(); j++) {
                final ProximityTree tree = trees.get(j);
                final Instances treeTrainEstimateData;
                if(trainEstimateMethod instanceof OutOfBag) {
                    treeTrainEstimateData = treeOobTestDatas.get(j);
                } else if(trainEstimateMethod instanceof CrossValidation) {
                    treeTrainEstimateData = trainData;
                } else {
                    throw new UnsupportedOperationException("unknown trainEstimateMethod: " + trainEstimateMethod);
                }
                final ClassifierResults treeTrainResults = this.treeTrainResults.get(j);
                for(int i = 0; i < treeTrainEstimateData.size(); i++) {
                    long time = System.nanoTime();
                    final Instance instance = treeTrainEstimateData.get(i);
                    final int instanceIndex = ((Indexer.IndexedInstance) instance).getIndex();
                    final double[] distribution = treeTrainResults.getProbabilityDistribution(i);
                    if(finalDistributions[instanceIndex] == null) {
                        finalDistributions[instanceIndex] = new double[getNumClasses()];
                    }
                    vote(finalDistributions[instanceIndex], distribution, treeTrainResults);
                    time = System.nanoTime() - time;
                    time += treeTrainResults.getPredictionTime(i);
                    times[instanceIndex] = time;
                }
            }
            for(int i = 0; i < finalDistributions.length; i++) {
                long time = System.nanoTime();
                if(finalDistributions[i] == null) {
                    finalDistributions[i] = ArrayUtilities.uniformDistribution(getNumClasses());
                } else {
                    ArrayUtilities.normaliseInPlace(finalDistributions[i]);
                }
                time = System.nanoTime() - time;
                times[i] += time;
            }
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
        logger.fine("build complete");
        checkpointer.saveFinalCheckpoint();
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        testTimer.resetAndStart();
        long longestTestStageTimeNanos = 0;
        final double[] finalDistribution = new double[getNumClasses()];
        for(int i = 0;
            i < trees.size()
            &&
            (testTimeLimitNanos <= 0 || testTimer.lap() + longestTestStageTimeNanos < testTimeLimitNanos)
                ; i++) {
            testStageTimer.resetAndStart();
            ProximityTree tree = trees.get(i);
            final double[] distribution = tree.distributionForInstance(instance);
            ClassifierResults treeTrainResult = null;
            if(estimateOwnPerformance) {
                treeTrainResult = treeTrainResults.get(i);
            }
            vote(finalDistribution, distribution, treeTrainResult);
            testStageTimer.stop();
            longestTestStageTimeNanos = Math.max(longestTestStageTimeNanos, testStageTimer.getTime());
        }
        ArrayUtilities.normaliseInPlace(finalDistribution);
        testTimer.stop();
        return finalDistribution;
    }

    public boolean insideNumTreeLimit() {
        return !hasNumTreeLimit() || trees.size() < numTreeLimit;
    }

    private void vote(double[] finalDistribution, double[] distribution, ClassifierResults treeTrainResults) {
        double weight = 1;
        if(weightTreesByTrainEstimate && treeTrainResults != null) {
            weight = treeTrainResults.getAcc();
        }
        if(useDistributionInVoting) {
            if(weightTreesByTrainEstimate) {
                ArrayUtilities.multiplyInPlace(distribution, weight);
            }
            ArrayUtilities.addInPlace(finalDistribution, distribution);
        } else {
            // majority vote
            if(!weightTreesByTrainEstimate) {
                weight = 1;
            }
            double index = Utilities.argMax(distribution, rand);
            finalDistribution[(int) index] += weight;
        }
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


}
