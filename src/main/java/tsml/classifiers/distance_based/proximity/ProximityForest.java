package tsml.classifiers.distance_based.proximity;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifier_mixins.Configurer;
import tsml.classifiers.distance_based.utils.classifier_mixins.Utils;
import tsml.classifiers.distance_based.utils.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
import tsml.classifiers.distance_based.utils.results.ResultUtils;
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

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ProximityForest extends BaseClassifier implements ContractedTrain, ContractedTest,
    TimedTrain, TimedTrainEstimate, TimedTest, WatchedMemory {

    public enum Config implements Configurer<ProximityForest> {
        DEFAULT() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(final B proximityForest) {
                proximityForest.setTrainEstimateMethod(new OutOfBag(true));
                proximityForest.setTrainTimeLimit(0);
                proximityForest.setTestTimeLimit(0);
                proximityForest.setNumTreeLimit(100);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.DEFAULT);
                proximityForest.setUseDistributionInVoting(false);
                return proximityForest;
            }
        },
        R1() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(final B proximityForest) {
                DEFAULT.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.R1);
                return proximityForest;
            }
        },
        R5() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(final B proximityForest) {
                DEFAULT.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.R5);
                return proximityForest;
            }
        },
        R10() {
            @Override
            public <B extends ProximityForest> B applyConfigTo(final B proximityForest) {
                DEFAULT.applyConfigTo(proximityForest);
                proximityForest.setProximityTreeConfig(ProximityTree.Config.R10);
                return proximityForest;
            }
        },
        ;
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
    // test set indices for each tree
    private List<Instances> treeOobTestDatas;

    public TrainEstimateMethod getTrainEstimateMethod() {
        return trainEstimateMethod;
    }

    public void setTrainEstimateMethod(
            final TrainEstimateMethod trainEstimateMethod) {
        this.trainEstimateMethod = trainEstimateMethod;
    }

    public ProximityForest() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        Config.DEFAULT.applyConfigTo(this);
    }

    public static void main(String[] args) throws Exception {
        Thread.sleep(10000);
        for(int i = 0; i < 1; i++) {
            int seed = i;
            ProximityForest classifier = new ProximityForest();
            classifier.setEstimateOwnPerformance(false);
            classifier.setSeed(seed);
            Config.R1.applyConfigTo(classifier);
            //            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            Utils.trainTestPrint(classifier, DatasetLoading.sampleDataset("/bench/datasets/uni2018/",
                "GunPoint", seed));
            //            Utils.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed));
        }
        //        Thread.sleep(10000);
    }

    @Override
    public void buildClassifier(final Instances trainData) throws Exception {
        // kick off resource monitors
        memoryWatcher.start();
        trainTimer.start();
        trainEstimateTimer.checkStopped();
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
                trainResults = new ClassifierResults();
                treeTrainResults = new ArrayList<>(numTreeLimit);
                if(trainEstimateMethod instanceof OutOfBag) {
                    treeOobTestDatas = new ArrayList<>(numTreeLimit);
                }
            } else {
                trainResults = null;
                treeTrainResults = null;
                treeOobTestDatas = null;
            }
        }
        Indexer.index(trainData); // todo overhead of multiple calls per contract
        while(
            insideNumTreeLimit()
                &&
                insideTrainTimeLimit(trainTimer.lap() + longestTrainStageTimeNanos)
        ) {
            LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimitNanos, logger, "train");
            trainStageTimer.resetAndStart();
            int treeIndex = trees.size();
            logger.fine("building tree " + treeIndex);
            ProximityTree tree = new ProximityTree();
            trees.add(tree);
            proximityTreeConfig.applyConfigTo(tree);
            tree.setRandom(rand);
            if(estimateOwnPerformance) {
                trainEstimateTimer.start();
                trainTimer.stop();
                ClassifierResults results;
                if(trainEstimateMethod instanceof OutOfBag) {
                    // build a new oob train / test data
                    final Instances oobTrain = new Instances(trainData, 0);
                    final List<Integer> oobTrainIndices = new ArrayList<>();
                    final Set<Integer> oobTestSetIndices = new HashSet<>(ArrayUtilities.sequence(trainData.size()));
                    for(int i = 0; i < trainData.size(); i++) {
                        int index = rand.nextInt(trainData.size());
                        Instance instance = trainData.get(index);
                        oobTrain.add(instance);
                        oobTrainIndices.add(index);
                        oobTestSetIndices.remove(index);
                    }
                    // quick check that oob test / train are independent
                    for(Integer i : oobTrainIndices) {
                        Assert.assertFalse(oobTestSetIndices.contains(i));
                    }
                    final List<Integer> oobTestIndices = new ArrayList<>(oobTestSetIndices);
                    treeOobTestDatas.add(oobTestIndices);
                    final Instances oobTest = new Instances(trainData, 0);
                    for(int index : oobTestIndices) {
                        oobTest.add(trainData.get(index));
                    }
                    tree.buildClassifier(oobTrain);
                    results = new ClassifierResults();
                    treeTrainResults.add(results);
                    Utils.addPredictions(tree, oobTest, results);
                    if(((OutOfBag) trainEstimateMethod).isRebuildAfterBagging()) {
                        tree.buildClassifier(trainData);
                    }
                } else if(trainEstimateMethod instanceof CrossValidation) {
                    CrossValidationEvaluator evaluator = new CrossValidationEvaluator();
                    evaluator.setSeed(seed);
                    evaluator.setCloneData(true);
                    evaluator.setSetClassMissing(true);
                    evaluator.setNumFolds(((CrossValidation) trainEstimateMethod).getNumFolds());
                    results = evaluator.evaluate(tree, trainData);
                    tree.buildClassifier(trainData);
                } else {
                    throw new UnsupportedOperationException("unknown trainEstimateMethod: " + trainEstimateMethod);
                }
                trainTimer.start();
                trainEstimateTimer.stop();
                ResultUtils.setInfo(results, tree, trainData);
                results.setErrorEstimateMethod(trainEstimateMethod.toString());
                results.setErrorEstimateTime(trainStageTimer.lap());
            } else {
                tree.buildClassifier(trainData);
            }
            trainStageTimer.stop();
            longestTrainStageTimeNanos = Math.max(longestTrainStageTimeNanos, trainStageTimer.getTime());
        }
        logger.info("finished building trees");
        LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimitNanos, logger, "train");
        if(getEstimateOwnPerformance()) {
            trainEstimateTimer.start();
            final double[][] finalDistributions = new double[trainData.size()][];
            final long[] times = new long[trainData.size()];
            for(int j = 0; j < trees.size(); j++) {
                final ProximityTree tree = trees.get(j);
                final List<Integer> oobTestIndices = treeOobTestDatas.get(j);
                final ClassifierResults treeTrainResults = this.treeTrainResults.get(j);
                for(int i = 0; i < oobTestIndices.size(); i++) {
                    long time = System.nanoTime();
                    int index = oobTestIndices.get(i);
                    double[] distribution = treeTrainResults.getProbabilityDistribution(i);
                    if(finalDistributions[index] == null) {
                        finalDistributions[index] = new double[getNumClasses()];
                    }
                    vote(finalDistributions[index], distribution, treeTrainResults);
                    time = System.nanoTime() - time;
                    time += treeTrainResults.getPredictionTime(i);
                    times[index] = time;
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
        if(estimateOwnPerformance) {
            ResultUtils.setInfo(trainResults, this, trainData);
        }
        logger.info("build complete");
    }

    private void vote(double[] finalDistribution, double[] distribution, double weight) {
        if(useDistributionInVoting) {
            ArrayUtilities.addInPlace(finalDistribution, distribution);
        } else {
            // majority vote
            double index = Utilities.argMax(distribution, rand);
            finalDistribution[(int) index] += weight;
        }
    }

    private void vote(double[] finalDistribution, double[] distribution) {
        vote(finalDistribution, distribution, 1);
    }

    private void vote(double[] finalDistribution, double[] distribution, ClassifierResults results) {
        vote(finalDistribution, distribution, results.getAcc());
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
            if(getEstimateOwnPerformance()) {
                vote(finalDistribution, distribution, treeTrainResults.get(i));
            } else {
                vote(finalDistribution, distribution);
            }
            testStageTimer.stop();
            longestTestStageTimeNanos = Math.max(longestTestStageTimeNanos, testStageTimer.getTime());
        }
        ArrayUtilities.normaliseInPlace(finalDistribution);
        testTimer.stop();
        return finalDistribution;
    }

    public boolean hasNumTreeLimit() {
        return numTreeLimit > 0;
    }

    public boolean insideNumTreeLimit() {
        return !hasNumTreeLimit() || trees.size() < numTreeLimit;
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

    public boolean isOOB() {
        return oob;
    }

    public void setOOB(final boolean oob) {
        this.oob = oob;
    }

    public boolean isCV() {
        return cv;
    }

    public void setCV(final boolean cv) {
        this.cv = cv;
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

}
