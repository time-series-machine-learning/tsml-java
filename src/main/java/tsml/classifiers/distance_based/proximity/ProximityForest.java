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
import tsml.classifiers.distance_based.utils.classifier_mixins.Utils;
import tsml.classifiers.distance_based.utils.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
import tsml.classifiers.distance_based.utils.results.ResultUtils;
import tsml.classifiers.distance_based.utils.stopwatch.StopWatch;
import tsml.classifiers.distance_based.utils.stopwatch.TimedTest;
import tsml.classifiers.distance_based.utils.stopwatch.TimedTrain;
import tsml.classifiers.distance_based.utils.stopwatch.TimedTrainEstimate;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.filters.CachedFilter;
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

    private final StopWatch trainEstimaterTimer = new StopWatch();
    private final StopWatch trainTimer = new StopWatch();
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private final StopWatch testTimer = new StopWatch();
    private List<ProximityTree> trees = new ArrayList<>();
    private int numTreeLimit = 100;
    private long trainTimeLimitNanos = 0;
    private long testTimeLimitNanos = 0;
    private long longestTreeBuildTimeNanos = 0;
    private ConstituentConfig constituentConfig = ProximityTree::setConfigR1;
    private boolean useDistributionInVoting = false;
    private Instances oobTrain;
    private Instances oobTest;
    private List<Integer> oobTestIndices;
    private List<Integer> oobTrainIndices;
    private List<ClassifierResults> treeTrainResults;
    private boolean oob = false;
    private boolean cv = false;
    public ProximityForest() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }

    public static void main(String[] args) throws Exception {
        Thread.sleep(10000);
        for(int i = 0; i < 1; i++) {
            int seed = i;
            ProximityForest classifier = new ProximityForest();
            classifier.setEstimateOwnPerformance(false);
            classifier.setSeed(seed);
            classifier.setNumTreeLimit(100);
            classifier.setConstituentConfig(tree -> {
                tree.setConfigR1();
                return tree;
            });
            //            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            Utils.trainTestPrint(classifier, DatasetLoading.sampleDataset("/bench/datasets/uni2018/",
                "GunPoint", seed));
            //            Utils.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed));
        }
        //        Thread.sleep(10000);
    }

    public ProximityForest setConfigOriginalR1() {
        setNumTreeLimit(100);
        setConstituentConfig(ProximityTree::setConfigR1);
        return this;
    }

    public ProximityForest setConfigOriginalR5() {
        setNumTreeLimit(100);
        setConstituentConfig(ProximityTree::setConfigR5);
        return this;
    }

    public ProximityForest setConfigOriginalR10() {
        setNumTreeLimit(100);
        setConstituentConfig(ProximityTree::setConfigR10);
        return this;
    }

    @Override
    public void buildClassifier(final Instances trainData) throws Exception {
        memoryWatcher.enable();
        trainTimer.enable();
        trainEstimaterTimer.checkDisabled();
        LogUtils.logTimeContract(trainTimer.getTimeNanos(), trainTimeLimitNanos, getLogger(), "train");
        if(isRebuild()) {
            trainEstimaterTimer.resetAndDisable();
            memoryWatcher.resetAndEnable();
            trainTimer.resetAndEnable();
            super.buildClassifier(trainData);
            trees = new ArrayList<>();
            trainResults = new ClassifierResults();
            treeTrainResults = new ArrayList<>();
            longestTreeBuildTimeNanos = 0;
            if(getEstimateOwnPerformance()) {
                if(oob) {
                    trainResults.setErrorEstimateMethod("oob");
                    oobTrain = new Instances(trainData, 0);
                    oobTrainIndices = new ArrayList<>();
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
                    oobTestIndices = new ArrayList<>(oobTestSetIndices);
                    oobTest = new Instances(trainData, 0);
                    for(int index : oobTestIndices) {
                        oobTest.add(trainData.get(index));
                    }
                } else if(cv) {
                    trainResults.setErrorEstimateMethod("10foldCv");
                } else {
                    throw new IllegalStateException("no train estimate method enabled");
                }
            }
        }
        CachedFilter.hashInstances(trainData);
        trainTimer.lap();
        LogUtils.logTimeContract(trainTimer.getTimeNanos(), trainTimeLimitNanos, getLogger(), "train");
        // todo contract train of trees?
        while(
            insideNumTreeLimit()
                &&
                insideTrainTimeLimit(trainTimer.getTimeNanos() + longestTreeBuildTimeNanos)
        ) {
            int treeIndex = trees.size();
            getLogger().info(() -> "building tree " + treeIndex);
            LogUtils.logTimeContract(trainTimer.getTimeNanos(), trainTimeLimitNanos, getLogger(), "train");
            ProximityTree tree = new ProximityTree();
            tree = constituentConfig.setConfig(tree);
            tree.setSeed(getRandom().nextInt());
            trees.add(tree);
            long timestamp = System.nanoTime();
            if(getEstimateOwnPerformance()) {
                getLogger().info(() -> "building tree " + treeIndex + "  train estimate");
                trainEstimaterTimer.enable();
                ClassifierResults results;
                if(oob) {
                    tree.buildClassifier(oobTrain);
                    results = new ClassifierResults();
                    treeTrainResults.add(results);
                    for(Instance instance : oobTest) {
                        Utils.addPrediction(tree, instance, results);
                    }
                } else if(cv) {
                    CrossValidationEvaluator evaluator = new CrossValidationEvaluator();
                    evaluator.setSeed(seed);
                    evaluator.setCloneData(true);
                    evaluator.setSetClassMissing(true);
                    evaluator.setNumFolds(10);
                    results = evaluator.evaluate(tree, trainData);
                } else {
                    throw new IllegalStateException("no train estimate method set");
                }
                ResultUtils.setInfo(results, tree, trainData);
                trainEstimaterTimer.disable();
                getLogger().info(() -> "finished building tree " + treeIndex + "  train estimate");
            }
            tree.setRebuild(true);
            tree.buildClassifier(trainData);
            longestTreeBuildTimeNanos = Math.max(longestTreeBuildTimeNanos, System.nanoTime() - timestamp);
            trainTimer.lap();
        }
        getLogger().info("finished building trees");
        LogUtils.logTimeContract(trainTimer.getTimeNanos(), trainTimeLimitNanos, getLogger(), "train");
        if(getEstimateOwnPerformance()) {
            trainEstimaterTimer.enable();
            double[][] finalDistributions = new double[trainData.size()][];
            long[] times = new long[trainData.size()];
            for(int j = 0; j < trees.size(); j++) {
                ProximityTree tree = trees.get(j);
                ClassifierResults treeTrainResults = this.treeTrainResults.get(j);
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
            trainEstimaterTimer.disable();
        }
        trainTimer.disable();
        memoryWatcher.disable();
        getLogger().info("build complete");
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
        testTimer.resetAndEnable();
        long longestPredictTime = 0;
        final double[] finalDistribution = new double[getNumClasses()];
        for(int i = 0;
            i < trees.size()
                &&
                (testTimeLimitNanos <= 0 || testTimer.getTimeNanos() + longestPredictTime < testTimeLimitNanos)
            ; i++) {
            final long timestamp = System.nanoTime();
            ProximityTree tree = trees.get(i);
            final double[] distribution = tree.distributionForInstance(instance);
            if(getEstimateOwnPerformance()) {
                vote(finalDistribution, distribution, treeTrainResults.get(i));
            } else {
                vote(finalDistribution, distribution);
            }
            longestPredictTime = System.nanoTime() - timestamp;
            testTimer.lap();
        }
        ArrayUtilities.normaliseInPlace(finalDistribution);
        testTimer.disable();
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

    public ProximityForest setNumTreeLimit(final int numTreeLimit) {
        this.numTreeLimit = numTreeLimit;
        return this;
    }

    public ConstituentConfig getConstituentConfig() {
        return constituentConfig;
    }

    public ProximityForest setConstituentConfig(
        final ConstituentConfig constituentConfig) {
        Assert.assertNotNull(constituentConfig);
        this.constituentConfig = constituentConfig;
        return this;
    }

    public boolean isUseDistributionInVoting() {
        return useDistributionInVoting;
    }

    public ProximityForest setUseDistributionInVoting(final boolean useDistributionInVoting) {
        this.useDistributionInVoting = useDistributionInVoting;
        return this;
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

    public ProximityForest setOOB(final boolean oob) {
        this.oob = oob;
        return this;
    }

    public boolean isCV() {
        return cv;
    }

    public ProximityForest setCV(final boolean cv) {
        this.cv = cv;
        return this;
    }

    @Override
    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    @Override
    public StopWatch getTrainEstimateTimer() {
        return trainEstimaterTimer;
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

    public interface ConstituentConfig {

        ProximityTree setConfig(ProximityTree tree);
    }
}
