package tsml.classifiers.distance_based.proximity;

import com.beust.jcommander.internal.Lists;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.junit.Assert;
import tsml.classifiers.TestTimeContractable;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.TrainTimeable;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.proximity.splitting.BestOfNSplits;
import tsml.classifiers.distance_based.proximity.splitting.Split;
import tsml.classifiers.distance_based.proximity.splitting.Splitter;
import tsml.classifiers.distance_based.proximity.splitting.exemplar_based.ContinuousDistanceFunctionConfigs;
import tsml.classifiers.distance_based.proximity.splitting.exemplar_based.RandomDistanceFunctionPicker;
import tsml.classifiers.distance_based.proximity.splitting.exemplar_based.RandomExemplarPerClassPicker;
import tsml.classifiers.distance_based.proximity.splitting.exemplar_based.RandomExemplarProximitySplit;
import tsml.classifiers.distance_based.proximity.stopping_conditions.Pure;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifier_mixins.TestTimeable;
import tsml.classifiers.distance_based.utils.classifier_mixins.Utils;
import tsml.classifiers.distance_based.utils.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.iteration.LinearListIterator;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.classifiers.distance_based.utils.stopwatch.StopWatch;
import tsml.classifiers.distance_based.utils.stopwatch.TimedTest;
import tsml.classifiers.distance_based.utils.stopwatch.TimedTrain;
import tsml.classifiers.distance_based.utils.stopwatch.TimedTrainEstimate;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatchable;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.tree.BaseTree;
import tsml.classifiers.distance_based.utils.tree.BaseTreeNode;
import tsml.classifiers.distance_based.utils.tree.Tree;
import tsml.classifiers.distance_based.utils.tree.TreeNode;
import tsml.filters.CachedFilter;
import utilities.ArrayUtilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: proximity tree
 * <p>
 * Contributors: goastler
 */
public class ProximityTree extends BaseClassifier implements ContractedTest, ContractedTrain,
    TimedTrain, TimedTest, WatchedMemory {

    public static void main(String[] args) throws Exception {
        for(int i = 2; i < 3; i++) {
            int seed = i;
            ProximityTree classifier = new ProximityTree();
            classifier.setSeed(seed);
            classifier.setConfigR1();
//            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            Utils.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed));
        }
    }

    // -------------------- configs --------------------

    public ProximityTree setConfigR1() {
        setBuildUntilPure();
        return setSingleSplit();
    }

    public ProximityTree setConfigR5() {
        setBuildUntilPure();
        return set5Splits();
    }

    public ProximityTree setConfigR10() {
        setBuildUntilPure();
        return set10Splits();
    }

    // -------------------- end configs --------------------

    public ProximityTree() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
    }

    public ProximityTree setBuildUntilPure() {
        return setStoppingCondition(new Pure());
    }

    private void setSingleSplitFromTrainData(Instances trainData) {
        final Random random = getRandom();
        final RandomExemplarPerClassPicker exemplarPicker = new RandomExemplarPerClassPicker(random);
        final List<ParamSpace> paramSpaces = Lists.newArrayList(
            DistanceMeasureConfigs.buildEdSpace(),
            // these aren't in orig PF
//            DistanceMeasureConfigs.buildFullDtwSpace(),
//            DistanceMeasureConfigs.buildFullDdtwSpace(),
            ContinuousDistanceFunctionConfigs.buildDtwSpace(trainData),
            ContinuousDistanceFunctionConfigs.buildDdtwSpace(trainData),
            ContinuousDistanceFunctionConfigs.buildErpSpace(trainData),
            ContinuousDistanceFunctionConfigs.buildLcssSpace(trainData),
            DistanceMeasureConfigs.buildMsmSpace(),
            ContinuousDistanceFunctionConfigs.buildWdtwSpace(),
            ContinuousDistanceFunctionConfigs.buildWddtwSpace(),
            DistanceMeasureConfigs.buildTwedSpace()
        );
        final RandomDistanceFunctionPicker distanceFunctionPicker = new RandomDistanceFunctionPicker(
            random, paramSpaces);
        setSplitter(data -> {
            RandomExemplarProximitySplit split = new RandomExemplarProximitySplit(
                random, exemplarPicker,
                distanceFunctionPicker);
            split.setData(data);
            return split;
        });
    }

    public ProximityTree setSingleSplit() {
        setRebuildListener(this::setSingleSplitFromTrainData);
        return this;
    }

    private void setMultipleSplitsFromTrainData(Instances trainData, int numSplits) {
        setSingleSplitFromTrainData(trainData);
        Splitter splitter = getSplitter();
        setSplitter(data -> {
            BestOfNSplits bestOfNSplits = new BestOfNSplits(splitter, this.getRandom(), numSplits);
            bestOfNSplits.setData(data);
            return bestOfNSplits;
        });
    }

    public ProximityTree setMultipleSplits(int numSplits) {
        Assert.assertTrue(numSplits > 0);
        setRebuildListener(trainData -> {
            setMultipleSplitsFromTrainData(trainData, numSplits);
        });
        return this;
    }

    public ProximityTree set5Splits() {
        return setMultipleSplits(5);
    }

    public ProximityTree set10Splits() {
        return setMultipleSplits(10);
    }

    private final Tree<Split> tree = new BaseTree<>();
    private final StopWatch trainTimer = new StopWatch();
    private final StopWatch testTimer = new StopWatch();
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private long trainTimeLimitNanos = 0;
    private long testTimeLimitNanos = 0;
    private long longestNodeBuildTimeNanos = 0;
    private ListIterator<TreeNode<Split>> nodeBuildQueue = new LinearListIterator<>();
    private StoppingCondition stoppingCondition;
    private Splitter splitter;
    public static final String STOPPING_CONDITION_FLAG = "c";
    public static final String SPLITTER_FLAG = "s";
    public static final String SPLITTER_BUILDER_FLAG = "b";

    @Override
    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    @Override
    public StopWatch getTestTimer() {
        return testTimer;
    }

    @Override
    public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    @Override
    public void setTestTimeLimit(final long nanos) {
        testTimeLimitNanos = nanos;
    }

    @Override
    public long getTestTimeLimit() {
        return testTimeLimitNanos;
    }

    @Override
    public void setTrainTimeLimit(final long nanos) {
        trainTimeLimitNanos = nanos;
    }

    @Override
    public long getTrainTimeLimit() {
        return trainTimeLimitNanos;
    }

    public interface StoppingCondition extends Serializable {
        boolean shouldStop(TreeNode<Split> node);
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        memoryWatcher.enable();
        trainTimer.enable();
        if(isRebuild()) {
            // reset
            memoryWatcher.resetAndEnable();
            trainTimer.resetAndEnable();
            super.buildClassifier(trainData);
            tree.clear();
            longestNodeBuildTimeNanos = 0;
            nodeBuildQueue = new LinearListIterator<>();
            final TreeNode<Split> root = buildNode(trainData, null);
            tree.setRoot(root);
        }
        CachedFilter.hashInstances(trainData);
        trainTimer.lap();
        while(
            // there is enough time for another split to be built
            insideTrainTimeLimit(trainTimer.getTimeNanos() + longestNodeBuildTimeNanos)
            // and there's remaining nodes to be built
            &&
            nodeBuildQueue.hasNext()
        ) {
            long time = System.nanoTime();
            final TreeNode<Split> node = nodeBuildQueue.next();
            // partition the data at the node
            Split split = node.getElement();
            split.buildSplit();
            List<Instances> partitions = split.getPartitions();
            // for each partition of data
            for(Instances partition : partitions) {
                // try to build a child node
                buildNode(partition, node);
            }
            longestNodeBuildTimeNanos = Math.max(longestNodeBuildTimeNanos, System.nanoTime() - time);
            trainTimer.lap();
        }
        trainTimer.disable();
        memoryWatcher.disable();
    }

    private TreeNode<Split> buildNode(Instances data, TreeNode<Split> parent) {
        // split the data into multiple partitions, housed in a Split object
        final Split split = splitter.buildSplit(data);
        // build a new node
        final TreeNode<Split> node = new BaseTreeNode<>(split);
        // set tree relationship
        node.setParent(parent);
        // check the stopping condition hasn't been hit
        final boolean stop = stoppingCondition.shouldStop(node);
        if(!stop) {
            // if not hit the stopping condition then add node to the build queue
            nodeBuildQueue.add(node);
        }
        return node;
    }

    public StoppingCondition getStoppingCondition() {
        return stoppingCondition;
    }

    public ProximityTree setStoppingCondition(
        final StoppingCondition stoppingCondition) {
        Assert.assertNotNull(stoppingCondition);
        this.stoppingCondition = stoppingCondition;
        return this;
    }

    public Splitter getSplitter() {
        return splitter;
    }

    public ProximityTree setSplitter(final Splitter splitter) {
        Assert.assertNotNull(splitter);
        this.splitter = splitter;
        return this;
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        // enable resource monitors
        testTimer.resetAndEnable();
        long longestPredictTime = 0;
        // start at the tree node
        TreeNode<Split> node = tree.getRoot();
        if(!node.hasChildren()) {
            // root node has not been built, just return random guess
            return ArrayUtilities.uniformDistribution(getNumClasses());
        }
        int index = -1;
        Split split = node.getElement();
        // traverse the tree downwards from root
        while(!node.isLeaf() && insideTestTimeLimit(testTimer.getTimeNanos() + longestPredictTime)) {
            final long timestamp = System.nanoTime();
            // get the split at that node
            split = node.getElement();
            // work out which branch to go to next
            index = split.getPartitionIndexFor(instance);
            final List<TreeNode<Split>> children = node.getChildren();
            // make this the next node to visit
            node = children.get(index);
            longestPredictTime = System.nanoTime() - timestamp;
        };
        // hit a leaf node
        // get the parent of the leaf node to work out distribution
        node = node.getParent();
        split = node.getElement();
        double[] distribution = split.distributionForInstance(instance, index);
        // disable the resource monitors
        testTimer.disable();
        return distribution;
    }
}
