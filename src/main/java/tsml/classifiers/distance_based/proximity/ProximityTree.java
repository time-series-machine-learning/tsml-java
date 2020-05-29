package tsml.classifiers.distance_based.proximity;

import com.beust.jcommander.internal.Lists;
import experiments.data.DatasetLoading;
import java.util.LinkedList;
import java.util.List;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifier_mixins.Utils;
import tsml.classifiers.distance_based.utils.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.system.timing.TimedTest;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrain;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.collections.tree.BaseTree;
import tsml.classifiers.distance_based.utils.collections.tree.BaseTreeNode;
import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import tsml.classifiers.distance_based.utils.collections.tree.TreeNode;
import tsml.filters.CachedFilter;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: proximity tree
 * <p>
 * Contributors: goastler
 */
public class ProximityTree extends BaseClassifier implements ContractedTest, ContractedTrain,
    TimedTrain, TimedTest, WatchedMemory {

    private Tree<ProximitySplit> tree;
    private final StopWatch trainTimer = new StopWatch();
    private final StopWatch testTimer = new StopWatch();
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private final StopWatch trainStageTimer = new StopWatch();
    private final StopWatch testStageTimer = new StopWatch();
    private long trainTimeLimitNanos;
    private long testTimeLimitNanos;
    private long longestNodeBuildTimeNanos;
    private int r;
    private boolean earlyAbandon;
    private boolean randomTieBreak;
    private LinkedList<TreeNode<ProximitySplit>> nodeBuildQueue;
    private boolean breadthFirst = false;
    private List<DistanceFunctionSpaceBuilder> distanceFunctionSpaceBuilders;

    public List<DistanceFunctionSpaceBuilder> getDistanceFunctionSpaceBuilders() {
        return distanceFunctionSpaceBuilders;
    }

    public ProximityTree setDistanceFunctionSpaceBuilders(
        final List<DistanceFunctionSpaceBuilder> distanceFunctionSpaceBuilders) {
        this.distanceFunctionSpaceBuilders = distanceFunctionSpaceBuilders;
        return this;
    }

    public ProximityTree() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        setConfigDefault();
    }

    public static void main(String[] args) throws Exception {
        for(int i = 0; i < 1; i++) {
            int seed = i;
            ProximityTree classifier = new ProximityTree();
            classifier.setSeed(seed);
            classifier.setConfigR1();
            //            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            Utils.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed));
        }
    }

    public ProximityTree setConfigDefault() {
        setR(5);
        setEarlyAbandon(false);
        setRandomTieBreak(false);
        setTrainTimeLimit(0);
        setTestTimeLimit(0);
        setDistanceFunctionSpaceBuilders(Lists.newArrayList(
            DistanceFunctionSpaceBuilder.ED,
            DistanceFunctionSpaceBuilder.FULL_DTW,
            DistanceFunctionSpaceBuilder.DTW,
            DistanceFunctionSpaceBuilder.FULL_DDTW,
            DistanceFunctionSpaceBuilder.DDTW,
            DistanceFunctionSpaceBuilder.WDTW,
            DistanceFunctionSpaceBuilder.WDDTW,
            DistanceFunctionSpaceBuilder.LCSS,
            DistanceFunctionSpaceBuilder.ERP,
            DistanceFunctionSpaceBuilder.TWED,
            DistanceFunctionSpaceBuilder.MSM
        ));
        return this;
    }

    public ProximityTree setConfigR1() {
        setConfigDefault();
        setR(1);
        return this;
    }

    public ProximityTree setConfigR5() {
        setConfigDefault();
        setR(5);
        return this;
    }

    public ProximityTree setConfigR10() {
        setConfigDefault();
        setR(10);
        return this;
    }

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
    public long getTestTimeLimit() {
        return testTimeLimitNanos;
    }

    @Override
    public void setTestTimeLimit(final long nanos) {
        testTimeLimitNanos = nanos;
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
    public void buildClassifier(Instances trainData) throws Exception {
        memoryWatcher.start();
        trainTimer.start();
        if(isRebuild()) {
            // reset
            memoryWatcher.resetAndStart();
            trainTimer.resetAndStart();
            tree = new BaseTree<>();
            nodeBuildQueue = new LinkedList<>();
            longestNodeBuildTimeNanos = 0;
            super.buildClassifier(trainData);
            final TreeNode<ProximitySplit> root = buildNode(trainData, null);
            tree.setRoot(root);
        }
        CachedFilter.hashInstances(trainData);
        while(
            // there is enough time for another split to be built
            insideTrainTimeLimit(trainTimer.lap() + longestNodeBuildTimeNanos)
                // and there's remaining nodes to be built
                &&
                !nodeBuildQueue.isEmpty()
        ) {
            trainStageTimer.resetAndStart();
            final TreeNode<ProximitySplit> node = nodeBuildQueue.removeFirst();
            // partition the data at the node
            ProximitySplit split = node.getElement();
            split.buildSplit();
            List<Instances> partitions = split.getPartitions();
            // for each partition of data
            for(Instances partition : partitions) {
                // try to build a child node
                buildNode(partition, node);
            }
            trainStageTimer.stop();
            longestNodeBuildTimeNanos = Math.max(longestNodeBuildTimeNanos, trainStageTimer.getTime());
        }
        trainTimer.stop();
        memoryWatcher.stop();
    }

    private TreeNode<ProximitySplit> buildNode(Instances data, TreeNode<ProximitySplit> parent) {
        // split the data into multiple partitions, housed in a ProximitySplit object
        final ProximitySplit split = setupSplit(data);
        // build a new node
        final TreeNode<ProximitySplit> node = new BaseTreeNode<>(split);
        // set tree relationship
        node.setParent(parent);
        // check the stopping condition hasn't been hit
        if(!Utilities.isHomogeneous(data)) {
            // if not hit the stopping condition then add node to the build queue
            if(breadthFirst) {
                nodeBuildQueue.addLast(node);
            } else {
                nodeBuildQueue.addFirst(node);
            }
        }
        return node;
    }

    private ProximitySplit setupSplit(Instances data) {
        ProximitySplit split = new ProximitySplit(getRandom());
        split.setData(data);
        split.setR(r);
        split.setEarlyAbandon(earlyAbandon);
        split.setRandomTieBreak(randomTieBreak);
        split.setDistanceFunctionSpaceBuilders(distanceFunctionSpaceBuilders);
        return split;
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        // enable resource monitors
        testTimer.resetAndStart();
        long longestPredictTime = 0;
        // start at the tree node
        TreeNode<ProximitySplit> node = tree.getRoot();
        if(!node.hasChildren()) {
            //             root node has not been built, just return random guess
            return ArrayUtilities.uniformDistribution(getNumClasses());
        }
        int index = -1;
        int i = 0;
        ProximitySplit split = node.getElement();
        // traverse the tree downwards from root
        while(
            !node.isLeaf()
                &&
                insideTestTimeLimit(testTimer.getTime() + longestPredictTime)
        ) {
            final long timestamp = System.nanoTime();
            // get the split at that node
            split = node.getElement();
            // work out which branch to go to next
            index = split.getPartitionIndexFor(instance);
            final List<TreeNode<ProximitySplit>> children = node.getChildren();
            // make this the next node to visit
            node = children.get(index);
            longestPredictTime = System.nanoTime() - timestamp;
        }
        // hit a leaf node
        // get the parent of the leaf node to work out distribution
        node = node.getParent();
        split = node.getElement();
        double[] distribution = split.distributionForInstance(instance, index);
        // disable the resource monitors
        testTimer.stop();
        return distribution;
    }

    public int height() {
        return tree.height();
    }

    public int size() {
        return tree.size();
    }

    public int getR() {
        return r;
    }

    public ProximityTree setR(final int r) {
        Assert.assertTrue(r > 0);
        this.r = r;
        return this;
    }

    public boolean isEarlyAbandon() {
        return earlyAbandon;
    }

    public ProximityTree setEarlyAbandon(final boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
        return this;
    }

    public boolean isRandomTieBreak() {
        return randomTieBreak;
    }

    public ProximityTree setRandomTieBreak(final boolean randomTieBreak) {
        this.randomTieBreak = randomTieBreak;
        return this;
    }

    public boolean isBreadthFirst() {
        return breadthFirst;
    }

    public ProximityTree setBreadthFirst(final boolean breadthFirst) {
        this.breadthFirst = breadthFirst;
        return this;
    }

}
