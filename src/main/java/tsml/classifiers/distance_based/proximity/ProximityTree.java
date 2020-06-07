package tsml.classifiers.distance_based.proximity;

import com.beust.jcommander.internal.Lists;
import experiments.data.DatasetLoading;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifier_mixins.Utils;
import tsml.classifiers.distance_based.utils.collections.tree.BaseTree;
import tsml.classifiers.distance_based.utils.collections.tree.BaseTreeNode;
import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import tsml.classifiers.distance_based.utils.collections.tree.TreeNode;
import tsml.classifiers.distance_based.utils.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.scoring.Scorer;
import tsml.classifiers.distance_based.utils.scoring.Scorer.GiniImpurityEntropy;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.system.timing.TimedTest;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrain;
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

    // train timer
    private final StopWatch trainTimer = new StopWatch();
    // test / predict timer
    private final StopWatch testTimer = new StopWatch();
    // memory watcher
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    // train stage timer used for predicting whether there's enough time for more training within the train contract
    private final StopWatch trainStageTimer = new StopWatch();
    // test stage timer used for predicting whether there's enough time for more prediction within the test contract
    private final StopWatch testStageTimer = new StopWatch();
    // the tree of splits
    private Tree<ProximitySplit> tree;
    // the train time limit / contract
    private long trainTimeLimitNanos;
    // the test time limit / contract
    private long testTimeLimitNanos;
    // the longest time taken to build a node / split
    private long maxTimePerInstanceForNodeBuilding;
    // r number of splits attempted at each node
    private int r;
    // the scoring method
    private Scorer scorer;
    // whether to early abandon distance measurements
    private boolean earlyAbandonDistances;
    // whether to random tie break exemplars in splits
    private boolean randomTieBreakDistances;
    // whether to random tie break split canaidates when r > 1
    private boolean randomTieBreakR;
    // whether to randomly choose R at each split
    private boolean randomR;
    // the queue of nodes left to build
    private Deque<TreeNode<ProximitySplit>> nodeBuildQueue;
    // whether to build in breadth first or depth first order
    private boolean breadthFirst;
    // the list of distance function space builders to produce distance functions in splits
    private List<DistanceFunctionSpaceBuilder> distanceFunctionSpaceBuilders;
    // whether to match the original Proximity Forest results exactly. This is only useful if mirroring PF parameters
    // exactly
    private boolean matchOriginalPFRandomCalls;
    // whether to check for exemplar matching inside the loop (original) or before any looping (improved method)
    private boolean exemplarCheckOriginal;

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

    public List<DistanceFunctionSpaceBuilder> getDistanceFunctionSpaceBuilders() {
        return distanceFunctionSpaceBuilders;
    }

    public ProximityTree setDistanceFunctionSpaceBuilders(
        final List<DistanceFunctionSpaceBuilder> distanceFunctionSpaceBuilders) {
        this.distanceFunctionSpaceBuilders = distanceFunctionSpaceBuilders;
        return this;
    }

    /**
     * set the default config
     * @return
     */
    public ProximityTree setConfigDefault() {
        setR(5);
        setEarlyAbandonDistances(false);
        setRandomTieBreakDistances(true);
        setRandomTieBreakR(false);
        setBreadthFirst(false);
        setRandomR(false);
        setScorer(new GiniImpurityEntropy());
        setTrainTimeLimit(0);
        setTestTimeLimit(0);
        setExemplarCheckOriginal(true);
        setRandomTieBreakTestDistributionDistances(false);
        setMatchOriginalPFRandomCalls(true);
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

    /**
     * the R=1 config from orig PF paper
     * @return
     */
    public ProximityTree setConfigR1() {
        setConfigDefault();
        setR(1);
        return this;
    }

    /**
     * the R=5 config from orig PF paper
     * @return
     */
    public ProximityTree setConfigR5() {
        setConfigDefault();
        setR(5);
        return this;
    }

    /**
     * the R=10 config from orig PF paper
     * @return
     */
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

    private long findNodeBuildTime(TreeNode<ProximitySplit> node, long time) {
        // assume that the time taken to build a node is proportional to the amount of instances at the node
        final Instances data = node.getElement().getData();
        final long timePerInstance = time / data.size();
        return timePerInstance + 1; // add 1 to account for precision error in div operation
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        // start monitoring resources
        memoryWatcher.start();
        trainTimer.start();
        if(isRebuild()) {
            // reset resources
            memoryWatcher.resetAndStart();
            trainTimer.resetAndStart();
            tree = new BaseTree<>();
            nodeBuildQueue = new LinkedList<>();
            maxTimePerInstanceForNodeBuilding = 0;
            super.buildClassifier(trainData);
            // setup the root node
            final TreeNode<ProximitySplit> root = setupNode(trainData, null);
            // add the root node to the tree
            tree.setRoot(root);
            // add the root node to the build queue
            nodeBuildQueue.add(root);
        }
        while(
            // there's remaining nodes to be built
            !nodeBuildQueue.isEmpty()
            &&
            // there is enough time for another split to be built
            insideTrainTimeLimit(trainTimer.lap() +
                maxTimePerInstanceForNodeBuilding * nodeBuildQueue.peekFirst().getElement().getData().size())
        ) {
            // time how long it takes to build the node
            trainStageTimer.resetAndStart();
            // get the next node to be built
            final TreeNode<ProximitySplit> node = nodeBuildQueue.removeFirst();
            // partition the data at the node
            ProximitySplit split = node.getElement();
            split.buildSplit();
            // for each partition of data build a child node
            final List<TreeNode<ProximitySplit>> children = setupChildNodes(node);
            // add the child nodes to the build queue
            enqueueNodes(children);
            // done building this node
            trainStageTimer.stop();
            // calculate the longest time taken to build a node given
            maxTimePerInstanceForNodeBuilding = findNodeBuildTime(node, trainStageTimer.getTime());
        }
        // stop resource monitoring
        trainTimer.stop();
        memoryWatcher.stop();
    }

    /**
     * add nodes to the build queue if they fail the stopping criteria
     * @param nodes
     */
    private void enqueueNodes(List<TreeNode<ProximitySplit>> nodes) {
        // for each node
        for(int i = 0; i < nodes.size(); i++) {
            TreeNode<ProximitySplit> node;
            if(breadthFirst) {
                // get the ith node if breath first
                node = nodes.get(i);
            } else {
                // get the nodes in reverse order if depth first (as we add to the front of the build queue, so need
                // to lookup nodes in reverse order here)
                node = nodes.get(nodes.size() - i - 1);
            }
            // check the stopping condition hasn't been hit
            // check the data at the node is not pure
            if(!Utilities.isHomogeneous(node.getElement().getData())) {
                // if not hit the stopping condition then add node to the build queue
                if(breadthFirst) {
                    nodeBuildQueue.addLast(node);
                } else {
                    nodeBuildQueue.addFirst(node);
                }
            }
        }
    }

    /**
     * setup the child nodes given the parent node
     * @param parent
     * @return
     */
    private List<TreeNode<ProximitySplit>> setupChildNodes(TreeNode<ProximitySplit> parent) {
        List<TreeNode<ProximitySplit>> children = new ArrayList<>();
        // for each child
        for(Instances partition : parent.getElement().getPartitions()) {
            // setup the node
            children.add(setupNode(partition, parent));
        }
        return children;
    }

    /**
     * setup a node with some given data
     * @param data
     * @param parent
     * @return
     */
    private TreeNode<ProximitySplit> setupNode(Instances data, TreeNode<ProximitySplit> parent) {
        // split the data into multiple partitions, housed in a ProximitySplit object
        final ProximitySplit split = setupSplit(data);
        // build a new node
        final TreeNode<ProximitySplit> node = new BaseTreeNode<>(split, parent);
        return node;
    }

    /**
     * setup the split. This houses all of the split config
     * @param data
     * @return
     */
    private ProximitySplit setupSplit(Instances data) {
        ProximitySplit split = new ProximitySplit(getRandom());
        split.setData(data);
        split.setR(r);
        split.setEarlyAbandonDistances(earlyAbandonDistances);
        split.setExemplarCheckOriginal(exemplarCheckOriginal);
        split.setRandomR(randomR);
        split.setMatchOriginalPFRandomCalls(matchOriginalPFRandomCalls);
        split.setRandomTieBreakDistances(randomTieBreakDistances);
        split.setRandomTieBreakR(randomTieBreakR);
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
            testStageTimer.resetAndStart();
            // get the split at that node
            split = node.getElement();
            // work out which branch to go to next
            index = split.getPartitionIndexFor(instance);
            final List<TreeNode<ProximitySplit>> children = node.getChildren();
            // make this the next node to visit
            node = children.get(index);
            testStageTimer.stop();
            longestPredictTime = testStageTimer.getTime();
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

    public boolean isEarlyAbandonDistances() {
        return earlyAbandonDistances;
    }

    public void setEarlyAbandonDistances(final boolean earlyAbandonDistances) {
        this.earlyAbandonDistances = earlyAbandonDistances;
    }

    public boolean isRandomTieBreakDistances() {
        return randomTieBreakDistances;
    }

    public void setRandomTieBreakDistances(final boolean randomTieBreakDistances) {
        this.randomTieBreakDistances = randomTieBreakDistances;
    }

    public boolean isBreadthFirst() {
        return breadthFirst;
    }

    public void setBreadthFirst(final boolean breadthFirst) {
        this.breadthFirst = breadthFirst;
    }

    public boolean isRandomTieBreakR() {
        return randomTieBreakR;
    }

    public void setRandomTieBreakR(final boolean randomTieBreakR) {
        this.randomTieBreakR = randomTieBreakR;
    }

    public Scorer getScorer() {
        return scorer;
    }

    public void setScorer(final Scorer scorer) {
        Assert.assertNotNull(scorer);
        this.scorer = scorer;
    }

    public boolean isMatchOriginalPFRandomCalls() {
        return matchOriginalPFRandomCalls;
    }

    public void setMatchOriginalPFRandomCalls(final boolean matchOriginalPFRandomCalls) {
        this.matchOriginalPFRandomCalls = matchOriginalPFRandomCalls;
    }

    public boolean isExemplarCheckOriginal() {
        return exemplarCheckOriginal;
    }

    public void setExemplarCheckOriginal(final boolean exemplarCheckOriginal) {
        this.exemplarCheckOriginal = exemplarCheckOriginal;
    }

    public boolean isRandomR() {
        return randomR;
    }

    public void setRandomR(final boolean randomR) {
        this.randomR = randomR;
    }

}
