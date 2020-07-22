package tsml.classifiers.distance_based.proximity;

import com.google.common.collect.Lists;
import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifiers.*;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointer;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.BaseCheckpointer;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.collections.tree.BaseTree;
import tsml.classifiers.distance_based.utils.collections.tree.BaseTreeNode;
import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import tsml.classifiers.distance_based.utils.collections.tree.TreeNode;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.stats.scoring.ChiSquaredGain;
import tsml.classifiers.distance_based.utils.stats.scoring.GiniGain;
import tsml.classifiers.distance_based.utils.stats.scoring.InfoGain;
import tsml.classifiers.distance_based.utils.stats.scoring.InfoEntropy;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.system.timing.TimedTest;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrain;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

/**
 * Purpose: proximity tree
 * <p>
 * Contributors: goastler
 */
public class ProximityTree extends BaseClassifier implements ContractedTest, ContractedTrain, TimedTrain, TimedTest, WatchedMemory, Checkpointed {

    public static void main(String[] args) throws Exception {
        for(int i = 0; i < 1; i++) {
            int seed = i;
            ProximityTree classifier = new ProximityTree();
            classifier.setSeed(seed);
            Config.PT_R5.configureFromEnum(classifier);
            classifier.setCheckpointDirPath("checkpoints");
            classifier.getLogger().setLevel(Level.ALL);
            //            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            Utils.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed));
        }
    }

    // the various configs for this classifier
    public enum Config implements EnumBasedConfigurer<ProximityTree> {
        PT_R1() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree.setBreadthFirst(false);
                proximityTree.setTrainTimeLimit(0);
                proximityTree.setTestTimeLimit(0);
                proximityTree.setDistanceFunctionSpaceBuilders(Lists.newArrayList(
                        ParamSpaceBuilder.ED,
                        ParamSpaceBuilder.FULL_DTW,
                        ParamSpaceBuilder.DTW,
                        ParamSpaceBuilder.FULL_DDTW,
                        ParamSpaceBuilder.DDTW,
                        ParamSpaceBuilder.WDTW,
                        ParamSpaceBuilder.WDDTW,
                        ParamSpaceBuilder.LCSS,
                        ParamSpaceBuilder.ERP,
                        ParamSpaceBuilder.TWED,
                        ParamSpaceBuilder.MSM
                ));
                proximityTree.setProximitySplitConfig(ProximitySplit.Config.PS_R1);
                return proximityTree;
            }
        },
        PT_R1_QUICK() {
            @Override public <B extends ProximityTree> B configureFromEnum(B  classifier) {
                classifier = PT_R1.configureFromEnum(classifier);
                classifier.setProximitySplitConfig(ProximitySplit.Config.PS_R1_QUICK);
                return classifier;
            }
        },
        PT_R5_QUICK() {
            @Override public <B extends ProximityTree> B configureFromEnum(B  classifier) {
                classifier = PT_R5.configureFromEnum(classifier);
                classifier.setProximitySplitConfig(ProximitySplit.Config.PS_R5_QUICK);
                return classifier;
            }
        },
        PT_R10_QUICK() {
            @Override public <B extends ProximityTree> B configureFromEnum(B  classifier) {
                classifier = PT_R10.configureFromEnum(classifier);
                classifier.setProximitySplitConfig(ProximitySplit.Config.PS_R10_QUICK);
                return classifier;
            }
        },
        PT_R5_CHI() {
            @Override public <B extends ProximityTree> B configureFromEnum(B  classifier) {
                classifier = PT_R5.configureFromEnum(classifier);
                classifier.setProximitySplitConfig(classifier.getProximitySplitConfig().and(
                        new Configurer<ProximitySplit>() {
                            @Override public <B extends ProximitySplit> B configure(final B classifier) {
                                classifier.setEarlyAbandonDistances(true);
                                classifier.setPartitionScorer(new ChiSquaredGain());
                                return classifier;
                            }
                        }));
                return classifier;
            }
        },
        PT_R5_IG() {
            @Override public <B extends ProximityTree> B configureFromEnum(B  classifier) {
                classifier = PT_R5.configureFromEnum(classifier);
                classifier.setProximitySplitConfig(classifier.getProximitySplitConfig().and(
                        new Configurer<ProximitySplit>() {
                            @Override public <B extends ProximitySplit> B configure(final B classifier) {
                                classifier.setEarlyAbandonDistances(true);
                                classifier.setPartitionScorer(new InfoGain());
                                return classifier;
                            }
                        }));
                return classifier;
            }
        },
        PT_R5_GG() {
            @Override public <B extends ProximityTree> B configureFromEnum(B  classifier) {
                classifier = PT_R5.configureFromEnum(classifier);
                classifier.setProximitySplitConfig(classifier.getProximitySplitConfig().and(
                        new Configurer<ProximitySplit>() {
                            @Override public <B extends ProximitySplit> B configure(final B classifier) {
                                classifier.setEarlyAbandonDistances(true);
                                classifier.setPartitionScorer(new GiniGain());
                                return classifier;
                            }
                        }));
                return classifier;
            }
        },
        PT_R5_IE() {
            @Override public <B extends ProximityTree> B configureFromEnum(B  classifier) {
                classifier = PT_R5.configureFromEnum(classifier);
                classifier.setProximitySplitConfig(classifier.getProximitySplitConfig().and(
                        new Configurer<ProximitySplit>() {
                            @Override public <B extends ProximitySplit> B configure(final B classifier) {
                                classifier.setEarlyAbandonDistances(true);
                                classifier.setPartitionScorer(new InfoEntropy());
                                return classifier;
                            }
                        }));
                return classifier;
            }
        },
        PT_R5_O() {
            @Override public <B extends ProximityTree> B configureFromEnum(B  classifier) {
                classifier = PT_R5.configureFromEnum(classifier);
                classifier.setDistanceFunctionSpaceBuilders(newArrayList(
                        ParamSpaceBuilder.DTW,
                        ParamSpaceBuilder.DDTW,
                        ParamSpaceBuilder.LCSS,
                        ParamSpaceBuilder.ERP
                ));
                classifier.setProximitySplitConfig(classifier.getProximitySplitConfig().and(
                        new Configurer<ProximitySplit>() {
                            @Override public <B extends ProximitySplit> B configure(final B classifier) {
                                classifier.setEarlyAbandonDistances(true);
                                return classifier;
                            }
                        }));
                return classifier;
            }
        },
        PT_R5_OU() {
            @Override public <B extends ProximityTree> B configureFromEnum(B  classifier) {
                classifier = PT_R5.configureFromEnum(classifier);
                classifier.setDistanceFunctionSpaceBuilders(newArrayList(
                        ParamSpaceBuilder.UDTW,
                        ParamSpaceBuilder.UDDTW,
                        ParamSpaceBuilder.ULCSS,
                        ParamSpaceBuilder.UERP
                ));
                classifier.setProximitySplitConfig(classifier.getProximitySplitConfig().and(
                        new Configurer<ProximitySplit>() {
                            @Override public <B extends ProximitySplit> B configure(final B classifier) {
                                classifier.setEarlyAbandonDistances(true);
                                return classifier;
                            }
                        }));
                return classifier;
            }
        },
        PT_R5_U() {
            @Override public <B extends ProximityTree> B configureFromEnum(B  classifier) {
                classifier = PT_R5.configureFromEnum(classifier);
                classifier.setDistanceFunctionSpaceBuilders(newArrayList(
                        ParamSpaceBuilder.ED,
                        ParamSpaceBuilder.FULL_DTW,
                        ParamSpaceBuilder.UDTW,
                        ParamSpaceBuilder.FULL_DDTW,
                        ParamSpaceBuilder.UDDTW,
                        ParamSpaceBuilder.WDTW,
                        ParamSpaceBuilder.WDDTW,
                        ParamSpaceBuilder.ULCSS,
                        ParamSpaceBuilder.UERP,
                        ParamSpaceBuilder.TWED,
                        ParamSpaceBuilder.MSM
                ));
                classifier.setProximitySplitConfig(classifier.getProximitySplitConfig().and(
                        new Configurer<ProximitySplit>() {
                            @Override public <B extends ProximitySplit> B configure(final B classifier) {
                                classifier.setEarlyAbandonDistances(true);
                                return classifier;
                            }
                        }));
                return classifier;
            }
        },
        PT_R5() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R1.configureFromEnum(proximityTree);
                proximityTree.setProximitySplitConfig(ProximitySplit.Config.PS_R5);
                return proximityTree;
            }
        },
        PT_R10() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R1.configureFromEnum(proximityTree);
                proximityTree.setProximitySplitConfig(ProximitySplit.Config.PS_R10);
                return proximityTree;
            }
        },
        PT_R5_I() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R5.configureFromEnum(proximityTree);
                proximityTree.setProximitySplitConfig(ProximitySplit.Config.PS_R5_I);
                return proximityTree;
            }
        },
        PT_R10_I() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R10.configureFromEnum(proximityTree);
                proximityTree.setProximitySplitConfig(ProximitySplit.Config.PS_R10_I);
                return proximityTree;
            }
        },
        PT_R1_I() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R1.configureFromEnum(proximityTree);
                proximityTree.setProximitySplitConfig(ProximitySplit.Config.PS_R1_I);
                return proximityTree;
            }
        },
        PT_RR5_I() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_RR5.configureFromEnum(proximityTree);
                proximityTree.setProximitySplitConfig(ProximitySplit.Config.PS_R5_I);
                return proximityTree;
            }
        },
        PT_RR10_I() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_RR10.configureFromEnum(proximityTree);
                proximityTree.setProximitySplitConfig(ProximitySplit.Config.PS_R10_I);
                return proximityTree;
            }
        },
        PT_RR5() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R5.configureFromEnum(proximityTree);
                proximityTree.setProximitySplitConfig(ProximitySplit.Config.PS_RR5);
                return proximityTree;
            }
        },
        PT_RR10() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R10.configureFromEnum(proximityTree);
                proximityTree.setProximitySplitConfig(ProximitySplit.Config.PS_RR10);
                return proximityTree;
            }
        },
        PT_R5_STUMP() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R5.configureFromEnum(proximityTree);
                proximityTree.setMaxHeight(1);
                return proximityTree;
            }
        },
        PT_R1_STUMP() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R1.configureFromEnum(proximityTree);
                proximityTree.setMaxHeight(1);
                return proximityTree;
            }
        },
        PT_R10_STUMP() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R10.configureFromEnum(proximityTree);
                proximityTree.setMaxHeight(1);
                return proximityTree;
            }
        }
        ;
    }

    public ProximityTree() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        Config.PT_R5.configureFromEnum(this);
    }

    private static final long serialVersionUID = 1;
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
    private transient long trainTimeLimitNanos;
    // the test time limit / contract
    private transient long testTimeLimitNanos;
    // the longest time taken to build a node / split
    private long maxTimePerInstanceForNodeBuilding;
    // the queue of nodes left to build
    private Deque<TreeNode<ProximitySplit>> nodeBuildQueue;
    // whether to build in breadth first or depth first order
    private boolean breadthFirst;
    // the list of distance function space builders to produce distance functions in splits
    private List<ParamSpaceBuilder> distanceFunctionSpaceBuilders;
    // method of setting up split config
    private Configurer<ProximitySplit> proximitySplitConfig;
    // checkpoint config
    private transient final Checkpointer checkpointer = new BaseCheckpointer(this);
    // max tree height
    private int maxHeight = -1;

    public boolean hasMaxHeight() {
        return maxHeight > 0;
    }

    public Checkpointer getCheckpointer() {
        return checkpointer;
    }

    public Configurer<ProximitySplit> getProximitySplitConfig() {
        return proximitySplitConfig;
    }

    public void setProximitySplitConfig(
            final Configurer<ProximitySplit> proximitySplitConfig) {
        Assert.assertNotNull(proximitySplitConfig);
        this.proximitySplitConfig = proximitySplitConfig;
    }

    public List<ParamSpaceBuilder> getDistanceFunctionSpaceBuilders() {
        return distanceFunctionSpaceBuilders;
    }

    public ProximityTree setDistanceFunctionSpaceBuilders(
            final List<ParamSpaceBuilder> distanceFunctionSpaceBuilders) {
        this.distanceFunctionSpaceBuilders = distanceFunctionSpaceBuilders;
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
        final Logger logger = getLogger();
        loadCheckpoint();
        // start monitoring resources
        memoryWatcher.start();
        trainTimer.start();
        final boolean rebuild = isRebuild();
        super.buildClassifier(trainData);
        if(rebuild) {
            // reset resources
            memoryWatcher.resetAndStart();
            trainTimer.resetAndStart();
            tree = new BaseTree<>();
            nodeBuildQueue = new LinkedList<>();
            maxTimePerInstanceForNodeBuilding = 0;
            // setup the root node
            final TreeNode<ProximitySplit> root = setupNode(trainData, null);
            // add the root node to the tree
            tree.setRoot(root);
            // add the root node to the build queue
            nodeBuildQueue.add(root);
        }
        LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimitNanos, logger, "train");
        trainTimer.lap();
        while(
            // there's remaining nodes to be built
                !nodeBuildQueue.isEmpty()
                &&
                // there is enough time for another split to be built
                insideTrainTimeLimit( trainTimer.getTime() +
                                     maxTimePerInstanceForNodeBuilding *
                                     nodeBuildQueue.peekFirst().getElement().getTrainData().size())
        ) {
            LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimitNanos, logger, "train");
            // time how long it takes to build the node
            trainStageTimer.resetAndStart();
            // get the next node to be built
            final TreeNode<ProximitySplit> node = nodeBuildQueue.removeFirst();
            // partition the data at the node
            ProximitySplit split = node.getElement();
            split.buildClassifier();
            // for each partition of data build a child node
            final List<TreeNode<ProximitySplit>> children = setupChildNodes(node);
            // add the child nodes to the build queue
            enqueueNodes(children);
            // done building this node
            trainStageTimer.stop();
            // calculate the longest time taken to build a node given
            maxTimePerInstanceForNodeBuilding = findNodeBuildTime(node, trainStageTimer.getTime());
            // checkpoint if necessary
            checkpointIfIntervalExpired();
            // update the train timer
            trainTimer.lap();
        }
        // stop resource monitoring
        trainTimer.stop();
        memoryWatcher.stop();
        ResultUtils.setInfo(trainResults, this, trainData);
        // checkpoint if work has been done since loading the first checkpoint
        checkpointIfWorkDone();
    }

    /**
     * setup a node with some given data
     *
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
     * setup the child nodes given the parent node
     *
     * @param parent
     * @return
     */
    private List<TreeNode<ProximitySplit>> setupChildNodes(TreeNode<ProximitySplit> parent) {
        final List<Instances> partitions = parent.getElement().getPartitions();
        List<TreeNode<ProximitySplit>> children = new ArrayList<>(partitions.size());
        // for each child
        for(Instances partition : partitions) {
            // setup the node
            children.add(setupNode(partition, parent));
        }
        return children;
    }

    /**
     * add nodes to the build queue if they fail the stopping criteria
     *
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
            // check the node's level in the tree is not beyond the max height
            if(hasMaxHeight() && node.getLevel() > maxHeight) {
                // if so then do not build the node
                continue;
            }
            // check the data at the node is not pure
            if(!Utilities.isHomogeneous(node.getElement().getTrainData())) {
                // if not hit the stopping condition then add node to the build queue
                if(breadthFirst) {
                    nodeBuildQueue.addLast(node);
                } else {
                    nodeBuildQueue.addFirst(node);
                }
            }
        }
    }

    private long findNodeBuildTime(TreeNode<ProximitySplit> node, long time) {
        // assume that the time taken to build a node is proportional to the amount of instances at the node
        final Instances data = node.getElement().getTrainData();
        final long timePerInstance = time / data.size();
        return Math.max(maxTimePerInstanceForNodeBuilding, timePerInstance + 1); // add 1 to account for precision
        // error in div operation
    }

    /**
     * setup the split. This houses all of the split config
     *
     * @param data
     * @return
     */
    private ProximitySplit setupSplit(Instances data) {
        ProximitySplit split = new ProximitySplit();
        split.setSeed(rand.nextInt());
        split.setTrainData(data);
        split.setDistanceFunctionSpaceBuilders(distanceFunctionSpaceBuilders);
        proximitySplitConfig.configure(split);
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

    public boolean isBreadthFirst() {
        return breadthFirst;
    }

    public void setBreadthFirst(final boolean breadthFirst) {
        this.breadthFirst = breadthFirst;
    }

    public int getMaxHeight() {
        return maxHeight;
    }

    public void setMaxHeight(final int maxHeight) {
        this.maxHeight = maxHeight;
    }
}
