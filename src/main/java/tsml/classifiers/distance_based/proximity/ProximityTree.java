package tsml.classifiers.distance_based.proximity;

import com.google.common.collect.Lists;
import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceConfigs;
import tsml.classifiers.distance_based.distances.ed.EDistanceConfigs;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceConfigs;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistanceConfigs;
import tsml.classifiers.distance_based.distances.msm.MSMDistanceConfigs;
import tsml.classifiers.distance_based.distances.transformed.BaseTransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.twed.TWEDistanceConfigs;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistanceConfigs;
import tsml.classifiers.distance_based.utils.classifiers.*;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointer;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.BaseCheckpointer;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.collections.intervals.Interval;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearchIterator;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap;
import tsml.classifiers.distance_based.utils.collections.tree.BaseTree;
import tsml.classifiers.distance_based.utils.collections.tree.BaseTreeNode;
import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import tsml.classifiers.distance_based.utils.collections.tree.TreeNode;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.stats.scoring.*;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.system.timing.TimedTest;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrain;
import tsml.transformers.IntervalTransform;
import tsml.transformers.TransformPipeline;
import utilities.ArrayUtilities;
import utilities.ClassifierTools;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
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
//            classifier.setCheckpointDirPath("checkpoints");
            classifier.getLogger().setLevel(Level.ALL);
            //            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            ClassifierTools.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed), seed);
        }
    }

    // the various configs for this classifier
    public enum Config implements EnumBasedConfigurer<ProximityTree> {
        PT_R1() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree.setDistanceFunctionSpaceBuilders(Lists.newArrayList(
                        new EDistanceConfigs.EDSpaceBuilder(),
                        new DTWDistanceConfigs.FullWindowDTWSpaceBuilder(),
                        new DTWDistanceConfigs.RestrictedContinuousDTWSpaceBuilder(),
                        new DTWDistanceConfigs.FullWindowDDTWSpaceBuilder(),
                        new DTWDistanceConfigs.RestrictedContinuousDDTWSpaceBuilder(),
                        new WDTWDistanceConfigs.ContinuousWDTWSpaceBuilder(),
                        new WDTWDistanceConfigs.ContinuousWDDTWSpaceBuilder(),
                        new LCSSDistanceConfigs.RestrictedContinuousLCSSSpaceBuilder(),
                        new ERPDistanceConfigs.RestrictedContinuousERPSpaceBuilder(),
                        new TWEDistanceConfigs.TWEDSpaceBuilder(),
                        new MSMDistanceConfigs.MSMSpaceBuilder()
                ));
                proximityTree.setRandomTieBreakDistances(true);
                proximityTree.setEarlyAbandonDistances(false);
                proximityTree.setPartitionScorer(new GiniEntropy());
                proximityTree.setReduceSplitTestSize(false);
                proximityTree.setImprovedExemplarCheck(false);
                proximityTree.setR(1);
                proximityTree.setRandomIntervals(false);
                proximityTree.setMinIntervalSize(-1);
                proximityTree.setRandomR(false);
                proximityTree.setRPatience(false);
                return proximityTree;
            }
        },
        PT_R5() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R1.configureFromEnum(proximityTree);
                proximityTree.setR(5);
                return proximityTree;
            }
        },
        PT_R10() {
            @Override
            public <B extends ProximityTree> B configureFromEnum(B proximityTree) {
                proximityTree = PT_R1.configureFromEnum(proximityTree);
                proximityTree.setR(10);
                return proximityTree;
            }
        },
        ;
    }

    public ProximityTree() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        setTrainTimeLimit(-1);
        setTestTimeLimit(-1);
        setBreadthFirst(false);
        setMaxHeight(-1);
        Config.PT_R1.configureFromEnum(this);
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
    private Tree<Split> tree;
    // the train time limit / contract
    private transient long trainTimeLimit;
    // the test time limit / contract
    private transient long testTimeLimit;
    // the longest time taken to build a node / split
    private long maxTimePerInstanceForNodeBuilding;
    // the queue of nodes left to build
    private Deque<TreeNode<Split>> nodeBuildQueue;
    // whether to build in breadth first or depth first order
    private boolean breadthFirst;
    // the list of distance function space builders to produce distance functions in splits
    private List<ParamSpaceBuilder> distanceFunctionSpaceBuilders;
    // checkpoint config
    private transient final Checkpointer checkpointer = new BaseCheckpointer(this);
    // max tree height
    private int maxHeight;
    // whether to early abandon distance measurements for distance between instances (data) and exemplars
    private boolean earlyAbandonDistances;
    // whether to random tie break distances (e.g. exemplar A and B have a distance of 3.5 to instance X, which to
    // choose?)
    private boolean randomTieBreakDistances;
    // the number of splits to consider for this split
    private int r;
    // a method of scoring the split of data into partitions
    private PartitionScorer partitionScorer;
    // whether to check for exemplar matching inside the loop (original) or before any looping (improved method)
    private boolean improvedExemplarCheck;
    // whether to use intervals
    private boolean randomIntervals;
    // the min interval size if using intervals
    private int minIntervalSize;
    // whether to reduce the number of instances used in testing split quality
    private boolean reduceSplitTestSize;
    // the number of exemplars to pick per class per split
    private int numExemplarsPerClass = 1;
    // whether to randomise the R parameter
    private boolean randomR;
    // whether to use patience in the R parameter
    private boolean rPatience;

    public boolean hasMaxHeight() {
        return maxHeight > 0;
    }

    public Checkpointer getCheckpointer() {
        return checkpointer;
    }

    public List<ParamSpaceBuilder> getDistanceFunctionSpaceBuilders() {
        return distanceFunctionSpaceBuilders;
    }

    public void setDistanceFunctionSpaceBuilders(final List<ParamSpaceBuilder> distanceFunctionSpaceBuilders) {
        this.distanceFunctionSpaceBuilders = distanceFunctionSpaceBuilders;
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
        return testTimeLimit;
    }

    @Override
    public void setTestTimeLimit(final long nanos) {
        testTimeLimit = nanos;
    }

    @Override
    public long getTrainTimeLimit() {
        return trainTimeLimit;
    }

    @Override
    public void setTrainTimeLimit(final long nanos) {
        trainTimeLimit = nanos;
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        // start monitoring resources
        memoryWatcher.start();
        trainTimer.start();
        final Logger logger = getLogger();
        // if checkpoint exists then skip initialisation
        if(!loadCheckpoint()) {
            // no checkpoint exists so check whether rebuilding is enabled
            super.buildClassifier(trainData);
            // if rebuilding (i.e. building from scratch) initialise the classifier
            if(isRebuild()) {
                // reset resources
                memoryWatcher.resetAndStart();
                trainTimer.resetAndStart();
                tree = new BaseTree<>();
                nodeBuildQueue = new LinkedList<>();
                maxTimePerInstanceForNodeBuilding = 0;
                // setup the root node
                final TreeNode<Split> root = new BaseTreeNode<>(new Split(trainData), null);
                // add the root node to the tree
                tree.setRoot(root);
                // add the root node to the build queue
                nodeBuildQueue.add(root);
            }
        }
        // update the timings
        trainTimer.lap();
        LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimit, logger, "train");
        while(
                // there's remaining nodes to be built
                !nodeBuildQueue.isEmpty()
                &&
                // there is enough time for another split to be built
                insideTrainTimeLimit( trainTimer.getTime() +
                                     maxTimePerInstanceForNodeBuilding *
                                     nodeBuildQueue.peekFirst().getElement().getData().size())
        ) {
            // time how long it takes to build the node
            trainStageTimer.resetAndStart();
            // get the next node to be built
            final TreeNode<Split> node = nodeBuildQueue.removeFirst();
            // partition the data at the node
            Split split = node.getElement();
            // find the best of R partitioning attempts
            split = buildSplit(split);
            node.setElement(split);
            // for each partition of data build a child node
            final List<TreeNode<Split>> children = setupChildNodes(node);
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
            LogUtils.logTimeContract(trainTimer.getTime(), trainTimeLimit, logger, "train");
        }
        // stop resource monitoring
        trainTimer.stop();
        memoryWatcher.stop();
        ResultUtils.setInfo(trainResults, this, trainData);
        // checkpoint if work has been done since (i.e. tree has been built further)
        checkpointIfWorkDone();
    }

    /**
     * setup the child nodes given the parent node
     *
     * @param parent
     * @return
     */
    private List<TreeNode<Split>> setupChildNodes(TreeNode<Split> parent) {
        final List<Instances> partitions = parent.getElement().getPartitionedData();
        List<TreeNode<Split>> children = new ArrayList<>(partitions.size());
        // for each child
        for(Instances partition : partitions) {
            // setup the node
            children.add(new BaseTreeNode<>(new Split(partition), parent));
        }
        return children;
    }

    /**
     * add nodes to the build queue if they fail the stopping criteria
     *
     * @param nodes
     */
    private void enqueueNodes(List<TreeNode<Split>> nodes) {
        // for each node
        for(int i = 0; i < nodes.size(); i++) {
            TreeNode<Split> node;
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

    private long findNodeBuildTime(TreeNode<Split> node, long time) {
        // assume that the time taken to build a node is proportional to the amount of instances at the node
        final Instances data = node.getElement().getData();
        final long timePerInstance = time / data.size();
        return Math.max(maxTimePerInstanceForNodeBuilding, timePerInstance + 1); // add 1 to account for precision
        // error in div operation
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        // enable resource monitors
        testTimer.resetAndStart();
        long longestPredictTime = 0;
        // start at the tree node
        TreeNode<Split> node = tree.getRoot();
        if(!node.hasChildren()) {
            //             root node has not been built, just return random guess
            return ArrayUtilities.uniformDistribution(getNumClasses());
        }
        int index = -1;
        int i = 0;
        Split split = node.getElement();
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
            index = split.findPartitionIndexFor(instance);
            final List<TreeNode<Split>> children = node.getChildren();
            // make this the next node to visit
            node = children.get(index);
            testStageTimer.stop();
            longestPredictTime = testStageTimer.getTime();
        }
        // hit a leaf node
        // get the parent of the leaf node to work out distribution
        node = node.getParent();
        split = node.getElement();
        // use the partition index to get the partition and then the distribution for instance for that partition
        double[] distribution = split.getPartitions().get(index).distributionForInstance(instance);
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

    public int getR() {
        return r;
    }

    public void setR(final int r) {
        this.r = r;
    }

    public PartitionScorer getPartitionScorer() {
        return partitionScorer;
    }

    public void setPartitionScorer(final PartitionScorer partitionScorer) {
        this.partitionScorer = partitionScorer;
    }

    public boolean isImprovedExemplarCheck() {
        return improvedExemplarCheck;
    }

    public void setImprovedExemplarCheck(final boolean improvedExemplarCheck) {
        this.improvedExemplarCheck = improvedExemplarCheck;
    }

    public boolean isRandomIntervals() {
        return randomIntervals;
    }

    public void setRandomIntervals(final boolean randomIntervals) {
        this.randomIntervals = randomIntervals;
    }

    public int getMinIntervalSize() {
        return minIntervalSize;
    }

    public void setMinIntervalSize(final int minIntervalSize) {
        this.minIntervalSize = minIntervalSize;
    }

    public boolean isReduceSplitTestSize() {
        return reduceSplitTestSize;
    }

    public void setReduceSplitTestSize(final boolean reduceSplitTestSize) {
        this.reduceSplitTestSize = reduceSplitTestSize;
    }

    public int getNumExemplarsPerClass() {
        return numExemplarsPerClass;
    }

    public void setNumExemplarsPerClass(final int numExemplarsPerClass) {
        this.numExemplarsPerClass = numExemplarsPerClass;
    }

    public boolean isRandomR() {
        return randomR;
    }

    public void setRandomR(final boolean randomR) {
        this.randomR = randomR;
    }

    private Split buildSplit(Split unbuiltSplit) {
        final Instances data = unbuiltSplit.getData();
        double bestSplitScore = Double.NEGATIVE_INFINITY;
        Split bestSplit = null;
        int r = this.r;
        if(randomR) {
            r = rand.nextInt(r) + 1;
        }
        // need to find the best of R splits
        for(int i = 0; i < r; i++) {
            // construct a new split
            Split split = new Split(data);
            split.partitionData();
            final double score = split.getScore();
            if(score > bestSplitScore) {
                bestSplit = split;
                bestSplitScore = score;
                if(rPatience) {
                    i = 0;
                }
            }
        }
        return bestSplit;
    }

    public boolean isRPatience() {
        return rPatience;
    }

    public void setRPatience(final boolean rPatience) {
        this.rPatience = rPatience;
    }

    private static class Partition {

        private Partition(final Instances data, final List<Instance> exemplars) {
            this.data = data;
            this.exemplars = exemplars;
        }

        // data in this partition
        private final Instances data;
        // exemplar instances representing this partition
        private final List<Instance> exemplars;

        /**
         * find the distribution for the given instance and partition index it belongs to
         *
         * @param instance the instance
         * @return the distribution
         */
        public double[] distributionForInstance(final Instance instance) {
            // this is a simple majority vote over all the exemplars in the exemplars group at the given partition
            // get the corresponding closest exemplars
            final double[] distribution = new double[instance.numClasses()];
            // for each exemplar
            for(Instance exemplar : exemplars) {
                // vote for the exemplar's class
                double classValue = exemplar.classValue();
                distribution[(int) classValue]++;
            }
            ArrayUtilities.normalise(distribution);
            return distribution;
        }

        public Instances getData() {
            return data;
        }

        public List<Instance> getExemplars() {
            return exemplars;
        }

    }

    private class Split {

        public Split() {}

        public Split(Instances data) {
            setData(data);
        }

        // the distance function for comparing instances to exemplars
        private DistanceFunction distanceFunction;
//        // the exemplars for each partition
        private Map<Instance, Integer> exemplarIndexToPartitionIndexMap;
        // the score of this split
        private double score = -1;
        // the data at this split (i.e. before being partitioned)
        private Instances data;
        // the partitions of the data, each containing data for the partition and exemplars representing the partition
        private List<Partition> partitions;

        public double getScore() {
            return score;
        }

        public void setupDistanceFunction() {
            Instances dataForParamSpaceBuilding = data;
            IntervalTransform intervalTransform = null;
            if(randomIntervals) {
                // suppose we're looking at instances of length 41.
                // the last value is the class label, therefore there's a ts of 40.
                // the max length of an interval is therefore numAttributes() - 1. +1 for random call is cancelled out
                // by the -1 for num attributes including the class label
                // if a min interval size is then included, say 3, then the max size of the interval should be 40 - 3 =
                // 37. The min size can be subtracted from the rand call and added after to ensure rand numbers between
                // min and max length (3 and 40).
                final int length = rand.nextInt(data.numAttributes() - minIntervalSize) + minIntervalSize;
                Assert.assertTrue(length > 0);
                // the start point is dependent on the length. Max length of 40 then the start can only be 0. Min length
                // of 3 then the start can be anywhere between 0..37 inclusively.
                // The start can therefore lie anywhere from 0 to tsLen - intervalLen inclusively. (shortest interval
                // would be 3, 40 - 3 = 37, checks out). +1 for random call is cancelled out by the -1 for num attributes
                // including the class label
                final int start = rand.nextInt(data.numAttributes() - length);
                final Interval interval = new Interval(start, length);
                intervalTransform = new IntervalTransform(interval);
                dataForParamSpaceBuilding = intervalTransform.transform(data);
            }
            // pick a random space
            ParamSpaceBuilder distanceFunctionSpaceBuilder = RandomUtils.choice(distanceFunctionSpaceBuilders, rand);
            // built that space
            ParamSpace distanceFunctionSpace = distanceFunctionSpaceBuilder.build(dataForParamSpaceBuilding);
            // randomly pick the distance function / parameters from that space
            final ParamSet paramSet = RandomSearchIterator.choice(rand, distanceFunctionSpace);
            // there is only one distance function in the ParamSet returned
            distanceFunction = (DistanceFunction) paramSet.getSingle(DistanceMeasure.DISTANCE_MEASURE_FLAG);
            if(randomIntervals) {
                if(distanceFunction instanceof TransformDistanceMeasure) {
                    final TransformDistanceMeasure tdf = (TransformDistanceMeasure) distanceFunction;
                    tdf.setTransformer(new TransformPipeline(newArrayList(tdf.getTransformer(), intervalTransform)));
                } else {
                    distanceFunction = new BaseTransformDistanceMeasure(DistanceMeasure.getName(distanceFunction), intervalTransform, distanceFunction);
                }
            }
            Assert.assertNotNull(distanceFunction);
        }

        /**
         * pick exemplars from the given dataset
         */
        public void setupExemplarsAndPartitions() {
            // change the view of the data into per class
            final Map<Double, List<Integer>> instancesByClass = Utilities.instancesByClass(this.data);
            // pick exemplars per class
            final int numPartitions = instancesByClass.size();
            final int totalNumExemplars = numExemplarsPerClass * numPartitions;
            partitions = new ArrayList<>(numPartitions);
            // populate an exemplar lookup for the improved exemplar check
            exemplarIndexToPartitionIndexMap = null;
            if(improvedExemplarCheck) {
                exemplarIndexToPartitionIndexMap = new HashMap<>(totalNumExemplars, 1);
            }
            // generate a partition per class
            for(Double classLabel : instancesByClass.keySet()) {
                final List<Integer> sameClassInstanceIndices = instancesByClass.get(classLabel);
                // if there are less instances to pick from than exemplars requested
                final List<Integer> exemplarIndices;
                if(numExemplarsPerClass >= sameClassInstanceIndices.size()) {
                    // then use all instances in the class
                    exemplarIndices = sameClassInstanceIndices;
                } else {
                    // otherwise randomly pick the exemplars
                    exemplarIndices = RandomUtils.choice(sameClassInstanceIndices, rand, numExemplarsPerClass);
                }
                // find the exemplars in the data
                final List<Instance> exemplars = new ArrayList<>(exemplarIndices.size());
                for(Integer i : exemplarIndices) {
                    exemplars.add(data.get(i));
                }
                // generate the partition with empty data and the chosen exemplar instances
                final Partition partition = new Partition(new Instances(data, 0), exemplars);
                partitions.add(partition);
                final int partitionIndex = partitions.size() - 1;
                if(improvedExemplarCheck) {
                    // add exemplar to partition mapping
                    for(Instance exemplar : exemplars) {
                        exemplarIndexToPartitionIndexMap.put(exemplar, partitionIndex);
                    }
                }
            }
            // sanity checks
            Assert.assertEquals(instancesByClass.size(), partitions.size());
            Assert.assertFalse(partitions.isEmpty());
        }

        public void partitionData() {
            // pick the distance function
            setupDistanceFunction();
            // pick the exemplars
            setupExemplarsAndPartitions();
            // setup the distance function
            distanceFunction.setInstances(data);
            // go through every instance and find which partition it should go into. This should be the partition
            // with the closest exemplar associate
            for(int i = 0; i < data.size(); i++) {
                final Instance instance = data.get(i);
                final Partition closestPartition = findPartitionFor(instance);
                // add the instance to the partition
                closestPartition.getData().add(instance);
            }
            // find the score of this split attempt, i.e. how good it is
            score = partitionScorer.findScore(data, getPartitionedData());
        }

        public DistanceFunction getDistanceFunction() {
            return distanceFunction;
        }

        /**
         * get the partition of the given instance. The returned partition is the set of data the given instance belongs to based on its proximity to the exemplar instances representing the partition.
         *
         * @param instance
         * @return
         */
        public int findPartitionIndexFor(final Instance instance) {
            // the instance may be an exemplar, so lookup the partition index
            if(improvedExemplarCheck) {
                Integer index = exemplarIndexToPartitionIndexMap.get(instance);
                if(index != null) {
                    // the instance is an exemplar and the partition must therefore be the partition that exemplar represents
                    return index;
                }
                // the instance is not an exemplar, so find the distances to each exemplar and pick partition from that
                // get the partition for this instance (based on proximity to corresponding exemplars)
            }
            // the limit for early abandon
            double limit = Double.POSITIVE_INFINITY;
            // a map to maintain the closest partition indices
            PrunedMultimap<Double, Integer> distanceToPartitionMap = PrunedMultimap.asc();
            if(randomTieBreakDistances) {
                // let the map keep all ties and randomly choose at the end
                distanceToPartitionMap.setSoftLimit(1);
            } else {
                // only keep 1 partition at any point in time, even if multiple partitions are equally close
                distanceToPartitionMap.setHardLimit(1);
                // discard the newest on tie break situation
                distanceToPartitionMap.setDiscardType(PrunedMultimap.DiscardType.NEWEST);
            }
            // loop through exemplars
            for(int i = 0; i < partitions.size(); i++) {
                final Partition partition = partitions.get(i);
                for(Instance exemplar : partition.getExemplars()) {
                    // for each exemplar
                    // check the instance isn't an exemplar
                    if(!improvedExemplarCheck) {
                        if(instance == exemplar) {
                            return i;
                        }
                    }
                    // find the distance
                    final double distance = distanceFunction.distance(instance, exemplar, limit);
                    // adjust early abandon limit
                    if(earlyAbandonDistances) {
                        limit = Math.min(distance, limit);
                    }
                    // add the distance and partition to the map
                    distanceToPartitionMap.put(distance, i);
                }
            }
            // get the smallest distance from the map
            final Double smallestDistance = distanceToPartitionMap.firstKey();
            // find the list of corresponding partitions which the instance could belong to
            final List<Integer> partitionIndices = distanceToPartitionMap.get(smallestDistance);
            if(!randomTieBreakDistances) {
                Assert.assertEquals(partitionIndices.size(), 1);
            }
            // random pick the best partition for the instance
            return RandomUtils.choice(partitionIndices, rand);
        }

        public Partition findPartitionFor(Instance instance) {
            return partitions.get(findPartitionIndexFor(instance));
        }

        public Instances getData() {
            return data;
        }

        public void setData(Instances data) {
            Assert.assertNotNull(data);
            this.data = data;
        }

        @Override
        public String toString() {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.append(getClass().getSimpleName() + "{" +
                                         "score=" + score +
                                         ", dataSize=" + data.size());
            stringBuilder.append(", df=");
            if(distanceFunction != null) {
                stringBuilder.append(distanceFunction.toString());
            } else {
                stringBuilder.append("null");
            }
            final List<Instances> partitionedData = getPartitionedData();
            if(partitionedData != null) {
                int i = 0;
                for(Instances instances : partitionedData) {
                    stringBuilder.append(", p" + i + "=" + instances.size());
                    i++;
                }
            }
            stringBuilder.append("}");
            return stringBuilder.toString();
        }

        public List<Partition> getPartitions() {
            return partitions;
        }

        public List<Instances> getPartitionedData() {
            if(partitions == null) {
                return null;
            }
            List<Instances> partitionDatas = new ArrayList<>(partitions.size());
            for(Partition partition : partitions) {
                partitionDatas.add(partition.getData());
            };
            return partitionDatas;
        }
    }
}
