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
import tsml.classifiers.distance_based.distances.twed.TWEDistanceConfigs;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistanceConfigs;
import tsml.classifiers.distance_based.utils.classifiers.*;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearch;
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
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatchable;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import utilities.ArrayUtilities;
import utilities.ClassifierTools;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.logging.Level;

/**
 * Proximity tree
 * <p>
 * Contributors: goastler
 */
public class ProximityTree extends BaseClassifier implements ContractedTest, ContractedTrain, MemoryWatchable, Checkpointed {

    public static void main(String[] args) throws Exception {
        for(int i = 0; i < 1; i++) {
            int seed = i;
            ProximityTree classifier = Config.PT_R5.build();
            classifier.setSeed(seed);
//            classifier.setCheckpointDirPath("checkpoints");
            classifier.setLogLevel(Level.ALL);
            //            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            ClassifierTools.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed), seed);
        }
    }

    // the various configs for this classifier
    public enum Config implements ClassifierFromEnum<ProximityTree> {
        PT_R1() {
            @Override
            public <B extends ProximityTree> B configure(B proximityTree) {
                proximityTree.setClassifierName(name());
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
                proximityTree.setPartitionScorer(new GiniEntropy());
                proximityTree.setR(1);
                proximityTree.setTrainTimeLimit(-1);
                proximityTree.setTestTimeLimit(-1);
                proximityTree.setBreadthFirst(false);
                return proximityTree;
            }
        },
        PT_R5() {
            @Override
            public <B extends ProximityTree> B configure(B proximityTree) {
                proximityTree = PT_R1.configure(proximityTree);
                proximityTree.setClassifierName(name());
                proximityTree.setR(5);
                return proximityTree;
            }
        },
        PT_R10() {
            @Override
            public <B extends ProximityTree> B configure(B proximityTree) {
                proximityTree = PT_R1.configure(proximityTree);
                proximityTree.setClassifierName(name());
                proximityTree.setR(10);
                return proximityTree;
            }
        },
        ;
    }

    public ProximityTree() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        Config.PT_R1.configure(this);
    }

    private static final long serialVersionUID = 1;
    // train timer
    private final StopWatch trainTimer = new StopWatch();
    // test / predict timer
    private final StopWatch testTimer = new StopWatch();
    // memory watcher
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    // the tree of splits
    private Tree<Split> tree;
    // the train time limit / contract
    private transient long trainTimeLimit;
    // the test time limit / contract
    private transient long testTimeLimit;
    // the longest time taken to build a node / split
    private long longestTimePerInstanceDuringNodeBuild;
    // the queue of nodes left to build
    private Deque<TreeNode<Split>> nodeBuildQueue;
    // the list of distance function space builders to produce distance functions in splits
    private List<ParamSpaceBuilder> distanceFunctionSpaceBuilders;
    // the number of splits to consider for this split
    private int r;
    // a method of scoring the split of data into partitions
    private PartitionScorer partitionScorer;
    // checkpoint config
    private long lastCheckpointTimeStamp = -1;
    private String checkpointPath;
    private String checkpointFileName = Checkpointed.DEFAULT_CHECKPOINT_FILENAME;
    private boolean checkpointLoadingEnabled = true;
    private long checkpointInterval = Checkpointed.DEFAULT_CHECKPOINT_INTERVAL;
    // whether to build the tree depth first or breadth first
    private boolean breadthFirst = false;

    public boolean isBreadthFirst() {
        return breadthFirst;
    }

    public void setBreadthFirst(final boolean breadthFirst) {
        this.breadthFirst = breadthFirst;
    }

    public List<ParamSpaceBuilder> getDistanceFunctionSpaceBuilders() {
        return distanceFunctionSpaceBuilders;
    }

    public void setDistanceFunctionSpaceBuilders(final List<ParamSpaceBuilder> distanceFunctionSpaceBuilders) {
        this.distanceFunctionSpaceBuilders = distanceFunctionSpaceBuilders;
    }

    @Override public long getTrainTime() {
        return trainTimer.getTime();
    }

    @Override public long getTestTime() {
        return testTimer.getTime();
    }

    @Override public long getMaxMemoryUsage() {
        return memoryWatcher.getMaxMemoryUsage();
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
        // load from checkpoint
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
                longestTimePerInstanceDuringNodeBuild = 0;
                // setup the root node
                final TreeNode<Split> root = new BaseTreeNode<>(new Split(trainData), null);
                // add the root node to the tree
                tree.setRoot(root);
                // add the root node to the build queue
                nodeBuildQueue.add(root);
            }
        }
        // update the timings
        LogUtils.logTimeContract(trainTimer.lap(), trainTimeLimit, getLog(), "train");
        final StopWatch trainStageTimer = new StopWatch();
        while(
                // there's remaining nodes to be built
                !nodeBuildQueue.isEmpty()
                &&
                // there is enough time for another split to be built
                insideTrainTimeLimit( trainTimer.getTime() +
                                     longestTimePerInstanceDuringNodeBuild *
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
            longestTimePerInstanceDuringNodeBuild = findNodeBuildTime(node, trainStageTimer.getTime());
            // checkpoint if necessary
            saveCheckpoint();
            // update the train timer
            LogUtils.logTimeContract(trainTimer.lap(), trainTimeLimit, getLog(), "train");
        }
        // stop resource monitoring
        trainTimer.stop();
        memoryWatcher.stop();
        ResultUtils.setInfo(trainResults, this, trainData);
        // checkpoint if work has been done since (i.e. tree has been built further)
        forceSaveCheckpoint();
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
        return Math.max(longestTimePerInstanceDuringNodeBuild, timePerInstance + 1); // add 1 to account for precision
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
            // root node has not been built, just return random guess
            return ArrayUtilities.uniformDistribution(getNumClasses());
        }
        int index = -1;
        int i = 0;
        Split split = node.getElement();
        final StopWatch testStageTimer = new StopWatch();
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

    private Split buildSplit(Split unbuiltSplit) {
        final Instances data = unbuiltSplit.getData();
        double bestSplitScore = Double.NEGATIVE_INFINITY;
        Split bestSplit = null;
        // need to find the best of R splits
        for(int i = 0; i < r; i++) {
            // construct a new split
            Split split = new Split(data);
            split.buildSplit();
            final double score = split.getScore();
            if(score > bestSplitScore) {
                bestSplit = split;
                bestSplitScore = score;
            }
        }
        return bestSplit;
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
        // the score of this split
        private double score = -1;
        // the data at this split (i.e. before being partitioned)
        private Instances data;
        // the partitions of the data, each containing data for the partition and exemplars representing the partition
        private List<Partition> partitions;

        public double getScore() {
            return score;
        }

        private void setupDistanceFunction() {
            // pick a random space
            ParamSpaceBuilder distanceFunctionSpaceBuilder = RandomUtils.choice(distanceFunctionSpaceBuilders, rand);
            // built that space
            ParamSpace distanceFunctionSpace = distanceFunctionSpaceBuilder.build(data);
            // randomly pick the distance function / parameters from that space
            final ParamSet paramSet = RandomSearch.choice(distanceFunctionSpace, getRandom());
            // there is only one distance function in the ParamSet returned
            distanceFunction = Objects.requireNonNull((DistanceFunction) paramSet.getSingle(DistanceMeasure.DISTANCE_MEASURE_FLAG));
        }

        /**
         * pick exemplars from the given dataset
         */
        private void setupExemplarsAndPartitions() {
            // change the view of the data into per class
            final Map<Double, List<Integer>> instancesByClass = Utilities.instancesByClass(this.data);
            // pick exemplars per class
            partitions = new ArrayList<>(instancesByClass.size());
            // generate a partition per class
            for(Double classLabel : instancesByClass.keySet()) {
                final List<Integer> sameClassInstanceIndices = instancesByClass.get(classLabel);
                // if there are less instances to pick from than exemplars requested
                final List<Integer> exemplarIndices = RandomUtils.choiceWithNoSkip(sameClassInstanceIndices, rand, 1);
                // find the exemplars in the data
                final List<Instance> exemplars = new ArrayList<>(exemplarIndices.size());
                for(Integer i : exemplarIndices) {
                    exemplars.add(data.get(i));
                }
                // generate the partition with empty data and the chosen exemplar instances
                final Partition partition = new Partition(new Instances(data, 0), exemplars);
                partitions.add(partition);
            }
            // sanity checks
            Assert.assertEquals(instancesByClass.size(), partitions.size());
            Assert.assertFalse(partitions.isEmpty());
        }

        /**
         * Partition the data and derive score for this split.
         */
        public void buildSplit() {
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
            // a map to maintain the closest partition indices
            PrunedMultimap<Double, Integer> distanceToPartitionMap = PrunedMultimap.asc();
            // let the map keep all ties and randomly choose at the end
            distanceToPartitionMap.setSoftLimit(1);
            // loop through exemplars
            for(int i = 0; i < partitions.size(); i++) {
                final Partition partition = partitions.get(i);
                for(Instance exemplar : partition.getExemplars()) {
                    // for each exemplar
                    // check the instance isn't an exemplar
                    if(instance == exemplar) {
                        return i;
                    }
                    // find the distance
                    final double distance = distanceFunction.distance(instance, exemplar);
                    // add the distance and partition to the map
                    distanceToPartitionMap.put(distance, i);
                }
            }
            // get the smallest distance from the map
            final Double smallestDistance = distanceToPartitionMap.firstKey();
            // find the list of corresponding partitions which the instance could belong to
            final List<Integer> partitionIndices = distanceToPartitionMap.get(smallestDistance);
            // random pick the best partition for the instance
            return RandomUtils.choiceWithNoSkip(partitionIndices, rand);
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

    @Override public long getLastCheckpointTimeStamp() {
        return lastCheckpointTimeStamp;
    }

    @Override public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
    }

    @Override public String getCheckpointFileName() {
        return checkpointFileName;
    }

    @Override public void setCheckpointFileName(final String checkpointFileName) {
        this.checkpointFileName = checkpointFileName;
    }

    @Override public String getCheckpointPath() {
        return checkpointPath;
    }

    @Override public boolean setCheckpointPath(final String checkpointPath) {
        this.checkpointPath = checkpointPath;
        return true;
    }

    @Override public void setCheckpointLoadingEnabled(final boolean checkpointLoadingEnabled) {
        this.checkpointLoadingEnabled = checkpointLoadingEnabled;
    }

    @Override public boolean isCheckpointLoadingEnabled() {
        return checkpointLoadingEnabled;
    }

    @Override public long getCheckpointInterval() {
        return checkpointInterval;
    }

    @Override public void setCheckpointInterval(final long checkpointInterval) {
        this.checkpointInterval = checkpointInterval;
    }
}
