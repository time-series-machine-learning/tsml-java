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
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.ArrayUtilities;
import utilities.ClassifierTools;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.logging.Level;
import java.util.stream.Collectors;

/**
 * Proximity tree
 * <p>
 * Contributors: goastler
 */
public class ProximityTree extends BaseClassifier implements ContractedTest, ContractedTrain, Checkpointed {

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
        PT() {
            @Override public <B extends ProximityTree> B configure(final B classifier) {
                return PT_R5.configure(classifier);
            }
        },
        PT_R1() {
            @Override
            public <B extends ProximityTree> B configure(B proximityTree) {
                proximityTree.setClassifierName(name());
                proximityTree.setDistanceMeasureSpaceBuilders(Lists.newArrayList(
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

        @Override public ProximityTree newInstance() {
            return new ProximityTree();
        }
    }

    public ProximityTree() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        Config.PT_R1.configure(this);
    }

    private static final long serialVersionUID = 1;
    // store the train data
    private TimeSeriesInstances trainData;
    // train timer
    private final StopWatch runTimer = new StopWatch();
    // test / predict timer
    private final StopWatch testTimer = new StopWatch();
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
    private List<ParamSpaceBuilder> distanceMeasureSpaceBuilders;
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
    private StopWatch checkpointTimer = new StopWatch();
    // whether to build the tree depth first or breadth first
    private boolean breadthFirst = false;

    @Override public long getCheckpointTime() {
        return checkpointTimer.elapsedTime();
    }

    @Override public boolean isModelFullyBuilt() {
        return nodeBuildQueue != null && nodeBuildQueue.isEmpty() && tree != null && tree.getRoot() != null;
    }

    public boolean isBreadthFirst() {
        return breadthFirst;
    }

    public void setBreadthFirst(final boolean breadthFirst) {
        this.breadthFirst = breadthFirst;
    }

    public List<ParamSpaceBuilder> getDistanceMeasureSpaceBuilders() {
        return distanceMeasureSpaceBuilders;
    }

    public void setDistanceMeasureSpaceBuilders(final List<ParamSpaceBuilder> distanceMeasureSpaceBuilders) {
        this.distanceMeasureSpaceBuilders = distanceMeasureSpaceBuilders;
    }

    @Override public long getTrainTime() {
        return getRunTime() - getCheckpointTime();
    }

    @Override public long getRunTime() {
        return runTimer.elapsedTime();
    }

    @Override public long getTestTime() {
        return testTimer.elapsedTime();
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
    public void buildClassifier(TimeSeriesInstances trainData) throws Exception {
        // timings:
            // train time tracks the time spent processing the algorithm. This should not be used for contracting.
            // run time tracks the entire time spent processing, whether this is work towards the algorithm or otherwise (e.g. saving checkpoints to disk). This should be used for contracting.
            // evaluation time tracks the time spent evaluating the quality of the classifier, i.e. producing an estimate of the train data error.
            // checkpoint time tracks the time spent loading / saving the classifier to disk.
        // record the start time
        final long startTimeStamp = System.nanoTime();
        runTimer.start(startTimeStamp);
        // check the other timers are disabled
        checkpointTimer.checkStopped();
        // several scenarios for entering this method:
            // 1) from scratch: isRebuild() is true
                // 1a) checkpoint found and loaded, resume from wherever left off
                // 1b) checkpoint not found, therefore initialise classifier and build from scratch
            // 2) rebuild off, i.e. buildClassifier has been called before and already handled 1a or 1b. We can safely continue building from current state. This is often the case if a smaller contract has been executed (e.g. 1h), then the contract is extended (e.g. to 2h) and we enter into this method again. There is no need to reinitialise / discard current progress - simply continue on under new constraints.
        if(isRebuild()) {
            // case (1)
            // load from a checkpoint
            // use a separate timer to handle this as the instance-based timer variables get overwritten with the ones from the checkpoint
            StopWatch loadCheckpointTimer = new StopWatch(true);
            boolean checkpointLoaded = loadCheckpoint();
            // finished loading the checkpoint
            loadCheckpointTimer.stop();
            // if there was a checkpoint and it was loaded        
            if(checkpointLoaded) {
                // case (1a)
                // update the run timer with the start time of this session
                runTimer.start(startTimeStamp);
                // just carry on with build as loaded from a checkpoint
                getLog().info("checkpoint loaded");
                // sanity check timer states
                runTimer.checkStarted();
                checkpointTimer.checkStopped();
            } else {
                // case (1b)
                // let super build anything necessary (will handle isRebuild accordingly in super class)
                super.buildClassifier(trainData);
                // if rebuilding
                // then init vars
                // build timer is already started so just clear any time already accrued from previous builds. I.e. keep the time stamp of when the timer was started, but clear any record of accumulated time
                runTimer.resetElapsedTime();
                // clear other timers entirely
                checkpointTimer.stopAndReset();
                // store the train data
                this.trainData = trainData;
                // setup the tree vars
                tree = new BaseTree<>();
                nodeBuildQueue = new LinkedList<>();
                longestTimePerInstanceDuringNodeBuild = 0;
                // setup the root node
                final TreeNode<Split> root = new BaseTreeNode<>(new Split(trainData, ArrayUtilities.sequence(trainData.numInstances())), null);
                // add the root node to the tree
                tree.setRoot(root);
                // add the root node to the build queue
                nodeBuildQueue.add(root);
            }
            // add the time to load the checkpoint onto the checkpoint timer (irrelevant of whether rebuilding or not)
            checkpointTimer.add(loadCheckpointTimer.elapsedTime());
        } // else case (2)
        LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLog(), "train");
        boolean workDone = false;
        final StopWatch trainStageTimer = new StopWatch();
        while(
                // there's remaining nodes to be built
                !nodeBuildQueue.isEmpty()
                &&
                // there is enough time for another split to be built
                insideTrainTimeLimit( runTimer.elapsedTime() +
                                     longestTimePerInstanceDuringNodeBuild *
                                     nodeBuildQueue.peekFirst().getValue().getData().numInstances())
        ) {
            // time how long it takes to build the node
            trainStageTimer.resetAndStart();
            // get the next node to be built
            final TreeNode<Split> node = nodeBuildQueue.removeFirst();
            // partition the data at the node
            Split split = node.getValue();
            // find the best of R partitioning attempts
            split = buildSplit(split);
            node.setValue(split);
            // for each partition of data build a child node
            final List<TreeNode<Split>> children = setupChildNodes(node);
            // add the child nodes to the build queue
            enqueueNodes(children);
            // done building this node
            trainStageTimer.stop();
            workDone = true;
            // calculate the longest time taken to build a node given
            longestTimePerInstanceDuringNodeBuild = findNodeBuildTime(node, trainStageTimer.elapsedTime());
            // checkpoint if necessary
            checkpointTimer.start();
            if(saveCheckpoint()) getLog().info("saved checkpoint");
            checkpointTimer.stop();
            // update the train timer
            LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLog(), "train");
        }
        // save the final checkpoint
        if(workDone) {
            checkpointTimer.start();
            if(forceSaveCheckpoint()) getLog().info("saved final checkpoint");
            checkpointTimer.stop();
        }
        // stop resource monitoring
        runTimer.stop();
        ResultUtils.setInfo(trainResults, this, trainData);
        // checkpoint if work has been done since (i.e. tree has been built further)
    }

    /**
     * setup the child nodes given the parent node
     *
     * @param parent
     * @return
     */
    private List<TreeNode<Split>> setupChildNodes(TreeNode<Split> parent) {
        final List<Partition> partitions = parent.getValue().getPartitions();
        List<TreeNode<Split>> children = new ArrayList<>(partitions.size());
        // for each child
        for(Partition partition : partitions) {
            // setup the node
            final Split split = new Split();
            split.setData(partition.getData(), partition.getDataIndices());
            children.add(new BaseTreeNode<>(split, parent));
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
            final List<Integer> uniqueClassLabelIndices =
                    node.getValue().getData().stream().map(TimeSeriesInstance::getLabelIndex).distinct()
                            .collect(Collectors.toList());
            if(uniqueClassLabelIndices.size() > 1) {
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
        final TimeSeriesInstances data = node.getValue().getData();
        final long timePerInstance = time / data.numInstances();
        return Math.max(longestTimePerInstanceDuringNodeBuild, timePerInstance + 1); // add 1 to account for precision
        // error in div operation
    }

    @Override
    public double[] distributionForInstance(final TimeSeriesInstance instance) throws Exception {
        // enable resource monitors
        testTimer.resetAndStart();
        long longestPredictTime = 0;
        // start at the tree node
        TreeNode<Split> node = tree.getRoot();
        if(node.isEmpty()) {
            // root node has not been built, just return random guess
            return ArrayUtilities.uniformDistribution(getNumClasses());
        }
        int index = -1;
        int i = 0;
        Split split = node.getValue();
        final StopWatch testStageTimer = new StopWatch();
        // traverse the tree downwards from root
        while(
                !node.isLeaf()
                &&
                insideTestTimeLimit(testTimer.elapsedTime() + longestPredictTime)
        ) {
            testStageTimer.resetAndStart();
            // get the split at that node
            split = node.getValue();
            // work out which branch to go to next
            index = split.findPartitionIndexFor(instance);
            // make this the next node to visit
            node = node.get(index);
            testStageTimer.stop();
            longestPredictTime = testStageTimer.elapsedTime();
        }
        // hit a leaf node
        // get the parent of the leaf node to work out distribution
        node = node.getParent();
        split = node.getValue();
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
        Assert.assertTrue(r > 0);
        this.r = r;
    }

    public PartitionScorer getPartitionScorer() {
        return partitionScorer;
    }

    public void setPartitionScorer(final PartitionScorer partitionScorer) {
        this.partitionScorer = partitionScorer;
    }

    @Override public String toString() {
        return "ProximityTree{tree=" + tree + "}";
    }

    private Split buildSplit(Split unbuiltSplit) {
        double bestSplitScore = Double.NEGATIVE_INFINITY;
        Split bestSplit = null;
        // need to find the best of R splits
        for(int i = 0; i < r; i++) {
            // construct a new split
            final Split split = new Split(unbuiltSplit.getData(), unbuiltSplit.getDataIndices());
            split.buildSplit();
            final double score = split.getScore();
            if(score > bestSplitScore) {
                bestSplit = split;
                bestSplitScore = score;
            }
        }
        return Objects.requireNonNull(bestSplit);
    }

    private static class Partition implements Serializable {

        private Partition(final String[] classLabels) {
            dataIndices = new ArrayList<>();
            this.data = new TimeSeriesInstances(classLabels);
            exemplars = new TimeSeriesInstances(classLabels);
            exemplarIndices = new ArrayList<>();
        }

        // data in this partition / indices of that data in the train data
        private final List<Integer> dataIndices;
        private final TimeSeriesInstances data;
        // exemplar instances representing this partition / indices of those exemplars in the train data
        private final TimeSeriesInstances exemplars;
        private final List<Integer> exemplarIndices;
        
        public void addData(TimeSeriesInstance instance, int i) {
            data.add(Objects.requireNonNull(instance));
            dataIndices.add(i);
        }
        
        public void addExemplar(TimeSeriesInstance instance, int i) {
            exemplars.add(Objects.requireNonNull(instance));
            exemplarIndices.add(i);
        }

        /**
         * find the distribution for the given instance and partition index it belongs to
         *
         * @param instance the instance
         * @return the distribution
         */
        public double[] distributionForInstance(final TimeSeriesInstance instance) {
            // this is a simple majority vote over all the exemplars in the exemplars group at the given partition
            // get the corresponding closest exemplars
            final double[] distribution = new double[instance.numClasses()];
            // for each exemplar
            for(TimeSeriesInstance exemplar : exemplars) {
                // vote for the exemplar's class
                int classValue = exemplar.getLabelIndex();
                distribution[classValue]++;
            }
            ArrayUtilities.normalise(distribution);
            return distribution;
        }

        public TimeSeriesInstances getExemplars() {
            return exemplars;
        }

        public List<Integer> getDataIndices() {
            return dataIndices;
        }
        
        public TimeSeriesInstances getData() {
            return data;
        }
        
        public List<Integer> getExemplarIndices() {
            return exemplarIndices;
        }

        @Override public String toString() {
            return getClass().getSimpleName() + "{" +
                           "exemplarIndices=" + exemplarIndices +
                           ", dataIndices=" + dataIndices +
                           '}';
        }
    }

    private class Split implements Serializable {

        public Split() {}

        public Split(TimeSeriesInstances data, List<Integer> dataIndices) {
            setData(data, dataIndices);
        }
        
        // the distance function for comparing instances to exemplars
        private DistanceMeasure distanceMeasure;
        // the score of this split
        private double score = -1;
        // the data at this split (i.e. before being partitioned)
        private TimeSeriesInstances data;
        private List<Integer> dataIndices;
        // the partitions of the data, each containing data for the partition and exemplars representing the partition
        private List<Partition> partitions;

        public double getScore() {
            return score;
        }

        private void setupDistanceFunction() {
            // pick a random space
            ParamSpaceBuilder distanceMeasureSpaceBuilder = RandomUtils.choice(distanceMeasureSpaceBuilders, rand);
            // built that space
            ParamSpace distanceMeasureSpace = distanceMeasureSpaceBuilder.build(data);
            // randomly pick the distance function / parameters from that space
            final ParamSet paramSet = RandomSearch.choice(distanceMeasureSpace, getRandom());
            // there is only one distance function in the ParamSet returned
            distanceMeasure = Objects.requireNonNull((DistanceMeasure) paramSet.getSingle(DistanceMeasure.DISTANCE_MEASURE_FLAG));
        }

        /**
         * pick exemplars from the given dataset
         */
        private void setupExemplarsAndPartitions() {
            // change the view of the data into per class
            final List<List<Integer>> instIndicesByClass = data.indicesByClass();
            if(instIndicesByClass.size() != data.numClasses()) {
                // may happen if any insts exist that are unclassified
                throw new IllegalStateException("incorrect number of subsets whilst dividing data by class");
            }
            // pick exemplars per class
            partitions = new ArrayList<>(instIndicesByClass.size());
            // generate a partition per class
            for(final List<Integer> sameClassInstIndices : instIndicesByClass) {
                // get the indices of all instances with the specified class
                // random pick exemplars from this 
                final List<Integer> exemplarIndices = RandomUtils.choice(sameClassInstIndices, rand, 1);
                // generate the partition with empty data and the chosen exemplar instances
                final Partition partition = new Partition(data.getClassLabels());
                for(Integer exemplarIndexInSplitData : exemplarIndices) {
                    // find the index of the exemplar in the dataIndices (i.e. the exemplar may be the 5th instance 
                    // in the data but the 5th instance may have index 33 in the train data)
                    final TimeSeriesInstance exemplar = data.get(exemplarIndexInSplitData);
                    final Integer exemplarIndexInTrainData = dataIndices.get(exemplarIndexInSplitData);
                    partition.addExemplar(exemplar, exemplarIndexInTrainData);
                }
                partitions.add(partition);
            }
            // sanity checks
            Assert.assertEquals(instIndicesByClass.size(), partitions.size());
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
            distanceMeasure.setInstances(data);
            // go through every instance and find which partition it should go into. This should be the partition
            // with the closest exemplar associate
            for(int i = 0; i < data.numInstances(); i++) {
                final TimeSeriesInstance instance = data.get(i);
                final Partition closestPartition = findPartitionFor(instance);
                // add the instance to the partition
                closestPartition.addData(data.get(i), dataIndices.get(i));
            }
            // find the score of this split attempt, i.e. how good it is
            score = partitionScorer.findScore(data, getPartitionedData());
        }

        public DistanceFunction getDistanceMeasure() {
            return distanceMeasure;
        }

        /**
         * get the partition of the given instance. The returned partition is the set of data the given instance belongs to based on its proximity to the exemplar instances representing the partition.
         *
         * @param instance
         * @return
         */
        public int findPartitionIndexFor(final TimeSeriesInstance instance) {
            // a map to maintain the closest partition indices
            PrunedMultimap<Double, Integer> distanceToPartitionMap = PrunedMultimap.asc();
            // let the map keep all ties and randomly choose at the end
            distanceToPartitionMap.setSoftLimit(1);
            // loop through exemplars
            for(int i = 0; i < partitions.size(); i++) {
                final Partition partition = partitions.get(i);
                for(TimeSeriesInstance exemplar : partition.getExemplars()) {
                    // for each exemplar
                    // check the instance isn't an exemplar
                    if(instance == exemplar) {
                        return i;
                    }
                    // find the distance
                    final double distance = distanceMeasure.distance(instance, exemplar);
                    // add the distance and partition to the map
                    distanceToPartitionMap.put(distance, i);
                }
            }
            // get the smallest distance from the map
            final Double smallestDistance = distanceToPartitionMap.firstKey();
            // find the list of corresponding partitions which the instance could belong to
            final List<Integer> partitionIndices = distanceToPartitionMap.get(smallestDistance);
            // random pick the best partition for the instance
            return RandomUtils.choice(partitionIndices, rand);
        }

        public Partition findPartitionFor(TimeSeriesInstance instance) {
            return partitions.get(findPartitionIndexFor(instance));
        }

        public TimeSeriesInstances getData() {
            return data;
        }

        public void setData(TimeSeriesInstances data, List<Integer> dataIndices) {
            this.dataIndices = dataIndices;
            this.data = Objects.requireNonNull(data);
        }

        public List<Integer> getDataIndices() {
            return dataIndices;
        }

        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder();
            sb.append(getClass().getSimpleName()).append("{");
            sb.append("dataIndices=").append(dataIndices);
            if(partitions != null) {
                sb.append(", score=").append(score);
                sb.append(", partitions=").append(partitions);
            }
            if(distanceMeasure != null) {
                sb.append(", df=").append(distanceMeasure);
            }
            sb.append("}");
                    
            return sb.toString();
        }

        public List<Partition> getPartitions() {
            return partitions;
        }

        public List<TimeSeriesInstances> getPartitionedData() {
            if(partitions == null) {
                return null;
            }
            List<TimeSeriesInstances> partitionDatas = new ArrayList<>(partitions.size());
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
