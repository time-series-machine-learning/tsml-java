package tsml.classifiers.distance_based.proximity;

import com.google.common.collect.Lists;
import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.IndependentDistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.spaces.*;
import tsml.classifiers.distance_based.distances.ed.spaces.EDistanceSpace;
import tsml.classifiers.distance_based.distances.erp.spaces.ERPDistanceRestrictedContinuousSpace;
import tsml.classifiers.distance_based.distances.lcss.spaces.LCSSDistanceRestrictedContinuousSpace;
import tsml.classifiers.distance_based.distances.msm.spaces.MSMDistanceSpace;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.twed.spaces.TWEDistanceSpace;
import tsml.classifiers.distance_based.distances.wdtw.spaces.WDDTWDistanceContinuousSpace;
import tsml.classifiers.distance_based.distances.wdtw.spaces.WDTWDistanceContinuousSpace;
import tsml.classifiers.distance_based.optimised.PrunedMap;
import tsml.classifiers.distance_based.utils.classifiers.*;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.CheckpointConfig;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Chkpt;
import tsml.classifiers.distance_based.utils.collections.lists.IndexList;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearch;
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
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.CachedTransformer;
import tsml.transformers.Derivative;
import tsml.transformers.Transformer;
import utilities.ArrayUtilities;
import utilities.ClassifierTools;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.stream.Collectors;

import static tsml.classifiers.distance_based.utils.collections.checks.Checks.requireReal;

/**
 * Proximity tree
 * <p>
 * Contributors: goastler
 */
public class ProximityTree extends BaseClassifier implements ContractedTest, ContractedTrain, Chkpt, MemoryWatchable {

    public static void main(String[] args) throws Exception {
//        System.out.println(CONFIGS);
        for(int i = 0; i < 1; i++) {
            int seed = i;
            ProximityTree classifier = CONFIGS.get("PT_R5").build();
            classifier.setSeed(seed);
//            classifier.setCheckpointDirPath("checkpoints");
            classifier.setLogLevel(Level.ALL);
            classifier.setDebug(true);
//            classifier.setDistanceMode(DistanceMode.DEPENDENT);
//            classifier.setDimensionConversion(DimensionConversionMode.NONE);
//            classifier.setDimensionSamplingMode(DimensionSamplingMode.ALL);
//            classifier.setMultivariateMode(DimensionSamplingMode.CONCAT_TO_UNIVARIATE);
//            classifier.setEarlyAbandonDistances(true);
//            classifier.setEarlyExemplarCheck(true);
//            classifier.setPartitionExaminationReordering(true);
            //            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
//            classifier.setCheckpointPath("checkpoints");
            classifier.setCheckpointInterval(10, TimeUnit.SECONDS);
            classifier.setTrainTimeLimit(5, TimeUnit.SECONDS);
            ClassifierTools.trainTestPrint(classifier, DatasetLoading.sampleItalyPowerDemand(seed), seed);
        }
    }
    
    public final static Configs<ProximityTree> CONFIGS = buildConfigs().immutable();
    
    public static Configs<ProximityTree> buildConfigs() {
        final Configs<ProximityTree> configs = new Configs<>();
        configs.add("PT_R1", "Proximity tree with a single split per node", ProximityTree::new,
               pt -> {
                        pt.setDistanceMeasureSpaceBuilders(Lists.newArrayList(
                                new EDistanceSpace(),
                                new DTWDistanceFullWindowSpace(),
                                new DTWDistanceRestrictedContinuousSpace(),
                                new DDTWDistanceFullWindowSpace(),
                                new DDTWDistanceRestrictedContinuousSpace(),
                                new WDTWDistanceContinuousSpace(),
                                new WDDTWDistanceContinuousSpace(),
                                new LCSSDistanceRestrictedContinuousSpace(),
                                new ERPDistanceRestrictedContinuousSpace(),
                                new TWEDistanceSpace(),
                                new MSMDistanceSpace()
                        ));
                        pt.setSplitScorer(new GiniEntropy());
                        pt.setR(1);
                        pt.setTrainTimeLimit(-1);
                        pt.setTestTimeLimit(-1);
                        pt.setBreadthFirst(false);
                        pt.setPartitionExaminationReordering(false);
                        pt.setEarlyExemplarCheck(false);
                        pt.setEarlyAbandonDistances(false);
                        pt.setDimensionConversion(DimensionConversionMode.NONE);
                        pt.setDistanceMode(DistanceMode.DEPENDENT);
                        pt.setDimensionSamplingMode(DimensionSamplingMode.SINGLE);
                        pt.setCacheTransforms(false);
                });
        
        configs.add("PT_R5", "5 random splits per node", "PT_R1", pt -> pt.setR(5));
        
        configs.add("PT_R10", "10 random splits per node", "PT_R1", pt -> pt.setR(10));
        
        for(DimensionSamplingMode samplingMode : DimensionSamplingMode.values()) {
            for(DimensionConversionMode conversionMode : DimensionConversionMode.values()) {
                for(DistanceMode distanceMode : DistanceMode.values()) {
                    String base = "PF_R5";
                    String name = base
                                          + "_" + (samplingMode.equals(DimensionSamplingMode.SINGLE) ? '1' :
                                          samplingMode.name().charAt(0)) 
                                          + "_" + conversionMode.name().charAt(0)
                                          + "_" + distanceMode.name().charAt(0);
                    configs.add(name, "", "PT_R5", pt -> {
                        pt.setDimensionSamplingMode(samplingMode);
                        pt.setDimensionConversion(conversionMode);
                        pt.setDistanceMode(distanceMode);
                    });
                }
            }
        }
        
        return configs;
    }

    public ProximityTree() {
        CONFIGS.get("PT_R1").configure(this);
    }

    private static final long serialVersionUID = 1;
    // train timer
    private final StopWatch runTimer = new StopWatch();
    // test / predict timer
    private final StopWatch testTimer = new StopWatch();
    // method of tracking memory
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    // the tree of splits
    private Tree<Split> tree;
    // the train time limit / contract
    private transient long trainTimeLimit;
    // the test time limit / contract
    private transient long testTimeLimit;
    // the longest time taken to build a node / split
    private long longestTrainStageTime;
    // the queue of nodes left to build
    private Deque<TreeNode<Split>> nodeBuildQueue;
    // the list of distance function space builders to produce distance functions in splits
    private List<ParamSpaceBuilder> distanceMeasureSpaceBuilders;
    // the number of splits to consider for this split
    private int r;
    // a method of scoring the split of data into partitions
    private SplitScorer splitScorer;
    // checkpoint config
    private final CheckpointConfig checkpointConfig = new CheckpointConfig();
    // whether to build the tree depth first or breadth first
    private boolean breadthFirst = false;
    // whether to use early abandon in the distance computations
    private boolean earlyAbandonDistances;
    // whether to use a quick check for exemplars
    private boolean earlyExemplarCheck;
    // enhanced early abandon distance computation via ordering partition examination to hit the most likely closest exemplar sooner
    private boolean partitionExaminationReordering;
    // cache certain transformers to avoid repetition
    private Map<Transformer, CachedTransformer> transformerCache;

    public DistanceMode getDistanceMode() {
        return distanceMode;
    }

    public void setDistanceMode(
            final DistanceMode distanceMode) {
        this.distanceMode = Objects.requireNonNull(distanceMode);
    }

    public DimensionConversionMode getDimensionConversion() {
        return dimensionConversionMode;
    }

    public void setDimensionConversion(
            final DimensionConversionMode dimensionConversionMode) {
        this.dimensionConversionMode = Objects.requireNonNull(dimensionConversionMode);
    }

    // what strategy to use for handling multivariate data
    private DimensionSamplingMode dimensionSamplingMode;
    // multivariate conversion mode to convert multivariate data into an alternate form
    private DimensionConversionMode dimensionConversionMode;
    // multivariate distance can be interpreted as several isolated univariate pairings or delegated to the distance measure to manage
    private DistanceMode distanceMode;
    
    public DimensionSamplingMode getDimensionSamplingMode() {
        return dimensionSamplingMode;
    }

    public void setDimensionSamplingMode(
            final DimensionSamplingMode dimensionSamplingMode) {
        this.dimensionSamplingMode = Objects.requireNonNull(dimensionSamplingMode);
    }

    public CheckpointConfig getCheckpointConfig() {
        return checkpointConfig;
    }

    public boolean isCacheTransforms() {
        return transformerCache != null;
    }

    public void setCacheTransforms(final boolean cacheTransforms) {
        if(cacheTransforms) {
            transformerCache = new HashMap<>();
        } else {
            transformerCache = null;
        }
    }

    /**
     * Set the cache to an external cache
     * @param cache
     */
    public void setCacheTransforms(final Map<Transformer, CachedTransformer> cache) {
        transformerCache = cache;
    }

    public enum DimensionSamplingMode {
        SINGLE, // randomly pick a single dimension, discarding others
        MULTIPLE, // randomly pick a subset of dimensions (between 1 and all dimensions) and discard others
        ALL, // retain all dimensions
        SHUFFLE, // retain all dimensions but shuffle the order, which is helpful when stratifying or concat'ing
        ;
    }
    
    public enum DimensionConversionMode {
        NONE, // don't convert dimensions whatsoever
        CONCAT, // concatenate dimensions into a single, long univariate time series
        STRATIFY, // stratify dimensions into a single, long univariate time series
        RANDOM, // random pick between the above conversions
        ;
    }
    
    public enum DistanceMode {
        DEPENDENT, // let the distance measure consider all dimensions to compute distance
        INDEPENDENT, // independently compute distance on each dimension, then sum for final distance
        RANDOM, // randomly choose independent or dependent
        ;
    }

    @Override public boolean isFullyBuilt() {
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
        this.distanceMeasureSpaceBuilders = Objects.requireNonNull(distanceMeasureSpaceBuilders);
        Assert.assertFalse(distanceMeasureSpaceBuilders.isEmpty());
    }

    @Override public long getTrainTime() {
        return getRunTime() - getCheckpointingTime();
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
        final long timeStamp = System.nanoTime();
        memoryWatcher.start();
        checkpointConfig.setLogger(getLogger());
        // several scenarios for entering this method:
            // 1) from scratch: isRebuild() is true
                // 1a) checkpoint found and loaded, resume from wherever left off
                // 1b) checkpoint not found, therefore initialise classifier and build from scratch
            // 2) rebuild off, i.e. buildClassifier has been called before and already handled 1a or 1b. We can safely continue building from current state. This is often the case if a smaller contract has been executed (e.g. 1h), then the contract is extended (e.g. to 2h) and we enter into this method again. There is no need to reinitialise / discard current progress - simply continue on under new constraints.
        if(isRebuild()) {
            // case (1)
            // load from a checkpoint
            if(loadCheckpoint()) {
                memoryWatcher.start();
                checkpointConfig.setLogger(getLogger());
            } else {
                // case (1b)
                // let super build anything necessary (will handle isRebuild accordingly in super class)
                super.buildClassifier(trainData);
                // if rebuilding
                // then init vars
                // build timer is already started so just clear any time already accrued from previous builds. I.e. keep the time stamp of when the timer was started, but clear any record of accumulated time
                runTimer.reset();
                // setup the tree vars
                tree = new BaseTree<>();
                nodeBuildQueue = new LinkedList<>();
                longestTrainStageTime = 0;
                if(isCacheTransforms()) {
                    // clear out any old cached versions
                    transformerCache = new HashMap<>();
                }
                // setup the root node
                final TreeNode<Split> root = new BaseTreeNode<>(new Split(trainData, new IndexList(trainData.numInstances())), null);
                // add the root node to the tree
                tree.setRoot(root);
                // add the root node to the build queue
                nodeBuildQueue.add(root);
            }  // else case (1a)

        } // else case (2)
        
        // update the run timer with the start time of this session 
        // as the runtimer has been overwritten with the one from the checkpoint (if loaded)
        // or the classifier has been initialised from scratch / resumed and can just start from the timestamp
        runTimer.start(timeStamp);
        
        LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLogger(), "train");
        boolean workDone = false;
        // maintain a timer for how long nodes take to build
        final StopWatch trainStageTimer = new StopWatch();
        while(
                // there's remaining nodes to be built
                !nodeBuildQueue.isEmpty()
                &&
                // there is enough time for another split to be built
                insideTrainTimeLimit( runTimer.elapsedTime() + longestTrainStageTime)
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
            // checkpoint if necessary
            saveCheckpoint();
            // update the train timer
            LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLogger(), "train");
            // calculate the longest time taken to build a node given
            longestTrainStageTime = Math.max(longestTrainStageTime, trainStageTimer.elapsedTime());
        }
        // stop resource monitoring
        memoryWatcher.stop();
        runTimer.stop();
        // save the final checkpoint / info
        if(workDone) {
            ResultUtils.setInfo(trainResults, this, trainData);
            forceSaveCheckpoint();
        }
    }

    public Tree<Split> getTree() {
        return tree;
    }

    @Override public long getMaxMemoryUsage() {
        return memoryWatcher.getMaxMemoryUsage();
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
            split.setData(partition.getData(), partition.getDataIndicesInTrainData());
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
                    node.getValue().getInsts().stream().map(TimeSeriesInstance::getLabelIndex).distinct()
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

    public SplitScorer getSplitScorer() {
        return splitScorer;
    }

    public void setSplitScorer(final SplitScorer splitScorer) {
        this.splitScorer = splitScorer;
    }

    @Override public String toString() {
        return "ProximityTree{tree=" + tree + "}";
    }

    private Split buildSplit(Split unbuiltSplit) {
        Split bestSplit = null;
        final TimeSeriesInstances data = unbuiltSplit.getInsts();
        final List<Integer> dataIndices = unbuiltSplit.getInstIndicesInTrainData();
        // need to find the best of R splits
        // linearly go through r splits and select the best
        for(int i = 0; i < r; i++) {
            // construct a new split
            final Split split = new Split(data, dataIndices);
            split.buildSplit();
            final double score = split.getScore();
            if(bestSplit == null || score > bestSplit.getScore()) {
                bestSplit = split;
            }
        }
        return Objects.requireNonNull(bestSplit);
    }

    public boolean isEarlyExemplarCheck() {
        return earlyExemplarCheck;
    }

    public void setEarlyExemplarCheck(final boolean earlyExemplarCheck) {
        this.earlyExemplarCheck = earlyExemplarCheck;
    }

    public boolean isEarlyAbandonDistances() {
        return earlyAbandonDistances;
    }

    public void setEarlyAbandonDistances(final boolean earlyAbandonDistances) {
        this.earlyAbandonDistances = earlyAbandonDistances;
    }

    public boolean isPartitionExaminationReordering() {
        return partitionExaminationReordering;
    }

    public void setPartitionExaminationReordering(final boolean partitionExaminationReordering) {
        this.partitionExaminationReordering = partitionExaminationReordering;
    }

    private static class Partition implements Serializable {

        private Partition(final String[] classLabels) {
            dataIndicesInTrainData = new ArrayList<>();
            this.data = new TimeSeriesInstances(classLabels);
            exemplars = new TimeSeriesInstances(classLabels);
            transformedExemplars = new TimeSeriesInstances(classLabels);
            exemplarIndicesInTrainData = new ArrayList<>();
        }

        // data in this partition / indices of that data in the train data
        private final List<Integer> dataIndicesInTrainData;
        private final TimeSeriesInstances data;
        // exemplar instances representing this partition / indices of those exemplars in the train data
        private final TimeSeriesInstances exemplars;
        private final List<Integer> exemplarIndicesInTrainData;
        // exemplar instances need to be stripped to the chosen dimensions every distance computation. This houses the stripped down version to avoid recomputation. I.e. given an exemplar with 7 dimensions, if dim 1,2,4,5 have been chosen (from the multivariate strategy) then the stripped exemplars are the same as the given exemplars but without dim 3,6,7
        private final TimeSeriesInstances transformedExemplars;
        
        public void addData(TimeSeriesInstance instance, int i) {
            data.add(Objects.requireNonNull(instance));
            dataIndicesInTrainData.add(i);
        }
        
        public void addExemplar(TimeSeriesInstance instance, int i, TimeSeriesInstance strippedExemplar) {
            exemplars.add(Objects.requireNonNull(instance));
            transformedExemplars.add(Objects.requireNonNull(strippedExemplar));
            exemplarIndicesInTrainData.add(i);
        }

        public TimeSeriesInstances getTransformedExemplars() {
            return transformedExemplars;
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

        public List<Integer> getDataIndicesInTrainData() {
            return dataIndicesInTrainData;
        }
        
        public TimeSeriesInstances getData() {
            return data;
        }
        
        public List<Integer> getExemplarIndicesInTrainData() {
            return exemplarIndicesInTrainData;
        }

        @Override public String toString() {
            return getClass().getSimpleName() + "{" +
                           "exemplarIndices=" + exemplarIndicesInTrainData +
                           ", dataIndices=" + dataIndicesInTrainData +
                           '}';
        }
    }
    

    private class Split implements Serializable {

        public Split() {}

        public Split(TimeSeriesInstances insts, List<Integer> instIndicesInTrainData) {
            setData(insts, instIndicesInTrainData);
        }
        
        // the distance function for comparing instances to exemplars
        private DistanceMeasure distanceMeasure;
        // the data at this split (i.e. before being partitioned)
        private TimeSeriesInstances insts;
        private List<Integer> instIndicesInTrainData;
        // the partitions of the data, each containing data for the partition and exemplars representing the partition
        private List<Partition> partitions;
        
        // partitionIndices houses all the partitions to look at when partitioning. This obviously stays consistent (i.e. look at all partitions in order) when not using early abandon
        private List<Integer> partitionIndices = null;
        
        // maintain a list of desc partition sizes per class. This ensures (when enabled) partitions are examined in
        // most likely first order
        private List<List<Integer>> partitionOrderByClass;
        
        // exemplars are normally checked ad-hoc during distance computation. Obviously checking which partition and exemplar belongs to is a waste of computation, as the distance will be zero and trump all other exemplars distances for other partitions. Therefore, it is important to check first. Original pf checked for exemplars as it went along, meaning for partition 5 it would compare exemplar 5 to exemplar 1..4 before realising it's an exemplar. Therefore, we can store the exemplar mapping to partition index and do a quick lookup before calculating distances. This is likely to only save a small amount of time, but increases as the breadth of trees / classes increases. I.e. for a 100 class problem, looking through 99 exemplars before realising we're examining the exemplar for the 100th partition is a large waste.
        private Map<Integer, Integer> exemplarIndexInTrainDataToPartitionIndex = null;

        // cache the scores
        private boolean findScore = true;
        // the score of this split
        private double score = -1;
        
        // track the stage of building
        private int instIndexInSplitData = -1;

        // list of dimensions to use when comparing insts
        private List<Integer> dimensionIndices;
        // settings for handling multivariate distance measurements
        private DistanceMode distanceMode;
        private DimensionConversionMode dimensionConversionMode;

        public List<Integer> getPartitionIndices() {
            return partitionIndices;
        }

        public List<List<Integer>> getPartitionOrderByClass() {
            return partitionOrderByClass;
        }

        private Labels<Integer> getParentLabels() {
            return new Labels<>(new AbstractList<Integer>() {
                @Override public Integer get(final int i) {
                    return insts.get(i).getLabelIndex();
                }

                @Override public int size() {
                    return insts.numInstances();
                }
            }); // todo weights
        }

        public double getScore() {
            if(findScore) {
                findScore = false;
                score = splitScorer.score(getParentLabels(), partitions.stream().map(partition -> new Labels<>(new AbstractList<Integer>() {
                    @Override public Integer get(final int i) {
                        return partition.getData().get(i).getLabelIndex();
                    }

                    @Override public int size() {
                        return partition.getData().numInstances();
                    }
                })).collect(Collectors.toList()));
                requireReal(score);
            }
            return score;
        }

        public Iterator<Integer> getBuildIterator() {
            // init build info
            preBuild();
            // return iterator to progressively build split
            // go through every instance and find which partition it should go into. This should be the partition
            // with the closest exemplar associate
            return new Iterator<Integer>() {
                
                @Override public boolean hasNext() {
                    return instIndexInSplitData + 1 < insts.numInstances();
                }

                @Override public Integer next() {
                    // shift i along to look at next inst
                    instIndexInSplitData++;
                    // mark that scores need recalculating, as we'd have added a new inst to a partition by the end of this method
                    findScore = true;
                    // get the inst to be partitioned
                    final TimeSeriesInstance inst = insts.get(instIndexInSplitData);
                    final Integer instIndexInTrainData = instIndicesInTrainData.get(instIndexInSplitData);
                    Integer closestPartitionIndex = null;
                    if(earlyExemplarCheck) {
                        // check for exemplars. If the inst is an exemplar, we already know what partition it represents and therefore belongs to
                        closestPartitionIndex = exemplarIndexInTrainDataToPartitionIndex.get(instIndexInTrainData);
                    }
                    
                    // if null then not exemplar / not doing quick exemplar checking
                    List<Integer> partitionIndicesOrder = null;
                    int closestPartitionIndexIndex = -1;
                    if(closestPartitionIndex == null) {
                        if(partitionExaminationReordering) {
                            // use the desc order of partition size for the given class
                            partitionIndicesOrder = partitionOrderByClass.get(inst.getLabelIndex());
                        } else {
                            // otherwise just loop through all partitions in order looking for the closest. Order is static and never changed
                            partitionIndicesOrder = partitionIndices;
                        }
                        closestPartitionIndexIndex = findPartitionIndexIndexFor(inst, instIndexInTrainData, partitionIndicesOrder);
                        closestPartitionIndex = partitionIndicesOrder.get(closestPartitionIndexIndex);
                    }
                    
                    final Partition closestPartition = partitions.get(closestPartitionIndex);
                    // add the instance to the partition
                    closestPartition.addData(insts.get(instIndexInSplitData), instIndexInTrainData);
                    
                    // if using partition reordering and order has been set
                    if(partitionExaminationReordering && partitionIndicesOrder != null) {
                        // we know the partition which the inst will be allocated to
                        // need to update the partition order to maintain desc size
                        partitionIndicesOrder.set(closestPartitionIndexIndex, closestPartition.getData().numInstances());
                        
                        // continue shifting up the current partition until it is in the correct ascending order
                        // e.g. index: [2,4,0,3,1]
                        //      sizes: [3,2,2,2,1]
                        //      would become (after incrementing size of partition 3, the closest partition, say):
                        //      index: [2,4,0,3,1]
                        //      sizes: [3,2,2,3,1]
                        //      shift the partition 3 upwards until desc order restored:
                        //      index: [2,3,4,0,1]
                        //      sizes: [3,3,2,2,1]
                        int i = closestPartitionIndexIndex - 1;
                        while(i >= 1 && partitionIndicesOrder.get(i - 1) > partitionIndicesOrder.get(i)) {
                            Collections.swap(partitionIndicesOrder, i - 1, i);
                        }
                        
                    }
                    
                    // quick check that partitions line up with num insts
                    if(isDebug() && !hasNext()) {
                        final HashSet<Integer> set = new HashSet<>();
                        partitions.forEach(partition -> set.addAll(partition.getDataIndicesInTrainData()));
                        if(!new HashSet<>(instIndicesInTrainData).containsAll(set)) {
                            throw new IllegalStateException("data indices mismatch against partitions: " + set + " should contain the same as " +
                                                                    instIndicesInTrainData);
                        }
                    }
                    
                    return closestPartitionIndex;
                }
            };
        }

        /**
         * Get the cached version of a transformer. The cached version can persist transforms to avoid repetition.
         * @param transformer
         * @return
         */
        private Transformer getCachedTransformer(Transformer transformer) {
            if(transformerCache != null) {
                // get from internal source
                return transformerCache.computeIfAbsent(transformer, x -> new CachedTransformer(transformer));
            } else {
                return transformer;
            }
        }
        
        private void setupDistanceMeasure() {
            
            // pick the distance function
            // pick a random space
            ParamSpaceBuilder distanceMeasureSpaceBuilder = RandomUtils.choice(distanceMeasureSpaceBuilders, getRandom());
            // if using certain dimensions for multivariate, must strip the dimensions not being examined before building the space
            final TimeSeriesInstances dataForSpace = transformInsts(insts);
            // built that space
            ParamSpace distanceMeasureSpace = distanceMeasureSpaceBuilder.build(dataForSpace);
            // randomly pick the distance function / parameters from that space
            final ParamSet paramSet = RandomSearch.choice(distanceMeasureSpace, getRandom());
            // there is only one distance function in the ParamSet returned
            distanceMeasure = Objects.requireNonNull((DistanceMeasure) paramSet.get(DistanceMeasure.DISTANCE_MEASURE_FLAG));
            
            // if we can cache the transforms
            if(isCacheTransforms()) {
                // check whether the distance measure involves a transform
                if(distanceMeasure instanceof TransformDistanceMeasure) {
                    Transformer transformer = ((TransformDistanceMeasure) distanceMeasure).getTransformer();
                    // check if transformer is of a type which can be cached
                    if(transformer instanceof Derivative) {
                        // cache all der transforms as they're simple, pure functions
                        transformer = getCachedTransformer(transformer);
                    }
                    // update the transformer with the cached version
                    ((TransformDistanceMeasure) distanceMeasure).setTransformer(transformer);
                }
            }
            
            // setup the distance function
            distanceMeasure.buildDistanceMeasure(dataForSpace);

            // if data is mv then apply the distance mode
            if(insts.isMultivariate()) {
                // apply the distance mode
                distanceMode = ProximityTree.this.distanceMode;
                // if randomly picking distance mode
                if(distanceMode.equals(DistanceMode.RANDOM)) {
                    // then random pick from the remaining modes
                    final Integer index = RandomUtils
                                                  .choiceIndexExcept(DistanceMode.values().length, getRandom(),
                                                          DistanceMode.RANDOM.ordinal());
                    distanceMode = DistanceMode.values()[index];
                }
                // if in independent mode
                if(distanceMode.equals(DistanceMode.INDEPENDENT)) {
                    // then wrap the distance measure to evaluate each dimension in isolation
                    distanceMeasure = new IndependentDistanceMeasure(Split.this.distanceMeasure);
                }
            }

        }
        
        private void setupExemplars() {
            // pick the exemplars
            // change the view of the data into per class
            final List<List<Integer>> instIndicesByClass = insts.getInstIndicesByClass();
            // pick exemplars per class
            partitions = new ArrayList<>();
            // generate a partition per class
            for(final List<Integer> sameClassInstIndices : instIndicesByClass) {
                // avoid empty classes, no need to create partition / exemplars from them
                if(!sameClassInstIndices.isEmpty()) {
                    // get the indices of all instances with the specified class
                    // random pick exemplars from this 
                    final List<Integer> exemplarIndices = RandomUtils.choice(sameClassInstIndices, rand, 1);
                    // generate the partition with empty data and the chosen exemplar instances
                    final Partition partition = new Partition(insts.getClassLabels());
                    for(Integer exemplarIndexInSplitData : exemplarIndices) {
                        // find the index of the exemplar in the dataIndices (i.e. the exemplar may be the 5th instance 
                        // in the data but the 5th instance may have index 33 in the train data)
                        final TimeSeriesInstance exemplar = insts.get(exemplarIndexInSplitData);
                        final Integer exemplarIndexInTrainData = instIndicesInTrainData.get(exemplarIndexInSplitData);
                        final TimeSeriesInstance transformedExemplar = transformInst(exemplar);
                        partition.addExemplar(exemplar, exemplarIndexInTrainData, transformedExemplar);
                    }
                    // add the partition to list of partitions
                    partitions.add(partition);
                }
            }
        }
        
        private void setupMisc() {
            // the list of partition indices to browse through when allocating an inst to a partition
           partitionIndices = new IndexList(partitions.size());
            if(partitionExaminationReordering) {
                // init the desc order of partitions for each class
                
                // for each class, make a list holding the partition indices in desc order of size
                // this order will be maintained as insts are allocated to partitions, hence maintaining a list of
                // the most likely partition to end up in given an inst is of a certain class
                partitionOrderByClass = new ArrayList<>(insts.numClasses());
                for(int i = 0; i < insts.numClasses(); i++) {
                    partitionOrderByClass.add(new ArrayList<>(partitionIndices));
                }
            }


            if(earlyExemplarCheck) {
                exemplarIndexInTrainDataToPartitionIndex = new HashMap<>();
                // chuck all exemplars in a map to check against before doing distance computation
                for(int partitionIndex = 0; partitionIndex < partitions.size(); partitionIndex++) {
                    final Partition partition = partitions.get(partitionIndex);
                    for(Integer exemplarIndexInTrainData : partition.getExemplarIndicesInTrainData()) {
                        exemplarIndexInTrainDataToPartitionIndex.put(exemplarIndexInTrainData, partitionIndex);
                    }
                }
            }
        }
        
        private void setupDimensions() {
            
            // if data is mv
            if(insts.isMultivariate()) {
                // then need to handle sampling mode for dimensions
                if(dimensionSamplingMode.equals(DimensionSamplingMode.ALL)) {
                    // use all dims
                    // set the dimIndices to null to indicate no slicing of dimensions needs to be done
                    dimensionIndices = null;
                } else if(dimensionSamplingMode.equals(DimensionSamplingMode.SINGLE)) {
                    // pick a single random dim
                    dimensionIndices = Collections.singletonList(RandomUtils.choiceIndex(insts.getMaxNumDimensions(), getRandom()));
                } else if(dimensionSamplingMode.equals(DimensionSamplingMode.MULTIPLE)) {
                    // pick at least 1 dimension
                    final int numChoices = RandomUtils.choiceIndex(insts.getMaxNumDimensions(), getRandom()) + 1;
                    // randomly select that number of dimensions
                    dimensionIndices = RandomUtils.choiceIndex(insts.getMaxNumDimensions(), getRandom(), numChoices);
                } else if(dimensionSamplingMode.equals(DimensionSamplingMode.SHUFFLE)) {
                    dimensionIndices = RandomUtils.shuffleIndices(insts.getMaxNumDimensions(), getRandom());
                } else {
                    throw new UnsupportedOperationException(dimensionSamplingMode + " unsupported");
                }

                dimensionConversionMode = ProximityTree.this.dimensionConversionMode;
                 if(dimensionConversionMode.equals(DimensionConversionMode.RANDOM)) {
                    // randomly choose an alternative mode than random
                    final Integer index = RandomUtils
                                                  .choiceIndexExcept(DimensionConversionMode.values().length, getRandom(), Arrays.asList(DimensionConversionMode.RANDOM.ordinal(), DimensionConversionMode.NONE.ordinal()));
                    dimensionConversionMode = DimensionConversionMode.values()[index];
                }
            }
        }
        
        private void preBuild() {
            setupDimensions();
            setupDistanceMeasure();
            setupExemplars();
            setupMisc();
        }
        
        /**
         * Partition the data and derive score for this split.
         */
        public void buildSplit() {
            Iterator<Integer> it = getBuildIterator();
            while(it.hasNext()) {
                it.next();
            }
        }

        public DistanceMeasure getDistanceMeasure() {
            return distanceMeasure;
        }
        
        public int findPartitionIndexFor(final TimeSeriesInstance inst, int instIndexInTrainData, List<Integer> partitionIndices) {
            final int partitionIndexIndex = findPartitionIndexIndexFor(inst, instIndexInTrainData, partitionIndices);
            return partitionIndices.get(partitionIndexIndex);
        }

        /**
         * get the partition of the given instance. The returned partition is the set of data the given instance belongs to based on its proximity to the exemplar instances representing the partition.
         *
         * @param inst
         * @param instIndexInTrainData the index of the inst in the data at this node. If the inst is not in the data at this node then set this to -1
         * @return
         */
        public int findPartitionIndexIndexFor(final TimeSeriesInstance inst, int instIndexInTrainData, List<Integer> partitionIndicesIterator) {
            // a map to maintain the closest partition indices
            final PrunedMap<Double, Integer> filter = PrunedMap.asc(1);
            // maintain a limit on distance computation
            double limit = Double.POSITIVE_INFINITY;
            // loop through exemplars / partitions
            for(int i = 0; i < partitionIndicesIterator.size(); i++) {
                final Integer partitionIndex = partitionIndicesIterator.get(i);
                final Partition partition = partitions.get(partitionIndex);
                final TimeSeriesInstances exemplars = partition.getExemplars();
                final List<Integer> exemplarIndicesInTrainData = partition.getExemplarIndicesInTrainData();
                final TimeSeriesInstances transformedExemplars = partition.getTransformedExemplars();
                for(int j = 0; j < exemplars.numInstances(); j++) {
                    // for each exemplar
                    final TimeSeriesInstance exemplar = exemplars.get(j);
                    final Integer exemplarIndexInTrainData = exemplarIndicesInTrainData.get(j);
                    // check the instance isn't an exemplar
                    if(!earlyExemplarCheck && instIndexInTrainData == exemplarIndexInTrainData) {
                        return i;
                    }
                    // get the stripped version of the inst and exemplar. This optionally trims down the dimensions 
                    // depending on the strategy in the multivariate case
                    final TimeSeriesInstance transformedExemplar = transformedExemplars.get(j);
                    final TimeSeriesInstance transformedInstance = transformInst(inst);
                    // find the distance
                    final double distance = distanceMeasure.distance(transformedInstance, transformedExemplar, limit);
                    // add the distance and partition to the map
                    if(filter.add(distance, i)) {
                        // new min dist
                        // set new limit if early abandon enabled
                        if(earlyAbandonDistances) {
                            limit = distance;
                        }
                    }
                }
            }
            
            // random pick the best partition for the instance
            return RandomUtils.choice(filter.valuesList(), rand);
        }

        /**
         * Find the partition index of an unseen instance (i.e. a test inst)
         * @param inst
         * @return
         */
        public int findPartitionIndexFor(final TimeSeriesInstance inst) {
            return findPartitionIndexFor(inst, -1, partitionIndices);
        }

        public TimeSeriesInstances getInsts() {
            return insts;
        }

        public void setData(TimeSeriesInstances data, List<Integer> dataIndices) {
            this.instIndicesInTrainData = Objects.requireNonNull(dataIndices);
            this.insts = Objects.requireNonNull(data);
            Assert.assertEquals(data.numInstances(), dataIndices.size());
        }

        public List<Integer> getInstIndicesInTrainData() {
            return instIndicesInTrainData;
        }
        
        private TimeSeriesInstance transformInst(TimeSeriesInstance inst) {
            
            // only need to transform mv data
            if(!inst.isMultivariate()) {
                return inst;
            }
            
            // sample the dimensions as required
            if(dimensionIndices != null) {
                inst = inst.getHSlice(dimensionIndices);
            }
            
            // convert the dimensions into a different form
            if(dimensionConversionMode.equals(DimensionConversionMode.CONCAT)) {
                
                final LinkedList<Double> values = new LinkedList<>();
                for(TimeSeries series : inst) {
                    for(Double value : series) {
                        values.add(value);
                    }
                }
                inst =  new TimeSeriesInstance(Collections.singletonList(values), inst.getLabelIndex(), inst.getClassLabels());
                
            } else if(dimensionConversionMode.equals(DimensionConversionMode.STRATIFY)) {
                
                final LinkedList<Double> values = new LinkedList<>();
                for(int j = 0; j < inst.getMaxLength(); j++) {
                    for(int i = 0; i < inst.getNumDimensions(); i++) {
                        values.add(inst.get(i).get(j));
                    }
                }
                inst = new TimeSeriesInstance(Collections.singletonList(values), inst.getLabelIndex(), inst.getClassLabels());
                
            } else if(dimensionConversionMode.equals(DimensionConversionMode.NONE)) {
                // do nothing
            } else {
                throw new UnsupportedOperationException("unknown conversion method " + dimensionConversionMode);
            }
            
            return inst;
        }
        
        private TimeSeriesInstances transformInsts(TimeSeriesInstances data) {
            if(!data.isMultivariate()) {
                return data;
            }
            return new TimeSeriesInstances(data.stream().map(this::transformInst).collect(Collectors.toList()));
        }
        
        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder();
            sb.append(getClass().getSimpleName()).append("{");
            sb.append("dataIndices=").append(instIndicesInTrainData);
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

        public List<Integer> getDimensionIndices() {
            return dimensionIndices;
        }

    }

}
