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
import tsml.classifiers.distance_based.utils.stats.scoring.GiniEntropy;
import tsml.classifiers.distance_based.utils.stats.scoring.SplitScorer;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import utilities.ArrayUtilities;
import utilities.ClassifierTools;

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
//            classifier.setEarlyAbandonDistances(true);
//            classifier.setEarlyExemplarCheck(true);
//            classifier.setPartitionExaminationReordering(true);
//            classifier.setEarlyAbandonSplits(true);
            //            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            ClassifierTools.trainTestPrint(classifier, DatasetLoading.sampleItalyPowerDemand(seed), seed);
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
                proximityTree.setSplitScorer(new GiniEntropy());
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
    private SplitScorer splitScorer;
    // checkpoint config
    private long lastCheckpointTimeStamp = -1;
    private String checkpointPath;
    private String checkpointFileName = Checkpointed.DEFAULT_CHECKPOINT_FILENAME;
    private boolean checkpointLoadingEnabled = true;
    private long checkpointInterval = Checkpointed.DEFAULT_CHECKPOINT_INTERVAL;
    private StopWatch checkpointTimer = new StopWatch();
    // whether to build the tree depth first or breadth first
    private boolean breadthFirst = false;
    // whether to use early abandon in the distance computations
    private boolean earlyAbandonDistances = false;
    // whether to use a quick check for exemplars
    private boolean earlyExemplarCheck = false;
    // enhanced early abandon distance computation via ordering partition examination to hit the most likely closest exemplar sooner
    private boolean partitionExaminationReordering = false;
    // concurrent split building allows splits to be abandoned if they're worse than the best split seen so far, reducing build time
    private boolean earlyAbandonSplits = false;
    // what strategy to use for handling multivariate data
    private MultivariateStrategy multivariateStrategy = MultivariateStrategy.RANDOM_SINGLE_DIMENSION;

    public MultivariateStrategy getMultivariateStrategy() {
        return multivariateStrategy;
    }

    public void setMultivariateStrategy(
            final MultivariateStrategy multivariateStrategy) {
        this.multivariateStrategy = Objects.requireNonNull(multivariateStrategy);
    }

    public boolean isEarlyAbandonSplits() {
        return earlyAbandonSplits;
    }

    public void setEarlyAbandonSplits(final boolean earlyAbandonSplits) {
        this.earlyAbandonSplits = earlyAbandonSplits;
    }

    public enum MultivariateStrategy {
        RANDOM_SINGLE_DIMENSION,
        RANDOM_MULTIPLE_DIMENSION,
        ALL_DIMENSIONS;
    }

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
        final TimeSeriesInstances data = unbuiltSplit.getData();
        final List<Integer> dataIndices = unbuiltSplit.getDataIndices();
        // need to find the best of R splits
        if(!earlyAbandonSplits) {
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
        } else {
            // build splits concurrently, attempting to eliminate poor splits part way through and save time
            // container for the split and an iterator for building said split
            class SplitBuilder {
                Split split;
                Iterator<Integer> iterator;
                double prevBestPotentialScore;
            }
            // build the pool of splits
            final TreeMap<Double, List<SplitBuilder>> splitPoolByBestPotentialScore = new TreeMap<>();
            // build the pool of fully built splits
            final List<Split> eliminatedSplits = new LinkedList<>();
            final List<SplitBuilder> allSplits = new ArrayList<>(r);
            for(int i = 0; i < r; i++) {
                // build a new split
                final SplitBuilder splitBuilder = new SplitBuilder();
                splitBuilder.split = new Split(data, dataIndices);
                // build an iterator to iteratively build the split
                splitBuilder.iterator = splitBuilder.split.getBuildIterator();
                splitBuilder.prevBestPotentialScore = splitBuilder.split.getBestPotentialScore();
                allSplits.add(splitBuilder);
                // add the split to the pool under the best potential scores
                splitPoolByBestPotentialScore.computeIfAbsent(splitBuilder.split.getBestPotentialScore(), x -> new ArrayList<>(1)).add(splitBuilder);
            }
            // while there's remaining splits to be built
            while(!splitPoolByBestPotentialScore.isEmpty()) {
                // select the split with the best score
                final Map.Entry<Double, List<SplitBuilder>> lastEntry = splitPoolByBestPotentialScore.lastEntry();
                final List<SplitBuilder> splits = lastEntry.getValue();
                // random tie break draws
                final SplitBuilder splitBuilder = splits.remove((int) RandomUtils.choiceIndex(splits.size(), getRandom()));
                // if there's no more splits left at the given score then remove the empty list
                if(splits.isEmpty()) {
                    splitPoolByBestPotentialScore.pollLastEntry();
                }
                if(!splitBuilder.iterator.hasNext()) {
                    throw new IllegalStateException("split cannot be built further");
                }
                // increment the build
                splitBuilder.iterator.next();
                // 3 cases: split got better (1), worse (2) or stayed the same (3)
                final Split split = splitBuilder.split;
                // track whether the current split has become so poor it must be eliminated
                boolean eliminate = false;
                if(split.getBestPotentialScore() > splitBuilder.prevBestPotentialScore) {
                    // case (1), got better
                    // split got better, therefore the worst potential score for the split will have raised. Other splits may now have a best potential score less than this and can be eliminated
                    final List<Double> toRemove = new ArrayList<>();
                    for(Map.Entry<Double, List<SplitBuilder>> entry : splitPoolByBestPotentialScore.entrySet()) {
                        final List<SplitBuilder> list = entry.getValue();
                        for(int i = list.size() - 1; i >= 0; i--) {
                            final SplitBuilder other = list.get(i);
                            if(other.split.getBestPotentialScore() <= split.getWorstPotentialScore()) {
                                list.remove(i);
                            }
                        }
                        if(list.isEmpty()) {
                            toRemove.add(entry.getKey());
                        }
                    }
                    for(Double d : toRemove) {
                        splitPoolByBestPotentialScore.remove(d);
                    }
                } else {
                    // case (2) and (3), got worse or stayed same
                    // split got worse, therefore the worst potential score for the split may be below that of other splits, thus this split may be eligible for elimination
                    if(bestSplit != null && bestSplit.getScore() > split.getBestPotentialScore()) {
                        eliminate = true;
                    } else {
                        for(Map.Entry<Double, List<SplitBuilder>> entry : splitPoolByBestPotentialScore.entrySet()) {
                            for(SplitBuilder other : entry.getValue()) {
                                if(other.split.getWorstPotentialScore() > split.getBestPotentialScore()) {
                                    eliminate = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                // by here we know whether the current split should be eliminated and whether it is fully built or not
                // if eliminated
                if(!splitBuilder.iterator.hasNext()) {
                    // split is fully built and hasn't been eliminated through early abandon
                    // thus it is a contender for being the best split (although the improvement will be very minor if early abandon couldn't weed it out!)
                    if(bestSplit == null || split.getScore() > bestSplit.getScore()) {
                        bestSplit = split;
                    }
                } else {
                    // split still has building left to complete later on
                    // if the split has been eliminated
                    if(eliminate) {
                        // don't bother adding the split back to the pool
                        eliminatedSplits.add(split);
                    } else {
                        // otherwise add it back into the pool of splits
                        // add the current split back into the pool
                        splitPoolByBestPotentialScore.computeIfAbsent(split.getBestPotentialScore(), d -> new ArrayList<>(1)).add(splitBuilder);
                    }
                }
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
        if(partitionExaminationReordering) {
            setEarlyAbandonDistances(true);
        }
    }

    private static class Partition implements Serializable {

        private Partition(final String[] classLabels) {
            dataIndices = new ArrayList<>();
            this.data = new TimeSeriesInstances(classLabels);
            exemplars = new TimeSeriesInstances(classLabels);
            strippedExemplars = new TimeSeriesInstances(classLabels);
            exemplarIndices = new ArrayList<>();
        }

        // data in this partition / indices of that data in the train data
        private final List<Integer> dataIndices;
        private final TimeSeriesInstances data;
        // exemplar instances representing this partition / indices of those exemplars in the train data
        private final TimeSeriesInstances exemplars;
        private final List<Integer> exemplarIndices;
        // exemplar instances need to be stripped to the chosen dimensions every distance computation. This houses the stripped down version to avoid recomputation. I.e. given an exemplar with 7 dimensions, if dim 1,2,4,5 have been chosen (from the multivariate strategy) then the stripped exemplars are the same as the given exemplars but without dim 3,6,7
        private final TimeSeriesInstances strippedExemplars;
        
        public void addData(TimeSeriesInstance instance, int i) {
            data.add(Objects.requireNonNull(instance));
            dataIndices.add(i);
        }
        
        public void addExemplar(TimeSeriesInstance instance, int i, TimeSeriesInstance strippedExemplar) {
            exemplars.add(Objects.requireNonNull(instance));
            strippedExemplars.add(Objects.requireNonNull(strippedExemplar));
            exemplarIndices.add(i);
        }

        public TimeSeriesInstances getStrippedExemplars() {
            return strippedExemplars;
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
        // the data at this split (i.e. before being partitioned)
        private TimeSeriesInstances data;
        private List<Integer> dataIndices;
        // the partitions of the data, each containing data for the partition and exemplars representing the partition
        private List<Partition> partitions;
        
        // similarly partitionIndices houses all the partitions to look at when partitioning. This obviously stays consistent (i.e. look at all partitions in order) when not using early abandon
        private List<Integer> partitionIndices = null;
        
        // when using early abandon for splitting the order that partitions are examined may be altered. Therefore we maintain two maps which contain partition index to a count of the number of insts in that partition so far and vice versa for the other map. By maintaining ordering of the maps, we can create a desc order partition list of the most popular partition to the least. This then takes benefit of the assumption that most insts are going to go to the majority partition, thus examining in desc popularity order should make best use of early abandon in the distance calculations.
        // for example, if we had 3 partitions (3 exemplars) and had no yet built the split. We have no idea about the distribution of insts between partitions, but we can be fairly certain there will be a more popular partition as most splits are lop sided (some partitions are bigger than others). Suppose we jump forward in time to half way through our data. part1 has 30% of the data, part2 has 55% and part3 has 15% (of the data seen so far). We can assume for the remaining data, most of it would end up in part2, closely followed by part1 and the rest in part3. Thus when calculating the distances to each exemplar for each partition, we should examine the exemplars / partitions in the order part2, part1, part3. This gives the highest likelihood of getting a low distance for part2 (as majority of data lies here) and therefore provide a lower bound on distance computation against part1 and part3's exemplars.
        // we can take this further and specialise by class, as a good split will split the data into classes. Therefore, if we can say that the majority of insts with class 2 seems to end up in part3, class1 in part1 and class0 in part2, we can order the partitions ahead of time based on where we believe the inst will be partitioned given its class value. 
        // obviously the order of partitions is ever evolving as we go through more and more data. Therefore, the early abandon for split building will be an increasing returns as the split is built.
        private HashMap<Integer, PartitionCounts> classToPartitionCounts = null;
        
        private class PartitionCounts {
            private final TreeMap<Integer, List<Integer>> descCountToPartitionIndex = new TreeMap<>(((Comparator<Integer>) Integer::compare).reversed());
            private final HashMap<Integer, Integer> partitionIndexToCount = new HashMap<>();
            
            public PartitionCounts() {
                // all partitions begin with a count of zero
                for(int i = 0; i < partitions.size(); i++) {
                    partitionIndexToCount.put(i, 0);
                }
                // therefore the desc mapping is just zero -> all partitions (ties broken by random choice)
                descCountToPartitionIndex.put(0, RandomUtils.choiceIndex(partitions.size(), getRandom(), partitions.size()));
            }
            
            public List<Integer> getPartitionIndexByDescCount() {
                final ArrayList<Integer> indices = new ArrayList<>(partitionIndexToCount.size());
                descCountToPartitionIndex.values().forEach(indices::addAll);
                return indices;
            }
            
            public void increment(Integer partitionIndex) {
                final Integer count = partitionIndexToCount.get(partitionIndex);
                final Integer newCount = count + 1; // increment the count to reflect a new inst in the given partition
                // increment the count for the corresponding partition
                partitionIndexToCount.put(partitionIndex, count + 1);
                // get the partition indices associated with the orig count
                List<Integer> partitionIndices = descCountToPartitionIndex.get(count);
                if(partitionIndices == null) {
                    throw new IllegalStateException("expected count to have list associated");
                }
                final boolean removed = partitionIndices.remove(partitionIndex);
                if(!removed) {
                    throw new IllegalStateException("failed to remove partition index where expected");
                }
                // if the list is now empty then remove the mapping
                if(partitionIndices.isEmpty()) {
                    descCountToPartitionIndex.remove(count);
                }
                // add the partition index to the new count
                partitionIndices = descCountToPartitionIndex.computeIfAbsent(newCount, i -> new ArrayList<>(1));
                // if the partition indices are empty
                if(partitionIndices.isEmpty()) {
                    // just add
                    partitionIndices.add(partitionIndex);
                } else {
                    // otherwise add in a random place
                    partitionIndices.add(RandomUtils.choiceIndex(partitions.size(), getRandom()), partitionIndex);
                }
            }

            @Override public String toString() {
                final StringBuilder sb = new StringBuilder();
                for(Integer i : getPartitionIndexByDescCount()) {
                    final Integer count = partitionIndexToCount.get(i);
                    sb.append("p").append(i).append("=").append(count);
                    sb.append(", ");
                }
                return sb.toString();
            }
        }
        
        // exemplars are normally checked ad-hoc during distance computation. Obviously checking which partition and exemplar belongs to is a waste of computation, as the distance will be zero and trump all other exemplars distances for other partitions. Therefore, it is important to check first. Original pf checked for exemplars as it went along, meaning for partition 5 it would compare exemplar 5 to exemplar 1..4 before realising it's an exemplar. Therefore, we can store the exemplar mapping to partition index and do a quick lookup before calculating distances. This is likely to only save a small amount of time, but increases as the breadth of trees / classes increases. I.e. for a 100 class problem, looking through 99 exemplars before realising we're examining the exemplar for the 100th partition is a large waste.
        private Map<Integer, Integer> exemplarPartitionIndices = null;

        // cache the scores
        private boolean findWorstPotentialScore = true;
        private boolean findBestPotentialScore = true;
        private boolean findScore = true;
        // the score of this split
        private double score = -1;
        private double bestPotentialScore = -1;
        private double worstPotentialScore = -1;
        
        // track the stage of building
        private int instIndex = 0;
        
        private void incrementInstIndexToBePartitioned() {
            instIndex++;
        }
        
        // list of dimensions to use when comparing insts
        private List<Integer> dimensionIndices;
        
        public double getBestPotentialScore() {
            if(findBestPotentialScore) {
                findBestPotentialScore = false;
//                bestPotentialScore = 
            }
            return bestPotentialScore;
        }
        
        public double getWorstPotentialScore() {
            if(findWorstPotentialScore) {
                findWorstPotentialScore = false;
//                worstPotentialScore = 
            }
            return worstPotentialScore;
        }

        public double getScore() {
            if(findScore) {
                findScore = false;
//                score = splitScorer.score(new Labels<>(new AbstractList<Integer>() {
//                    @Override public Integer get(final int i) {
//                        return data.get(i).getLabelIndex();
//                    }
//
//                    @Override public int size() {
//                        return data.numInstances();
//                    }
//                }), partitions.stream().map(partition -> new Labels<>(new AbstractList<Integer>() {
//                    @Override public Integer get(final int i) {
//                        return partition.getData().get(i).getLabelIndex();
//                    }
//
//                    @Override public int size() {
//                        return partition.getData().numInstances();
//                    }
//                })).collect(Collectors.toList()));
                score = splitScorer.findScore(Converter.toArff(data), getPartitionedData().stream().map(Converter::toArff).collect(Collectors.toList()));
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
                    return instIndex < data.numInstances();
                }

                @Override public Integer next() {
                    // mark that scores need recalculating, as we'd have added a new inst to a partition by the end of this method
                    findScore = true;
                    findBestPotentialScore = true;
                    findWorstPotentialScore = true;
                    // get the inst to be partitioned
                    TimeSeriesInstance inst = data.get(instIndex);
                    Integer closestPartitionIndex = null;
                    if(earlyExemplarCheck) {
                        // check for exemplars. If the inst is an exemplar, we already know what partition it represents and therefore belongs to
                        closestPartitionIndex = exemplarPartitionIndices.get(instIndex);
                    }
                    // if null then not exemplar / not doing quick exemplar checking
                    TreeSet<Integer> descSizePartitionIndices = null;
                    if(closestPartitionIndex == null) {
                        if(partitionExaminationReordering) {
                            // enhanced early abandon reorders the partitions to attempt to hit the lowest distance first and create largest benefit of early abandon during distance computation
                            // get the ordering of partitions for the inst's class
                            final PartitionCounts partitionCounts = classToPartitionCounts
                                                                            .computeIfAbsent(inst.getLabelIndex(),
                                                                                    index -> new PartitionCounts());
                            closestPartitionIndex = findPartitionIndexFor(inst, partitionCounts.getPartitionIndexByDescCount());
                            // another inst is assigned to that partition, so adjust the sorted set of partitions indices to maintain desc count
                            partitionCounts.increment(closestPartitionIndex);
                        } else {
                            // otherwise just loop through all partitions in order looking for the closest. Order is static and never changed
                            closestPartitionIndex = findPartitionIndexFor(inst, partitionIndices);
                        }
                    }
                    Partition closestPartition = partitions.get(closestPartitionIndex);
                    // add the instance to the partition
                    closestPartition.addData(data.get(instIndex), dataIndices.get(
                            instIndex));
                    // shift i along to look at next inst
                    instIndex++;
                    return closestPartitionIndex;
                }
            };
        }
        
        private void pickDistanceMeasure() {
            // pick the distance function
            // pick a random space
            ParamSpaceBuilder distanceMeasureSpaceBuilder = RandomUtils.choice(distanceMeasureSpaceBuilders, rand);
            // built that space
            ParamSpace distanceMeasureSpace = distanceMeasureSpaceBuilder.build(data);
            // randomly pick the distance function / parameters from that space
            final ParamSet paramSet = RandomSearch.choice(distanceMeasureSpace, getRandom());
            // there is only one distance function in the ParamSet returned
            distanceMeasure = Objects.requireNonNull((DistanceMeasure) paramSet.getSingle(DistanceMeasure.DISTANCE_MEASURE_FLAG));
            // setup the distance function
            distanceMeasure.buildDistanceMeasure(data);
        }
        
        private void pickExemplars() {
            // pick the exemplars
            // change the view of the data into per class
            final List<List<Integer>> instIndicesByClass = data.indicesByClass();
            // pick exemplars per class
            partitions = new ArrayList<>(instIndicesByClass.size());
            // generate a partition per class
            for(final List<Integer> sameClassInstIndices : instIndicesByClass) {
                if(!sameClassInstIndices.isEmpty()) {
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
                        final TimeSeriesInstance strippedExemplar = exemplar.getHSlice(dimensionIndices);
                        partition.addExemplar(exemplar, exemplarIndexInTrainData, strippedExemplar);
                    }
                    partitions.add(partition);
                }
            }
            if(earlyExemplarCheck) {
                exemplarPartitionIndices = new HashMap<>();
                // chuck all exemplars in a map to check against before doing distance computation
                for(int partitionIndex = 0; partitionIndex < partitions.size(); partitionIndex++) {
                    final Partition partition = partitions.get(partitionIndex);
                    for(Integer exemplarIndex : partition.getExemplarIndices()) {
                        exemplarPartitionIndices.put(exemplarIndex, partitionIndex);
                    }
                }
            }
        }
        
        private void setupEarlyAbandon() {
            // init the maps of partition counts / desc order of partitions for each class
            classToPartitionCounts = new HashMap<>();
            if(!partitionExaminationReordering) {
                partitionIndices = ArrayUtilities.sequence(partitions.size());
            }
        }
        
        private void pickDimensions() {
            // handle multivariate data by applying the corresponding strategy
            final int numDimensions = data.getMaxNumDimensions();
            if(multivariateStrategy == MultivariateStrategy.ALL_DIMENSIONS) {
                dimensionIndices = ArrayUtilities.sequence(numDimensions);
            } else if(multivariateStrategy == MultivariateStrategy.RANDOM_SINGLE_DIMENSION) {
                dimensionIndices = Collections.singletonList(RandomUtils.choiceIndex(numDimensions, getRandom()));
            } else if(multivariateStrategy == MultivariateStrategy.RANDOM_MULTIPLE_DIMENSION) {
                // pick at least 1 dimension
                final int numChoices = RandomUtils.choiceIndex(numDimensions, getRandom()) + 1;
                // randomly select that number of dimensions
                dimensionIndices = RandomUtils.choiceIndex(numDimensions, getRandom(), numChoices);
            } else {
                throw new IllegalStateException("cannot handling multivariate strategy " + multivariateStrategy);
            }
        }
        
        private void preBuild() {
            pickDistanceMeasure();
            pickDimensions();
            pickExemplars();
            setupEarlyAbandon();
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

        public int findPartitionIndexFor(final TimeSeriesInstance instance, Iterable<Integer> partitionIndices) {
            return findPartitionIndexFor(instance, partitionIndices.iterator());
        }
        
        /**
         * get the partition of the given instance. The returned partition is the set of data the given instance belongs to based on its proximity to the exemplar instances representing the partition.
         *
         * @param instance
         * @return
         */
        public int findPartitionIndexFor(final TimeSeriesInstance instance, Iterator<Integer> partitionIndicesIterator) {
            // a map to maintain the closest partition indices
            PrunedMultimap<Double, Integer> distanceToPartitionMap = PrunedMultimap.asc();
            // let the map keep all ties and randomly choose at the end
            distanceToPartitionMap.setSoftLimit(1);
            // loop through exemplars / partitions
            while(partitionIndicesIterator.hasNext()) {
                final int i = partitionIndicesIterator.next();
                final Partition partition = partitions.get(i);
                final TimeSeriesInstances exemplars = partition.getExemplars();
                final TimeSeriesInstances strippedExemplars = partition.getStrippedExemplars();
                for(int j = 0; j < exemplars.numInstances(); j++) {
                    // for each exemplar
                    final TimeSeriesInstance exemplar = exemplars.get(j);
                    // check the instance isn't an exemplar
                    if(!earlyExemplarCheck && instance == exemplar) {
                        return i;
                    }
                    // get the stripped version of the inst and exemplar. This optionally trims down the dimensions depending on the strategy in the multivariate case
                    final TimeSeriesInstance strippedExemplar = strippedExemplars.get(j);
                    final TimeSeriesInstance strippedInstance = instance.getHSlice(dimensionIndices);
                    // find the distance
                    final double distance = distanceMeasure.distance(strippedInstance, strippedExemplar);
                    // add the distance and partition to the map
                    distanceToPartitionMap.put(distance, i);
                }
            }
            // get the smallest distance from the map
            final Double smallestDistance = distanceToPartitionMap.firstKey();
            // find the list of corresponding partitions which the instance could belong to
            final List<Integer> bestPartitionIndices = distanceToPartitionMap.get(smallestDistance);
            // random pick the best partition for the instance
            return RandomUtils.choice(bestPartitionIndices, rand);
        }
        
        public int findPartitionIndexFor(final TimeSeriesInstance instance) {
            return findPartitionIndexFor(instance, ArrayUtilities.sequence(partitions.size()));
        }

        public Partition findPartitionFor(TimeSeriesInstance instance) {
            final int index = findPartitionIndexFor(instance);
            return partitions.get(index);
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

        public List<Integer> getDimensionIndices() {
            return dimensionIndices;
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
