package tsml.classifiers.distance_based.knn;

import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.TrainEstimateTimeable;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.ed.EDistance;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.collections.iteration.BaseRandomIterator;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import tsml.classifiers.distance_based.utils.collections.lists.UnorderedArrayList;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import utilities.ClassifierTools;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

import static utilities.ArrayUtilities.normalise;
import static utilities.ArrayUtilities.uniformDistribution;

public class K extends BaseClassifier implements ContractedTrain, ContractedTest, TrainEstimateTimeable, Checkpointed {

    public static void main(String[] args) throws Exception {
        for(int i = 1; i < 2; i++) {
            int seed = i;
            K classifier = new K();
            classifier.setEstimateOwnPerformance(true);
            classifier.setSeed(seed);
            classifier.setTrainTimeLimit(30, TimeUnit.SECONDS);
            ClassifierTools
                    .trainTestPrint(classifier, DatasetLoading
                                                        .sampleDataset("/bench/phd/data/all", "SyntheticControl", seed), seed);
        }
    }
    
    public K() {
        super(true);
        setNeighbourLimit(-1);
        setK(1);
        setDistanceMeasure(new EDistance());
    }
    
    private static final long serialVersionUID = 1;
    // the train time limit / contract
    private long trainTimeLimit;
    // the test time limit / contract
    private long testTimeLimit;
    // how long this took to build. THIS INCLUDES THE TRAIN ESTIMATE!
    private final StopWatch trainTimer = new StopWatch();
    // how long the train estimate took
    private final StopWatch trainEstimateTimer = new StopWatch();
    // how long testing took
    private final StopWatch testTimer = new StopWatch();
    // the longest train stage time, e.g. max time to add a neighbour to every neighbourhood
    private long longestComparisonTime;
    // checkpoint config
    private long lastCheckpointTimeStamp = -1;
    private String checkpointPath;
    private String checkpointFileName = Checkpointed.DEFAULT_CHECKPOINT_FILENAME;
    private boolean checkpointLoadingEnabled = true;
    private long checkpointInterval = Checkpointed.DEFAULT_CHECKPOINT_INTERVAL;
    // the train data
    private Instances trainData;
    // distance function for comparing instances
    private DistanceMeasure distanceMeasure;
    // neighbourhoods for each train instances
    private List<Neighbourhood> neighbourhoods;
    // the neighbours not yet added to the neighbourhoods
    private List<Integer> missingNeighbourIndices;
    // the neighbours added to the neighbourhoods
    private List<Integer> neighbourIndices;
    // the number of neighbours to look for
    private int k;
    // whether the results need rebuilding
    private boolean rebuildTrainEstimateResults;
    // limit the number of neighbours used in the loocv
    private int neighbourLimit;
    // the last time the train contract was logged
    private long trainContractLogTimeStamp;

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        // kick off resource monitors
        //        // track the time from start to after loading the checkpoint. Loading checkpoints overwrites the trainTimer, not recording any time taken to load the checkpoint / do things before loading the checkpoint. This timer records that.
        //        final StopWatch loadCheckpointTimer = new StopWatch(true);
        trainTimer.start();
        trainEstimateTimer.checkStopped();
        // IDE may say this var is redundant - it isn't because may be overwritten in load checkpoint
        final StopWatch trainTimerBeforeLoadCheckpoint = trainTimer;
        // load from checkpoint
        // if checkpoint exists then skip initialisation
        if(loadCheckpoint()) {
            getLog().info("loaded from checkpoint");
            // train timer has been replaced with one from checkpoint. Need to add on any time spent between the start of this func and here
            trainTimer.add(trainTimerBeforeLoadCheckpoint.elapsedTime());
        } else {
            // no checkpoint exists
            super.buildClassifier(trainData);
            // if rebuilding (i.e. building from scratch) initialise the classifier
            if(isRebuild()) {
                // reset resources
                trainTimer.resetElapsedTime();
                trainEstimateTimer.resetElapsedTime();
                // zero tree build time so the first tree build will always set the bar
                longestComparisonTime = 0;
                this.trainData = trainData; 
                if(estimateOwnPerformance) {
                    trainContractLogTimeStamp = -1;
                    neighbourIndices = new ArrayList<>(trainData.size());
                    missingNeighbourIndices = new UnorderedArrayList<>(trainData.size());
                    neighbourhoods = new UnorderedArrayList<>(trainData.size());
                    for(int i = 0; i < trainData.size(); i++) {
                        Instance instance = trainData.get(i);
                        // every instance has a neighbourhood which is empty
                        neighbourhoods.add(new Neighbourhood(instance, i));
                        // to begin with all neighbours are missing from the neighbourhoods
                        missingNeighbourIndices.add(i);
                    }
                }
            }
        }
        trainContractLogTimeStamp = LogUtils.logTimeContract(trainTimer.elapsedTime(), trainTimeLimit, getLog(), "train", trainContractLogTimeStamp);
        if(estimateOwnPerformance) {
            rebuildTrainEstimateResults = false;
            while(insideTrainTimeLimit(trainTimer.elapsedTime() + longestComparisonTime * trainData.size()) && !missingNeighbourIndices.isEmpty()) {
                // pick the next neighbour
                final Integer candidateNeighbourIndex = RandomUtils.pick(missingNeighbourIndices, getRandom());
                neighbourIndices.add(candidateNeighbourIndex);
                // compare the chosen neighbour to each neighbourhood
                for(int i = 0; i < neighbourhoods.size(); i++) {
                    if(i != candidateNeighbourIndex) {
                        compareNeighbours(i, candidateNeighbourIndex);
                    }
                }
                rebuildTrainEstimateResults = true;
                saveCheckpoint();
                trainContractLogTimeStamp = LogUtils.logTimeContract(trainTimer.elapsedTime(), trainTimeLimit, getLog(), "train", trainContractLogTimeStamp);
            }
            if(rebuildTrainEstimateResults) {
                for(int i = 0; i < trainData.size(); i++) {
                    final long timeStamp = System.nanoTime();
                    final Neighbourhood neighbourhood = neighbourhoods.get(i);
                    final Instance instance = neighbourhood.getTarget();
                    final double classValue = instance.classValue();
                    final double[] distribution = neighbourhood.distributionForInstance();
                    final int prediction = neighbourhood.classifyInstance(distribution);
                    final long time = (System.nanoTime() - timeStamp) + neighbourhood.getComparisonTime();
                    trainResults.addPrediction(classValue, distribution, prediction, time, "");
                }
            }
        }
        trainTimer.stop();
        trainEstimateTimer.checkStopped();
        ResultUtils.setInfo(trainResults, this, trainData);
        forceSaveCheckpoint();
    }

    private void compareNeighbours(int targetIndex, int candidateIndex) {
        // avoid using the evaluation instance as a neighbour to itself!
        if(targetIndex == candidateIndex) {
            throw new IllegalStateException("should not have itself as a neighbour!");
        }
        // time the comparison
        final StopWatch comparisonTimer = new StopWatch(true);
        // the candidate is a potential nearest candidate to the target instance, i.e. the neighbour to be added
        // get the corresponding neighbourhood for the candidate instance
        final Neighbourhood candidateNeighbourhood = neighbourhoods.get(candidateIndex);
        // get the limit for the candidate
        final double candidateNeighbourhoodLimit = candidateNeighbourhood.getLimit();
        // get the candidate instance, i.e. the neighbour to be added
        final Instance candidate = candidateNeighbourhood.getTarget();
        // get the target and candidate instance
        final Neighbourhood targetNeighbourhood = neighbourhoods.get(targetIndex);
        // the target is the instance to which the nearest candidate is being found
        final Instance target = targetNeighbourhood.getTarget();
        // get the limit for the neighbourhood (no need to compute distance beyond what the further candidate so far is). This must be the max of the limit for both the target's neighbourhood and the candidate's neighbourhood as the distance will be used in both of those neighbourhoods (only applies if dealing with symmetric distance measure)!
        double limit = targetNeighbourhood.getLimit();
        if(distanceMeasure.isSymmetric()) {
            limit = Math.max(limit, candidateNeighbourhoodLimit);
        }
        // compute the similarity of the two instances
        final double distance = distanceMeasure.distance(target, candidate, limit);
        // add the candidate to the target's neighbourhood
        targetNeighbourhood.add(distance, candidate);
        if(distanceMeasure.isSymmetric()) {
            // and vice versa, add the target to the candidate's neighbourhood
            candidateNeighbourhood.add(distance, target);
        }
        final long comparisonTime = comparisonTimer.elapsedTime();
        // add the comparison time to the corresponding neighbourhoods, sharing equally only if symmetric
        if(distanceMeasure.isSymmetric()) {
            targetNeighbourhood.incrementTime(comparisonTime / 2);
            candidateNeighbourhood.incrementTime(comparisonTime - comparisonTime / 2);
        } else {
            targetNeighbourhood.incrementTime(comparisonTime);
        }
        // track the longest time taken to compare two instances
        longestComparisonTime = Math.max(longestComparisonTime, comparisonTime);
    }

    @Override public double[] distributionForInstance(final Instance instance) throws Exception {
        testTimer.resetAndStart();
        // randomly iterate through the train data
        final RandomIterator<Instance> iterator = new BaseRandomIterator<>();
        iterator.setRandom(getRandom());
        iterator.buildIterator(trainData);
        // store a neighbourhood for the test instance
        final Neighbourhood neighbourhood = new Neighbourhood(instance);
        // assume the comparison time will be similar to during training
        long longestTestComparisonTime = longestComparisonTime;
        // while there's remaining neighbours and inside the test contract
        while(iterator.hasNext() && insideTestTimeLimit(testTimer.elapsedTime() + longestTestComparisonTime)) {
            final long timeStamp = System.nanoTime();
            // compute the distance to the neighbour
            final Instance neighbour = iterator.next();
            final double distance = distanceMeasure.distance(instance, neighbour);
            // add the neighbour to the neighbourhood
            neighbourhood.add(distance, neighbour);
            // update longest comparison time found during testing
            final long time = System.nanoTime() - timeStamp;
            longestTestComparisonTime = Math.max(longestTestComparisonTime, time);
        }
        // return the distribution of neighbours
        final double[] distribution = neighbourhood.distributionForInstance();
        testTimer.stop();
        return distribution;
    }

    @Override public long getTrainTimeLimit() {
        return trainTimeLimit;
    }

    @Override public void setTrainTimeLimit(final long trainTimeLimit) {
        this.trainTimeLimit = trainTimeLimit;
    }

    @Override public long getTestTimeLimit() {
        return testTimeLimit;
    }

    @Override public void setTestTimeLimit(final long testTimeLimit) {
        this.testTimeLimit = testTimeLimit;
    }

    @Override public long getLastCheckpointTimeStamp() {
        return lastCheckpointTimeStamp;
    }

    @Override public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
    }

    @Override public String getCheckpointPath() {
        return checkpointPath;
    }

    @Override public boolean setCheckpointPath(final String checkpointPath) {
        this.checkpointPath = checkpointPath;
        return true;
    }

    @Override public String getCheckpointFileName() {
        return checkpointFileName;
    }

    @Override public void setCheckpointFileName(final String checkpointFileName) {
        this.checkpointFileName = checkpointFileName;
    }

    @Override public boolean isCheckpointLoadingEnabled() {
        return checkpointLoadingEnabled;
    }

    @Override public void setCheckpointLoadingEnabled(final boolean checkpointLoadingEnabled) {
        this.checkpointLoadingEnabled = checkpointLoadingEnabled;
    }

    @Override public long getCheckpointInterval() {
        return checkpointInterval;
    }

    @Override public void setCheckpointInterval(final long checkpointInterval) {
        this.checkpointInterval = checkpointInterval;
    }

    @Override public long getTrainTime() {
        return trainTimer.elapsedTime() - getTrainEstimateTime();
    }

    @Override public long getTrainEstimateTime() {
        return trainEstimateTimer.elapsedTime();
    }

    @Override public long getTestTime() {
        return testTimer.elapsedTime();
    }

    public int getNeighbourLimit() {
        return neighbourLimit;
    }

    public void setNeighbourLimit(final int neighbourLimit) {
        this.neighbourLimit = neighbourLimit;
    }
    
    public boolean hasNeighbourLimit() {
        return neighbourLimit >= 0;
    }
    
    public boolean insideNeighbourLimit() {
        return (!hasNeighbourLimit() || neighbourIndices.size() < neighbourLimit);
    }

    /**
     * hold the contents of a neighbourhood search
     */
    private class Neighbourhood {
        private Neighbourhood(final Instance target) {
            this(target, -1);
        }

        private Neighbourhood(final int targetIndex) {
            this(trainData.get(targetIndex), targetIndex);
        }

        private Neighbourhood(final Instance target, final int targetIndex) {
            // set the target
            this.target = Objects.requireNonNull(target);
            this.targetIndex = targetIndex;
            // make a neighbour map. This tracks the k nearest neighbours, keeping >k if ties occur
            neighbourMap = PrunedMultimap.asc();
            neighbourMap.setSoftLimit(k);
            neighbourMap.disableHardLimit();
        }

        // the index of the instance in the train data. Set to -1 if not in the train data, i.e. during testing
        private final int targetIndex;
        // the maximum distance of the neighbourhood so far (i.e. the nearest neighbour, one of the k nearest neighbours, with the furthest distance)
        private double limit = Double.POSITIVE_INFINITY;
        // the map of distances to the nearest neighbours
        private final PrunedMultimap<Double, Instance> neighbourMap;
        // the total comparison time taken so far to examine the neighbourhood
        private long comparisonTime = 0;
        // looking for the nearest neighbours of the target instance
        private final Instance target;
        // the total number of seen neighbours
        private int size = 0;
        // the distribution of nearest neighbours. Adjusted on demand when a prediction is requested and a better neighbour has been found since the last distribution call
        private boolean findDistribution = true;
        private double[] distribution;
        // the last prediction. Only adjust when a better neighbour is found
        private boolean findPrediction = true;
        private int prediction = -1;

        public int getTargetIndex() {
            return targetIndex;
        }

        public double getLimit() {
            return limit;
        }

        public void setLimit(final double limit) {
            Assert.assertTrue(limit >= 0);
            this.limit = limit;
        }

        public PrunedMultimap<Double, Instance> getNeighbourMap() {
            return neighbourMap;
        }

        public Instance getTarget() {
            return target;
        }

        public long getComparisonTime() {
            return comparisonTime;
        }

        public void incrementTime(long time) {
            comparisonTime += time;
        }

        public void add(double distance, Instance instance) {
            if(neighbourMap.put(distance, instance)) {
                distribution = null;
                findDistribution = true;
                prediction = -1;
                findPrediction = true;
            }
            limit = neighbourMap.lastKey();
            size++;
        }

        public int size() {
            return size;
        }

        public List<Integer> getMissingNeighbourIndices() {
            return missingNeighbourIndices;
        }

        public double[] distributionForInstance() {
            if(findDistribution) {
                findDistribution = false;
                // if there are no neighbours then equally predict every class
                if(neighbourMap.isEmpty()) {
                    distribution = uniformDistribution(getNumClasses());
                } else {
                    // build distribution of closest k neighbours
                    distribution = new double[getNumClasses()];
                    for(Map.Entry<Double, Instance> entry : neighbourMap.entries()) {
                        final Instance instance = entry.getValue();
                        final int i = (int) instance.classValue();
                        distribution[i]++;
                    }
                    normalise(distribution);
                }
            }
            return distribution;
        }

        public int classifyInstance(double[] distribution) {
            if(findPrediction) {
                findPrediction = false;
                final int[] indices = Utilities.argMax(distribution);
                // random tie break nearest neighbours for prediction
                prediction = indices[getRandom().nextInt(indices.length)];
            }
            return prediction;
        }

        public int classifyInstance() {
            return classifyInstance(distributionForInstance());
        }

    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
        this.distanceMeasure = Objects.requireNonNull(distanceMeasure);
    }

    public int getK() {
        return k;
    }

    public void setK(final int k) {
        if(k <= 0) throw new IllegalArgumentException("k cannot be l/e to zero: " + k);
        this.k = k;
    }
}
