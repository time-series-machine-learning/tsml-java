package tsml.classifiers.distance_based.knn;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.TrainEstimateTimeable;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.ed.EDistance;
import tsml.classifiers.distance_based.optimised.PrunedMap;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.CheckpointConfig;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Chkpt;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import tsml.classifiers.distance_based.utils.collections.checks.Checks;
import tsml.classifiers.distance_based.utils.collections.lists.UnorderedArrayList;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.ArrayUtilities;
import utilities.ClassifierTools;

public class N extends BaseClassifier implements ParamHandler, Chkpt, ContractedTrain, TrainEstimateTimeable,
                                                         ContractedTest {

    public static void main(String[] args) throws Exception {
        final int seed = 0;
        final N classifier = new N();
        classifier.setSeed(seed);
        classifier.setEstimateOwnPerformance(true);
        classifier.setEarlyPredict(true);

//        final List<Integer> neighboursByClass = Arrays.asList(5, 20, 15, 18, 40, 3, 22);
//        final List<Integer> nearestNeighbourClassDistribution = Arrays.asList(3, 0, 0, 0, 1, 0, 1);
//        final ArrayList<Integer> list = new ArrayList<>(new IndexList(neighboursByClass.size()));
//        Collections.sort(list, ((Comparator<Integer>) (a, b) -> Boolean.compare(nearestNeighbourClassDistribution.get(a) == 0,
//                nearestNeighbourClassDistribution.get(b) == 0))
//                                       .thenComparing(Comparator.comparingInt(neighboursByClass::get).reversed()).reversed());

//        Utilities.sleep(10000);
//        final Instances[] insts = DatasetLoading.sampleGunPoint(seed);
//        final TimeSeriesInstances data = Converter.fromArff(insts[0]);
//        
//        long timeStamp;
//        final int num = 5;
//        final long[] timeSums = new long[num];
//        final int numRepeats = 1000;
//        String output = null;
//        
//        for(int i = 0; i < numRepeats; i++) {
//            for(int j = 3; j < num; j++) {
//                if(j == 3) {
//                    classifier.earlyPredict = false;
//                } else {
//                    classifier.earlyPredict = true;
//                }
////                classifier.setLogLevel(Level.ALL);
//                timeStamp = System.nanoTime();
//                classifier.setCheckpointPath("checkpoints");
//                classifier.getCheckpointConfig().setInterval(1, TimeUnit.MINUTES);
//                classifier.buildClassifier(data);
//                final long time = System.nanoTime() - timeStamp;
//                timeSums[j] += time;
//                if(i == 0) {
//                    System.out.println(classifier.getTrainResults().writeFullResultsToString());
//                    if(output == null) {
//                        output = classifier.getTrainResults().writeFullResultsToString();
//                    } else {
//                        Assert.assertEquals(output, classifier.getTrainResults().writeFullResultsToString());
//                    }
//                }
//            }
//
//        }
//
//        System.out.println(Arrays.toString(Arrays.stream(timeSums).map(i -> i / numRepeats).toArray()));
        
//        classifier.setCheckpointPath("checkpoints");
        classifier.setK(20);
        classifier.setAutoK(true);
        classifier.setCheckpointInterval(1, TimeUnit.MINUTES);
        ClassifierTools.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed), seed);
    }
    
    private DistanceMeasure distanceMeasure;
    private int k;
    private TimeSeriesInstances trainData;
    private List<Search> searches;
    private List<Integer> remainingSearchIndices;
    // will ensure loocv searches use at least this many neighbours. The limit may be exceeded if the distance measure
    // is symmetric and neighbours are produced as a by product from other searches
    private int neighbourhoodSizeLimit;
    private double neighbourhoodSizeLimitProportional;
    private int neighbourhoodSize;
    private boolean earlyPredict;
    private boolean earlyAbandonDistances;
    private boolean autoK;
    private int bestK;
    
    // track the total run time of the build
    private final StopWatch runTimer = new StopWatch();
    // track the time to test an inst
    private final StopWatch testTimer = new StopWatch();
    // track the time to build the train estimate
    private final StopWatch trainEstimateTimer = new StopWatch();
    // the max amount of time taken to add a neighbour
    private long longestAddNeighbourTime;
    
    public static final String EARLY_PREDICT_FLAG = "p";
    public static final String EARLY_ABANDON_DISTANCES_FLAG = "e";
    public static final String NEIGHBOURHOOD_SIZE_LIMIT_FLAG = "n";
    public static final String AUTO_K_FLAG = "a";
    
    private final CheckpointConfig checkpointConfig = new CheckpointConfig();
    private long trainTimeLimit = -1;
    private long testTimeLimit = -1;

    @Override public long getRunTime() {
        return runTimer.elapsedTime();
    }

    @Override public long getTrainTime() {
        return getRunTime() - getCheckpointingTime() - getTrainEstimateTime();
    }

    @Override public long getTrainEstimateTime() {
        return trainEstimateTimer.elapsedTime();
    }

    @Override public CheckpointConfig getCheckpointConfig() {
        return checkpointConfig;
    }

    public N() {
        super(true);
        setK(1);
        setDistanceMeasure(new EDistance());
        setEarlyPredict(false);
        setEarlyAbandonDistances(false);
        setNeighbourhoodSizeLimit(-1);
        setNeighbourhoodSizeLimitProportional(1d);
    }

    @Override public ParamSet getParams() {
        return super.getParams()
                       .add(DistanceMeasure.DISTANCE_MEASURE_FLAG, distanceMeasure)
                       .add(EARLY_ABANDON_DISTANCES_FLAG, earlyAbandonDistances)
                       .add(EARLY_PREDICT_FLAG, earlyPredict)
                       .add(NEIGHBOURHOOD_SIZE_LIMIT_FLAG, neighbourhoodSizeLimit)
                       .add(AUTO_K_FLAG, autoK);
    }

    @Override public void setParams(final ParamSet params) throws Exception {
        super.setParams(params);
        setDistanceMeasure(params.get(DistanceMeasure.DISTANCE_MEASURE_FLAG, distanceMeasure));
        setEarlyAbandonDistances(params.get(EARLY_ABANDON_DISTANCES_FLAG, earlyAbandonDistances));
        setEarlyPredict(params.get(EARLY_PREDICT_FLAG, earlyPredict));
        setAutoK(params.get(AUTO_K_FLAG, autoK));
    }

    @Override public boolean isFullyBuilt() {
        final boolean inside = neighbourhoodSize >= neighbourhoodSizeLimit;
        final boolean inactive = neighbourhoodSizeLimit < 0;
        final boolean insideProp = getNeighbourhoodSizeProportional() >= neighbourhoodSizeLimitProportional;
        return !getEstimateOwnPerformance() || ((inside || inactive) && insideProp);
    }

    public int getNeighbourhoodSize() {
        return neighbourhoodSize;
    }
    
    public double getNeighbourhoodSizeProportional() {
        return (double) neighbourhoodSize / getMaxNeighbourhoodSize();
    }
    
    public boolean insideNeighbourhoodLimit() {
        final boolean inside = neighbourhoodSize < neighbourhoodSizeLimit;
        final boolean inactive = neighbourhoodSizeLimit < 0;
        final boolean insideProp = (double) neighbourhoodSize / getMaxNeighbourhoodSize() < neighbourhoodSizeLimitProportional;
        return (inactive || inside) && insideProp;
    }
    
    private int getMaxNeighbourhoodSize() {
        final int numInstances = trainData.numInstances();
        return numInstances * (numInstances - 1); // -1 because each left out inst cannot be a neighbour for itself
    }
    
    private List<List<Integer>> neighbourIndicesByClass(int instIndex) {

        // break the neighbours into classes
        final List<List<Integer>> neighboursByClass = new ArrayList<>();
        for(int i = 0; i < trainData.numClasses(); i++) {
            neighboursByClass.add(new UnorderedArrayList<>());
        }
        for(int i = 0; i < trainData.numInstances(); i++) {
            // if inst is in the neighbours data then skip it
            if(i != instIndex) {
                final int labelIndex = trainData.get(i).getLabelIndex();
                neighboursByClass.get(labelIndex).add(i);
            }
        }
        
        return neighboursByClass;
    }

    @Override public void buildClassifier(final TimeSeriesInstances data) throws Exception {
        final long timeStamp = System.nanoTime();
        runTimer.start(timeStamp);
        
        if(isRebuild()) {
            // attempt to load from a checkpoint
            if(loadCheckpoint()) {
                // internals of this object have been changed, so the run timer needs restarting
                runTimer.start(timeStamp); // start from same time point though, no time missed while dealing with chkp
            } else {
                // failed to load checkpoint, so initialise classifier from scratch
                super.buildClassifier(data);
                neighbourhoodSize = 0;
                trainData = data;
                longestAddNeighbourTime = 0;
                if(getEstimateOwnPerformance()) {
                    trainEstimateTimer.start();
                    // init the searches for loocv
                    searches = new ArrayList<>();
                    remainingSearchIndices = new UnorderedArrayList<>();
                    // build a search per left out inst
                    for(int i = 0; i < data.numInstances(); i++) {
                        final Search search = new Search(i);
                        searches.add(search);
                        if(search.hasNext()) {
                            remainingSearchIndices.add(i);
                        }
                    }
                    trainEstimateTimer.stop();
                }
            }
        }
        
        // set the last checkpoint time to now (as we've either loaded from a checkpoint, started from scratch or
        // already had some work done, all of which should already be saved in a checkpoint)
        checkpointConfig.setLastCheckpointTimeStamp(System.nanoTime());
        if(getEstimateOwnPerformance()) {
            estimatePerformance();
        }
        
        runTimer.stop();
        
        // we do this after the timers have been stopped, etc, otherwise times are inaccurate
        // this sets the classifier name / dataset name / timings / meta in the results
        ResultUtils.setInfo(trainResults, this, trainData);
    }
    
    private void estimatePerformance() throws Exception {
        trainEstimateTimer.start();
        // if neighbourhood is empty, then set workDone to true to regenerate the (not yet made) train results
        // otherwise, more neighbours must be added before the train results are regenerated
        boolean workDone = neighbourhoodSize == 0;
        // while there are remaining searches
        while(!remainingSearchIndices.isEmpty() && insideNeighbourhoodLimit() 
                      && insideTrainTimeLimit(getRunTime() + longestAddNeighbourTime)) {
            final long timeStamp = System.nanoTime();
            workDone = true;
            
            // note that due to a symmetric distance measure, the remaining neighbours / searches may not line up
            // exactly with the neighbour count. This is because the distance counts for both neighbours (i.e. a-->b 
            // and b-->a). So the neighbour count gets incremented twice, but the unseen neighbours / remaining searches
            // may not be updated until subsequent iterations of this loop - in which case they do a no-op and skip
            // the neighbour as we've already seen it. So just bare in mind that even though there's remaining searches
            // / unseen neighbours apparently remaining, they may have already been seen. The neighbourhoodSize is the
            // ground truth
            
            // randomly iterate over searches
            final int remainingSearchIndex = RandomUtils.choiceIndex(remainingSearchIndices.size(), getRandom());
            final int searchIndex = remainingSearchIndices.get(remainingSearchIndex);
            final Search search = searches.get(searchIndex);
            
            // add a neighbour to the search
            search.next();
            
            // remove the search if it has no more neighbours available
            if(!search.hasNext()) {
                remainingSearchIndices.remove(remainingSearchIndex);
            }
            
            // optionally update the longest time taken to add a neighbour to assist with contracting
            longestAddNeighbourTime = Math.max(longestAddNeighbourTime, System.nanoTime() - timeStamp);
            
            saveCheckpoint();
        }
        
        // when searches have been exhausted
        if(remainingSearchIndices.isEmpty()) {
            // sanity check all neighbours have been seen
            if(getMaxNeighbourhoodSize() != neighbourhoodSize) {
                throw new IllegalStateException("expected neighbourhood to be full: " + neighbourhoodSize + " != " + getMaxNeighbourhoodSize());                
            }
        }
        
        if(workDone) {
            
            if(autoK) {
                // take a backup of the searches as they are currently
                final List<Search> searchesBackup = new ArrayList<>();
                for(Search search : searches) {
                    searchesBackup.add(CopierUtils.deepCopy(search));
                }
                
                // loop through the searches, adjusted the k decrementally. Take best k and best score so far
                generateTrainResults();    
                bestK = k;
                double bestAcc = trainResults.getAcc();
                
                for(int i = k - 1; i > 0; i--) {
                    for(Search search : searches) {
                        search.setK(i);
                    }
                    generateTrainResults();
                    final double acc = trainResults.getAcc();
                    if(acc >= bestAcc) {
                        bestK = i;
                        bestAcc = acc;
                    }
                }
                
                // restore the original searches and trim down to the optimum k
                searches = searchesBackup;
                for(Search search : searches) {
                    search.setK(bestK);
                }
            }
            
        }
        
        // if work done or train results have been cleared
        if(workDone || trainResults.getPredClassVals() == null) {
            generateTrainResults();
            saveCheckpoint(true);
        }
        
        trainEstimateTimer.stop();
    }
    
    private void generateTrainResults() {
        trainResults = new ClassifierResults();
        for(int i = 0; i < trainData.numInstances(); i++) {
            final Search search = searches.get(i);
            final double[] distribution = search.predict();
            final int prediction = CollectionUtils.bestIndex(ArrayUtilities.asList(distribution), getRandom());
            final long time = search.getTime();
            final TimeSeriesInstance instance = trainData.get(i);
            trainResults.addPrediction(instance.getLabelIndex(), distribution, prediction, time, null);
        }
    }
    
    @Override public double[] distributionForInstance(final TimeSeriesInstance testInst) throws Exception {
        testTimer.resetAndStart();
        final Search search = new Search(testInst);
        if(autoK) {
            search.setK(bestK);
        }
        long longestAddNeighbourTime = 0;
        while(search.hasNext() && insideTestTimeLimit(getTestTime() + longestAddNeighbourTime)) {
            final long timeStamp = System.nanoTime();
            search.next();
            longestAddNeighbourTime = Math.max(longestAddNeighbourTime, System.nanoTime() - timeStamp);
        }
        testTimer.stop();
        return search.predict();
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
        this.k = k;
    }

    public int getNeighbourhoodSizeLimit() {
        return neighbourhoodSizeLimit;
    }

    public void setNeighbourhoodSizeLimit(final int neighbourhoodSizeLimit) {
        this.neighbourhoodSizeLimit = neighbourhoodSizeLimit;
    }

    public boolean isEarlyPredict() {
        return earlyPredict;
    }

    public void setEarlyPredict(final boolean earlyPredict) {
        this.earlyPredict = earlyPredict;
    }
    
    public boolean earlyPredictActive() {
        return k == 1 && earlyPredict;
    }

    public boolean isEarlyAbandonDistances() {
        return earlyAbandonDistances;
    }

    public void setEarlyAbandonDistances(final boolean earlyAbandonDistances) {
        this.earlyAbandonDistances = earlyAbandonDistances;
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

    @Override public long getTestTime() {
        return testTimer.elapsedTime();
    }

    public boolean isAutoK() {
        return autoK;
    }

    public void setAutoK(final boolean autoK) {
        this.autoK = autoK;
    }

    public double getNeighbourhoodSizeLimitProportional() {
        return neighbourhoodSizeLimitProportional;
    }

    public void setNeighbourhoodSizeLimitProportional(final double neighbourhoodSizeLimitProportional) {
        this.neighbourhoodSizeLimitProportional = Checks.requireUnitInterval(neighbourhoodSizeLimitProportional);
    }

    // class to search for the nearest neighbour for a given instance
    private class Search implements Iterator<Neighbour>, Serializable {
        public Search(final TimeSeriesInstance target) {
            this(-1, target);
        }
        
        public Search(final int i) {
            this(i, null);
        }
        
        private Search(final int i, final TimeSeriesInstance target) {
            targetIndexInTrainData = i;
            this.target = target;
            if(target == null && (i < 0 || i > trainData.numInstances() - 1)) {
                throw new IllegalStateException("target cannot be null and have invalid index in train data: " + targetIndexInTrainData);
            }
            unseenNeighbourIndicesByClass = neighbourIndicesByClass(targetIndexInTrainData);
            for(int j = 0; j < trainData.numClasses(); j++) {
                if(!unseenNeighbourIndicesByClass.get(j).isEmpty()) {
                    availableClassIndices.add(j);
                }
            }
            nearestNeighbourIndices = new PrunedMap<Double, Integer>(k, true);
        }
        
        private final TimeSeriesInstance target;
        private final int targetIndexInTrainData;
        private double limit = Double.POSITIVE_INFINITY;
        private final BitSet seenNeighbours = new BitSet(trainData.numInstances());
        private final PrunedMap<Double, Integer> nearestNeighbourIndices;
        private boolean updateDistribution = false;
        private final double[] distribution = ArrayUtilities.uniformDistribution(trainData.numClasses());
        private long time = 0;
        private int size = 0;
        private int homogeneousLabelIndex = -1;
        private final List<List<Integer>> unseenNeighbourIndicesByClass;
        private final List<Integer> availableClassIndices = new UnorderedArrayList<>();
        
        public boolean isTargetInTrainData() {
            return targetIndexInTrainData >= 0;
        }
        
        public TimeSeriesInstance getTarget() {
            if(isTargetInTrainData()) {
                return trainData.get(targetIndexInTrainData);
            } else {
                return target;
            }
        }

        public void setK(int k) {
            if(nearestNeighbourIndices.setLimit(k)) {
                updateDistribution = true;
            }
        }

        public int getUnseenCount() {
            int count = trainData.numInstances();
            if(isTargetInTrainData()) {
                count--;
            }
            count -= size();
            return count;
        }
        
        private int getNumUnavailableHomogeneousClasses() {
            return homogeneousLabelIndex >= 0 ? 1 : 0;
        }
        
        @Override public boolean hasNext() {
            // if there are active classes, then there remains unseen neighbours in those corresponding classes
            return availableClassIndices.size() - getNumUnavailableHomogeneousClasses() > 0;
        }
        
        private boolean add(int neighbourIndexInTrainData, double distance) {
            if(seenNeighbours.get(neighbourIndexInTrainData)) {
                throw new IllegalStateException("already examined neighbour " + neighbourIndexInTrainData + " for inst " + targetIndexInTrainData);
            }
            
            final boolean nearest = nearestNeighbourIndices.add(distance, neighbourIndexInTrainData);

            if(nearest) {
                if(earlyAbandonDistances) {
                    // update the limit for early abandoning distances
                    this.limit = nearestNeighbourIndices.lastKey();
                } // else leave limit at pos inf
                updateDistribution = true;
            }
            
//            getLog().info("adding neighbour " + neighbourIndexInTrainData + " to search " + targetIndexInTrainData + ", distance " + distance + (nearest ? ", nearest neighbour" : ""));

            seenNeighbours.set(neighbourIndexInTrainData, true);
            size++; // we've examined another neighbour
            neighbourhoodSize++;
            
            return nearest;
        }

        @Override public Neighbour next() {
            final long timeStamp = System.nanoTime();
            // pick an active class index
            int availableClassIndex = RandomUtils.choiceIndex(availableClassIndices.size() - getNumUnavailableHomogeneousClasses(), getRandom());
            int classIndex = availableClassIndices.get(availableClassIndex);
            if(classIndex == homogeneousLabelIndex) {
                // pick again if hit the homogeneous class label index (i.e. if the nearest neighbours are all from the
                // same class, look at neighbours from other classes to attempt to change the output of the knn
                availableClassIndex++;
                classIndex = availableClassIndices.get(availableClassIndex);
            }
            
            // pick a random neighbour from 
            final List<Integer> unseenNeighbourIndices = unseenNeighbourIndicesByClass.get(classIndex);
            final Integer neighbourIndexInTrainData = RandomUtils.remove(unseenNeighbourIndices, getRandom());
            if(unseenNeighbourIndices.isEmpty()) {
                // no more insts from this class so make class inactive
                availableClassIndices.remove(availableClassIndex);
            }

            // sanity checks
            if(neighbourIndexInTrainData == targetIndexInTrainData) {
                throw new IllegalArgumentException("cannot add itself as neighbour: " + neighbourIndexInTrainData);
            }

            final TimeSeriesInstance neighbour = trainData.get(neighbourIndexInTrainData);
            final int labelIndex = neighbour.getLabelIndex();
            if(labelIndex != classIndex) {
                throw new IllegalStateException("class label mismatch");
            }
                        
            // might have already seen the neighbour (because the distance measure is symmetric and distance was reused 
            // from adding us as a neighbour
            // therefore just skip over this distance computation and adding of neighbours to the nearest neighbours
            double distance = -1; // -1 indicates distance is invalid / cached from another search with symmetry
            boolean nearest = false;
            
            if(!seenNeighbours.get(neighbourIndexInTrainData)) {

                final boolean symmetric = symmetricNeighbours();
                double limit = this.limit;
                Search altSearch = null;
                if(symmetric) {
                    // this search is searching for the nearest neighbour for a train inst
                    // in this case, if the distance measure is symmetric we can add the target inst of this search as a
                    // neighbour to the corresponding neighbour search
                    altSearch = searches.get(neighbourIndexInTrainData);
                    // set the limit to the max of both, as we will reuse the distance in both searches to must adhere to
                    // the furthest distance in both searches respectively
                    limit = Math.max(this.limit, altSearch.getLimit());
                }

                // compute the distance to the neighbour
                distance = distanceMeasure.distance(getTarget(), neighbour, limit);
                nearest = add(neighbourIndexInTrainData, distance);
                
                if(nearest && earlyPredict) {
                    // neighbour is one of k nearest
                    
                    // update the early predict homogeneity of the nearest neighbours
                    if(!(isHomogeneousNearestNeighbours() && labelIndex == homogeneousLabelIndex)) {
                        // recalculate homogeneous-ness
                        boolean first = true;
                        for(Integer neighbourIndex : nearestNeighbourIndices.valuesList()) {
                            final int neighbourLabelIndex = trainData.get(neighbourIndex).getLabelIndex();
                            if(first) {
                                // haven't seen any neighbours at this point, so this first neighbour will indicate the
                                // potential homogeneous class
                                homogeneousLabelIndex = neighbourLabelIndex;
                                first = false;
                            } else if(neighbourLabelIndex != homogeneousLabelIndex) {
                                // looking at the second or later neighbour
                                homogeneousLabelIndex = -1;
                                break;
                            }
                        }
                    }
                    
                }

                if(symmetric) {
                    // then we can add this target inst as a neighbour to the corresponding search for the neighbour
                    altSearch.add(targetIndexInTrainData, distance);
                    // note that we DO NOT remove the corresponding unseen neighbour index in the alt search. This is
                    // because we'd have to do a linear removal on a list
                    // so instead, at some point, the alt search will find the target inst as a neighbour. It will check
                    // the seen neighbours and find that it has already been handled and just skip over it.
                    // likewise, this doesn't update the active classes for the alt search for the same reasons
                }
            }
            
            time += System.nanoTime() - timeStamp;
            
            return new Neighbour(distance, neighbourIndexInTrainData, nearest);
        }
        
        public boolean symmetricNeighbours() {
            // if the target is an inst in the train data AND distance measure is symmetric, we can reuse the distance
            // and provide the target and distance as a precomputed neighbour to the corresponding alternative search
            return targetIndexInTrainData >= 0 && distanceMeasure.isSymmetric();
        }

        public BitSet getSeenNeighbours() {
            return seenNeighbours;
        }

        public int getHomogeneousLabelIndex() {
            return homogeneousLabelIndex;
        }
        
        public boolean isHomogeneousNearestNeighbours() {
            return homogeneousLabelIndex >= 0;
        }
        
        public double[] predict() {
            if(updateDistribution) {
                updateDistribution = false;
                Arrays.fill(distribution, 0d);
                // note that more than k neighbours may be held as the nearest neighbours if there are ties.
                // it makes most sense to keep the ties. The ties should get the kth
                final Double lastKey = nearestNeighbourIndices.lastKey();
                for(Double distance : nearestNeighbourIndices.keySet()) {
                    final List<Integer> instIndices = nearestNeighbourIndices.get(distance);
                    final double weight;
                    if(distance.equals(lastKey)) {
                        // last list contains any tie breaks for the kth nearest neighbour
                        // give any ties equal share for the kth vote
                        weight = 1d / instIndices.size();
                    } else {
                        weight = 1d;
                    }
                    for(Integer i : instIndices) {
                        final TimeSeriesInstance nearestNeighbour = trainData.get(i);
                        distribution[nearestNeighbour.getLabelIndex()] += weight;
                    }
                }
                ArrayUtilities.normalise(distribution, true);
            }
            return distribution;
        }
        
        public int getTargetIndexInTrainData() {
            return targetIndexInTrainData;
        }
        
        public int size() {
            return size;
        }
        
        public double getLimit() {
            return limit;
        }
        
        public long getTime() {
            return time;
        }
        
    }
}

