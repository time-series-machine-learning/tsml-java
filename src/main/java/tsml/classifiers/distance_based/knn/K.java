package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.ed.EuclideanDistance;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.Configurer;
import tsml.classifiers.distance_based.utils.classifiers.Utils;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap.DiscardType;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.memory.WatchedMemory;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.system.timing.TimedTest;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrain;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrainEstimate;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.Map.Entry;

public class K extends BaseClassifier implements TimedTest, TimedTrain, TimedTrainEstimate, ContractedTest,
                                                 ContractedTrain, WatchedMemory {

    public static void main(String[] args) throws Exception {
        for(int i = 0; i < 1; i++) {
            int seed = i;
            K classifier = new K();
            Config.DEFAULT.applyConfigTo(classifier);
            classifier.setEstimateOwnPerformance(true);
            classifier.setSeed(seed);
            //            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            Utils.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed));
        }
    }

    public enum Config implements Configurer<K> {
        DEFAULT() {
            @Override
            public <B extends K> B applyConfigTo(final B k) {
                k.setTestTimeLimit(0);
                k.setTrainTimeLimit(0);
                k.setEstimateOwnPerformance(false);
//                k.setK(1);
//                k.setOptimiseK(false);
//                k.setKMax(-1);
                k.setKMax(10);
                k.setComparisonCountLimit(-1);
                k.setOptimiseK(true);
                k.setDistanceFunction(new EuclideanDistance());
                return k;
            }
        },
        ;
    }

    public K() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        Config.DEFAULT.applyConfigTo(this);
    }

    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private final StopWatch testTimer = new StopWatch();
    private final StopWatch trainTimer = new StopWatch();
    private final StopWatch trainEstimateTimer = new StopWatch();
    private final StopWatch trainStageTimer = new StopWatch();
    private final StopWatch testStageTimer = new StopWatch();
    private List<Instance> trainData;
    private List<Search> searches;
    private Map<Instance, Search> searchLookup;
    private ListIterator<Search> targetSearchIterator;
    private DistanceFunction distanceFunction;
    private int k;
    private int kMax;
    private boolean optimiseK;
    private long trainTimeLimit;
    private long testTimeLimit;
    private long longestNeighbourComparisonTimeTrain;
    private long longestNeighbourComparisonTimeTest;
    private long comparisonCount;
    private long comparisonCountLimit;


    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        memoryWatcher.start();
        trainTimer.start();
        trainEstimateTimer.checkStopped();
        super.buildClassifier(trainData);
        // index the train data
        if(isRebuild()) {
            trainTimer.resetAndStart();
            trainEstimateTimer.resetAndStop();
            this.trainData = trainData;
            if(getEstimateOwnPerformance()) {
                // need a matrix of pairwise distances between instances. The matrix will be symmetrical, therefore
                // only need to consider instances in the lower or upper triangle discounting the main diagonal as
                // this include self comparisons (going against LOOCV).
                // the searches, one per instance to search their corresponding neighbourhood
                searches = new ArrayList<>(trainData.size());
                searchLookup = new HashMap<>(trainData.size(), 1);
                // the neighbourhood for each instance. As only the lower or upper triangle is being considered the
                // neighbourhood can be expanded by one for every instance (i.e. row of the matrix) looked at,
                // forming the lower triangle
                final List<Instance> neighbourhood = new ArrayList<>(trainData.size());
                // method of iterating through the searches
                targetSearchIterator = new RandomIterator<>(rand, new ArrayList<>(trainData.size()), true);
                // for each row of the matrix (i.e. neighbourhood search)
                for(int i = 0; i < trainData.size(); i++) {
                    final Instance target = trainData.get(i);
                    final ArrayList<Instance> neighbourhoodCopy = new ArrayList<>(neighbourhood);
                    final Search search = new Search(target, neighbourhoodCopy,
                                                     new RandomIterator<>(rand, neighbourhoodCopy, false), true);
                    if(i > 0) {
                        // neighbourhood is not empty, add the search to the iterator
                        targetSearchIterator.add(search);
                    }
                    neighbourhood.add(target);
                    searches.add(search);
                    searchLookup.put(target, search);
                }
            }
            comparisonCount = 0;
            longestNeighbourComparisonTimeTrain = 0;
        }
        if(estimateOwnPerformance) {
            trainEstimateTimer.start();
            trainTimer.stop();
            while(targetSearchIterator.hasNext() && insideTrainTimeLimit(
                    trainTimer.getTime() + trainEstimateTimer.getTime() + longestNeighbourComparisonTimeTrain) && insideComparisonCountLimit()) {
                // start another stage
                comparisonCount++;
//                System.out.println(comparisonCount);
                trainStageTimer.resetAndStart();
                // fetch the next target
                final Search targetSearch = targetSearchIterator.next();
                final Instance target = (Instance) targetSearch.getTarget();
                // get the remaining candidate neighbours for the target search
                final List<Instance> neighbourhood = targetSearch.getNeighbourhood();
                // pick one candidate
                final Iterator<Instance> neighbourIterator = targetSearch.getIterator();
                // remove that candidate, as it should not be assessed as a potential neighbour again
                final Instance candidate = neighbourIterator.next();
                // remove the search from the iterator if there's no more neighbours to assess
                if(!neighbourIterator.hasNext()) {
                    targetSearchIterator.remove();
                }
//                System.out.println(target.getIndex() + "," + candidate.getIndex());
                // find the search associated with the candidate
                final Search candidateSearch = searchLookup.get(candidate);
                // find the max limit of both the candidate's neighbour search and the target's neighbour search
                final double limit = Math.max(targetSearch.getLimit(), candidateSearch.getLimit());
                // find the distance between the target and candidate
                final double distance = distanceFunction.distance(target, candidate, limit);
                // add the candidate to the target's neighbourhood
                targetSearch.add(candidate, distance);
                // add the target to the candidate's neighbourhood
                candidateSearch.add(target, distance);
                // stage complete
                trainStageTimer.stop();
                longestNeighbourComparisonTimeTrain = Math.max(longestNeighbourComparisonTimeTrain, trainStageTimer.getTime());
                trainEstimateTimer.lap();
            }
            // if all work was completed
            if(!targetSearchIterator.hasNext()) {
                // check every search has seen every possible neighbour
                for(Search search : searches) {
                    Assert.assertEquals(this.trainData.size() - 1, search.getNeighbourCount());
                }
            }
            // get the prediction of each searcher and add to train results for LOOCV.
            // need to optimise k here, looking at most at k smallest distances for each searcher (i.e. each left out
            // instance.
            if(optimiseK) {
                final PrunedMultimap<Double, KNeighbourResults> trainResultsMap = PrunedMultimap.descSoftSingle();
                for(int k = kMax; k > 0; k--) {
                    final KNeighbourResults kNeighbourResults = buildKNeighbourResults(k);
                    trainResultsMap.put(kNeighbourResults.getResults().getAcc(), kNeighbourResults);
                }
                final KNeighbourResults bestKNeighbourResults = RandomUtils.choice(trainResultsMap.values(), getRandom());
                setK(bestKNeighbourResults.k);
                trainResults = bestKNeighbourResults.results;
            } else {
                trainResults = buildKNeighbourResults(k).getResults();
            }
            System.out.println(comparisonCount);
            ResultUtils.setInfo(trainResults, this, trainData);
            trainTimer.start();
            trainEstimateTimer.stop();
        }
        trainTimer.stop();
        trainEstimateTimer.checkStopped();
        memoryWatcher.stop();
    }

    public boolean hasComparisonCountLimit() {
        return comparisonCountLimit > 0;
    }

    public boolean insideComparisonCountLimit() {
        return !hasComparisonCountLimit() || comparisonCount < comparisonCountLimit;
    }

    private KNeighbourResults buildKNeighbourResults(int k) {
        final ClassifierResults results = new ClassifierResults();
        for(Search search : searches) {
            final PrunedMultimap<Double, Instance> distanceMap = search.getDistanceMap();
            distanceMap.setDiscardType(DiscardType.NEWEST);
            distanceMap.setHardLimit(k);
            distanceMap.prune();
            final double[] distribution = search.predict();
            final int prediction = Utilities.argMax(distribution, rand);
            final long testTime = search.getTestTime();
            final double classValue = search.getTarget().classValue();
            results.addPrediction(classValue, distribution, prediction, testTime, "");
        }
        return new KNeighbourResults(results, k);
    }

    public static int getMaxComparisonCount(int numInstances) {
        // the size of the lower triangle of the distance matrix
        return numInstances * (numInstances + 1) / 2 - numInstances;
    }

    @Override
    public double[] distributionForInstance(final Instance target) throws Exception {
        testTimer.resetAndStart();
        longestNeighbourComparisonTimeTest = 0;
        final List<Instance> neighbourhood = new ArrayList<>(trainData);
        final Iterator<Instance> iterator = new RandomIterator<>(rand, neighbourhood, false);
        final Search search = new Search(target, neighbourhood, iterator, false);
        while(iterator.hasNext() && insideTestTimeLimit(
                testTimer.getTime() + longestNeighbourComparisonTimeTest)) {
            testStageTimer.resetAndStart();
            final Instance candidate = iterator.next();
            final double distance = distanceFunction.distance(target, candidate, search.getLimit());
            search.add(candidate, distance);
            testStageTimer.stop();
            longestNeighbourComparisonTimeTest = Math.max(longestNeighbourComparisonTimeTest, testStageTimer.getTime());
            testTimer.lap();
        }
        final double[] distribution = search.predict();
        testTimer.stop();
        return distribution;
    }

    public boolean isOptimiseK() {
        return optimiseK;
    }

    public void setOptimiseK(final boolean optimiseK) {
        this.optimiseK = optimiseK;
    }

    public int getKMax() {
        return kMax;
    }

    public void setKMax(final int kMax) {
        this.kMax = kMax;
    }

    @Override
    public long getTestTimeLimit() {
        return testTimeLimit;
    }

    @Override
    public void setTestTimeLimit(final long testTimeLimit) {
        this.testTimeLimit = testTimeLimit;
    }

    @Override
    public long getTrainTimeLimit() {
        return trainTimeLimit;
    }

    @Override
    public void setTrainTimeLimit(final long trainTimeLimit) {
        this.trainTimeLimit = trainTimeLimit;
    }

    public int getK() {
        return k;
    }

    public void setK(final int k) {
        this.k = k;
    }

    @Override
    public StopWatch getTestTimer() {
        return testTimer;
    }

    @Override
    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    @Override
    public StopWatch getTrainEstimateTimer() {
        return trainEstimateTimer;
    }

    @Override public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    public void setDistanceFunction(final DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }

    public long getComparisonCountLimit() {
        return comparisonCountLimit;
    }

    public void setComparisonCountLimit(final long comparisonCountLimit) {
        this.comparisonCountLimit = comparisonCountLimit;
    }

    private static class KNeighbourResults {

        KNeighbourResults(final ClassifierResults results, final int k) {
            this.results = results;
            this.k = k;
        }

        private final ClassifierResults results;
        private final int k;

        public ClassifierResults getResults() {
            return results;
        }

        public int getK() {
            return k;
        }
    }

    private class Search implements TimedTest {

        private Search(final Instance target, final List<Instance> neighbourhood,
                       final Iterator<Instance> iterator, boolean training) {
            this.target = target;
            this.neighbourhood = neighbourhood;
            this.iterator = iterator;
            distanceMap = PrunedMultimap.asc();
            if(training && optimiseK && kMax > 0) {
                distanceMap.setSoftLimit(kMax);
            } else if(k > 0) {
                distanceMap.setSoftLimit(k);
            } else {
                distanceMap.disableSoftLimit();
            }
            limit = Double.POSITIVE_INFINITY;
        }

        private final Instance target;
        private final PrunedMultimap<Double, Instance> distanceMap;
        private final List<Instance> neighbourhood;
        private final Iterator<Instance> iterator;
        private final StopWatch testTimer = new StopWatch();
        private double limit;
        private int neighbourCount = 0;

        public void add(Instance instance, double distance) {
            distanceMap.put(distance, instance);
            // only update the limit if we've hit max number of neighbours
            if(distanceMap.size() >= distanceMap.getSoftLimit()) {
                limit = Math.min(limit, distanceMap.lastKey());
            }
            neighbourCount++;
        }

        public Instance getTarget() {
            return target;
        }

        public List<Instance> getNeighbourhood() {
            return neighbourhood;
        }

        public double getLimit() {
            return limit;
        }

        public PrunedMultimap<Double, Instance> getDistanceMap() {
            return distanceMap;
        }

        public double[] predict() {
            testTimer.resetAndStart();
            double[] distribution = new double[getNumClasses()];
            if(distanceMap.isEmpty()) {
                distribution = ArrayUtilities.uniformDistribution(distribution.length);
            } else {
                for(Entry<Double, Instance> entry : distanceMap.entries()) {
                    final Instance neighbour = entry.getValue();
                    final Double distance = entry.getKey();
                    distribution[(int) neighbour.classValue()]++;
                }
                ArrayUtilities.normaliseInPlace(distribution);
            }
            testTimer.stop();
            return distribution;
        }

        public StopWatch getTestTimer() {
            return testTimer;
        }

        @Override
        public int hashCode() {
            return target.hashCode();
        }

        @Override
        public boolean equals(Object o) {
            if(o instanceof Search) {
                return hashCode() == o.hashCode();
            } else {
                return false;
            }
        }

        @Override public String toString() {
            return "Searcher{" +
                   "target=" + target +
                   '}';
        }

        public int getNeighbourCount() {
            return neighbourCount;
        }

        public Iterator<Instance> getIterator() {
            return iterator;
        }
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }
}
