package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.proximity.ProximityTree;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifier_mixins.Configurer;
import tsml.classifiers.distance_based.utils.classifier_mixins.Utils;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap.DiscardType;
import tsml.classifiers.distance_based.utils.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.iteration.RandomIterator;
import tsml.classifiers.distance_based.utils.iteration.RandomListIterator;
import tsml.classifiers.distance_based.utils.random.RandomUtils;
import tsml.classifiers.distance_based.utils.results.ResultUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.system.timing.TimedTest;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrain;
import tsml.classifiers.distance_based.utils.system.timing.TimedTrainEstimate;
import tsml.transformers.Indexer;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map.Entry;

public class K extends BaseClassifier implements TimedTest, TimedTrain, TimedTrainEstimate, ContractedTest, ContractedTrain {

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
                k.setK(1);
                k.setOptimiseK(false);
                k.setkMax(-1);
                k.setDistanceFunction(new DTWDistance());
                return k;
            }
        },
        ;
    }

    public K() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        Config.DEFAULT.applyConfigTo(this);
    }

    private final StopWatch testTimer = new StopWatch();
    private final StopWatch trainTimer = new StopWatch();
    private final StopWatch trainEstimateTimer = new StopWatch();
    private final StopWatch trainStageTimer = new StopWatch();
    private final StopWatch testStageTimer = new StopWatch();
    private transient Instances trainData;
    private Iterator<Searcher> targetIterator;
    private List<Searcher> searchers;
    private DistanceFunction distanceFunction;
    private int k;
    private int kMax;
    private boolean optimiseK;
    private long trainTimeLimit;
    private long testTimeLimit;
    private long longestNeighbourComparisonTimeTrain;
    private long longestNeighbourComparisonTimeTest;

    @Override
    public void buildClassifier(final Instances trainData) throws Exception {
        trainTimer.start();
        trainEstimateTimer.checkStopped();
        super.buildClassifier(trainData);
        // index the train data
        Indexer.index(trainData);
        if(isRebuild()) {
            trainTimer.resetAndStart();
            trainEstimateTimer.resetAndStop();
            this.trainData = trainData;
            if(getEstimateOwnPerformance()) {
                // each row should have its own iterator
                searchers = new ArrayList<>(trainData.size());
                for(Instance instance : trainData) {
                    searchers.add(new Searcher(instance, new RandomIterator<>(rand, trainData, false)));
                }
                // generate an iterator to iterate over the rows
                targetIterator = new RandomIterator<>(rand, searchers, true);
                longestNeighbourComparisonTimeTrain = 0;
            }
        }
        if(estimateOwnPerformance) {
            trainEstimateTimer.start();
            trainTimer.stop();
            while(targetIterator.hasNext() && insideTrainTimeLimit(
                    trainTimer.getTime() + trainEstimateTimer.getTime() + longestNeighbourComparisonTimeTrain)) {
                trainStageTimer.resetAndStart();
                final Searcher targetSearcher = targetIterator.next();
                final Iterator<Instance> targetSearcherCandidateIterator = targetSearcher.getIterator();
                final Instance candidate = targetSearcherCandidateIterator.next();
                if(!targetSearcherCandidateIterator.hasNext()) {
                    targetIterator.remove();
                }
                if(!(candidate instanceof Indexer.IndexedInstance)) {
                    throw new IllegalStateException("instance should be indexed");
                }
                final Searcher candidateSearcher = searchers.get(((Indexer.IndexedInstance) candidate).getIndex());
                final Instance target = targetSearcher.getTarget();
                if(candidate == target) {
                    continue;
                }
                final double limit = Math.max(targetSearcher.getLimit(), candidateSearcher.getLimit());
                final double distance = distanceFunction.distance(target, candidate, limit);
                targetSearcher.add(candidate, distance);
                candidateSearcher.add(target, distance);
                trainStageTimer.stop();
                longestNeighbourComparisonTimeTrain = Math.max(longestNeighbourComparisonTimeTrain,
                                                               trainStageTimer.getTime());
                trainEstimateTimer.lap();
            }
            // get the prediction of each searcher and add to train results for LOOCV.
            // need to optimise k here, looking at most at k smallest distances for each searcher (i.e. each left out
            // instance.
            final PrunedMultimap<Double, KResults> trainResultsMap = PrunedMultimap.descSoftSingle();
            for(int k = kMax; k > 0; k++) {
                final ClassifierResults results = new ClassifierResults();
                for(Searcher searcher : searchers) {
                    final PrunedMultimap<Double, Instance> distanceMap = searcher.getDistanceMap();
                    distanceMap.setDiscardType(DiscardType.NEWEST);
                    distanceMap.setHardLimit(k);
                    distanceMap.prune();
                    final double[] distribution = searcher.predict();
                    final int prediction = Utilities.argMax(distribution, rand);
                    final long testTime = searcher.getTestTime();
                    final double classValue = searcher.getTarget().classValue();
                    results.addPrediction(classValue, distribution, prediction, testTime, "");
                }
                trainResultsMap.put(results.getAcc(), new KResults(results, k));
            }
            final KResults bestKResults = RandomUtils.choice(trainResultsMap.values(), getRandom());
            setK(bestKResults.k);
            trainResults = bestKResults.results;
            ResultUtils.setInfo(trainResults, this, trainData);
            trainTimer.start();
            trainEstimateTimer.stop();
        }
        trainTimer.stop();
        trainEstimateTimer.checkStopped();
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        testTimer.resetAndStart();
        longestNeighbourComparisonTimeTest = 0;
        final Searcher searcher = new Searcher(instance, new RandomIterator<>(getRandom(), trainData));
        while(searcher.getIterator().hasNext() && insideTestTimeLimit(
                testTimer.getTime() + longestNeighbourComparisonTimeTest)) {
            testStageTimer.resetAndStart();
            final Instance candidate = searcher.getIterator().next();
            final double distance = distanceFunction.distance(instance, candidate, searcher.getLimit());
            searcher.add(candidate, distance);
            testStageTimer.stop();
            longestNeighbourComparisonTimeTest = Math.max(longestNeighbourComparisonTimeTest, testStageTimer.getTime());
            testTimer.lap();
        }
        final double[] distribution = searcher.predict();
        testTimer.stop();
        return distribution;
    }

    public boolean isOptimiseK() {
        return optimiseK;
    }

    public void setOptimiseK(final boolean optimiseK) {
        this.optimiseK = optimiseK;
    }

    public int getkMax() {
        return kMax;
    }

    public void setkMax(final int kMax) {
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

    public void setDistanceFunction(final DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }

    private static class KResults {

        KResults(final ClassifierResults results, final int k) {
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

    private class Searcher implements TimedTest {

        private Searcher(final Instance target, final Iterator<Instance> iterator) {
            this.target = target;
            this.iterator = iterator;
            distanceMap = PrunedMultimap.asc();
            if(optimiseK && kMax > 0) {
                distanceMap.setSoftLimit(kMax);
            } else if(k > 0) {
                distanceMap.setSoftLimit(k);
            } else {
                distanceMap.disableLimits();
            }
            limit = Double.POSITIVE_INFINITY;
        }

        private final Instance target;
        private final PrunedMultimap<Double, Instance> distanceMap;
        private final Iterator<Instance> iterator;
        private final StopWatch testTimer = new StopWatch();
        private double limit;

        public void add(Instance instance, double distance) {
            distanceMap.put(distance, instance);
            limit = Math.min(limit, distanceMap.lastKey());
        }

        public Instance getTarget() {
            return target;
        }

        public Iterator<Instance> getIterator() {
            return iterator;
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
            for(Entry<Double, Instance> entry : distanceMap.entries()) {
                final Instance neighbour = entry.getValue();
                final Double distance = entry.getKey();
                distribution[(int) neighbour.classValue()]++;
            }
            testTimer.stop();
            return distribution;
        }

        public StopWatch getTestTimer() {
            return testTimer;
        }

        @Override public String toString() {
            return "Searcher{" +
                   "target=" + target +
                   '}';
        }

    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }
}
