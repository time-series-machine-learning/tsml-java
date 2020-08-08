package tsml.classifiers.distance_based.knn.strategies;

import com.google.common.collect.ImmutableSet;
import evaluation.storage.ClassifierResults;
import java.util.logging.Logger;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import tsml.classifiers.distance_based.tuned.*;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.IndexedParameterSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearchIterator;
import utilities.*;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import tsml.classifiers.distance_based.utils.collections.box.Box;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

import static tsml.classifiers.distance_based.utils.strings.StrUtils.extractNameAndParams;
import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.replace;

/**
 * Purpose: reinforce-learn a knn. This explores two dimensions: parameters and number of neighbours. In this case,
 * we're exploring parameter for a distance measure. The idea here is we can reduce the number of neighbours examined
 * whilst still identifying a *suitable* parameter. This class produces benchmarks / classifiers. Each benchmark is
 * either a new parameter knn OR a previously seen knn with more neighbours. The RLTunedClassifier sources these
 * benchmarks and builds the classifier. Afterwards, it is passed back to this class via the feedback() function.
 * Inside the feedback function we can examine the performance (score) of the benchmark using the train results and
 * decide what we will do next (i.e. examine another parameter or increase the number of neighbours for more reliable
 * performance estimation).
 *
 * We maintain three sets: one for improved benchmarks, one for the next batch of improved benchmarks, one of
 * unimproveable benchmarks (because the maximum number of neighbours have been hit). The former two hold improvement
 * batches. These have to be in batches to ensure we increase the neighbourhood fairly between benchmarks (as we're
 * randomly sampling parameters therefore we may choose to improve the neighbours of the same classifier twice or
 * more otherwise!)
 *
 * At the moment, the current stategy is:
 * 1) sample 10% of params
 * 2) train those params to 10% neighbourhood
 * 3) sample another 40% of params (50% overall), training each new param to 10% neighbourhood
 * 4) train params to 50% neighbourhood
 * 5) sample another 50% of params (100% overall), training each new param to 50% neighbourhood
 * 6) train params to full neighbourhood
 *
 * Contributors: goastler
 */
public class RLTunedKNNSetup implements RLTunedClassifier.TrainSetupFunction, Loggable {

    @Override
    public Logger getLogger() {
        return rlTunedClassifier.getLogger();
    }

    // the rl tuner we're working on
    private RLTunedClassifier rlTunedClassifier = new RLTunedClassifier();
    // the param space to explore
    private ParamSpace paramSpace;
    // the agent to explore the space
    private Agent agent = null;
    // an iterator to iterate over the unseen params
    private Iterator<ParamSet> paramSetIterator;
    // space limits
    private int neighbourhoodSizeLimit = -1;
    private int paramSpaceSizeLimit = -1;
    private int maxParamSpaceSize = -1; // max number of params
    private int maxNeighbourhoodSize = -1; // max number of neighbours
    private Box<Integer> neighbourCount; // current number of neighbours
    private Box<Integer> paramCount; // current number of params
    // track maximum time taken for another param to run
    private long longestExploreTimeNanos;
    // track max time taken for an addition of neighbours. Note this is addition of 1 neighbour to 1 of the current
    // benchmarks
    private long longestExploitTimeNanos;
    // limits in percentage rather than raw value
    private double neighbourhoodSizeLimitPercentage = -1;
    private double paramSpaceSizeLimitPercentage = -1;
    // the absolute maximum sizes which may be less than the limits (e.g. limiting neighbours to 100 when there's
    // only 25 instances, say)
    private int fullParamSpaceSize = -1;
    private int fullNeighbourhoodSize = -1;
    // true means we're going to incrementally add each neighbour, false means train straight up to neighbour limit
    private boolean incrementalMode = true;
    // builds the param space
    private ParamSpaceBuilder paramSpaceBuilder;
    // iterate over the unseen space
    private Iterator<EnhancedAbstractClassifier> explorer;
    // iterator over the seen space
    private Iterator<EnhancedAbstractClassifier> exploiter;
    // if we can both explore and exploit, the Strategy should tell us which to do
    private Stategy stategy;
    // supplier for a classifier to apply params to
    private Supplier<KNNLOOCV> knnSupplier;
    // set of the next possible benchmarks
    private Set<EnhancedAbstractClassifier> nextImproveableBenchmarks;
    // current set of benchmarks
    private Set<EnhancedAbstractClassifier> improveableBenchmarks;
    // set of complete benchmarks
    private Set<EnhancedAbstractClassifier> unimproveableBenchmarks;
    // iterator to explore improveable benchmarks
    private Iterator<EnhancedAbstractClassifier> improveableBenchmarkIterator;
    // whether to train the final benchmarks up to full neighbourhood or leave as is
    private boolean trainSelectedBenchmarksFully = true;
    // whether we're in full training selected benchmarks mode. If this is true then the iterators will be training
    // neighbourhoods to full. This is just a simple flag to indicate when we've examined as much of the space as we
    // would like and want to fully train the resultant benchmarks from that investigation - i.e. we've explored some
    // parameters, added some neighbours and found a final benchmark which is the best. We now want to fully train
    // that selected benchmark to indicate its quality.
    private boolean trainSelectedBenchmarksFullyHasBeenSetup = false;
    // the set of best benchmarks
    private PrunedMultimap<Double, EnhancedAbstractClassifier> finalBenchmarks;
    // whether we're exploring (true) or exploiting (false)
    private boolean exploreState;
    // id of the next benchmark (as we need to individually address each on)
    private int id = 0;
    private BenchmarkIteratorBuilder improveableBenchmarkIteratorBuilder = new BenchmarkIteratorBuilder() {
        @Override
        public Iterator<EnhancedAbstractClassifier> apply(List<EnhancedAbstractClassifier> benchmarks) {
            // use a random iterator to explore the improveable benchmarks
            getLogger().info(() -> "building new improveable benchmark iterator for " + benchmarks.size() + " benchmarks");
            RandomIterator<EnhancedAbstractClassifier> iterator = new RandomIterator<>(new Random(
                rlTunedClassifier.getSeed()), new ArrayList<>(
                RLTunedKNNSetup.this.improveableBenchmarks));
            return iterator;
        }
    };

    /**
     * Builds a benchmark iterator given a list of benchmarks
     */
    public interface BenchmarkIteratorBuilder extends Function<List<EnhancedAbstractClassifier>, Iterator<EnhancedAbstractClassifier>> {

    }

    /**
     * builds a parameter space given a dataset
     */
    public interface ParamSpaceBuilder extends Function<Instances, ParamSpace>, Serializable {

    }

    public boolean isIncrementalMode() {
        return incrementalMode;
    }

    public void setIncrementalMode(boolean incrementalMode) {
        this.incrementalMode = incrementalMode;
    }

    /**
     * the main agent to run knn tuning
     */
    private class KnnAgent implements Agent {

        // simple checks and balances fields
        private boolean hasNextCalled = false;
        private boolean hasNext = false;
        private Set<EnhancedAbstractClassifier> finalClassifiers = null;

        @Override
        public long predictNextTimeNanos() {
            long time;
            final boolean hasNext = hasNext();
            if(hasNext) {
                // if we're going to explore then report the longest exploration time we've seen (examining another
                // parameter)
                if(exploreState) {
                    time = longestExploreTimeNanos;
                } else {
                    // otherwise the longest exploitation time we've seen (adding more neighbours)
                    time = longestExploitTimeNanos;
                }
            } else {
                // we're not doing anymore work
                time = 0;
            }
            getLogger().info(() -> "predicted next time to be: " + time);
            return time;
        }

        /**
         * build the final set of benchmarks
         * @return set of the 1 best benchmark
         */
        @Override
        public Set<EnhancedAbstractClassifier> getFinalClassifiers() {
            if(finalClassifiers == null) {
                getLogger().info(() -> "finding final classifiers");
                // randomly pick 1 of the best classifiers
                final Collection<EnhancedAbstractClassifier> benchmarks = finalBenchmarks.values();
                final List<EnhancedAbstractClassifier> selectedBenchmarks = null;
//                        Utilities.randPickN(benchmarks, 1,
//                    rlTunedClassifier.getRandom());
                if(selectedBenchmarks.size() > 1) {
                    throw new IllegalStateException("there shouldn't be more than 1");
                }
                getLogger().info(() -> "picked final classifier to be: " + selectedBenchmarks.get(0));
                return new HashSet<>(selectedBenchmarks);
            } else {
                getLogger().info(() -> "final classifiers already found");
                return finalClassifiers;
            }
        }

        /**
         * given an improved / new classifier, examine the improvement (both in score and classifier properties (e.g.
         * num neighbours)). If the score is good enough then it will become the best benchmark so far. If the
         * classifier is not improveable (i.e. no more neighbours / hit limit) then don't add it back into the
         * improveable set
         * @param classifier
         * @return
         */
        @Override
        public boolean feedback(EnhancedAbstractClassifier classifier) {
            finalBenchmarks.put(scorer.findScore(classifier), classifier); // add the benchmark back to the final benchmarks under the new score (which may be worse, hence why we have to remove the original benchmark first
            getLogger().info(() -> "score of " + scorer.findScore(classifier) + " for " + extractNameAndParams(classifier));
            boolean result;
            // if the classifier cannot be improved
            if(!isImproveable(classifier)) {
                // put it in the unimproveable pile
                rlTunedClassifier
                        .getLogger().info(() -> "unimproveable classifier " + extractNameAndParams(classifier));
                CollectionUtils.put(classifier, unimproveableBenchmarks);
                // we won't be using that benchmark again
                result = false;
            } else {
                // else the classifier can be improved, so put it in the improveable pile
                rlTunedClassifier
                        .getLogger().info(() -> "improveable classifier " + extractNameAndParams(classifier));
                CollectionUtils.put(classifier, nextImproveableBenchmarks);
                // we will be using that benchmark again
                result = true;
            }
            // update timings depending on our action
            long time = classifier.getTrainResults().getBuildPlusEstimateTime();
            if(!exploreState) {
                // we've done some exploiting so update the exploit time (i.e. time taken to add a neighbour to a
                // benchmark)
                longestExploitTimeNanos = Math.max(time, longestExploitTimeNanos);
                getLogger().info(() -> "longest exploit time: " + longestExploitTimeNanos);
            }
            // we've either seen another benchmark OR update a current benchmark so update the explore time, which
            // would be the time if we were to explore another benchmark next time round. We do this because we have
            // either just explored a new parameter or just added another neighbour to a parameter. If we've just
            // added another neighbour, the neighbour count may have risen which would also raise the time taken to
            // explore a new parameter next time (as the new parameter is trained to the neighbour count)
            longestExploreTimeNanos = Math.max(time, longestExploreTimeNanos);
            getLogger().info(() -> "longest explore time: " + longestExploreTimeNanos);
            return result;
        }

        @Override
        public EnhancedAbstractClassifier next() {
            if(!hasNext) {
                throw new IllegalStateException("next() called but hasNext() returned false");
            }
            hasNextCalled = false;
            EnhancedAbstractClassifier result;
            if(exploreState) {
                // explore next classifier
                result = explorer.next();
                getLogger().info(() -> "picked " + result);
            } else {
                // improve a current classifier
                result = exploiter.next();
            }
            return result;
        }

        @Override
        public boolean hasNext() {
            // if not already called
            if(!hasNextCalled) {
                hasNextCalled = true;
                boolean hasNext;
                // can we explore and / or exploit
                boolean explore = hasNextExplore();
                boolean exploit = hasNextExploit();
                getLogger().info(() -> "can explore: " + explore);
                getLogger().info(() -> "can exploit: " + exploit);
                exploreState = false;
                if(!exploit && explore) {
                    exploreState = true;
                    hasNext = true;
                } else if(exploit && explore) {
                    // if both then let optimiser decide
                    exploreState = stategy.shouldExplore();
                    hasNext = true;
                } else if(trainSelectedBenchmarksFully && finalClassifiers == null) {
                    getLogger().info(() -> "switching over improveable benchmarks to enable full training");
                    // this should only be hit once
                    // we're training the final benchmarks fully
                    // we need to find the final benchmarks
                    finalClassifiers = getFinalClassifiers();
                    // then make the final classifiers the only improveable benchmarks
                    improveableBenchmarks.clear();
                    nextImproveableBenchmarks.clear();
                    for(EnhancedAbstractClassifier classifier : finalClassifiers) {
                        if(isImproveable(classifier)) {
                            // we can improve the neighbourhood so add the classifier to the improvers
                            nextImproveableBenchmarks.add(classifier);
                        }
                    }
                    // switch the next improvements over to the current and build iterator
                    switchImproveableBenchmarks();
                    // make the neighbour count full neighbourhood
                    neighbourCount.set(fullNeighbourhoodSize);
                    // now the final classifiers are the only thing to be improved. They will be improved up to
                    // the full neighbourhood
                    // hasNext true if the final classifiers can be improved (i.e. aren't at full neighbourhood
                    // already)
                    hasNext = hasNextExploit();
                } else {
                    // fully explored, fully exploited, fully trained (if enabled)
                    hasNext = false;
                }
                this.hasNext = hasNext;
            }
            return hasNext;
        }

        @Override
        public boolean isExploringOrExploiting() {
            return exploreState;
        }
    }

    /**
     * optimiser to decide whether to explore or exploit when both are available
     */
    public interface Stategy {
        boolean shouldExplore();
    }

    public Scorer getScorer() {
        return scorer;
    }

    public void setScorer(Scorer scorer) {
        this.scorer = scorer;
    }

    public boolean isTrainSelectedBenchmarksFully() {
        return trainSelectedBenchmarksFully;
    }

    public void setTrainSelectedBenchmarksFully(boolean trainSelectedBenchmarksFully) {
        this.trainSelectedBenchmarksFully = trainSelectedBenchmarksFully;
    }

    /**
     * method to quantify how good a benchmark is
     */
    public interface Scorer extends Serializable {
        double findScore(EnhancedAbstractClassifier eac);
    }

    private Scorer scorer = eac -> {
        final ClassifierResults results = eac.getTrainResults();
        final double acc = results.getAcc();
        return acc;
    };

    // todo param handling

    /**
     * build the RL tuned classifier. This should already be set, all we're doing here is setting the classifier up
     * with this class and returning it.
     * @return
     */
    public RLTunedClassifier build() {
        rlTunedClassifier.setTrainSetupFunction(this);
        return rlTunedClassifier;
    }

    /**
     * find the limit of a integer field given a maximum size, a raw limit which truncates that size, and a
     * percentage limit which truncates the size to said percentage
     * @param size
     * @param rawLimit
     * @param percentageLimit
     * @return
     */
    private int findLimit(int size, int rawLimit, double percentageLimit) {
        if(size == 0) {
            throw new IllegalArgumentException();
        }
        int result = size;
        if(rawLimit >= 0) {
            result = rawLimit;
        }
        if(NumUtils.isPercentage(percentageLimit)) {
            result = (int) (size * percentageLimit);
        }
        if(result == 0) {
            result = 1;
        }
        return result;
    }

    // checker methods for limits
    private boolean hasLimits() {
        return hasLimitedNeighbourhoodSize() || hasLimitedParamSpaceSize();
    }

    private boolean hasLimitedParamSpaceSize() {
        return paramSpaceSizeLimit >= 0 || NumUtils.isPercentage(paramSpaceSizeLimitPercentage);
    }

    private boolean hasLimitedNeighbourhoodSize() {
        return neighbourhoodSizeLimit >= 0 || NumUtils.isPercentage(neighbourhoodSizeLimitPercentage);
    }

    public double getParamSpaceSizeLimitPercentage() {
        return paramSpaceSizeLimitPercentage;
    }

    public RLTunedKNNSetup setParamSpaceSizeLimitPercentage(final double paramSpaceSizeLimitPercentage) {
        this.paramSpaceSizeLimitPercentage = paramSpaceSizeLimitPercentage;
        return this;
    }

    public Set<EnhancedAbstractClassifier> getImproveableBenchmarks() {
        return ImmutableSet.copyOf(improveableBenchmarks);
    }

    public Set<EnhancedAbstractClassifier> getUnimproveableBenchmarks() {
        return ImmutableSet.copyOf(unimproveableBenchmarks);
    }

    /**
     * all possible benchmarks are housed in the improveable and unimproveable sets
     * @return
     */
    public Set<EnhancedAbstractClassifier> getAllBenchmarks() {
        final HashSet<EnhancedAbstractClassifier> benchmarks = new HashSet<>();
        benchmarks.addAll(unimproveableBenchmarks);
        benchmarks.addAll(this.improveableBenchmarks);
        return benchmarks;
    }

    /**
     * check whether a benchmark can be improved, i.e. more neighbours can be added
     * @param benchmark
     * @return
     */
    private boolean isImproveable(EnhancedAbstractClassifier benchmark) {
        try {
            final KNNLOOCV knn = (KNNLOOCV) benchmark;
            return knn.getNeighbourLimit() + 1 <= maxNeighbourhoodSize;
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private boolean hasNextExploreTime() {
//        return !rlTunedClassifier.hasTrainTimeLimit() || longestExploreTimeNanos < rlTunedClassifier.getRemainingTrainTimeNanos();
        return true;
    }

    private boolean hasNextExploitTime() {
//        return !rlTunedClassifier.hasTrainTimeLimit() || longestExploitTimeNanos < rlTunedClassifier.getRemainingTrainTimeNanos();
        return true;
    }

    private boolean hasNextExploreTick() {
        return explorer.hasNext();
    }

    private boolean hasNextExploitTick() {
        return exploiter.hasNext();
    }

    private boolean hasNextExplore() {
        return hasNextExploreTime() && hasNextExploreTick();
    }

    public boolean hasNextExploit() {
        return hasNextExploitTime() && hasNextExploitTick();
    }

    @Override
    public void accept(Instances trainData) {
        neighbourCount = new Box<>(1); // must start at 1 otherwise the loocv produces no train estimate
        paramCount = new Box<>(0);
        longestExploreTimeNanos = 0;
        id = 0;
        longestExploitTimeNanos = 0;
        nextImproveableBenchmarks = new HashSet<>();
        improveableBenchmarks = new HashSet<>();
        unimproveableBenchmarks = new HashSet<>();
        switchImproveableBenchmarks();
        finalBenchmarks = PrunedMultimap.desc(ArrayList::new);
        finalBenchmarks.setSoftLimit(1);
        final int seed = rlTunedClassifier.getSeed();
        paramSpace = paramSpaceBuilder.apply(trainData);
        paramSetIterator = new RandomSearchIterator(new Random(seed), this.paramSpace, seed).setReplacement(true);
        fullParamSpaceSize = new IndexedParameterSpace(this.paramSpace).size();
        fullNeighbourhoodSize = trainData.size(); // todo check all seeds set
        maxNeighbourhoodSize = findLimit(fullNeighbourhoodSize, neighbourhoodSizeLimit, neighbourhoodSizeLimitPercentage);
        maxParamSpaceSize = findLimit(fullParamSpaceSize, paramSpaceSizeLimit, paramSpaceSizeLimitPercentage);
        if(!incrementalMode) {
            neighbourCount.set(maxNeighbourhoodSize);
        }
        // transform classifiers into benchmarks
        explorer = new ParamExplorer();
        // setup an iterator to improve benchmarks
        exploiter = new NeighbourExploiter();
        stategy = new LeeStategy();
        agent = new KnnAgent();
        // set corresponding iterators in the incremental tuned classifier
        rlTunedClassifier.setAgent(agent);
        rlTunedClassifier.setEnsembler(Ensembler.single());
        // todo make sure the seeds are set for everything
    }

    private class ParamExplorer implements Iterator<EnhancedAbstractClassifier> {
        @Override public EnhancedAbstractClassifier next() {
            ParamSet paramSet = paramSetIterator.next();
            paramCount.set(paramCount.get() + 1);
            final KNNLOOCV knn = knnSupplier.get();
            try {
                knn.setParams(paramSet);
            } catch(Exception e) {
                throw new IllegalStateException(e);
            }
            final String name = knn.getClassifierName() + "_" + (id++);
            knn.setClassifierName(name);
            knn.setNeighbourLimit(neighbourCount.get());
            getLogger().info(() -> "exploring: " + extractNameAndParams(knn));
            return knn;
        }

        @Override public boolean hasNext() {
            return paramSetIterator.hasNext() && withinParamSpaceSizeLimit();
        }

        private boolean withinParamSpaceSizeLimit() {
            return paramCount.get() < maxParamSpaceSize;
        }
    }

    private void switchImproveableBenchmarks() {
        improveableBenchmarks = nextImproveableBenchmarks;
        nextImproveableBenchmarks = new HashSet<>();
        improveableBenchmarkIterator = improveableBenchmarkIteratorBuilder.apply(new ArrayList<>(
            improveableBenchmarks));
    }

    private class NeighbourExploiter implements Iterator<EnhancedAbstractClassifier> {

        @Override
        public EnhancedAbstractClassifier next() {
            if(!improveableBenchmarkIterator.hasNext()) {
                final int origNeighbourCount = neighbourCount.get();
                final int nextNeighbourCount = origNeighbourCount + 1;
                getLogger().info(() -> "neighbourhood " + origNeighbourCount + " --> " + nextNeighbourCount);
                neighbourCount.set(nextNeighbourCount);
                switchImproveableBenchmarks();
                if(!improveableBenchmarkIterator.hasNext()) {
                    throw new IllegalStateException("this shouldn't happen, if we get to this point we should always "
                        + "have improveable neighbours");
                }
            }
            final EnhancedAbstractClassifier classifier = improveableBenchmarkIterator.next();
            improveableBenchmarkIterator.remove();
            final KNNLOOCV knn = (KNNLOOCV) classifier;
            final int nextNeighbourCount = neighbourCount.get();
            // sanity check that the neighbours are increasing
            final int currentNeighbourLimit = knn.getNeighbourLimit();
            if(nextNeighbourCount <= currentNeighbourLimit) {
                throw new IllegalStateException("no improvement to the number of neighbours");
            }
            knn.setNeighbourLimit(nextNeighbourCount);
            finalBenchmarks.remove(scorer.findScore(classifier), classifier); // remove the current classifier from the final benchmarks
            return knn;
        }

        @Override
        public boolean hasNext() {
            boolean result = improveableBenchmarkIterator.hasNext() || !nextImproveableBenchmarks.isEmpty();
            return result;
        }
    }

    private class LeeStategy implements Stategy {

        @Override
        public boolean shouldExplore() {
            // only called when *both* improvements and source remain
            final int neighbours = neighbourCount.get();
            final int params = paramCount.get();
            boolean result;
            if(params < maxParamSpaceSize / 10) {
                // 10% params, 0% neighbours
                result = true;
            } else if(neighbours < maxNeighbourhoodSize / 10) {
                // 10% params, 10% neighbours
                result = false;
            } else if(params < maxParamSpaceSize / 2) {
                // 50% params, 10% neighbours
                result = true;
            } else if(neighbours < maxNeighbourhoodSize / 2) {
                // 50% params, 50% neighbours
                result = false;
            } else if(params < maxParamSpaceSize) {
                // 100% params, 50% neighbours
                result = true;
            }
            else {
                // by this point all params have been hit. Therefore, shouldSource should not be called at
                // all as only improvements will remain, if any.
                throw new IllegalStateException("invalid source / improvement state");
            }
            getLogger().info(() -> "strategy has chosen to explore: " + result);
            return result;
        }
    }

    public BenchmarkIteratorBuilder getImproveableBenchmarkIteratorBuilder() {
        return improveableBenchmarkIteratorBuilder;
    }

    public void setImproveableBenchmarkIteratorBuilder(BenchmarkIteratorBuilder improveableBenchmarkIteratorBuilder) {
        this.improveableBenchmarkIteratorBuilder = improveableBenchmarkIteratorBuilder;
    }

    public Agent getAgent() {
        return agent;
    }

    public int getFullParamSpaceSize() {
        return fullParamSpaceSize;
    }

    public int getFullNeighbourhoodSize() {
        return fullNeighbourhoodSize;
    }

    public Set<EnhancedAbstractClassifier> getNextImproveableBenchmarks() {
        return ImmutableSet.copyOf(nextImproveableBenchmarks);
    }

    public Iterator<EnhancedAbstractClassifier> getImproveableBenchmarkIterator() {
        return improveableBenchmarkIterator;
    }

    public PrunedMultimap<Double, EnhancedAbstractClassifier> getFinalBenchmarks() {
        return finalBenchmarks;
    }

    public boolean isExploreState() {
        return exploreState;
    }

    public RLTunedClassifier getRlTunedClassifier() {
        return rlTunedClassifier;
    }

    public RLTunedKNNSetup setRlTunedClassifier(final RLTunedClassifier rlTunedClassifier) {
        this.rlTunedClassifier = rlTunedClassifier;
        return this;
    }

    public ParamSpace getParamSpace() {
        return paramSpace;
    }

    public RLTunedKNNSetup setParamSpace(final ParamSpace paramSpace) {
        this.paramSpace = paramSpace;
        return this;
    }

    public Iterator<ParamSet> getParamSetIterator() {
        return paramSetIterator;
    }

    public RLTunedKNNSetup setParamSetIterator(final Iterator<ParamSet> paramSetIterator) {
        this.paramSetIterator = paramSetIterator;
        return this;
    }

    public int getMaxParamSpaceSize() {
        return maxParamSpaceSize;
    }

    public RLTunedKNNSetup setParamSpaceSizeLimit(final int limit) {
        this.neighbourhoodSizeLimit = limit;
        return this;
    }

    public int getMaxNeighbourhoodSize() {
        return maxNeighbourhoodSize;
    }

    public RLTunedKNNSetup setNeighbourhoodSizeLimit(final int limit) {
        this.neighbourhoodSizeLimit = limit;
        return this;
    }

    public int getNeighbourhoodSizeLimit() {
        return neighbourhoodSizeLimit;
    }

    public int getParamSpaceSizeLimit() {
        return paramSpaceSizeLimit;
    }

    public Integer getNeighbourCount() {
        return neighbourCount.get();
    }

    public Integer getParamCount() {
        return paramCount.get();
    }

    public long getLongestExploreTimeNanos() {
        return longestExploreTimeNanos;
    }

    public RLTunedKNNSetup setLongestExploreTimeNanos(final long longestExploreTimeNanos) {
        this.longestExploreTimeNanos = longestExploreTimeNanos;
        return this;
    }

    public long getLongestExploitTimeNanos() {
        return longestExploitTimeNanos;
    }

    public RLTunedKNNSetup setLongestExploitTimeNanos(final long longestExploitTimeNanos) {
        this.longestExploitTimeNanos = longestExploitTimeNanos;
        return this;
    }

    public ParamSpaceBuilder getParamSpaceBuilder() {
        return paramSpaceBuilder;
    }

    public RLTunedKNNSetup setParamSpaceBuilder(final ParamSpaceBuilder paramSpaceBuilder) {
        this.paramSpaceBuilder = paramSpaceBuilder;
        return this;
    }

    public Stategy getStategy() {
        return stategy;
    }

    public RLTunedKNNSetup setStategy(final Stategy stategy) {
        this.stategy = stategy;
        return this;
    }

    // todo change supplier to own interface
    public Supplier<KNNLOOCV> getKnnSupplier() {
        return knnSupplier;
    }

    public RLTunedKNNSetup setKnnSupplier(final Supplier<KNNLOOCV> knnSupplier) {
        this.knnSupplier = knnSupplier;
        return this;
    }

    public RLTunedKNNSetup setParamSpace(ParamSpaceBuilder func) {
        return setParamSpaceBuilder(func);
    }

    public RLTunedKNNSetup setParamSpaceFunction(Supplier<ParamSpace> supplier) {
        return setParamSpace(i -> supplier.get());
    }

    public RLTunedKNNSetup setParamSpace(Supplier<ParamSpace> supplier) {
        return setParamSpaceFunction(supplier);
    }

    public double getNeighbourhoodSizeLimitPercentage() {
        return neighbourhoodSizeLimitPercentage;
    }

    public RLTunedKNNSetup setNeighbourhoodSizeLimitPercentage(final double neighbourhoodSizeLimitPercentage) {
        this.neighbourhoodSizeLimitPercentage = neighbourhoodSizeLimitPercentage;
        return this;
    }
}
