package tsml.classifiers.distance_based.utils.collections.params.iteration;

import tsml.classifiers.distance_based.utils.collections.iteration.BaseRandomIterator;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete.IndexedParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Purpose: randomly iterate over a parameter space. The random iteration occurs with replacement therefore the same
 * parameter set may be hit more than once.
 * <p>
 * Contributors: goastler
 */
public class RandomSearch extends AbstractSearch implements RandomIterator<ParamSet> {

    public static final int DEFAULT_ITERATION_LIMIT = 100;
    private int iterationLimit = DEFAULT_ITERATION_LIMIT;
    private final RandomIterator<ParamSet> randomIterator = new BaseRandomIterator<>();
    private boolean discrete;

    @Override public void setRandom(final Random random) {
        randomIterator.setRandom(random);
    }

    @Override public Random getRandom() {
        return randomIterator.getRandom();
    }

    @Override public void buildSearch(final ParamSpace paramSpace) {
        super.buildSearch(paramSpace);
        if(randomIterator.getRandom() == null) throw new IllegalStateException("random not set");
        // is the param space discrete?
        try {
            // if so then build the random iterator
            randomIterator.buildIterator(new IndexedParamSpace(paramSpace));
            discrete = true;
        } catch(IllegalArgumentException e) {
            // param space is not discrete. Do not use the random iterator
            discrete = false;
        }
        // should be able to sample an unlimited amount of paramsets with risk of finding repeats occurring
        randomIterator.setWithReplacement(true);
    }

    public boolean hasIterationLimit() {
        return getIterationLimit() >= 0;
    }

    public boolean insideIterationCountLimit() {
        return !hasIterationLimit() || getIterationCount() < getIterationLimit();
    }
    
    @Override protected boolean hasNextParamSet() {
        return insideIterationCountLimit() && (!discrete || randomIterator.hasNext());
    }

    @Override protected ParamSet nextParamSet() {
        // if dealing with a discrete space
        if(discrete) {
            // then use the random iterator to iterate over it
            return randomIterator.next();
        } else {
            // otherwise extract a random param set whilst managing discrete and continuous dimensions
            return extractRandomParamSet(getParamSpace(), getRandom());
        }
    }

    /**
     * CAUTION: this may return an item within a list, therefore mutations of said item will modify the same instance in the list!
     * @param values
     * @param random
     * @return
     */
    private static Object extractRandomValue(Object values, Random random) {
        final Object value;
        if(values instanceof Distribution<?>) {
            Distribution<?> distribution = (Distribution<?>) values;
            // same as below, but a distribution should make a new instance of the value already. Take a copy just in case.
            value = distribution.sample(random);
        } else if(values instanceof List<?>) {
            List<?> list = (List<?>) values;
            value = RandomUtils.choice(list, random);
        } else {
            throw new IllegalArgumentException("cannot handle type {" + values.getClass() + "} for dimension content");
        }
        return value;
    }

    public static ParamSet extractRandomParamSet(ParamSpace paramSpace, Random random) {
        final Map<String, List<ParamDimension<?>>> dimensionMap = paramSpace.getDimensionMap();
        final ParamSet paramSet = new ParamSet();
        for(Map.Entry<String, List<ParamDimension<?>>> entry : dimensionMap.entrySet()) {
            final String name = entry.getKey();
            List<ParamDimension<?>> dimensions = entry.getValue();
            ParamDimension<?> dimension = RandomUtils.choice(dimensions, random);
            final Object value = extractRandomValue(dimension.getValues(), random);
            final List<ParamSpace> subSpaces = dimension.getSubSpaces();
            final List<ParamSet> subParamSets = new ArrayList<>(subSpaces.size());
            for(ParamSpace subSpace : subSpaces) {
                final ParamSet subParamSet = extractRandomParamSet(subSpace, random);
                subParamSets.add(subParamSet);
            }
            paramSet.add(name, value, subParamSets);
        }
        return paramSet;
    }

    public int getIterationLimit() {
        return iterationLimit;
    }

    public RandomSearch setIterationLimit(final int iterationLimit) {
        this.iterationLimit = iterationLimit;
        return this;
    }

    public static ParamSet choice(ParamSpace paramSpace, Random random) {
        return choice(paramSpace, random, 1).get(0);
    }

    public static List<ParamSet> choice(ParamSpace paramSpace, Random random, int numChoices) {
        final RandomSearch iterator = new RandomSearch();
        iterator.setRandom(random);
        iterator.buildSearch(paramSpace);
        iterator.setIterationLimit(numChoices);
        return RandomUtils.choice(iterator, numChoices);
    }

    @Override public boolean withReplacement() {
        return randomIterator.withReplacement();
    }

    @Override public void setWithReplacement(final boolean withReplacement) {
        randomIterator.setWithReplacement(withReplacement);
    }
}
