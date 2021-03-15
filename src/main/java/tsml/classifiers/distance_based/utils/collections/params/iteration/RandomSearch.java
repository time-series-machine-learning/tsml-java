package tsml.classifiers.distance_based.utils.collections.params.iteration;

import tsml.classifiers.distance_based.utils.collections.iteration.BaseRandomIterator;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.continuous.ContinuousParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete.DiscreteParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete.GridParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;

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
    private GridParamSpace gridParamSpace; // only used if paramspace is discrete

    public RandomSearch() {
        setWithReplacement(false);
    }
    
    @Override public void setRandom(final Random random) {
        randomIterator.setRandom(random);
    }

    @Override public Random getRandom() {
        return randomIterator.getRandom();
    }

    @Override public void setSeed(final int seed) {
        randomIterator.setSeed(seed);
    }

    @Override public int getSeed() {
        return randomIterator.getSeed();
    }

    @Override public void buildSearch(final ParamSpace paramSpace) {
        super.buildSearch(paramSpace);
        checkRandom();

        // is the param space discrete?
        gridParamSpace = null;
        discrete = GridParamSpace.isDiscrete(paramSpace);
        if(discrete) {
            // if so then build the random iterator
            gridParamSpace = new GridParamSpace(paramSpace);
            randomIterator.buildIterator(gridParamSpace);
        } // else param space is not discrete. Do not use the random iterator
    }

    public boolean hasIterationLimit() {
        return getIterationLimit() >= 0;
    }

    public boolean insideIterationCountLimit() {
        return !hasIterationLimit() || getIterationCount() < getIterationLimit();
    }
    
    @Override protected boolean hasNextParamSet() {
        return insideIterationCountLimit() && (!discrete || getParamSpace().isEmpty() || randomIterator.hasNext());
    }

    @Override protected ParamSet nextParamSet() {
        // if dealing with a discrete space
        if(getParamSpace().isEmpty()) {
            return new ParamSet();
        } else if(discrete) {
            // then use the random iterator to iterate over it
            return randomIterator.next();
        } else {
            // otherwise extract a random param set whilst managing discrete and continuous dimensions
            return extractRandomParamSet(getParamSpace(), getRandom());
        }
    }

    /**
     * 
     * @param random
     * @return
     */
    private static Object extractRandomValue(ParamDimension<?> dimension, Random random) {
        final Object value;
        if(dimension instanceof ContinuousParamDimension<?>) {
            Distribution<?> distribution = ((ContinuousParamDimension<?>) dimension).getDistribution();
            // same as below, but a distribution should make a new instance of the value already. Take a copy just in case.
            value = distribution.sample(random);
        } else if(dimension instanceof DiscreteParamDimension<?>) {
            List<?> list = ((DiscreteParamDimension<?>) dimension).getValues();
            value = RandomUtils.choice(list, random);
        } else {
            throw new IllegalArgumentException("cannot handle dimension of type " + dimension.getClass().getSimpleName());
        }
        return value;
    }

    public static ParamSet extractRandomParamSet(ParamSpace paramSpace, Random random) {
        final ParamSet paramSet = new ParamSet();
        if(paramSpace.isEmpty()) {
            return paramSet;
        }
        final ParamMap paramMap = RandomUtils.choice(paramSpace, random);
        for(Map.Entry<String, List<ParamDimension<?>>> entry : paramMap.entrySet()) {
            final String name = entry.getKey();
            List<ParamDimension<?>> dimensions = entry.getValue();
            ParamDimension<?> dimension = RandomUtils.choice(dimensions, random);
            final Object value = extractRandomValue(dimension, random);
            final ParamSpace subSpace = dimension.getSubSpace();
            final ParamSet subParamSet = extractRandomParamSet(subSpace, random);
            paramSet.add(name, value, subParamSet);
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
        iterator.setIterationLimit(numChoices);
        iterator.buildSearch(paramSpace);
        return RandomUtils.choice(iterator, numChoices);
    }

    @Override public boolean withReplacement() {
        return randomIterator.withReplacement();
    }

    @Override public void setWithReplacement(final boolean withReplacement) {
        randomIterator.setWithReplacement(withReplacement);
    }

    @Override public int size() {
        int size = iterationLimit;
        if(getParamSpace().isEmpty()) {
            size = Math.min(size, 1);
        } else if(discrete) {
            size = Math.min(size, gridParamSpace.size());
        }
        return size;
    }
}
