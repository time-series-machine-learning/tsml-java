package tsml.classifiers.distance_based.utils.collections.params.iteration;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParameterDimension;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;

/**
 * Purpose: randomly iterate over a parameter space. The random iteration occurs with replacement therefore the same
 * parameter set may be hit more than once.
 * <p>
 * Contributors: goastler
 */
public class RandomSearchIterator extends RandomIterator<ParamSet> {

    private ParamSpace paramSpace;
    private static final int DEFAULT_ITERATION_LIMIT = 100;
    private int iterationLimit;
    private int iterationCount = 0;
    private boolean replacement;

    public int getIterationCount() {
        return iterationCount;
    }

    protected void setIterationCount(final int iterationCount) {
        this.iterationCount = iterationCount;
    }

    public boolean hasIterationLimit() {
        return getIterationLimit() >= 0;
    }

    public RandomSearchIterator disableIterationLimit() {
        return setIterationLimit(-1);
    }

    public boolean withinIterationLimit() {
        if(!hasIterationLimit()) {
            return true;
        } else {
            return getIterationCount() < getIterationLimit();
        }
    }

    private void setup(ParamSpace paramSpace, boolean replacement, int numIterations) {
        setParamSpace(paramSpace);
        setReplacement(replacement);
        setIterationLimit(numIterations);
    }

    public RandomSearchIterator(Random random, ParamSpace paramSpace, int iterationLimit) {
        this(random, paramSpace, iterationLimit, false);
    }

    public RandomSearchIterator(Random random, final ParamSpace paramSpace, final int iterationLimit, boolean replacement) {
        super(random);
        setup(paramSpace, replacement, iterationLimit);
    }

    public RandomSearchIterator(Random random, final ParamSpace paramSpace) {
        this(random, paramSpace, DEFAULT_ITERATION_LIMIT, false);
    }

    public ParamSpace getParamSpace() {
        return paramSpace;
    }

    public RandomSearchIterator setParamSpace(
        final ParamSpace paramSpace) {
        Assert.assertNotNull(paramSpace);
        this.paramSpace = paramSpace;
        if(!withReplacement()) {

        }
        return this;
    }

    public boolean withReplacement() {
        return replacement;
    }

    public RandomSearchIterator setReplacement(final boolean replacement) {
        this.replacement = replacement;
        return this;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    @Override
    public boolean hasNext() {
        return withinIterationLimit();
    }

    private Object extractRandomValue(Object values) {
        Object value;
        if(values instanceof Distribution<?>) {
            Distribution<?> distribution = (Distribution<?>) values;
            value = distribution.sample(getRandom());
        } else if(values instanceof List<?>) {
            List<?> list = (List<?>) values;
            value = RandomUtils.choice(list, getRandom());
        } else {
            throw new IllegalArgumentException("cannot handle type {" + values.getClass() + "} for dimension content");
        }
        return value;
    }

    private ParamSet extractRandomParamSet(ParamSpace paramSpace) {
        final Map<String, List<ParameterDimension<?>>> dimensionMap = paramSpace.getDimensionMap();
        final ParamSet paramSet = new ParamSet();
        for(Map.Entry<String, List<ParameterDimension<?>>> entry : dimensionMap.entrySet()) {
            List<ParameterDimension<?>> dimensions = entry.getValue();
            ParameterDimension<?> dimension = RandomUtils.choice(dimensions, getRandom());
            final Object value = extractRandomValue(dimension.getValues());
            final List<ParamSpace> subSpaces = dimension.getSubSpaces();
            final List<ParamSet> subParamSets = new ArrayList<>();
            for(ParamSpace subSpace : subSpaces) {
                final ParamSet subParamSet = extractRandomParamSet(subSpace);
                subParamSets.add(subParamSet);
            }
            final String name = entry.getKey();
            paramSet.add(name, value, subParamSets);
        }
        return paramSet;
    }

    @Override
    public ParamSet next() {
        ParamSet paramSet = extractRandomParamSet(paramSpace);
        setIterationCount(getIterationCount() + 1);
        return paramSet;
    }

    public int getIterationLimit() {
        return iterationLimit;
    }

    public RandomSearchIterator setIterationLimit(final int iterationLimit) {
        this.iterationLimit = iterationLimit;
        return this;
    }

    public static ParamSet choice(Random random, ParamSpace paramSpace) {
        return choice(random, paramSpace, 1).get(0);
    }

    public static List<ParamSet> choice(Random random, ParamSpace paramSpace, int numChoices) {
        final RandomSearchIterator iterator = new RandomSearchIterator(random, paramSpace);
        iterator.setIterationLimit(numChoices);
        return RandomUtils.choice(iterator, numChoices);
    }

}
