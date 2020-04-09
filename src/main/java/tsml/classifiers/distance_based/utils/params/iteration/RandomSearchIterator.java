package tsml.classifiers.distance_based.utils.params.iteration;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.iteration.RandomIteration;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.classifiers.distance_based.utils.params.dimensions.ParameterDimension;
import tsml.classifiers.distance_based.utils.params.distribution.Distribution;
import tsml.classifiers.distance_based.utils.random.BaseRandom;
import tsml.classifiers.distance_based.utils.random.RandomUtils;

/**
 * Purpose: randomly iterate over a parameter space. The random iteration occurs with replacement therefore the same
 * parameter set may be hit more than once.
 * <p>
 * Contributors: goastler
 */
public class RandomSearchIterator extends BaseRandom implements RandomIteration<ParamSet> {

    private ParamSpace paramSpace;
    private static final int DEFAULT_ITERATION_LIMIT = 100;
    private int iterationLimit;
    private int iterationCount = 0;
    private boolean replacement;

    public int getIterationCount() {
        return iterationCount;
    }

    protected RandomSearchIterator setIterationCount(final int iterationCount) {
        this.iterationCount = iterationCount;
        return this;
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

    public RandomSearchIterator(int seed, ParamSpace paramSpace, int iterationLimit) {
        this(seed, paramSpace, iterationLimit, false);
    }

    public RandomSearchIterator(int seed, final ParamSpace paramSpace, final int iterationLimit, boolean replacement) {
        super(seed);
        setup(paramSpace, replacement, iterationLimit);
    }

    public RandomSearchIterator(int seed, final ParamSpace paramSpace) {
        this(seed, paramSpace, DEFAULT_ITERATION_LIMIT, false);
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

    public static class UnitTests {

        @Test
        public void testIteration() {
            ParamSpace space = ParamSpace.UnitTests.build2DContinuousSpace();
            final int limit = 10;
            RandomSearchIterator iterator = new RandomSearchIterator(0, space, limit);
            iterator.setRandom(ParamSpace.UnitTests.buildRandom());
            StringBuilder stringBuilder = new StringBuilder();
            int count = 0;
            while(iterator.hasNext()) {
                count++;
                ParamSet paramSet = iterator.next();
                stringBuilder.append(paramSet);
                stringBuilder.append("\n");
            }
            System.out.println(stringBuilder.toString());
            Assert.assertEquals(count, limit);
            Assert.assertEquals(stringBuilder.toString(),
                "-a, \"0.4157204943935306\", -b, \"0.8187087126750541\"\n"
                    + "-a, \"0.058503304403612566\", -b, \"0.6666091997383249\"\n"
                    + "-a, \"0.3065178840223069\", -b, \"0.9395912589362401\"\n"
                    + "-a, \"0.08798840101774008\", -b, \"0.5644485754368884\"\n"
                    + "-a, \"0.35258737223772796\", -b, \"0.7733698785992328\"\n"
                    + "-a, \"0.2814748369491896\", -b, \"0.8125731817327797\"\n"
                    + "-a, \"0.00746354294055912\", -b, \"0.9953613928573914\"\n"
                    + "-a, \"0.43383933414698683\", -b, \"0.8665760350974969\"\n"
                    + "-a, \"0.006403325787859793\", -b, \"0.7633497173024331\"\n"
                    + "-a, \"0.49233707140341276\", -b, \"0.5415311991124574\"\n"
            );
        }
    }
}
