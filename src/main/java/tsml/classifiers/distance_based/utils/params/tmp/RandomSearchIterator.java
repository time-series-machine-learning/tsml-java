package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.BaseRandom;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.distribution.Distribution;
import tsml.classifiers.distance_based.utils.random.RandomUtils;

/**
 * Purpose: randomly iterate over a parameter space. The random iteration occurs with replacement therefore the same
 * parameter set may be hit more than once.
 * <p>
 * Contributors: goastler
 */
public class RandomSearchIterator extends BaseRandom implements Iterator<ParamSet> {

    private ParameterSpace parameterSpace;
    private int numIterations = 100;
    private int iterationCount = 0;

    public RandomSearchIterator(final ParameterSpace parameterSpace, final int numIterations) {
        this(parameterSpace);
        setNumIterations(numIterations);
    }

    public RandomSearchIterator(final ParameterSpace parameterSpace) {
        setParameterSpace(parameterSpace);
    }

    public ParameterSpace getParameterSpace() {
        return parameterSpace;
    }

    protected RandomSearchIterator setParameterSpace(
        final ParameterSpace parameterSpace) {
        Assert.assertNotNull(parameterSpace);
        this.parameterSpace = parameterSpace;
        return this;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    @Override
    public boolean hasNext() {
        return iterationCount < numIterations;
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

    private ParamSet extractRandomParamSet(ParameterSpace parameterSpace) {
        final Map<String, List<ParameterDimension<?>>> dimensionMap = parameterSpace.getDimensionMap();
        final ParamSet paramSet = new ParamSet();
        for(Map.Entry<String, List<ParameterDimension<?>>> entry : dimensionMap.entrySet()) {
            List<ParameterDimension<?>> dimensions = entry.getValue();
            ParameterDimension<?> dimension = RandomUtils.choice(dimensions, getRandom());
            final Object value = extractRandomValue(dimension.getValues());
            final List<ParameterSpace> subSpaces = dimension.getSubSpaces();
            final List<ParamSet> subParamSets = new ArrayList<>();
            for(ParameterSpace subSpace : subSpaces) {
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
        ParamSet paramSet = extractRandomParamSet(parameterSpace);
        iterationCount++;
        return paramSet;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public RandomSearchIterator setNumIterations(final int numIterations) {
        Assert.assertTrue(numIterations >= 0);
        this.numIterations = numIterations;
        return this;
    }

    public static class UnitTests {

        @Test
        public void testIteration() {
            ParameterSpace space = ParameterSpace.UnitTests.build2DContinuousSpace();
            final int limit = 10;
            RandomSearchIterator iterator = new RandomSearchIterator(space, limit);
            iterator.setRandom(ParameterSpace.UnitTests.buildRandom());
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
                "ParamSet{-a, \"0.4157204943935306\", -b, \"0.8187087126750541\"}\n"
                    + "ParamSet{-a, \"0.058503304403612566\", -b, \"0.6666091997383249\"}\n"
                    + "ParamSet{-a, \"0.3065178840223069\", -b, \"0.9395912589362401\"}\n"
                    + "ParamSet{-a, \"0.08798840101774008\", -b, \"0.5644485754368884\"}\n"
                    + "ParamSet{-a, \"0.35258737223772796\", -b, \"0.7733698785992328\"}\n"
                    + "ParamSet{-a, \"0.2814748369491896\", -b, \"0.8125731817327797\"}\n"
                    + "ParamSet{-a, \"0.00746354294055912\", -b, \"0.9953613928573914\"}\n"
                    + "ParamSet{-a, \"0.43383933414698683\", -b, \"0.8665760350974969\"}\n"
                    + "ParamSet{-a, \"0.006403325787859793\", -b, \"0.7633497173024331\"}\n"
                    + "ParamSet{-a, \"0.49233707140341276\", -b, \"0.5415311991124574\"}\n"
            );
        }
    }
}
