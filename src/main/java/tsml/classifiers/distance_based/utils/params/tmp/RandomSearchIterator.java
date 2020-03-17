package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Assert;
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

    private Object extractRandomValue(ParameterDimension<?> dimension) {
        Object value;
        if(dimension instanceof ContinuousParameterDimension) {
            ContinuousParameterDimension<?> continuousParameterDimension = (ContinuousParameterDimension<?>) dimension;
            Distribution<?> distribution = continuousParameterDimension.getValues();
            distribution.setRandom(getRandom());
            value = distribution.sample();
            distribution.setRandom(null);
        } else if(dimension instanceof DiscreteParameterDimension) {
            DiscreteParameterDimension<?> discreteParameterDimension = (DiscreteParameterDimension<?>) dimension;
            List<?> values = discreteParameterDimension.getValues();
            value = RandomUtils.choice(values, getRandom());
        } else {
            throw new IllegalArgumentException("cannot handle type {" + dimension.getClass() + "} for dimension");
        }
        return value;
    }

    private ParamSet extractRandomParamSet(ParameterSpace parameterSpace) {
        final Map<String, List<ParameterDimension<?>>> dimensionMap = parameterSpace.getDimensionMap();
        final ParamSet paramSet = new ParamSet();
        for(Map.Entry<String, List<ParameterDimension<?>>> entry : dimensionMap.entrySet()) {
            List<ParameterDimension<?>> dimensions = entry.getValue();
            ParameterDimension<?> dimension = RandomUtils.choice(dimensions, getRandom());
            final Object value = extractRandomValue(dimension);
            final List<ParameterSpace> subSpaces = dimension.getSubSpaces();
            final List<ParamSet> subParamSets = new ArrayList<>();
            for(ParameterSpace subSpace : subSpaces) {
                final ParamSet subParamSet = extractRandomParamSet(subSpace);
                subParamSets.add(subParamSet);
            }
            final String name = entry.getKey();
            paramSet.add(name,value, subParamSets);
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
}
