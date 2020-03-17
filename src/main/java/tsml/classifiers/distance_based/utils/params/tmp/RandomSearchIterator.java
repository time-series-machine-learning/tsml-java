package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.Deque;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Assert;
import sun.awt.image.ImageWatched.Link;
import tsml.classifiers.distance_based.utils.BaseRandom;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.distribution.Distribution;
import tsml.classifiers.distance_based.utils.random.RandomUtils;
import utilities.Utilities;

/**
 * Purpose: randomly iterate over a parameter space. The random iteration occurs with replacement therefore the same
 * parameter set may be hit more than once.
 * <p>
 * Contributors: goastler
 */
public class RandomSearchIterator extends BaseRandom implements Iterator<ParamSet> {

    private ParameterSpace parameterSpace;
    private int numIterations = 100;

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
        return numIterations >= 0;
    }

    private void abc(ParameterDimension<?> dimension) {
        final List<ParameterSpace> subSpaces = dimension.getSubSpaces();
        Object value;
        if(dimension instanceof ContinuousParameterDimension) {
            Distribution<?> distribution = ((ContinuousParameterDimension<?>) dimension).getDistribution();
            distribution.setRandom(getRandom());
            value = distribution.sample();
            distribution.setRandom(null);
        } else if(dimension instanceof DiscreteParameterDimension) {

        }
    }

    @Override
    public ParamSet next() {
        final ParamSet param = new ParamSet();
        final Random random = getRandom();
        LinkedList<ParameterSpace> stack = new LinkedList<>();
        stack.push(this.parameterSpace);
        while(!stack.isEmpty()) {
            ParameterSpace parameterSpace = stack.peek();
            parameterSpace.
        }


        for(Map.Entry<String, List<ParameterDimension<?>>> entry : parameterSpace.getDimensionMap().entrySet()) {
            List<ParameterDimension<?>> parameterDimensions = entry.getValue();
            ParameterDimension<?> parameterDimension = RandomUtils.choice(parameterDimensions, random);

            for(ParameterDimension<?> paramValues : parameterDimensions) {
                int size = paramValues.size();
                index -= size;
                if(index < 0) {
                    Object paramValue = paramValues.get(index + size);
                    try {
                        paramValue = Utilities.deepCopy(paramValue); // must copy objects otherwise every paramset
                        // uses the same object reference!
                    } catch(Exception e) {
                        throw new IllegalStateException("cannot copy value");
                    }
                    param.add(entry.getKey(), paramValue);
                    break;
                }
            }
            if(index >= 0) {
                throw new IndexOutOfBoundsException();
            }
        }
        return param;
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
