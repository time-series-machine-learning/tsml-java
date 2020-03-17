package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.Iterator;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.params.ParamSet;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class GridSearchIterator implements Iterator<ParamSet> {

    private ParameterSpace parameterSpace;

    public GridSearchIterator(final ParameterSpace parameterSpace) {
        setParameterSpace(parameterSpace);
    }

    public ParameterSpace getParameterSpace() {
        return parameterSpace;
    }

    protected void setParameterSpace(
        final ParameterSpace parameterSpace) {
        Assert.assertNotNull(parameterSpace);
        this.parameterSpace = parameterSpace;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public ParamSet next() {
        return null;
    }
}
