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

    private IndexedParameterSpace indexedParameterSpace;
    private int iterationCount = 0;
    private int numIterations = -1;

    public GridSearchIterator(final ParameterSpace parameterSpace) {
        setParameterSpace(parameterSpace);
    }

    public ParameterSpace getParameterSpace() {
        return indexedParameterSpace.getParameterSpace();
    }

    public IndexedParameterSpace getIndexedParameterSpace() {
        return indexedParameterSpace;
    }

    protected void setParameterSpace(
        final ParameterSpace parameterSpace) {
        indexedParameterSpace = new IndexedParameterSpace(parameterSpace);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{iterationCount=" + iterationCount + ", parameterSpace=" + indexedParameterSpace.getParameterSpace().toString() + "}";
    }

    @Override
    public boolean hasNext() {
        return iterationCount < numIterations;
    }

    @Override
    public ParamSet next() {
        ParamSet paramSet = getIndexedParameterSpace().get(iterationCount);
        iterationCount++;
        return paramSet;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public void setNumIterations(final int numIterations) {
        Assert.assertTrue(numIterations >= 0);
        this.numIterations = numIterations;
    }

    public static class UnitTests {

    }
}
