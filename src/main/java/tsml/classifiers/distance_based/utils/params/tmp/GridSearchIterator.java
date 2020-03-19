package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.Iterator;
import org.junit.Assert;
import org.junit.Test;
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
        numIterations = indexedParameterSpace.size();
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

        @Test
        public void testIteration() {
            ParameterSpace space = ParameterSpace.UnitTests.build2DDiscreteSpace();
            GridSearchIterator iterator = new GridSearchIterator(space);
            StringBuilder stringBuilder = new StringBuilder();
            while(iterator.hasNext()) {
                ParamSet paramSet = iterator.next();
                stringBuilder.append(paramSet);
                stringBuilder.append("\n");
            }
            System.out.println(stringBuilder.toString());
            Assert.assertEquals(stringBuilder.toString(),
                "ParamSet{-a, \"1\", -b, \"6\"}\n"
                + "ParamSet{-a, \"2\", -b, \"6\"}\n"
                + "ParamSet{-a, \"3\", -b, \"6\"}\n"
                + "ParamSet{-a, \"4\", -b, \"6\"}\n"
                + "ParamSet{-a, \"5\", -b, \"6\"}\n"
                + "ParamSet{-a, \"1\", -b, \"7\"}\n"
                + "ParamSet{-a, \"2\", -b, \"7\"}\n"
                + "ParamSet{-a, \"3\", -b, \"7\"}\n"
                + "ParamSet{-a, \"4\", -b, \"7\"}\n"
                + "ParamSet{-a, \"5\", -b, \"7\"}\n"
                + "ParamSet{-a, \"1\", -b, \"8\"}\n"
                + "ParamSet{-a, \"2\", -b, \"8\"}\n"
                + "ParamSet{-a, \"3\", -b, \"8\"}\n"
                + "ParamSet{-a, \"4\", -b, \"8\"}\n"
                + "ParamSet{-a, \"5\", -b, \"8\"}\n"
                + "ParamSet{-a, \"1\", -b, \"9\"}\n"
                + "ParamSet{-a, \"2\", -b, \"9\"}\n"
                + "ParamSet{-a, \"3\", -b, \"9\"}\n"
                + "ParamSet{-a, \"4\", -b, \"9\"}\n"
                + "ParamSet{-a, \"5\", -b, \"9\"}\n"
                + "ParamSet{-a, \"1\", -b, \"10\"}\n"
                + "ParamSet{-a, \"2\", -b, \"10\"}\n"
                + "ParamSet{-a, \"3\", -b, \"10\"}\n"
                + "ParamSet{-a, \"4\", -b, \"10\"}\n"
                + "ParamSet{-a, \"5\", -b, \"10\"}\n"
            );
        }
    }
}
