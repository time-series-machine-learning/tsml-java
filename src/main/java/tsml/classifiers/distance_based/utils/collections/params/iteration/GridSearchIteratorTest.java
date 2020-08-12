package tsml.classifiers.distance_based.utils.collections.params.iteration;

import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceTest;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class GridSearchIteratorTest {

    @Test
    public void testIteration() {
        ParamSpace space = new ParamSpaceTest().build2DDiscreteSpace();
        GridSearchIterator iterator = new GridSearchIterator(space);
        StringBuilder stringBuilder = new StringBuilder();
        while(iterator.hasNext()) {
            ParamSet paramSet = iterator.next();
            stringBuilder.append(paramSet);
            stringBuilder.append("\n");
        }
//        System.out.println(stringBuilder.toString());
        Assert.assertEquals(stringBuilder.toString(),
            "-a, \"1\", -b, \"6\"\n"
                + "-a, \"2\", -b, \"6\"\n"
                + "-a, \"3\", -b, \"6\"\n"
                + "-a, \"4\", -b, \"6\"\n"
                + "-a, \"5\", -b, \"6\"\n"
                + "-a, \"1\", -b, \"7\"\n"
                + "-a, \"2\", -b, \"7\"\n"
                + "-a, \"3\", -b, \"7\"\n"
                + "-a, \"4\", -b, \"7\"\n"
                + "-a, \"5\", -b, \"7\"\n"
                + "-a, \"1\", -b, \"8\"\n"
                + "-a, \"2\", -b, \"8\"\n"
                + "-a, \"3\", -b, \"8\"\n"
                + "-a, \"4\", -b, \"8\"\n"
                + "-a, \"5\", -b, \"8\"\n"
                + "-a, \"1\", -b, \"9\"\n"
                + "-a, \"2\", -b, \"9\"\n"
                + "-a, \"3\", -b, \"9\"\n"
                + "-a, \"4\", -b, \"9\"\n"
                + "-a, \"5\", -b, \"9\"\n"
                + "-a, \"1\", -b, \"10\"\n"
                + "-a, \"2\", -b, \"10\"\n"
                + "-a, \"3\", -b, \"10\"\n"
                + "-a, \"4\", -b, \"10\"\n"
                + "-a, \"5\", -b, \"10\"\n"
        );
    }
}
