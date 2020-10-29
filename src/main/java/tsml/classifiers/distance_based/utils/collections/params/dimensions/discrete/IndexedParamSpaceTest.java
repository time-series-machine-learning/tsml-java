package tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete;

import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceTest;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class IndexedParamSpaceTest {

    @Test(expected = IllegalArgumentException.class)
    public void testNonDiscreteParameterDimensionException() {
        IndexedParamSpace space = new IndexedParamSpace(new ParamSpaceTest().buildLParams());
        for(int i = 0; i < space.size(); i++) {
            space.get(i);
        }
    }

    @Test
    public void testUniquePermutations() {
        IndexedParamSpace space = new IndexedParamSpace(new ParamSpaceTest().buildWParams());
        int size = space.size();
        Set<ParamSet> paramSets = new HashSet<>();
        for(int i = 0; i < size; i++) {
            ParamSet paramSet = space.get(i);
            boolean added = paramSets.add(paramSet);
//            System.out.println(paramSet);
            if(!added) {
                Assert.fail("duplicate parameter set: " + paramSet);
            }
        }
    }

    @Test
    public void testEquals() {
        ParamSpace wParams = new ParamSpaceTest().buildWParams();
        IndexedParamSpace a = new IndexedParamSpace(wParams);
        IndexedParamSpace b = new IndexedParamSpace(wParams);
        ParamSpace alt = new ParamSpace();
        alt.add("letters", new DiscreteParamDimension<>(Arrays.asList("a", "b", "c")));
        IndexedParamSpace c = new IndexedParamSpace(alt);
        IndexedParamSpace d = new IndexedParamSpace(alt);
        Assert.assertEquals(a, b);
        Assert.assertEquals(a.hashCode(), b.hashCode());
        Assert.assertEquals(c, d);
        Assert.assertEquals(c.hashCode(), c.hashCode());
        Assert.assertNotEquals(a, c);
        Assert.assertNotEquals(a.hashCode(), c.hashCode());
        Assert.assertNotEquals(b, c);
        Assert.assertNotEquals(b.hashCode(), c.hashCode());
        Assert.assertNotEquals(a, d);
        Assert.assertNotEquals(a.hashCode(), d.hashCode());
        Assert.assertNotEquals(b, d);
        Assert.assertNotEquals(b.hashCode(), d.hashCode());
    }
}
