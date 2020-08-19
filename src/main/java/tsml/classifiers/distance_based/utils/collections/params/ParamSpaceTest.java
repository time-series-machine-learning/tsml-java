package tsml.classifiers.distance_based.utils.collections.params;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.WarpingDistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.DiscreteParameterDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParameterDimension;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import weka.core.DistanceFunction;

import static tsml.classifiers.distance_based.distances.dtw.DTWDistanceConfigs.newDDTWDistance;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ParamSpaceTest {

    public ParamSpaceTest() {
        // for users outside of this class, make sure we're all setup for tests
        before();
    }

    private int seed;
    private Random random;
    private ParamSpace wParams;
    private List<Integer> wParamValues;
    private ParamSpace lParams;
    private UniformDoubleDistribution eDist;
    private UniformDoubleDistribution dDist;
    private DiscreteParameterDimension<DistanceFunction> wDmParams;
    private DiscreteParameterDimension<DistanceFunction> lDmParams;
    private ParamSpace params;

    @Before
    public void before() {
        seed = 0;
        random = new Random(seed);
        wParamValues = buildWParamValues();
        wParams = buildWParams();
        lParams = buildLParams();
        eDist = buildEDist();
        dDist = buildDDist();
        wDmParams = buildWDmParams();
        lDmParams = buildLDmParams();
        params = buildParams();
    }

    public Random buildRandom() {
        return new Random(seed);
    }

    public List<Integer> buildWParamValues() {
        return Arrays.asList(1, 2, 3, 4, 5);
    }

    public ParamSpace buildWParams() {
        ParamSpace wParams = new ParamSpace();
        List<Integer> wParamValues = buildWParamValues();
        wParams.add(WarpingDistanceMeasure.WINDOW_SIZE_FLAG, wParamValues);
        return wParams;
    }

    public List<Integer> buildDummyValuesA() {
        return Arrays.asList(1,2,3,4,5);
    }

    public List<Integer> buildDummyValuesB() {
        return Arrays.asList(6,7,8,9,10);
    }

    public ParamSpace build2DDiscreteSpace() {
        ParamSpace space = new ParamSpace();
        List<Integer> a = buildDummyValuesA();
        List<Integer> b = buildDummyValuesB();
        space.add("a", a);
        space.add("b", b);
        return space;
    }

    public UniformDoubleDistribution buildDummyDistributionA() {
        UniformDoubleDistribution u = new UniformDoubleDistribution(0d, 0.5d);
        u.setRandom(buildRandom());
        return u;
    }

    public UniformDoubleDistribution buildDummyDistributionB() {
        UniformDoubleDistribution u = new UniformDoubleDistribution(0.5d, 1d);
        u.setRandom(buildRandom());
        return u;
    }

    public ParamSpace build2DContinuousSpace() {
        ParamSpace space = new ParamSpace();
        UniformDoubleDistribution a = buildDummyDistributionA();
        UniformDoubleDistribution b = buildDummyDistributionB();
        space.add("a", a);
        space.add("b", b);
        return space;
    }

    public UniformDoubleDistribution buildEDist() {
        UniformDoubleDistribution eDist = new UniformDoubleDistribution();
        eDist.setRandom(buildRandom());
        eDist.setMinAndMax(0d, 0.25);
        return eDist;
    }

    public UniformDoubleDistribution buildDDist() {
        UniformDoubleDistribution eDist = new UniformDoubleDistribution();
        eDist.setRandom(buildRandom());
        eDist.setMinAndMax(0.5, 1d);
        return eDist;
    }

    public ParamSpace buildLParams() {
        ParamSpace lParams = new ParamSpace();
        lParams.add(LCSSDistance.EPSILON_FLAG, buildEDist());
        lParams.add(LCSSDistance.WINDOW_SIZE_FLAG, buildDDist());
        return lParams;
    }

    public DiscreteParameterDimension<DistanceFunction> buildWDmParams() {
        DiscreteParameterDimension<DistanceFunction> wDmParams = new DiscreteParameterDimension<>(
            Arrays.asList(new DTWDistance(), newDDTWDistance()));
        wDmParams.addSubSpace(buildWParams());
        return wDmParams;
    }

    public DiscreteParameterDimension<DistanceFunction> buildLDmParams() {
        DiscreteParameterDimension<DistanceFunction> lDmParams = new DiscreteParameterDimension<>(
            Arrays.asList(new LCSSDistance()));
        lDmParams.addSubSpace(buildLParams());
        return lDmParams;
    }

    public ParamSpace buildParams() {
        ParamSpace params = new ParamSpace();
        params.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, lDmParams);
        params.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, wDmParams);
        return params;
    }

    @Test
    public void testAddAndGetForListOfValues() {
        List<ParameterDimension<?>> valuesOut = wParams.get(WarpingDistanceMeasure.WINDOW_SIZE_FLAG);
        Object value = valuesOut.get(0).getValues();
        Assert.assertEquals(value, wParamValues);
    }

    @Test
    public void testAddAndGetForDistributionOfValues() {
        List<ParameterDimension<?>> dimensions = lParams.get(LCSSDistance.EPSILON_FLAG);
        for(ParameterDimension<?> dimension : dimensions) {
            Object values = dimension.getValues();
            Assert.assertEquals(values, eDist);
        }
        dimensions = lParams.get(LCSSDistance.WINDOW_SIZE_FLAG);
        for(ParameterDimension<?> dimension : dimensions) {
            Object values = dimension.getValues();
            Assert.assertEquals(values, dDist);
        }
    }

    @Test
    public void testParamsToString() {
//        System.out.println(params.toString());
        Assert.assertEquals(params.toString(), "{d=[{values=[LCSSDistance -e \"0.01\" -ws \"-1\"], "
            + "subSpaces=[{e=[{values=UniformDoubleDistribution{min=0.0, max=0.25}}], "
            + "ws=[{values=UniformDoubleDistribution{min=0.5, max=1.0}}]}]}, {values=[DTWDistance -ws \"-1\", DDTWDistance -d \"tsml.classifiers.distance_based.distances.dtw.DTWDistance -ws \"-1\"\" -t \"tsml.transformers.CachedTransformer -f tsml.transformers.Derivative\"], "
            + "subSpaces=[{ws=[{values=[1, 2, 3, 4, 5]}]}]}]}");
    }

    @Test
    public void testWParamsToString() {
//        System.out.println(wParams.toString());
        Assert.assertEquals(wParams.toString(), "{ws=[{values=[1, 2, 3, 4, 5]}]}");
    }

    @Test
    public void testLParamsToString() {
//        System.out.println(lParams.toString());
        Assert.assertEquals(lParams.toString(), "{e=[{values=UniformDoubleDistribution{min=0.0, max=0.25}}], "
            + "ws=[{values=UniformDoubleDistribution{min=0.5, max=1.0}}]}");
    }

    @Test
    public void testWDmParamsToString() {
//        System.out.println(wDmParams.toString());
        Assert.assertEquals(wDmParams.toString(), "{values=[DTWDistance -ws \"-1\", DDTWDistance -d \"tsml.classifiers.distance_based.distances.dtw.DTWDistance -ws \"-1\"\" -t \"tsml.transformers.CachedTransformer -f tsml.transformers.Derivative\"], subSpaces=[{ws=[{values=[1, "
            + "2, 3, 4, 5]}]}]}");
    }

    @Test
    public void testLDmParamsToString() {
//        System.out.println(lDmParams.toString());
        Assert.assertEquals(lDmParams.toString(), "{values=[LCSSDistance -e \"0.01\" -ws \"-1\"], "
            + "subSpaces=[{e=[{values=UniformDoubleDistribution{min=0.0, max=0.25}}], "
            + "ws=[{values=UniformDoubleDistribution{min=0.5, max=1.0}}]}]}");
    }

    @Test
    public void testEquals() {
        ParamSpace a = wParams;
        ParamSpace b = wParams;
        ParamSpace c = lParams;
        ParamSpace d = lParams;
        Assert.assertEquals(a, b);
        Assert.assertEquals(c, d);
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
