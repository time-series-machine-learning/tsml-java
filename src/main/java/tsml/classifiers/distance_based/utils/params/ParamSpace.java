package tsml.classifiers.distance_based.utils.params;

import com.beust.jcommander.internal.Lists;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.ddtw.DDTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.utils.params.dimensions.ContinuousParameterDimension;
import tsml.classifiers.distance_based.utils.params.distribution.Distribution;
import tsml.classifiers.distance_based.utils.params.distribution.UniformDistribution;
import tsml.classifiers.distance_based.utils.params.dimensions.DiscreteParameterDimension;
import tsml.classifiers.distance_based.utils.params.dimensions.ParameterDimension;
import weka.core.DistanceFunction;

public class ParamSpace {

    // 1 to many mapping of param name to list of param dimensions
    private Map<String, List<ParameterDimension<?>>> dimensionMap = new LinkedHashMap<>();

    public Map<String, List<ParameterDimension<?>>> getDimensionMap() {
        return dimensionMap;
    }

    @Override
    public String toString() {
        return String.valueOf(dimensionMap);
    }

    public ParamSpace add(String name, ParameterDimension<?> dimension) {
        getDimensionMap().computeIfAbsent(name, s -> new ArrayList<>()).add(dimension);
        return this;
    }

    public <A> ParamSpace add(String name, List<A> values) {
        return add(name, new DiscreteParameterDimension<A>(values));
    }

    public <A> ParamSpace add(String name, List<A> values, List<ParamSpace> subSpaces) {
        return add(name, new DiscreteParameterDimension<>(values, subSpaces));
    }

    public <A> ParamSpace add(String name, List<A> values, ParamSpace subSpace) {
        List<ParamSpace> list = new ArrayList<>(Collections.singletonList(subSpace));
        return add(name, values, list);
    }

    public <A> ParamSpace add(String name, Distribution<A> values) {
        return add(name, new ContinuousParameterDimension<A>(values));
    }

    public <A> ParamSpace add(String name, Distribution<A> values, List<ParamSpace> subSpaces) {
        return add(name, new ContinuousParameterDimension<>(values, subSpaces));
    }

    public <A> ParamSpace add(String name, Distribution<A> values, ParamSpace subSpace) {
        List<ParamSpace> list = new ArrayList<>(Collections.singletonList(subSpace));
        return add(name, values, list);
    }

    public List<ParameterDimension<?>> get(String name) {
        return getDimensionMap().get(name);
    }

    public static class UnitTests {

        private static int seed = 0;
        private static Random random = buildRandom();
        private static ParamSpace wParams = buildWParams();
        private static List<Integer> wParamValues = buildWParamValues();
        private static ParamSpace lParams = buildLParams();
        private static UniformDistribution eDist = buildEDist();
        private static UniformDistribution dDist = buildDDist();
        private static DiscreteParameterDimension<DistanceFunction> wDmParams = buildWDmParams();
        private static DiscreteParameterDimension<DistanceFunction> lDmParams = buildLDmParams();
        private static ParamSpace params = buildParams();

        public static Random buildRandom() {
            return new Random(seed);
        }

        public static List<Integer> buildWParamValues() {
            return Arrays.asList(1, 2, 3, 4, 5);
        }

        public static ParamSpace buildWParams() {
            ParamSpace wParams = new ParamSpace();
            List<Integer> wParamValues = buildWParamValues();
            wParams.add(DTW.getWarpingWindowFlag(), wParamValues);
            return wParams;
        }

        public static List<Integer> buildDummyValuesA() {
            return Arrays.asList(1,2,3,4,5);
        }

        public static List<Integer> buildDummyValuesB() {
            return Arrays.asList(6,7,8,9,10);
        }

        public static ParamSpace build2DDiscreteSpace() {
            ParamSpace space = new ParamSpace();
            List<Integer> a = buildDummyValuesA();
            List<Integer> b = buildDummyValuesB();
            space.add("a", a);
            space.add("b", b);
            return space;
        }

        public static UniformDistribution buildDummyDistributionA() {
            return new UniformDistribution(0, 0.5, random);
        }

        public static UniformDistribution buildDummyDistributionB() {
            return new UniformDistribution(0.5, 1, random);
        }

        public static ParamSpace build2DContinuousSpace() {
            ParamSpace space = new ParamSpace();
            UniformDistribution a = buildDummyDistributionA();
            UniformDistribution b = buildDummyDistributionB();
            space.add("a", a);
            space.add("b", b);
            return space;
        }

        public static UniformDistribution buildEDist() {
            UniformDistribution eDist = new UniformDistribution(buildRandom());
            eDist.setRandom(random);
            eDist.setMinAndMax(0, 0.25);
            return eDist;
        }

        public static UniformDistribution buildDDist() {
            UniformDistribution eDist = new UniformDistribution(buildRandom());
            eDist.setRandom(random);
            eDist.setMinAndMax(0.5, 1);
            return eDist;
        }

        public static ParamSpace buildLParams() {
            ParamSpace lParams = new ParamSpace();
            lParams.add(LCSSDistance.getEpsilonFlag(), buildEDist());
            lParams.add(LCSSDistance.getDeltaFlag(), buildDDist());
            return lParams;
        }

        public static DiscreteParameterDimension<DistanceFunction> buildWDmParams() {
            DiscreteParameterDimension<DistanceFunction> wDmParams = new DiscreteParameterDimension<>(
                Arrays.asList(new DTWDistance(), new DDTWDistance()));
            wDmParams.addSubSpace(buildWParams());
            return wDmParams;
        }

        public static DiscreteParameterDimension<DistanceFunction> buildLDmParams() {
            DiscreteParameterDimension<DistanceFunction> lDmParams = new DiscreteParameterDimension<>(
                Arrays.asList(new LCSSDistance()));
            lDmParams.addSubSpace(buildLParams());
            return lDmParams;
        }

        public static ParamSpace buildParams() {
            ParamSpace params = new ParamSpace();
            params.add(DistanceMeasureable.getDistanceFunctionFlag(), lDmParams);
            params.add(DistanceMeasureable.getDistanceFunctionFlag(), wDmParams);
            return params;
        }

        @Before
        public void setup() {
            random.setSeed(seed);
        }

        @Test
        public void testAddAndGetForListOfValues() {
            List<ParameterDimension<?>> valuesOut = wParams.get(DTW.getWarpingWindowFlag());
            Object value = valuesOut.get(0).getValues();
            Assert.assertEquals(value, wParamValues);
        }

        @Test
        public void testAddAndGetForDistributionOfValues() {
            List<ParameterDimension<?>> dimensions = lParams.get(LCSSDistance.getEpsilonFlag());
            for(ParameterDimension<?> dimension : dimensions) {
                Object values = dimension.getValues();
                Assert.assertEquals(values, eDist);
            }
            dimensions = lParams.get(LCSSDistance.getDeltaFlag());
            for(ParameterDimension<?> dimension : dimensions) {
                Object values = dimension.getValues();
                Assert.assertEquals(values, dDist);
            }
        }

        @Test
        public void testParamsToString() {
            System.out.println(params.toString());
            Assert.assertEquals(params.toString(), "{d=[{values=[LCSSDistance], subSpaces=[{e=[{values=UniformDistribution{min=0.0, max=0.25}}], d=[{values=UniformDistribution{min=0.5, max=1.0}}]}]}, {values=[DTWDistance, DDTWDistance], subSpaces=[{w=[{values=[1, 2, 3, 4, 5]}]}]}]}");
        }

        @Test
        public void testWParamsToString() {
            System.out.println(wParams.toString());
            Assert.assertEquals(wParams.toString(), "{w=[{values=[1, 2, 3, 4, 5]}]}");
        }

        @Test
        public void testLParamsToString() {
            System.out.println(lParams.toString());
            Assert.assertEquals(lParams.toString(), "{e=[{values=UniformDistribution{min=0.0, max=0.25}}], d=[{values=UniformDistribution{min=0.5, max=1.0}}]}");
        }

        @Test
        public void testWDmParamsToString() {
            System.out.println(wDmParams.toString());
            Assert.assertEquals(wDmParams.toString(), "{values=[DTWDistance, DDTWDistance], subSpaces=[{w=[{values=[1, 2, 3, 4, 5]}]}]}");
        }

        @Test
        public void testLDmParamsToString() {
            System.out.println(lDmParams.toString());
            Assert.assertEquals(lDmParams.toString(), "{values=[LCSSDistance], subSpaces=[{e=[{values=UniformDistribution{min=0.0, max=0.25}}], d=[{values=UniformDistribution{min=0.5, max=1.0}}]}]}");
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

}
