package tsml.classifiers.distance_based.utils.params.tmp;

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
import tsml.classifiers.distance_based.utils.params.distribution.UniformDistribution;
import weka.core.DistanceFunction;

public class ParameterSpace {

    // 1 to many mapping of param name to list of param dimensions
    private Map<String, List<ParameterDimension<?>>> dimensionMap = new LinkedHashMap<>();

    public Map<String, List<ParameterDimension<?>>> getDimensionMap() {
        return dimensionMap;
    }

    @Override
    public String toString() {
        return String.valueOf(dimensionMap);
    }

    public void add(String name, ParameterDimension<?> dimension) {
        getDimensionMap().computeIfAbsent(name, s -> new ArrayList<>()).add(dimension);
    }

    public <A> void add(String name, A values) {
        add(name, new ParameterDimension<A>(values));
    }

    public <A> void add(String name, A values, List<ParameterSpace> subSpaces) {
        add(name, new ParameterDimension<>(values, subSpaces));
    }

    public <A> void add(String name, A values, ParameterSpace subSpace) {
        List<ParameterSpace> list = new ArrayList<>(Collections.singletonList(subSpace));
        add(name, values, list);
    }

    public List<ParameterDimension<?>> get(String name) {
        return getDimensionMap().get(name);
    }

    public static class UnitTests {

        private static int seed = 0;
        private static Random random = buildRandom();
        private static ParameterSpace wParams = buildWParams();
        private static List<Integer> wParamValues = buildWParamValues();
        private static ParameterSpace lParams = buildLParams();
        private static UniformDistribution eDist = buildEDist();
        private static UniformDistribution dDist = buildDDist();
        private static DiscreteParameterDimension<DistanceFunction> wDmParams = buildWDmParams();
        private static DiscreteParameterDimension<DistanceFunction> lDmParams = buildLDmParams();
        private static ParameterSpace params = buildParams();

        public static Random buildRandom() {
            return new Random(seed);
        }

        public static List<Integer> buildWParamValues() {
            return Arrays.asList(1, 2, 3, 4, 5);
        }

        public static ParameterSpace buildWParams() {
            ParameterSpace wParams = new ParameterSpace();
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

        public static ParameterSpace build2DDiscreteSpace() {
            ParameterSpace space = new ParameterSpace();
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

        public static ParameterSpace build2DContinuousSpace() {
            ParameterSpace space = new ParameterSpace();
            UniformDistribution a = buildDummyDistributionA();
            UniformDistribution b = buildDummyDistributionB();
            space.add("a", a);
            space.add("b", b);
            return space;
        }

        public static UniformDistribution buildEDist() {
            UniformDistribution eDist = new UniformDistribution();
            eDist.setRandom(random);
            eDist.setMinAndMax(0, 0.25);
            return eDist;
        }

        public static UniformDistribution buildDDist() {
            UniformDistribution eDist = new UniformDistribution();
            eDist.setRandom(random);
            eDist.setMinAndMax(0.5, 1);
            return eDist;
        }

        public static ParameterSpace buildLParams() {
            ParameterSpace lParams = new ParameterSpace();
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

        public static ParameterSpace buildParams() {
            ParameterSpace params = new ParameterSpace();
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
            ParameterSpace a = wParams;
            ParameterSpace b = wParams;
            ParameterSpace c = lParams;
            ParameterSpace d = lParams;
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
