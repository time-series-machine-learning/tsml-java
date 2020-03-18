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
import org.junit.BeforeClass;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.ddtw.DDTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.utils.params.distribution.UniformDistribution;

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

        public static void main(String[] args) {
            printToStrings();
        }

        private static int seed = 0;
        private static Random random = new Random();
        private static ParameterSpace wParams;
        private static List<Integer> wParamValues;
        private static ParameterSpace lParams;
        private static UniformDistribution eDist;
        private static UniformDistribution dDist;
        private static DiscreteParameterDimension<? extends BaseDistanceMeasure> wDmParams;
        private static DiscreteParameterDimension<? extends BaseDistanceMeasure> lDmParams;
        private static ParameterSpace params;

        @BeforeClass
        public static void setupOnce() {
            // build dtw / ddtw params
            wParams = new ParameterSpace();
            wParamValues = Arrays.asList(1, 2, 3, 4, 5);
            wParams.add(DTW.getWarpingWindowFlag(), wParamValues);
            lParams = new ParameterSpace();
            eDist = new UniformDistribution();
            eDist.setRandom(random);
            eDist.setMinAndMax(0, 0.25);
            lParams.add(LCSSDistance.getEpsilonFlag(), eDist);
            dDist = new UniformDistribution();
            dDist.setRandom(random);
            dDist.setMinAndMax(0.5, 1);
            lParams.add(LCSSDistance.getDeltaFlag(), dDist);
            // build dtw / ddtw space
            wDmParams = new DiscreteParameterDimension<>(
                Arrays.asList(new DTWDistance(), new DDTWDistance()));
            wDmParams.addSubSpace(wParams);
            // build lcss space
            lDmParams = new DiscreteParameterDimension<>(
                Arrays.asList(new LCSSDistance()));
            lDmParams.addSubSpace(lParams);
            // build overall space including ddtw, dtw and lcss WITH corresponding param spaces
            params = new ParameterSpace();
            params.add(DistanceMeasureable.getDistanceFunctionFlag(), lDmParams);
            params.add(DistanceMeasureable.getDistanceFunctionFlag(), wDmParams);
        }

        @Before
        public void setup() {
            random.setSeed(seed);
        }

        public static void printToStrings() {
            UnitTests unitTests = new UnitTests();
            setupOnce();
            unitTests.setup();
            System.out.println(wParams);
            System.out.println(lParams);
            System.out.println(wDmParams);
            System.out.println(lDmParams);
            System.out.println(params);
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
            Assert.assertEquals(params.toString(), "{d=[{values=[LCSSDistance], subSpaces=[{e=[{values=UniformDistribution{min=0.0, max=0.25}}], d=[{values=UniformDistribution{min=0.5, max=1.0}}]}]}, {values=[DTWDistance, DDTWDistance], subSpaces=[{w=[{values=[1, 2, 3, 4, 5]}]}]}]}");
        }

        @Test
        public void testWParamsToString() {
            Assert.assertEquals(wParams.toString(), "{w=[{values=[1, 2, 3, 4, 5]}]}");
        }

        @Test
        public void testLParamsToString() {
            Assert.assertEquals(lParams.toString(), "{e=[{values=UniformDistribution{min=0.0, max=0.25}}], d=[{values=UniformDistribution{min=0.5, max=1.0}}]}");
        }

        @Test
        public void testWDmParamsToString() {
            Assert.assertEquals(wDmParams.toString(), "{values=[DTWDistance, DDTWDistance], subSpaces=[{w=[{values=[1, 2, 3, 4, 5]}]}]}");
        }

        @Test
        public void testLDmParamsToString() {
            Assert.assertEquals(lDmParams.toString(), "{values=[LCSSDistance], subSpaces=[{e=[{values=UniformDistribution{min=0.0, max=0.25}}], d=[{values=UniformDistribution{min=0.5, max=1.0}}]}]}");
        }

    }

}
