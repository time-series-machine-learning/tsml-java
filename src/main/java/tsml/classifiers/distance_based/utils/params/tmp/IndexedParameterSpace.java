package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
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
import tsml.classifiers.distance_based.utils.collections.IndexedCollection;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.distribution.UniformDistribution;
import utilities.ArrayUtilities;
import utilities.Utilities;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class IndexedParameterSpace implements IndexedCollection<ParamSet> {

    private ParameterSpace parameterSpace;

    public IndexedParameterSpace(
        final ParameterSpace parameterSpace) {
        setParameterSpace(parameterSpace);
    }

    public ParameterSpace getParameterSpace() {
        return parameterSpace;
    }

    public IndexedParameterSpace setParameterSpace(
        final ParameterSpace parameterSpace) {
        Assert.assertNotNull(parameterSpace);
        this.parameterSpace = parameterSpace;
        return this;
    }

    @Override
    public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(o == null || getClass() != o.getClass()) {
            return false;
        }
        final IndexedParameterSpace space = (IndexedParameterSpace) o;
        return getParameterSpace().equals(space.getParameterSpace());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getParameterSpace());
    }

    @Override
    public ParamSet get(final int index) {
        return get(getParameterSpace(), index);
    }

    @Override
    public int size() {
        return size(getParameterSpace());
    }


    /**
     * get the value of the parameter dimension at the given index
     * @param dimension
     * @param index
     * @return
     */
    public static Object get(ParameterDimension<?> dimension, int index) {
        Object values = dimension.getValues();
        if(values instanceof List<?>) {
            final List<Integer> allSizes = sizes(dimension);
            final List<Integer> indices = Permutations.fromPermutation(index, allSizes);
            final Integer valueIndex = indices.remove(0);
            List<?> valuesList = (List<?>) values;
            Object value = valuesList.get(valueIndex);
            try {
                value = Utilities.deepCopy(value); // must copy objects otherwise every paramset
                // uses the same object reference!
            } catch(Exception e) {
                throw new IllegalStateException("cannot copy value");
            }
            ParamSet subParamSet = get(dimension.getSubSpaces(), indices);
            if(!subParamSet.isEmpty()) {
                if(value instanceof ParamHandler) {
                    ((ParamHandler) value).setParams(subParamSet);
                } else {
                    throw new IllegalStateException("{" + value.toString() + "} isn't an instance of ParamHandler, cannot "
                        + "set params");
                }
            }
            return value;
        } else {
            throw new IllegalArgumentException();
        }
    }

    /**
     * get the paramset corresponding to the given indices
     * @param spaces
     * @param indices
     * @return
     */
    public static ParamSet get(final List<ParameterSpace> spaces, final List<Integer> indices) {
        Assert.assertEquals(spaces.size(), indices.size());
        final ParamSet overallParamSet = new ParamSet();
        for(int i = 0; i < spaces.size(); i++) {
            final ParamSet paramSet = get(spaces.get(i), indices.get(i));
            overallParamSet.addAll(paramSet);
        }
        return overallParamSet;
    }

    /**
     * get the paramset given the permutation index
     * @param space
     * @param index
     * @return
     */
    public static ParamSet get(ParameterSpace space, int index) {
        final List<Integer> sizes = sizes(space);
        final List<Integer> indices = ArrayUtilities.fromPermutation(index, sizes);
        int i = 0;
        ParamSet param = new ParamSet();
        for(Map.Entry<String, List<ParameterDimension<?>>> entry : space.getDimensionMap().entrySet()) {
            index = indices.get(i);
            List<ParameterDimension<?>> dimensions = entry.getValue();
            Object value = get(dimensions, index);
            param.add(entry.getKey(), value);
            i++;
        }
        return param;
    }

    /**
     * get the object at the given index across several dimensions
     * @param dimensions
     * @param index
     * @return
     */
    public static Object get(List<ParameterDimension<?>> dimensions, int index) {
        for(ParameterDimension<?> dimension : dimensions) {
            int size = size(dimension);
            index -= size;
            if(index < 0) {
                index += size;
                return get(dimension, index);
            }
        }
        throw new IndexOutOfBoundsException();
    }

    public static int size(ParameterSpace space) {
        final List<Integer> sizes = sizes(space);
        return Permutations.numPermutations(sizes);
    }

    public static int size(ParameterDimension<?> dimension) {
        return Permutations.numPermutations(sizes(dimension));
    }

    public static List<Integer> sizesParameterSpace(List<ParameterSpace> spaces) {
        return Utilities.convert(spaces, IndexedParameterSpace::size);
    }

    public static int size(List<ParameterDimension<?>> dimensions) {
        return Permutations.numPermutations(sizesParameterDimension(dimensions));
    }

    public static List<Integer> sizesParameterDimension(List<ParameterDimension<?>> dimensions) {
        return Utilities.convert(dimensions, IndexedParameterSpace::size);
    }

    public static List<Integer> sizes(ParameterSpace space) {
        final Map<String, List<ParameterDimension<?>>> dimensionMap = space.getDimensionMap();
        final List<Integer> sizes = new ArrayList<>();
        for(final Map.Entry<String, List<ParameterDimension<?>>> entry : dimensionMap.entrySet()) {
            final List<ParameterDimension<?>> dimensions = entry.getValue();
            int size = 0;
            for(ParameterDimension<?> dimension : dimensions) {
                size += size(dimension);
            }
            sizes.add(size);
        }
        return sizes;
    }

    public static List<Integer> sizes(ParameterDimension<?> dimension) {
        List<Integer> sizes = sizesParameterSpace(dimension.getSubSpaces());
        Object values = dimension.getValues();
        if(values instanceof List<?>) {
            int size = ((List<?>) values).size();
            sizes.add(0, size);
        } else {
            throw new IllegalArgumentException("cannot handle dimension type");
        }
        return sizes;
    }

    public static class UnitTests {

        public static void main(String[] args) {

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

        @Test(expected = IllegalArgumentException.class)
        public void testNonDiscreteParameterDimensionException() {
            IndexedParameterSpace space = new IndexedParameterSpace(lParams);
            for(int i = 0; i < space.size(); i++) {
                space.get(i);
            }
        }

        @Test
        public void testUniquePermutations() {
            IndexedParameterSpace space = new IndexedParameterSpace(wParams);
            int size = space.size();
            Set<ParamSet> paramSets = new HashSet<>();
            for(int i = 0; i < size; i++) {
                ParamSet paramSet = space.get(i);
                boolean added = paramSets.add(paramSet);
                if(!added) {
                    Assert.fail("duplicate parameter set: " + paramSet);
                }
            }
        }

        @Test
        public void testEquals() {
            IndexedParameterSpace a = new IndexedParameterSpace(wParams);
            IndexedParameterSpace b = new IndexedParameterSpace(wParams);
            ParameterSpace alt = new ParameterSpace();
            alt.add("letters", new DiscreteParameterDimension<>(Arrays.asList("a", "b", "c")));
            IndexedParameterSpace c = new IndexedParameterSpace(alt);
            IndexedParameterSpace d = new IndexedParameterSpace(alt);
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
}
