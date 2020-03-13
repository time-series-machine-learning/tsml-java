package tsml.classifiers.distance_based.utils.params;

import com.google.common.collect.Range;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.ddtw.DDTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.distances.wddtw.WDDTWDistance;
import tsml.classifiers.distance_based.distances.wdtw.WDTW;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.classifiers.distance_based.proximity.RandomSource;
import utilities.ArrayUtilities;
import utilities.Utilities;
import tsml.classifiers.distance_based.utils.collections.DefaultList;

/**
 * Purpose: holds a mapping of parameter names to their corresponding values, where the values are stored as a
 * ParamValues object to allow for sub parameter spaces. Each ParamValues object stores a mapping of String (name) to
 * ParamValues list. ParamValues holds a list of raw values and a list of ParamSpace, facilitating subspaces. This
 * allows several values to share a sub space if required. As ParamSpace stores a list of ParamValues as the mapping,
 * you could have some values which share sub space A and another set of values sharing sub space B, all mapped under
 * the same parameter. This is designed to be very flexible to handle any parameter space situation.
 *
 * Contributors: goastler
 */
public class ParamSpace implements DefaultList<ParamSet> { // todo don't extend list

    private Map<String, List<ParamValues>> paramsMap = new LinkedHashMap<>(); // 1-many mapping of parameter names

    public static void main(String[] args) {
        ParamSpace params = new ParamSpace();
        ParamSpace wParams = new ParamSpace();
        wParams.add(DTW.getWarpingWindowFlag(), new ParamValues(Arrays.asList(1, 2, 3, 4, 5)));
        params.add(DistanceMeasureable.getDistanceFunctionFlag(),
            new ParamValues(Arrays.asList(new DTWDistance(), new DDTWDistance()),
                Arrays.asList(wParams)));
        ParamSpace lParams = new ParamSpace();
        lParams.add(WDTW.getGFlag(), new ParamValues(Range.closed(1D, 5D)));
        lParams.add(WDTW.getGFlag(), new ParamValues(Arrays.asList(1D, 2D, 3D)));
        lParams.add(LCSSDistance.getEpsilonFlag(), new ParamValues(Arrays.asList(1D, 2D, 3D, 4D)));
        params.add(DistanceMeasureable.getDistanceFunctionFlag(),
            new ParamValues(Arrays.asList(new WDTWDistance(), new WDDTWDistance()),
                Arrays.asList(lParams)));
        int size;
        size = wParams.size();
        size = lParams.size();
        size = params.size();
        for(int i = 0; i < size; i++) {
            //            System.out.println(i);
            ParamSet param = params.get(i);
            System.out.println(param);
        }

    }

    // todo remove function

    /**
     * gets a list of the sizes of each parameter.
     * @return
     */
    public List<Integer> getBins() {
        List<Integer> bins = new ArrayList<>();
        Iterator<Map.Entry<String, List<ParamValues>>> iterator = paramsMap.entrySet().iterator();
        for(int i = 0; i < paramsMap.size(); i++) {
            Map.Entry<String, List<ParamValues>> entry = iterator.next();
            int size = 0;
            for(ParamValues paramValues : entry.getValue()) {
                size += paramValues.size();
            }
            bins.add(size);
        }
        return bins;
    }

    /**
     * gets a ParamSet for the corresponding index in the ParamSpace.
     * @param index
     * @return
     */
    public ParamSet get(int index) {
        List<Integer> indices = ArrayUtilities.fromPermutation(index, getBins());
        int i = 0;
        ParamSet param = new ParamSet();
        for(Map.Entry<String, List<ParamValues>> entry : paramsMap.entrySet()) {
            index = indices.get(i);
            List<ParamValues> paramValuesList = entry.getValue();
            for(ParamValues paramValues : paramValuesList) {
                int size = paramValues.size();
                index -= size;
                if(index < 0) {
                    Object paramValue = paramValues.get(index + size);
                    try {
                        paramValue = Utilities.deepCopy(paramValue); // must copy objects otherwise every paramset
                        // uses the same object reference!
                    } catch(Exception e) {
                        throw new IllegalStateException("cannot copy value");
                    }
                    param.add(entry.getKey(), paramValue);
                    break;
                }
            }
            if(index >= 0) {
                throw new IndexOutOfBoundsException();
            }
            i++;
        }
        return param;
    }

    /**
     * finds the size of the parameter space.
     * @return
     */
    public int size() {
        return ArrayUtilities.numPermutations(getBins());
    }

    /**
     * add a parameter given ParamValues already setup.
     * @param name
     * @param param
     * @return
     */
    public ParamSpace add(String name, ParamValues param) {
        paramsMap.computeIfAbsent(name, k -> new ArrayList<>()).add(param);
        return this;
    }

    /**
     * add a parameter name mapping to several values.
     * @param name
     * @param values
     * @return
     */
    public ParamSpace add(String name, List<?> values) {
        add(name, new ParamValues(values));
        return this;
    }

    /**
     * add a parameter name mapping to several values, each sharing several sub spaces.
     * @param name
     * @param values
     * @param params
     * @return
     */
    public ParamSpace add(String name, List<?> values, List<ParamSpace> params) {
        add(name, new ParamValues(values, params));
        return this;
    }

    /**
     * add a parameter name mapping of several values with a single shared sub space.
     * @param name
     * @param values
     * @param params
     * @return
     */
    public ParamSpace add(String name, List<?> values, ParamSpace params) {
        add(name, values, new ArrayList<>(Collections.singletonList(params)));
        return this;
    }

    public void clear() {
        paramsMap.clear();
    }

    @Override
    public String toString() {
        return "Params" +
            //            "{" +
            //            "paramsMap=" +
            paramsMap
            //            +
            //            '}'
            ;
    }

    private static class DiscreteDistribution {

        public Discretiser getDiscretiser() {
            return discretiser;
        }

        public DiscreteDistribution setDiscretiser(
            final Discretiser discretiser) {
            Assert.assertNotNull(discretiser); // todo do asserts work in production? There's no <if running in test
            // mode>, right?!
            this.discretiser = discretiser;
            return this;
        }

        public interface Discretiser extends Serializable {
            double discretise(double value);
        }

        private Distribution distribution = new UniformDistribution();
        private Random random = null;
        private Discretiser discretiser = Math::round;

        public DiscreteDistribution() {

        }

        public DiscreteDistribution(Distribution distribution, Discretiser discretiser) {
            setDiscretiser(discretiser);
            setDistribution(distribution);
        }

        public DiscreteDistribution(Distribution distribution, Discretiser discretiser, Random random) {
            this(distribution, discretiser);
            setRandom(random);
        }

        public int sample() {
            final Distribution distribution = getDistribution();
            final Random origRandom = distribution.getRandom();
            distribution.setRandom(getRandom());
            final double value = distribution.sample();
            distribution.setRandom(origRandom);
            double discretisedValue = getDiscretiser().discretise(value);
            return (int) discretisedValue;
        }

        public Distribution getDistribution() {
            return distribution;
        }

        public void setDistribution(
            final Distribution distribution) {
            Assert.assertNotNull(distribution);
            this.distribution = distribution;
        }

        public Random getRandom() {
            return random;
        }

        public void setRandom(final Random random) {
            this.random = random;
        }
    }

    private static class RangedDistribution extends Distribution {
        private Range<Double> range = null;
        private Distribution distribution = new UniformDistribution();

        public RangedDistribution() {

        }

        public RangedDistribution(Distribution distribution) {
            setDistribution(distribution);
        }

        public RangedDistribution(Distribution distribution, Range<Double> range) {
            this(distribution);
            setRange(range);
        }

        public RangedDistribution(Distribution distribution, Random random) {
            this(distribution);
            setRandom(random);
        }

        public RangedDistribution(Distribution distribution, Range<Double> range, Random random) {
            this(distribution, range);
            setRandom(random);
        }

        public Distribution getDistribution() {
            return distribution;
        }

        public RangedDistribution setDistribution(
            final Distribution distribution) {
            Assert.assertNotNull(distribution);
            distribution.setRandom(getRandom());
            this.distribution = distribution;
            return this;
        }

        /**
         * produce a sample from the distribution
         * @return a double in the range of 0 to 1.
         */
        public final double sample() {
            final Distribution distribution = getDistribution();
            final Random origRandom = distribution.getRandom();
            distribution.setRandom(getRandom());
            double value = distribution.sample();
            distribution.setRandom(origRandom);
            if(isPdf()) {
                return value;
            } else {
                final Range<Double> range = getRange();
                return range.lowerEndpoint() + value * (range.upperEndpoint() - range.lowerEndpoint());
            }
        }

        public boolean isPdf() {
            return getRange() == null;
        }

        public Range<Double> getRange() {
            return range;
        }

        public Distribution setRange(final Range<Double> range) {
            this.range = range;
            return this;
        }
    }

    private static class UniformDistribution extends Distribution {

        @Override
        public double sample() {
            return getRandom().nextDouble();
        }
    }

    private static abstract class Distribution implements RandomSource {

        private Random random = null;

        public Distribution() {

        }

        public Distribution(Random random) {
            setRandom(random);
        }

        public abstract double sample();

        @Override
        public Random getRandom() {
            return random;
        }

        @Override
        public void setRandom(Random random) {
            this.random = random;
        }

    }

    /**
     * holds a set of values (e.g. DTW and DDTW) and a set of corresponding params for those values (e.g. a set of
     * warping windows). I.e. many values can map to many sub spaces.
     */
    public static class ParamValues {

        private List<?> values = new ArrayList<>(); // raw values
        private List<ParamSpace> paramsList = new ArrayList<>(); // sub spaces

        public ParamValues() {

        }

        public ParamValues(List<?> values, List<ParamSpace> params) {
            setValues(values);
            setParamsList(params);
        }

        public ParamValues(List<?> values) {
            this(values, null); // no sub param space
        }

        /**
         * get the size of each parameter in the space, adding the raw values also.
         * @return
         */
        public List<Integer> getBins() {
            List<Integer> bins = new ArrayList<>();

            for(ParamSpace paramSets : paramsList) {
                bins.add(paramSets.size());
            }
            bins.add(values.size());
            return bins;
        }

        public int size() {
            return ArrayUtilities.numPermutations(getBins());
        }

        /**
         * get the index of a value in the space.
         * @param index
         * @return
         */
        public Object get(final int index) {
            List<Integer> indices = ArrayUtilities.fromPermutation(index, getBins());
            Object value = values.get(indices.get(indices.size() - 1));
            if(!(value instanceof ParamHandler) && !paramsList.isEmpty()) {
                throw new IllegalStateException("value not param settable");
            }
            for(int i = 0; i < paramsList.size(); i++) {
                ParamHandler paramHandler = (ParamHandler) value;
                ParamSet param = paramsList.get(i).get(indices.get(i));
                paramHandler.setParams(param);
            }
            return value;
        }

        public void addParams(ParamSpace... params) {
            this.paramsList.addAll(Arrays.asList(params));
        }

        public List<?> getValues() {
            return values;
        }

        public void setValues(List<?> values) {
            if(values == null) {
                values = new ArrayList<>();
            }
            this.values = values;
        }

        public List<ParamSpace> getParamsList() {
            return paramsList;
        }

        public void setParamsList(List<ParamSpace> paramsList) {
            if(paramsList == null) {
                paramsList = new ArrayList<>();
            }
            this.paramsList = paramsList;
        }

        @Override
        public String toString() {
            return "ParamValues{" +
                "values=" + values +
                ", params=" + paramsList +
                '}';
        }
    }


}
