package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.params.Distribution;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.ArrayUtilities;
import utilities.Utilities;

public class ParameterSpace {

    // 1 to many mapping of param name to list of param dimensions
    private Map<String, List<ParameterDimension<?>>> paramsMap = new LinkedHashMap<>();

    /**
     * hold the parameter dimension. In here should be a method of retreiving values for the given parameter along
     * with sub parameter spaces to explore
     * @param <A>
     */
    public abstract static class ParameterDimension<A> {

        // list of subspaces to explore
        private List<ParameterSpace> subSpaces = new ArrayList<>();

        /**
         * some method of retreiving a parameter value (given an index, which may or may not be used as is context
         * dependent)
         * @param index
         * @return
         */
        public abstract A getParameterValue(int index);

        /**
         * get the number of possible values for this parameter NOT including the sub spaces!
         * @return <0 means an infinite number of values, >=0 is the finite size. An infinite size indicates the
         * parameter is continuous, finite indicates it is an enumerated set of values.
         */
        public abstract int getParameterDimensionSize();

        public List<ParameterSpace> getSubSpaces() {
            return subSpaces;
        }

        public void setSubSpaces(final List<ParameterSpace> subSpaces) {
            Assert.assertNotNull(subSpaces);
            this.subSpaces = subSpaces;
        }

        public boolean containsContinuousDimension() {
            if(getParameterDimensionSize() < 0) {
                return true;
            }
            List<Integer> sizes = getSubDimensionSizes();
            for(Integer size : sizes) {
                if(size < 0) {
                    return true;
                }
            }
            return false;
        }

        /**
         * the size of the parameter dimension. Note this is only the FINITE size! If the parameter is continuous
         * then there is no concept of size. A continuous parameter may have subspaces which are discrete and finite
         * however. For example, I may have a discrete parameter set of [dtw, ddtw]. I may then have a subspace which
         * is continuous, say [warpingWindow{0 : 1}]. This method would return 2, reflecting the two discrete options
         * (dtw and ddtw). It does not take into account the infinite range of the subspace parameter warpingWindow.
         * @return
         */
        public int size() { // todo make sure all of this works with index / size <0 or =0
            final List<Integer> sizes = getSubDimensionSizes();
            // put this parameter's dimension size on the front of the sub sizes
            sizes.add(0, getParameterDimensionSize());
            return Utilities.numPermutations(sizes);
        }

        public List<Integer> getSubDimensionSizes() {
            return Utilities.convert(getSubSpaces(), ParameterSpace::size);
        }

        /**
         * get a ParamSet using a given index. The index is only used to index the finite permutations of parameters
         * where their values have been enumerated. Infinite / continuous parameters are sampled using a pdf and have
         * no bearing on the index variable. If you are only using pdf parameters then the index variable is ignored.
         * @param index
         * @return
         */
        public A get(int index) {
            final List<Integer> sizes = getSubDimensionSizes();
            // find the indices of the dimension and each sub space using the permutation index
            final List<Integer> indices = Utilities.fromPermutation(index, sizes);
            // get this parameter's value index, leaving only sub space indices
            final int valueIndex = indices.remove(0);
            final A value = getParameterValue(valueIndex);
            final List<ParameterSpace> subSpaces = getSubSpaces();
            Assert.assertEquals(subSpaces.size(), indices.size());
            // if we've got any sub spaces
            if(!sizes.isEmpty()) {
                // check if we can set params
                if(!(value instanceof ParamHandler)) {
                    throw new IllegalStateException("value not param settable");
                } else {
                    // set parameter for each sub space using the indices
                    ParamHandler paramHandler = (ParamHandler) value;
                    for(int i = 0; i < subSpaces.size(); i++) {
                        int subSpaceValueIndex = indices.get(i);
                        ParameterSpace subSpace = subSpaces.get(i);
                        if(subSpaceValueIndex >= 0 || subSpace.containsContinuousDimension()) {
                            ParamSet param = subSpace.get(subSpaceValueIndex);
                            paramHandler.setParams(param);
                        }
                    }
                }
            }
            return value;
        }

    }

    /**
     * class to hold a finite set of values for a parameter
     * @param <A>
     */
    public static class DiscreteParameterDimension<A> extends ParameterDimension<A> {

        private List<? extends A> values;

        public DiscreteParameterDimension() {
            this(new ArrayList<>());
        }

        public DiscreteParameterDimension(List<? extends A> values) {
            this(values, new ArrayList<>());
        }

        public DiscreteParameterDimension(List<? extends A> values, List<ParameterSpace> subSpaces) {
            setValues(values);
            setSubSpaces(subSpaces);
        }

        @Override
        public A getParameterValue(final int index) {
            // simply grab the value at the given index
            return values.get(index);
        }

        @Override
        public int getParameterDimensionSize() {
            return values.size();
        }

        public List<? extends A> getValues() {
            return values;
        }

        public void setValues(final List<? extends A> values) {
            Assert.assertNotNull(values);
            this.values = values;
        }
    }

    /**
     * class to represent a continuous set of parameter values, i.e. a range of doubles between 0 and 1, say. The
     * range is sampled randomly and the probability density function should be controlled using the distribution
     * class functions (e.g. setting the range and type of pdf).
     * @param <A>
     */
    public static class ContinuousParameterDimension<A> extends ParameterDimension<A> {

        private Distribution<A> distribution;

        public ContinuousParameterDimension(Distribution<A> distribution, List<ParameterSpace> subSpaces) {
            setDistribution(distribution);
            setSubSpaces(subSpaces);
        }

        public ContinuousParameterDimension(Distribution<A> distribution) {
            this(distribution, new ArrayList<>());
        }

        @Override
        public A getParameterValue(final int index) {
            return distribution.sample();
        }

        @Override
        public int getParameterDimensionSize() {
            // -1 because there's an infinite amount of values in the distribution
            return -1;
        }

        public Distribution<A> getDistribution() {
            return distribution;
        }

        public void setDistribution(final Distribution<A> distribution) {
            Assert.assertNotNull(distribution);
            this.distribution = distribution;
        }
    }

    public int size() {
        final List<Integer> sizes = getDimensionSizes();
        return Utilities.numPermutations(sizes);
    }

    // todo move current size methods to another name, make size return int maxVal if continuous, else finite size

    /**
     * gets a ParamSet for the corresponding index in the ParamSpace.
     * @param index
     * @return
     */
    public ParamSet get(int index) {
        List<Integer> indices = ArrayUtilities.fromPermutation(index, getDimensionSizes());
        int i = 0;
        ParamSet param = new ParamSet();
        for(Map.Entry<String, List<ParameterDimension<?>>> entry : paramsMap.entrySet()) {
            index = indices.get(i);
            List<ParameterDimension<?>> paramValuesList = entry.getValue();
            for(ParameterDimension<?> paramValues : paramValuesList) {
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
     * gets a list of the sizes of each parameter. Remember, this is only the finite size, so if there's any
     * continuous infinite dimensions in this space they are not counted!
     * @return
     */
    public List<Integer> getDimensionSizes() {
        List<Integer> sizes = new ArrayList<>();
        for(Map.Entry<String, List<ParameterDimension<?>>> entry : paramsMap.entrySet()) {
            int size = 0;
            for(ParameterDimension<?> parameterDimension : entry.getValue()) {
                final int dimensionSize = parameterDimension.size();
                // ignore it if the dimension is continuous
                if(dimensionSize > 0) {
                    size += dimensionSize;
                }
            }
            sizes.add(size);
        }
        return sizes;
    }

    public boolean containsContinuousDimension() {
        List<Integer> sizes = getDimensionSizes();
        for(Integer size : sizes) {
            if(size < 0) {
                return true;
            }
        }
        return false;
    }
}
