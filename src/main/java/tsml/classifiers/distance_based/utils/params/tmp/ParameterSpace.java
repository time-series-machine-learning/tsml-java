package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.ddtw.DDTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.utils.collections.DefaultList;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.distribution.UniformDistribution;
import utilities.ArrayUtilities;
import utilities.Utilities;

public class ParameterSpace implements DefaultList<ParamSet> {

    // 1 to many mapping of param name to list of param dimensions
    private Map<String, List<ParameterDimension<?>>> paramsMap = new LinkedHashMap<>();

    public static void main(String[] args) {
        int seed = 0;
        Random random = new Random(0);
        // build dtw / ddtw params
        ParameterSpace wParams = new ParameterSpace();
        wParams.add(DTW.getWarpingWindowFlag(), new DiscreteParameterDimension<>(Arrays.asList(1, 2, 3, 4, 5)));
        System.out.println(wParams);
        System.out.println(wParams.size());
        // build lcss params
        ParameterSpace lParams = new ParameterSpace();
        UniformDistribution eDist = new UniformDistribution();
        eDist.setRandom(random);
        eDist.setMinAndMax(0, 0.25);
        lParams.add(LCSSDistance.getDeltaFlag(), new ContinuousParameterDimension<>(eDist));
        UniformDistribution dDist = new UniformDistribution();
        dDist.setRandom(random);
        dDist.setMinAndMax(0.5, 1);
        lParams.add(LCSSDistance.getEpsilonFlag(), new ContinuousParameterDimension<>(dDist));
        System.out.println(lParams);
        System.out.println(lParams.size());
        // build dtw / ddtw space
        DiscreteParameterDimension<? extends BaseDistanceMeasure> wDmParams = new DiscreteParameterDimension<>(
            Arrays.asList(new DTWDistance(), new DDTWDistance()));
        wDmParams.addSubSpace(wParams);
        System.out.println(wDmParams);
        System.out.println(wDmParams.size());
        // build lcss space
        DiscreteParameterDimension<? extends BaseDistanceMeasure> lDmParams = new DiscreteParameterDimension<>(
            Arrays.asList(new LCSSDistance()));
        lDmParams.addSubSpace(lParams); // todo can we shrink this into one func?
        System.out.println(lDmParams);
        System.out.println(lDmParams.size());
        // build overall space including ddtw, dtw and lcss WITH corresponding param spaces
        ParameterSpace params = new ParameterSpace();
        params.add(DistanceMeasureable.getDistanceFunctionFlag(), lDmParams);
        params.add(DistanceMeasureable.getDistanceFunctionFlag(), wDmParams);
        System.out.println(params);
        System.out.println(params.size());
        // check every combination of the overall space
        for(int i = 0; i < params.size(); i++) {

        }

        //        params.add(DistanceMeasureable.getDistanceFunctionFlag(),
        //            new ParamValues(Arrays.asList(new DTWDistance(), new DDTWDistance()),
        //                Arrays.asList(wParams)));
        //        ParamSpace lParams = new ParamSpace();
        //        lParams.add(WDTW.getGFlag(), new ParamValues(/*Range.closed(1D, 5D)*/));
        //        lParams.add(WDTW.getGFlag(), new ParamValues(Arrays.asList(1D, 2D, 3D)));
        //        lParams.add(LCSSDistance.getEpsilonFlag(), new ParamValues(Arrays.asList(1D, 2D, 3D, 4D)));
        //        params.add(DistanceMeasureable.getDistanceFunctionFlag(),
        //            new ParamValues(Arrays.asList(new WDTWDistance(), new WDDTWDistance()),
        //                Arrays.asList(lParams)));
        //        int size;
        //        size = wParams.size();
        //        size = lParams.size();
        //        size = params.size();
        //        for(int i = 0; i < size; i++) {
        //            //            System.out.println(i);
        //            ParamSet param = params.get(i);
        //            System.out.println(param);
        //        }

    }

    @Override
    public String toString() {
        return String.valueOf(paramsMap);
    }

    public void add(String name, ParameterDimension<?> dimension) {
        paramsMap.computeIfAbsent(name, s -> new ArrayList<>()).add(dimension);
    }

    public int size() {
        final List<Integer> sizes = getDimensionSizes();
        return Permutations.numPermutations(sizes);
    }

    // todo move current size methods to another name, make size return int maxVal if continuous, else finite size

    /**
     * gets a ParamSet for the corresponding index in the ParamSpace.
     *
     * @param index
     * @return
     */
    public ParamSet get(int index) {
        List<Integer> indices = ArrayUtilities.fromPermutation(index, getDimensionSizes());
        ParamSet param = new ParamSet();
        int i = 0;
        for(Map.Entry<String, List<ParameterDimension<?>>> entry : paramsMap.entrySet()) {
            index = indices.get(i++);
            List<ParameterDimension<?>> parameterDimensions = entry.getValue();
            for(ParameterDimension<?> paramValues : parameterDimensions) {
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
        }
        return param;
    }

    /**
     * gets a list of the sizes of each parameter. Remember, this is only the finite size, so if there's any continuous
     * infinite dimensions in this space they are not counted!
     *
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

    /**
     * Does this space contain a continuous dimension?
     *
     * @return
     */
    public boolean containsContinuousDimension() {
        for(Map.Entry<String, List<ParameterDimension<?>>> entry : paramsMap.entrySet()) {
            for(ParameterDimension<?> parameterDimension : entry.getValue()) {
                final int dimensionSize = parameterDimension.size();
                // ignore it if the dimension is continuous
                if(dimensionSize < 0) {
                    return true;
                }
            }
        }
        return false;
    }


}
