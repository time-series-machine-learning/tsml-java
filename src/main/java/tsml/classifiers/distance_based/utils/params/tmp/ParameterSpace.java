package tsml.classifiers.distance_based.utils.params.tmp;

import com.beust.jcommander.internal.Lists;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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

public class ParameterSpace {

    // 1 to many mapping of param name to list of param dimensions
    private Map<String, List<ParameterDimension<?>>> dimensionMap = new LinkedHashMap<>();

    public Map<String, List<ParameterDimension<?>>> getDimensionMap() {
        return dimensionMap;
    }

    public static void main(String[] args) {
        int seed = 0;
        Random random = new Random(seed);
        // build dtw / ddtw params
        ParameterSpace wParams = new ParameterSpace();
        wParams.add(DTW.getWarpingWindowFlag(), Arrays.asList(1, 2, 3, 4, 5));
        System.out.println(wParams);
        // build lcss params
        ParameterSpace lParams = new ParameterSpace();
        UniformDistribution eDist = new UniformDistribution();
        eDist.setRandom(random);
        eDist.setMinAndMax(0, 0.25);
        lParams.add(LCSSDistance.getDeltaFlag(), eDist);
        UniformDistribution dDist = new UniformDistribution();
        dDist.setRandom(random);
        dDist.setMinAndMax(0.5, 1);
        lParams.add(LCSSDistance.getEpsilonFlag(), dDist);
        System.out.println(lParams);
        // build dtw / ddtw space
        DiscreteParameterDimension<? extends BaseDistanceMeasure> wDmParams = new DiscreteParameterDimension<>(
            Arrays.asList(new DTWDistance(), new DDTWDistance()));
        wDmParams.addSubSpace(wParams);
        System.out.println(wDmParams);
        // build lcss space
        DiscreteParameterDimension<? extends BaseDistanceMeasure> lDmParams = new DiscreteParameterDimension<>(
            Arrays.asList(new LCSSDistance()));
        lDmParams.addSubSpace(lParams); // todo can we shrink this into one func?
        System.out.println(lDmParams);
        // build overall space including ddtw, dtw and lcss WITH corresponding param spaces
        ParameterSpace params = new ParameterSpace();
        params.add(DistanceMeasureable.getDistanceFunctionFlag(), lDmParams);
        params.add(DistanceMeasureable.getDistanceFunctionFlag(), wDmParams);
        System.out.println(params);
    }

    @Override
    public String toString() {
        return String.valueOf(dimensionMap);
    }

    public void add(String name, ParameterDimension<?> dimension) {
        dimensionMap.computeIfAbsent(name, s -> new ArrayList<>()).add(dimension);
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

}
