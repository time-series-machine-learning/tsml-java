package timeseriesweka.classifiers.distance_based.distances.ddtw;

import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.distance_based.distances.dtw.DtwParameterSpaceBuilder;

public class AllDdtwParameterSpaceBuilder
    extends DdtwParameterSpaceBuilder {

    @Override
    public ParameterSpace build() {
        ParameterSpace parameterSpace = super.build();
        parameterSpace.addAll(new EdDdtwParameterSpaceBuilder().build());
        parameterSpace.addAll(new FullDdtwParameterSpaceBuilder().build());
        return parameterSpace;
    }
}
