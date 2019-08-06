package timeseriesweka.classifiers.distance_based.distances.dtw;

import evaluation.tuning.ParameterSpace;

public class AllDtwParameterSpaceBuilder
    extends DtwParameterSpaceBuilder {

    @Override
    public ParameterSpace build() {
        ParameterSpace parameterSpace = super.build();
        parameterSpace.addAll(new EdParameterSpaceBuilder().build());
        parameterSpace.addAll(new FullDtwParameterSpaceBuilder().build());
        return parameterSpace;
    }
}
