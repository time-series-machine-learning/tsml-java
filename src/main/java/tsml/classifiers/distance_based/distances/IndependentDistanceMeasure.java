package tsml.classifiers.distance_based.distances;

import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Objects;

public class IndependentDistanceMeasure extends BaseDistanceMeasure {

    public IndependentDistanceMeasure(final DistanceMeasure distanceMeasure) {
        setDistanceMeasure(distanceMeasure);
    }

    private DistanceMeasure distanceMeasure;

    @Override public boolean isSymmetric() {
        return distanceMeasure.isSymmetric();
    }

    @Override public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b,
            final double limit) {
        double sum = 0;
        for(int i = 0; i < a.getNumDimensions(); i++) {
            final TimeSeriesInstance singleDimA = a.getHSlice(i);
            final TimeSeriesInstance singleDimB = b.getHSlice(i);
            sum += distanceMeasure.distance(singleDimA, singleDimB);
        }
        return sum;
    }

    @Override public String getName() {
        return distanceMeasure.getName() + "_I";
    }

    @Override public void buildDistanceMeasure(final TimeSeriesInstances data) {
        distanceMeasure.buildDistanceMeasure(data);
    }

    @Override public void buildDistanceMeasure(final Instances data) {
        distanceMeasure.buildDistanceMeasure(data);
    }
    
    private void setDistanceMeasure(DistanceMeasure distanceMeasure) {
        this.distanceMeasure = Objects.requireNonNull(distanceMeasure);
    }

    @Override public void setParams(final ParamSet paramSet) throws Exception {
        ParamHandlerUtils.setParam(paramSet, DISTANCE_MEASURE_FLAG, this::setDistanceMeasure);
    }

    @Override public ParamSet getParams() {
        return new ParamSet().add(DISTANCE_MEASURE_FLAG, distanceMeasure);
    }
}
