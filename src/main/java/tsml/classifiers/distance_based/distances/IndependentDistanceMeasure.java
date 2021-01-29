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
            // extract the single dim from each inst
            final TimeSeriesInstance singleDimA = a.getHSlice(i);
            final TimeSeriesInstance singleDimB = b.getHSlice(i);
            // compute the distance between the single dims
            // the limit will be the remainder of the limit after subtracting the current sum
            final double distance = distanceMeasure.distance(singleDimA, singleDimB, limit - sum);
            // distance will be inf if limit hit, so sum will coalesce to inf as well
            sum += distance;
            // if the last distance tipped the sum over the limit (or hit the limit itself and sum is now inf) return inf as over limit
            if(sum > limit) {
                return Double.POSITIVE_INFINITY;
            }
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

    @Override public String toString() {
        return getName() + " " + distanceMeasure.getParams();
    }
}
