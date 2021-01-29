package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.ed.EDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;
import tsml.transformers.Transformer;
import weka.core.Instance;

import java.util.Objects;

public class BaseTransformDistanceMeasure extends MatrixBasedDistanceMeasure implements TransformDistanceMeasure {

    public BaseTransformDistanceMeasure(String name, Transformer transformer,
                                        DistanceMeasure distanceMeasure) {
        setDistanceMeasure(distanceMeasure);
        setTransformer(transformer);
        setName(name);
    }

    public BaseTransformDistanceMeasure() {
        this(null, null, new EDistance());
    }

    public static final String TRANSFORMER_FLAG = "t";
    private DistanceMeasure distanceMeasure;
    private Transformer transformer;
    private String name;

    @Override public String getName() {
        return name;
    }

    public void setName(String name) {
        if(name == null) {
            name = distanceMeasure.getName();
            if(transformer != null) {
                name = transformer.getClass().getSimpleName() + "_" + name;
            }
        }
        this.name = name;
    }

    @Override public boolean isSymmetric() {
        return distanceMeasure.isSymmetric();
    }

    private static Instance transform(Transformer transformer, Instance instance) {
        if(transformer == null) {
            return instance;
        } else {
            return transformer.transform(instance);
        }
    }
    
    private static TimeSeriesInstance transform(Transformer transformer, TimeSeriesInstance instance) {
        if(transformer == null) {
            return instance;
        } else {
            return transformer.transform(instance);
        }
    }
    
    public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, final double limit) {
        try {
            // users must call fit method on transformer if required before calling distance
            final TimeSeriesInstance at = transform(transformer, a);
            // need to take the interval here, before the transform
            final TimeSeriesInstance bt = transform(transformer, b);
            return distanceMeasure.distance(at, bt, limit);
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(DistanceMeasure distanceMeasure) {
        this.distanceMeasure = Objects.requireNonNull(distanceMeasure);
    }

    @Override public ParamSet getParams() {
        final ParamSet paramSet = super.getParams();
        paramSet.add(TRANSFORMER_FLAG, transformer);
        paramSet.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, distanceMeasure);
        return paramSet;
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        ParamHandlerUtils.setParam(param, TRANSFORMER_FLAG, this::setTransformer);
        ParamHandlerUtils.setParam(param, DISTANCE_MEASURE_FLAG, this::setDistanceMeasure);
        super.setParams(param);
    }

    public Transformer getTransformer() {
        return transformer;
    }

    public void setTransformer(Transformer a) {
        transformer = a;
    }

    @Override public String toString() {
        return getName() + " " + distanceMeasure.getParams();
    }
}
