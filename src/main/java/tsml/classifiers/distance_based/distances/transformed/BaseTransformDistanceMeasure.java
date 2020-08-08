package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import org.junit.Assert;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.ed.EDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.transformers.TrainableTransformer;
import tsml.transformers.Transformer;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

public class BaseTransformDistanceMeasure extends BaseDistanceMeasure implements TransformDistanceMeasure {

    public BaseTransformDistanceMeasure(String name, Transformer transformer,
                                        DistanceFunction distanceFunction) {
        setName(name);
        setDistanceFunction(distanceFunction);
        setTransformer(transformer);
    }

    public BaseTransformDistanceMeasure() {
        this("", null, new EDistance());
        setName(getClass().getSimpleName());
    }

    public static final String TRANSFORMER_FLAG = "t";
    private DistanceFunction distanceFunction;
    private Transformer transformer;

    @Override public boolean isSymmetric() {
        return (distanceFunction instanceof DistanceMeasure && ((DistanceMeasure) distanceFunction).isSymmetric());
    }

    private static Instance transform(Transformer transformer, Instance instance) {
        if(transformer == null) {
            return instance;
        } else {
            return transformer.transform(instance);
        }
    }

    @Override
    public double findDistance(final Instance a, final Instance b, final double limit) {
        try {
            final Instance at = transform(transformer, a);
            // need to take the interval here, before the transform
            final Instance bt = transform(transformer, b);
            return distanceFunction.distance(at, bt, limit);
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void setInstances(Instances data) {
        super.setInstances(data);
        distanceFunction.setInstances(data);
        if(transformer != null) {
            if(transformer instanceof TrainableTransformer) {
                ((TrainableTransformer) transformer).fit(data);
            }
        }
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public void setDistanceFunction(DistanceFunction distanceFunction) {
        Assert.assertNotNull(distanceFunction);
        this.distanceFunction = distanceFunction;
    }

    @Override public ParamSet getParams() {
        final ParamSet paramSet = super.getParams();
        paramSet.add(TRANSFORMER_FLAG, transformer);
        paramSet.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, distanceFunction);
        return paramSet;
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        ParamHandlerUtils.setParam(param, TRANSFORMER_FLAG, this::setTransformer, Transformer.class);
        ParamHandlerUtils.setParam(param, DISTANCE_MEASURE_FLAG, this::setDistanceFunction, DistanceFunction.class);
        super.setParams(param);
    }

    public Transformer getTransformer() {
        return transformer;
    }

    public void setTransformer(Transformer a) {
        transformer = a;
    }

}
