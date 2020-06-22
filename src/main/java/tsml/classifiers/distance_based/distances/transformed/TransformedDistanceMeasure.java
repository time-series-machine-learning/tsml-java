package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import org.junit.Assert;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.utils.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.transformers.TrainableTransformer;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import tsml.transformers.Transformer;

public class TransformedDistanceMeasure extends BaseDistanceMeasure implements TransformedDistanceMeasureable {

    public TransformedDistanceMeasure(String name, Transformer transformer,
        DistanceFunction distanceFunction) {
        setName(name);
        setDistanceFunction(distanceFunction);
        setTransformer(transformer);
    }

    protected TransformedDistanceMeasure() {

    }

    private DistanceFunction distanceFunction;
    private Transformer transformer;
    private String name = getClass().getSimpleName();

    @Override
    protected boolean requiresDataToBeSet() {
        return true;
    }

    protected void setName(String name) {
        Assert.assertNotNull(name);
        this.name = name;
    }

    @Override
    public void setInstances(Instances data) {
        super.setInstances(data);
        distanceFunction.setInstances(data);
        if(transformer instanceof TrainableTransformer) {
            ((TrainableTransformer) transformer).fit(data);
        }
    }

    @Override
    public String getName() {
        return name;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    protected void setDistanceFunction(DistanceFunction distanceFunction) {
        Assert.assertNotNull(distanceFunction);
        this.distanceFunction = distanceFunction;
    }

    public Transformer getTransformer() {
        return transformer;
    }

    protected void setTransformer(Transformer transformer) {
        Assert.assertNotNull(transformer);
        this.transformer = transformer;
    }

    @Override
    public double findDistance(final Instance a, final Instance b, final double limit) {
        try {
            final Instance firstTransformed = transformer.transform(a);
            final Instance secondTransformed = transformer.transform(b);
            return distanceFunction.distance(firstTransformed, secondTransformed, limit);
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override public ParamSet getParams() {
        return super.getParams().add(TRANSFORMER_FLAG, transformer).add(DistanceMeasureable.DISTANCE_MEASURE_FLAG, distanceFunction);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        ParamHandlerUtils.setParam(param, TRANSFORMER_FLAG, this::setTransformer, Transformer.class);
        ParamHandlerUtils.setParam(param, DISTANCE_MEASURE_FLAG, this::setDistanceFunction, DistanceFunction.class);
        super.setParams(param);
    }

    public static final String TRANSFORMER_FLAG = "t";

}
