package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.transformers.Transformer;
import weka.core.DistanceFunction;

public class TransformDistanceMeasure extends TransformedDistanceMeasure implements TransformDistanceMeasureable {

    public TransformDistanceMeasure(String name,
        Transformer transformer, DistanceFunction distanceFunction) {
        super(name, transformer, distanceFunction);
    }

    @Override
    public void setDistanceFunction(DistanceFunction distanceFunction) {
        super.setDistanceFunction(distanceFunction);
    }

    @Override
    public void setTransformer(Transformer transformer) {
        super.setTransformer(transformer);
    }

    @Override
    public void setName(String name) {
        super.setName(name);
    }

    @Override public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, TRANSFORMER_FLAG, this::setTransformer, Transformer.class);
        ParamHandlerUtils
                .setParam(param, DistanceMeasureable.DISTANCE_MEASURE_FLAG, this::setDistanceFunction, DistanceFunction.class);
    }

}
