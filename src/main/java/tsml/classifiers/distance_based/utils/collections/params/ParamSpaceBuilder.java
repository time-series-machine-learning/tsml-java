package tsml.classifiers.distance_based.utils.collections.params;

import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.Instances;

import java.io.Serializable;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public interface ParamSpaceBuilder extends Serializable {

    default ParamSpace build(TimeSeriesInstances data) {
        return build(Converter.toArff(data));
    }
    
    default ParamSpace build(Instances data) {
        return build(Converter.fromArff(data));
    }

}
