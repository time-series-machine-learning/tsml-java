package tsml.classifiers.distance_based.utils.collections.params;

import tsml.classifiers.distance_based.distances.dtw.DTWDistanceConfigs;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceConfigs;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistanceConfigs;
import tsml.classifiers.distance_based.distances.msm.MSMDistanceConfigs;
import tsml.classifiers.distance_based.distances.twed.TWEDistanceConfigs;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistanceConfigs;
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
