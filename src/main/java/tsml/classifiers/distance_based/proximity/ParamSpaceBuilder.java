package tsml.classifiers.distance_based.proximity;

import tsml.classifiers.distance_based.distances.dtw.DTWDistanceConfigs;
import tsml.classifiers.distance_based.distances.ed.EuclideanDistanceConfigs;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceConfigs;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistanceConfigs;
import tsml.classifiers.distance_based.distances.msm.MSMDistanceConfigs;
import tsml.classifiers.distance_based.distances.twed.TWEDistanceConfigs;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistanceConfigs;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public interface ParamSpaceBuilder {

    ParamSpace build(Instances data);

    ParamSpaceBuilder ED = i -> EuclideanDistanceConfigs.buildEdSpace();
    ParamSpaceBuilder DTW = DTWDistanceConfigs::buildDtwSpaceRestrictedContinuous;
    ParamSpaceBuilder FULL_DTW = i -> DTWDistanceConfigs.buildDtwFullWindowSpace();
    ParamSpaceBuilder DDTW = DTWDistanceConfigs::buildDdtwSpaceRestrictedContinuous;
    ParamSpaceBuilder FULL_DDTW = i -> DTWDistanceConfigs.buildDdtwFullWindowSpace();
    ParamSpaceBuilder LCSS = LCSSDistanceConfigs::buildLcssSpaceContinuous;
    ParamSpaceBuilder ERP = ERPDistanceConfigs::buildErpSpaceContinuous;
    ParamSpaceBuilder MSM = i -> MSMDistanceConfigs.buildMsmSpace();
    ParamSpaceBuilder TWED = i -> TWEDistanceConfigs.buildTwedSpace();
    ParamSpaceBuilder WDTW = i -> WDTWDistanceConfigs.buildWdtwSpaceContinuous();
    ParamSpaceBuilder WDDTW = i -> WDTWDistanceConfigs.buildWddtwSpaceContinuous();

}
