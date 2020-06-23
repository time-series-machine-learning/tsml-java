package tsml.classifiers.distance_based.proximity;

import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public interface ParamSpaceBuilder {

    ParamSpace build(Instances data);

    ParamSpaceBuilder ED = i -> DistanceMeasureConfigs.buildEdSpace();
    ParamSpaceBuilder DTW = ContinuousDistanceFunctionConfigs::buildDtwSpace;
    ParamSpaceBuilder FULL_DTW = i -> DistanceMeasureConfigs.buildFullDtwSpace();
    ParamSpaceBuilder DDTW = ContinuousDistanceFunctionConfigs::buildDdtwSpace;
    ParamSpaceBuilder FULL_DDTW = i -> DistanceMeasureConfigs.buildFullDdtwSpace();
    ParamSpaceBuilder LCSS = ContinuousDistanceFunctionConfigs::buildLcssSpace;
    ParamSpaceBuilder ERP = ContinuousDistanceFunctionConfigs::buildErpSpace;
    ParamSpaceBuilder MSM = i -> DistanceMeasureConfigs.buildMsmSpace();
    ParamSpaceBuilder TWED = i -> DistanceMeasureConfigs.buildTwedSpace();
    ParamSpaceBuilder WDTW = i -> ContinuousDistanceFunctionConfigs.buildWdtwSpace();
    ParamSpaceBuilder WDDTW = i -> ContinuousDistanceFunctionConfigs.buildWddtwSpace();

}
