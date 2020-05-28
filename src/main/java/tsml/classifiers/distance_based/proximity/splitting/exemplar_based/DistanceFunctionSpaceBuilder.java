package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public interface DistanceFunctionSpaceBuilder {

    ParamSpace build(Instances data);

    DistanceFunctionSpaceBuilder ED = i -> DistanceMeasureConfigs.buildEdSpace();
    DistanceFunctionSpaceBuilder DTW = ContinuousDistanceFunctionConfigs::buildDtwSpace;
    DistanceFunctionSpaceBuilder FULL_DTW = i -> DistanceMeasureConfigs.buildFullDtwSpace();
    DistanceFunctionSpaceBuilder DDTW = ContinuousDistanceFunctionConfigs::buildDdtwSpace;
    DistanceFunctionSpaceBuilder FULL_DDTW = i -> DistanceMeasureConfigs.buildFullDdtwSpace();
    DistanceFunctionSpaceBuilder LCSS = ContinuousDistanceFunctionConfigs::buildLcssSpace;
    DistanceFunctionSpaceBuilder ERP = ContinuousDistanceFunctionConfigs::buildErpSpace;
    DistanceFunctionSpaceBuilder MSM = i -> DistanceMeasureConfigs.buildMsmSpace();
    DistanceFunctionSpaceBuilder TWED = i -> DistanceMeasureConfigs.buildTwedSpace();
    DistanceFunctionSpaceBuilder WDTW = i -> ContinuousDistanceFunctionConfigs.buildWdtwSpace();
    DistanceFunctionSpaceBuilder WDDTW = i -> ContinuousDistanceFunctionConfigs.buildWddtwSpace();

}
