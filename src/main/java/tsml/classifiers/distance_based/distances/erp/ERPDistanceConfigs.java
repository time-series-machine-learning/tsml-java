/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.distances.erp;

import com.beust.jcommander.internal.Lists;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;
import tsml.classifiers.distance_based.utils.collections.params.distribution.int_based.UniformIntDistribution;
import utilities.StatisticalUtilities;
import weka.core.Instances;

import java.util.List;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;
import static utilities.ArrayUtilities.range;
import static utilities.ArrayUtilities.unique;

public class ERPDistanceConfigs {

    public static class ERPSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildERPSpace(data);
        }
    }

    public static class RestrictedContinuousERPSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildRestrictedContinuousERPSpace(data);
        }
    }

    public static class ContinuousERPSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildContinuousERPSpace(data);
        }
    }

    public static ParamSpace buildERPSpace(Instances instances) {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new ERPDistance()),
                buildERPParams(instances));
    }

    public static ParamSpace buildERPParams(Instances instances) {
        double std = StatisticalUtilities.pStdDev(instances);
        double stdFloor = std * 0.2;
        int[] bandSizeValues = range(0, (instances.numAttributes() - 1) / 4, 10);
        double[] penaltyValues = range(stdFloor, std, 10);
        List<Double> penaltyValuesUnique = unique(penaltyValues);
        List<Integer> bandSizeValuesUnique = unique(bandSizeValues);
        ParamSpace params = new ParamSpace();
        params.add(ERPDistance.WINDOW_SIZE_FLAG, bandSizeValuesUnique);
        params.add(ERPDistance.G_FLAG, penaltyValuesUnique);
        return params;
    }

    public static ParamSpace buildRestrictedContinuousERPParams(Instances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(ERPDistance.G_FLAG, new UniformDoubleDistribution(0.2 * std, std));
        // pf implements this as randInt(len / 4 + 1), so range is from 0 to len / 4 inclusively
        // above doesn't consider class value, so -1 from len
        subSpace.add(ERPDistance.WINDOW_SIZE_FLAG, new UniformIntDistribution(0,
            (data.numAttributes() - 1) / 4));
        return subSpace;
    }

    public static ParamSpace buildRestrictedContinuousERPSpace(Instances data) {
        final ParamSpace space = new ParamSpace();
        space.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, Lists.newArrayList(new ERPDistance()),
                  buildRestrictedContinuousERPParams(data));
        return space;
    }

    public static ParamSpace buildContinuousERPParams(Instances data) {
        final double std = StatisticalUtilities.pStdDev(data);
        final ParamSpace subSpace = new ParamSpace();
        subSpace.add(ERPDistance.G_FLAG, new UniformDoubleDistribution(0.02 * std, std));
        subSpace.add(ERPDistance.WINDOW_SIZE_FLAG, new UniformIntDistribution(0, data.numAttributes() - 1 - 1)); // todo adjust this to use length instead of max index
        return subSpace;
    }


    public static ParamSpace buildContinuousERPSpace(Instances instances) {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new ERPDistance()),
                buildContinuousERPParams(instances));
    }
}
