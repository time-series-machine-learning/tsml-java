/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */   
package tsml.filters.shapelet_transforms;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;
import tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance;
import tsml.filters.shapelet_transforms.class_value.BinaryClassValue;
import tsml.filters.shapelet_transforms.class_value.NormalClassValue;
import tsml.filters.shapelet_transforms.search_functions.ShapeletSearch;
import tsml.filters.shapelet_transforms.search_functions.ShapeletSearchFactory;
import tsml.filters.shapelet_transforms.search_functions.ShapeletSearchOptions;
import tsml.filters.shapelet_transforms.distance_functions.CachedSubSeqDistance;
import tsml.filters.shapelet_transforms.distance_functions.DimensionDistance;
import tsml.filters.shapelet_transforms.distance_functions.ImprovedOnlineSubSeqDistance;
import tsml.filters.shapelet_transforms.distance_functions.MultivariateDependentDistance;
import tsml.filters.shapelet_transforms.distance_functions.MultivariateIndependentDistance;
import tsml.filters.shapelet_transforms.distance_functions.OnlineCachedSubSeqDistance;
import tsml.filters.shapelet_transforms.distance_functions.OnlineSubSeqDistance;
import tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType;
import utilities.rescalers.NoRescaling;
import utilities.rescalers.SeriesRescaler;
import utilities.rescalers.ZNormalisation;
import utilities.rescalers.ZStandardisation;

import static tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType.CACHED;
import static tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType.DEPENDENT;
import static tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType.DIMENSION;
import static tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType.IMPROVED_ONLINE;
import static tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType.INDEPENDENT;
import static tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType.NORMAL;
import static tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType.ONLINE;
import static tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType.ONLINE_CACHED;
import static tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance.RescalerType.*;

/**
 *
 * @author Aaron
 */
public class ShapeletTransformFactory {
     
    private static final Map<DistanceType, Supplier<SubSeqDistance>> distanceFunctions = createDistanceTable();
    private static Map<DistanceType, Supplier<SubSeqDistance>> createDistanceTable(){
        //DistanceType{NORMAL, ONLINE, IMP_ONLINE, CACHED, ONLINE_CACHED, DEPENDENT, INDEPENDENT};
        Map<DistanceType, Supplier<SubSeqDistance>> dCons = new HashMap<DistanceType, Supplier<SubSeqDistance>>();
        dCons.put(NORMAL, SubSeqDistance::new);
        dCons.put(ONLINE, OnlineSubSeqDistance::new);
        dCons.put(IMPROVED_ONLINE, ImprovedOnlineSubSeqDistance::new);
        dCons.put(CACHED, CachedSubSeqDistance::new);
        dCons.put(ONLINE_CACHED, OnlineCachedSubSeqDistance::new);
        dCons.put(DEPENDENT, MultivariateDependentDistance::new);
        dCons.put(INDEPENDENT, MultivariateIndependentDistance::new);
        dCons.put(DIMENSION, DimensionDistance::new);
        return dCons;
    }

    private static final Map<SubSeqDistance.RescalerType, Supplier<SeriesRescaler>> rescalingFunctions = createRescalerTable();
    private static Map<SubSeqDistance.RescalerType, Supplier<SeriesRescaler>> createRescalerTable(){
        //RescalerType{NONE, NORMALISATION, STANDARDISATION};
        Map<SubSeqDistance.RescalerType, Supplier<SeriesRescaler>> dCons = new HashMap<SubSeqDistance.RescalerType, Supplier<SeriesRescaler>>();
        dCons.put(NONE, NoRescaling::new);
        dCons.put(NORMALISATION, ZNormalisation::new);
        dCons.put(STANDARDISATION, ZStandardisation::new);

        return dCons;
    }
    
    ShapeletTransformFactoryOptions options;

    public ShapeletTransformFactory(ShapeletTransformFactoryOptions op){
        options = op;
    }
    
    public ShapeletTransform getTransform(){
        //build shapelet transform based on options.
        ShapeletTransform st = createTransform(options.isBalanceClasses());
        st.setClassValue(createClassValue(options.isBinaryClassValue()));
        st.setShapeletMinAndMax(options.getMinLength(), options.getMaxLength());
        st.setNumberOfShapelets(options.getkShapelets());
        st.setSubSeqDistance(createDistance(options.getDistance()));
        st.setRescaler(createRescaler(options.getRescalerType()));
        st.setSearchFunction(createSearch(options.getSearchOptions()));
        st.setQualityMeasure(options.getQualityChoice());
        st.setRoundRobin(options.useRoundRobin());
        st.setCandidatePruning(options.useCandidatePruning());
        //st.supressOutput();
        return st;
    }



    //time, number of cases, and length of series
    public static ShapeletTransform createDefaultTimedTransform(long numShapeletsToEvaluate, int n, int m, long seed){
        ShapeletSearchOptions sOp = new ShapeletSearchOptions.Builder()
                                        .setMin(3)
                                        .setMax(m)
                                        .setSearchType(ShapeletSearch.SearchType.IMPROVED_RANDOM)
                                        .setNumShapeletsToEvaluate(numShapeletsToEvaluate)
                                        .setSeed(seed)
                                        .build();
                
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .useClassBalancing()
                                            .useBinaryClassValue()
                                            .useCandidatePruning()
                                            .setKShapelets(Math.min(2000, n))
                                            .setDistanceType(DistanceType.NORMAL)
                                            .setSearchOptions(sOp)
                                            .setRescalerType(NORMALISATION)
                                            .build();
        
        return new ShapeletTransformFactory(options).getTransform();
    }
    
    private ShapeletSearch createSearch(ShapeletSearchOptions sOp){
        return new ShapeletSearchFactory(sOp).getShapeletSearch();
    }
    
    private NormalClassValue createClassValue(boolean classValue){
        return classValue ?  new BinaryClassValue() : new NormalClassValue();
    }
    
    private ShapeletTransform createTransform(boolean balance){
        return balance ?  new BalancedClassShapeletTransform() : new ShapeletTransform();
    }
    
    private SubSeqDistance createDistance(DistanceType dist){
            return distanceFunctions.get(dist).get();
    }

    private SeriesRescaler createRescaler(SubSeqDistance.RescalerType rescalerType) {
            return rescalingFunctions.get(rescalerType).get();
    }
    
    
    public static void main(String[] args) {
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                .useClassBalancing()
                .setKShapelets(1000)
                .setDistanceType(NORMAL)
                .setRescalerType(NORMALISATION)
                .setMinLength(3)
                .setMaxLength(100)
                .build();
        
        ShapeletTransformFactory factory = new ShapeletTransformFactory(options);
        ShapeletTransform st = factory.getTransform();
    }
}
