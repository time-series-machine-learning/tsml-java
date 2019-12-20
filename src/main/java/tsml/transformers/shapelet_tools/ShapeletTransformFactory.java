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
package tsml.transformers.shapelet_tools;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

import tsml.filters.shapelet_filters.BalancedClassShapeletFilter;
import tsml.filters.shapelet_filters.ShapeletFilter;
import tsml.transformers.ShapeletTransform;
import tsml.transformers.shapelet_tools.distance_functions.*;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance;
import tsml.transformers.shapelet_tools.class_value.BinaryClassValue;
import tsml.transformers.shapelet_tools.class_value.NormalClassValue;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchFactory;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType;
import utilities.rescalers.NoRescaling;
import utilities.rescalers.SeriesRescaler;
import utilities.rescalers.ZNormalisation;
import utilities.rescalers.ZStandardisation;

import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType.CACHED;
import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType.DEPENDENT;
import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType.DIMENSION;
import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType.IMPROVED_ONLINE;
import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType.INDEPENDENT;
import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType.NORMAL;
import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType.ONLINE;
import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType.ONLINE_CACHED;
import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.RescalerType.*;

/**
 *
 * @author Aaron
 */
public class ShapeletTransformFactory {
     
    private static final Map<DistanceType, Supplier<ShapeletDistance>> distanceFunctions = createDistanceTable();
    private static Map<DistanceType, Supplier<ShapeletDistance>> createDistanceTable(){
        //DistanceType{NORMAL, ONLINE, IMP_ONLINE, CACHED, ONLINE_CACHED, DEPENDENT, INDEPENDENT};
        Map<DistanceType, Supplier<ShapeletDistance>> dCons = new HashMap<DistanceType, Supplier<ShapeletDistance>>();
        dCons.put(NORMAL, ShapeletDistance::new);
        dCons.put(ONLINE, OnlineShapeletDistance::new);
        dCons.put(IMPROVED_ONLINE, ImprovedOnlineShapeletDistance::new);
        dCons.put(CACHED, CachedShapeletDistance::new);
        dCons.put(ONLINE_CACHED, OnlineCachedShapeletDistance::new);
        dCons.put(DEPENDENT, MultivariateDependentDistance::new);
        dCons.put(INDEPENDENT, MultivariateIndependentDistance::new);
        dCons.put(DIMENSION, DimensionDistance::new);
        return dCons;
    }

    private static final Map<ShapeletDistance.RescalerType, Supplier<SeriesRescaler>> rescalingFunctions = createRescalerTable();
    private static Map<ShapeletDistance.RescalerType, Supplier<SeriesRescaler>> createRescalerTable(){
        //RescalerType{NONE, NORMALISATION, STANDARDISATION};
        Map<ShapeletDistance.RescalerType, Supplier<SeriesRescaler>> dCons = new HashMap<ShapeletDistance.RescalerType, Supplier<SeriesRescaler>>();
        dCons.put(NONE, NoRescaling::new);
        dCons.put(NORMALISATION, ZNormalisation::new);
        dCons.put(STANDARDISATION, ZStandardisation::new);

        return dCons;
    }
    
    ShapeletTransformFactoryOptions options;

    public ShapeletTransformFactory(ShapeletTransformFactoryOptions op){
        options = op;
    }
    
    public ShapeletFilter getFilter(){
        //build shapelet transform based on options.
        ShapeletFilter st = createFilter(options.isBalanceClasses());
//        ShapeletTransform st = createTransform(options.isBalanceClasses());
        st.setUseBalancedClasses(options.isBalanceClasses());
        st.setClassValue(createClassValue(options.isBinaryClassValue()));
        st.setShapeletMinAndMax(options.getMinLength(), options.getMaxLength());
        st.setNumberOfShapelets(options.getkShapelets());
        st.setSubSeqDistance(createDistance(options.getDistance()));
        st.setRescaler(createRescaler(options.getRescalerType()));
        st.setSearchFunction(createSearch(options.getSearchOptions()));
        st.setQualityMeasure(options.getQualityChoice());
        st.setRoundRobin(options.useRoundRobin());
        st.setCandidatePruning(options.useCandidatePruning());
        st.setNumberOfShapelets(options.getkShapelets());
        st.setShapeletMinAndMax(options.getMinLength(),options.getMaxLength());
        //st.supressOutput();
        return st;
    }

    public ShapeletTransform getTransform(){
        //build shapelet transform based on options.
        ShapeletTransform st = new ShapeletTransform();//Removed the balance type
        st.setUseBalancedClasses(options.isBalanceClasses());
        st.setClassValue(createClassValue(options.isBinaryClassValue()));
        st.setShapeletMinAndMax(options.getMinLength(), options.getMaxLength());
        st.setNumberOfShapelets(options.getkShapelets());
        st.setShapeletDistance(createDistance(options.getDistance()));
        st.setRescaler(createRescaler(options.getRescalerType()));
        st.setSearchFunction(createSearch(options.getSearchOptions()));
        st.setQualityMeasure(options.getQualityChoice());
        st.setRoundRobin(options.useRoundRobin());
        st.setCandidatePruning(options.useCandidatePruning());
        st.setNumberOfShapelets(options.getkShapelets());
        st.setShapeletMinAndMax(options.getMinLength(),options.getMaxLength());
        //st.supressOutput();
        return st;
    }



    //time, number of cases, and length of series
    public static ShapeletFilter createDefaultTimedTransform(long numShapeletsToEvaluate, int n, int m, long seed){
        ShapeletSearchOptions sOp = new ShapeletSearchOptions.Builder()
                                        .setMin(3)
                                        .setMax(m)
                                        .setSearchType(ShapeletSearch.SearchType.IMPROVED_RANDOM)
                                        .setNumShapeletsToEvaluate(numShapeletsToEvaluate)
                                        .setSeed(seed)
                                        .build();
                
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .useClassBalancing()
                                            .useBinaryClassValue()
                                            .useCandidatePruning()
                                            .setKShapelets(Math.min(2000, n))
                                            .setDistanceType(DistanceType.NORMAL)
                                            .setSearchOptions(sOp)
                                            .setRescalerType(NORMALISATION)
                                            .build();
        
        return new ShapeletTransformFactory(options).getFilter();
    }
    
    private ShapeletSearch createSearch(ShapeletSearchOptions sOp){
        return new ShapeletSearchFactory(sOp).getShapeletSearch();
    }
    
    private NormalClassValue createClassValue(boolean classValue){
        return classValue ?  new BinaryClassValue() : new NormalClassValue();
    }

    private ShapeletFilter createFilter(boolean balance){
        return balance ?  new BalancedClassShapeletFilter() : new ShapeletFilter();
    }

    private ShapeletDistance createDistance(DistanceType dist){
            return distanceFunctions.get(dist).get();
    }

    private SeriesRescaler createRescaler(ShapeletDistance.RescalerType rescalerType) {
            return rescalingFunctions.get(rescalerType).get();
    }
    
    
    public static void main(String[] args) {
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                .useClassBalancing()
                .setKShapelets(1000)
                .setDistanceType(NORMAL)
                .setRescalerType(NORMALISATION)
                .setMinLength(3)
                .setMaxLength(100)
                .build();
        
        ShapeletTransformFactory factory = new ShapeletTransformFactory(options);
        ShapeletFilter st = factory.getFilter();
    }
}
