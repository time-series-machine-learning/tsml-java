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

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import static tsml.transformers.shapelet_tools.ShapeletTransformTimingUtilities.nanoToOp;

import tsml.filters.shapelet_filters.ShapeletFilter;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import static tsml.transformers.shapelet_tools.search_functions.ShapeletSearch.SearchType.FULL;
import static tsml.transformers.shapelet_tools.search_functions.ShapeletSearch.SearchType.IMPROVED_RANDOM;
import static tsml.transformers.shapelet_tools.search_functions.ShapeletSearch.SearchType.MAGNIFY;
import static tsml.transformers.shapelet_tools.search_functions.ShapeletSearch.SearchType.TABU;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import weka.core.Instances;
import utilities.TriFunction;

/**
 *
 * @author raj09hxu
 */
public class DefaultShapeletOptions {
    
    public static final Map<String, Function<Instances, ShapeletTransformFactoryOptions>> FACTORY_OPTIONS;
    static {
        Map<String, Function<Instances, ShapeletTransformFactoryOptions>> map = new HashMap();
        map.put("INDEPENDENT", DefaultShapeletOptions::createIndependentShapeletSearch);
        map.put("SHAPELET_I", DefaultShapeletOptions::createSHAPELET_I);
        map.put("SHAPELET_D", DefaultShapeletOptions::createSHAPELET_D);

        FACTORY_OPTIONS = Collections.unmodifiableMap(map);
    }
    
    public static final Map<String, TriFunction<Instances, Long, Long, ShapeletTransformFactoryOptions>> TIMED_FACTORY_OPTIONS;
    static {
        Map<String, TriFunction<Instances, Long, Long, ShapeletTransformFactoryOptions>> map = new HashMap();
        map.put("INDEPENDENT", DefaultShapeletOptions::createIndependentShapeletSearch_TIMED);
        map.put("SHAPELET_I", DefaultShapeletOptions::createSHAPELET_I_TIMED);
        map.put("SHAPELET_D", DefaultShapeletOptions::createSHAPELET_D_TIMED);
        map.put("SKIPPING", DefaultShapeletOptions::createSKIPPING_TIMED);
        map.put("TABU", DefaultShapeletOptions::createTABU_TIMED);
        map.put("RANDOM", DefaultShapeletOptions::createRANDOM_TIMED);
        map.put("MAGNIFY", DefaultShapeletOptions::createMAGNIFY_TIMED);

        TIMED_FACTORY_OPTIONS = Collections.unmodifiableMap(map);
    }
    
    /**
     * 
     * When calculating the timing for the number of shapelets. 
     * Its better to treat the timign as a univariate problem because in essence it is the same.
     * We just have more shapelets to consider, but calculating a single one is the same as unvariate times.
     * So the number we can calculate in a given time is the same.
     * 
     * @param train
     * @param time
     * @param seed
     * @return 
     */
    public static ShapeletTransformFactoryOptions createIndependentShapeletSearch_TIMED(Instances train, long time, long seed){  
        int n = train.numInstances();
        int m = utilities.multivariate_tools.MultivariateInstanceTools.channelLength(train);
        //create our search options.
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);
        searchBuilder.setSearchType(FULL); //default to FULL, if we need to sample will get overwrote.
        searchBuilder.setNumDimensions(utilities.multivariate_tools.MultivariateInstanceTools.numDimensions(train));
        
        //clamp K to 2000.
        int K = n > 2000 ? 2000 : n;   
        
        long numShapelets;
   
        //how much time do we have vs. how long our algorithm will take.
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        if(opCount.compareTo(opCountTarget) == 1){
            
            System.out.println("initiate timed");
            BigDecimal oct = new BigDecimal(opCountTarget);
            BigDecimal oc = new BigDecimal(opCount);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            
            //if we've not set a shapelet count, calculate one, based on the time set.
            numShapelets = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n,m,3,m);
            numShapelets *= prop.doubleValue();
             
             //we need to find atleast one shapelet in every series.
            searchBuilder.setSeed(seed);
            searchBuilder.setSearchType(IMPROVED_RANDOM);
            searchBuilder.setNumShapeletsToEvaluate(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(ShapeletDistance.DistanceType.DIMENSION)
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
    
    public static ShapeletTransformFactoryOptions createSHAPELET_I_TIMED(Instances train, long time, long seed){  
        int n = train.numInstances();
        int m = utilities.multivariate_tools.MultivariateInstanceTools.channelLength(train);
        //create our search options.
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);

        //clamp K to 2000.
        int K = n > 2000 ? 2000 : n;   
        
        long numShapelets;
   
        //how much time do we have vs. how long our algorithm will take.
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        //multiple the total opCount by K becauise for each comparison we do across dimensions.
        opCount = opCount.multiply(BigInteger.valueOf(utilities.multivariate_tools.MultivariateInstanceTools.numDimensions(train)));
        if(opCount.compareTo(opCountTarget) == 1){
            BigDecimal oct = new BigDecimal(opCountTarget);
            BigDecimal oc = new BigDecimal(opCount);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            
            //if we've not set a shapelet count, calculate one, based on the time set.
            numShapelets = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n,m,3,m);
            numShapelets *= prop.doubleValue();
             
             //we need to find atleast one shapelet in every series.
            searchBuilder.setSeed(seed);
            searchBuilder.setSearchType(IMPROVED_RANDOM);
            searchBuilder.setNumShapeletsToEvaluate(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(ShapeletDistance.DistanceType.INDEPENDENT)
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
    
    public static ShapeletTransformFactoryOptions createSHAPELET_D_TIMED(Instances train, long time, long seed){  
        int n = train.numInstances();
        int m = utilities.multivariate_tools.MultivariateInstanceTools.channelLength(train);
        //create our search options.
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);

        //clamp K to 2000.
        int K = n > 2000 ? 2000 : n;   
        
        long numShapelets;
   
        //how much time do we have vs. how long our algorithm will take.
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        opCount = opCount.multiply(BigInteger.valueOf(utilities.multivariate_tools.MultivariateInstanceTools.numDimensions(train)));
        //multiple the total opCount by K becauise for each comparison we do across dimensions.
        if(opCount.compareTo(opCountTarget) == 1){
            BigDecimal oct = new BigDecimal(opCountTarget);
            BigDecimal oc = new BigDecimal(opCount);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            
            //if we've not set a shapelet count, calculate one, based on the time set.
            numShapelets = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n,m,3,m);
            numShapelets *= prop.doubleValue();
             
             //we need to find atleast one shapelet in every series.
            searchBuilder.setSeed(seed);
            searchBuilder.setSearchType(IMPROVED_RANDOM);
            searchBuilder.setNumShapeletsToEvaluate(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(ShapeletDistance.DistanceType.DEPENDENT)
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
    
    public static ShapeletTransformFactoryOptions createSKIPPING_TIMED(Instances train, long time, long seed){  
        int n = train.numInstances();
        int m = train.numAttributes()-1;
        //create our search options.
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);

        //clamp K to 2000.
        int K = n > 2000 ? 2000 : n;   
           
        //how much time do we have vs. how long our algorithm will take.
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        //multiple the total opCount by K becauise for each comparison we do across dimensions.
        if(opCount.compareTo(opCountTarget) == 1){           
             //we need to find atleast one shapelet in every series.
            searchBuilder.setSeed(seed);
            searchBuilder.setSearchType(FULL);
            
            //find skipping values.
            int i = 1;
            while(ShapeletTransformTimingUtilities.calc(n, m, 3, m, i, i) > opCountTarget.doubleValue())
                    i++;
            System.out.println(i);
            searchBuilder.setPosInc(i);
            searchBuilder.setLengthInc(i);
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(ShapeletDistance.DistanceType.CACHED)
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
    
    
    public static ShapeletTransformFactoryOptions createTABU_TIMED(Instances train, long time, long seed){  
        int n = train.numInstances();
        int m = train.numAttributes()-1;
        //create our search options.
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);

        //clamp K to 2000.
        int K = n > 2000 ? 2000 : n;   
        
        long numShapelets;
   
        //how much time do we have vs. how long our algorithm will take.
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        //multiple the total opCount by K becauise for each comparison we do across dimensions.
        if(opCount.compareTo(opCountTarget) == 1){
            BigDecimal oct = new BigDecimal(opCountTarget);
            BigDecimal oc = new BigDecimal(opCount);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            
            //if we've not set a shapelet count, calculate one, based on the time set.
            numShapelets = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n,m,3,m);
            numShapelets *= prop.doubleValue();
             
             //we need to find atleast one shapelet in every series.
            searchBuilder.setSeed(seed);
            searchBuilder.setSearchType(TABU);
            searchBuilder.setNumShapeletsToEvaluate(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(ShapeletDistance.DistanceType.CACHED)
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
    
       public static ShapeletTransformFactoryOptions createRANDOM_TIMED(Instances train, long time, long seed){  
        int n = train.numInstances();
        int m = train.numAttributes()-1;
        //create our search options.
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);

        //clamp K to 2000.
        int K = n > 2000 ? 2000 : n;   
        
        long numShapelets;
   
        //how much time do we have vs. how long our algorithm will take.
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        //multiple the total opCount by K becauise for each comparison we do across dimensions.
        if(opCount.compareTo(opCountTarget) == 1){
            BigDecimal oct = new BigDecimal(opCountTarget);
            BigDecimal oc = new BigDecimal(opCount);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            
            //if we've not set a shapelet count, calculate one, based on the time set.
            numShapelets = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n,m,3,m);
            numShapelets *= prop.doubleValue();
             
             //we need to find atleast one shapelet in every series.
            searchBuilder.setSeed(seed);
            searchBuilder.setSearchType(IMPROVED_RANDOM);
            searchBuilder.setNumShapeletsToEvaluate(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(ShapeletDistance.DistanceType.CACHED)
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
       
    public static ShapeletTransformFactoryOptions createMAGNIFY_TIMED(Instances train, long time, long seed){  
        int n = train.numInstances();
        int m = train.numAttributes()-1;
        //create our search options.
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);

        //clamp K to 2000.
        int K = n > 2000 ? 2000 : n;   
        
        long numShapelets;
   
        //how much time do we have vs. how long our algorithm will take.
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        //multiple the total opCount by K becauise for each comparison we do across dimensions.
        if(opCount.compareTo(opCountTarget) == 1){
            BigDecimal oct = new BigDecimal(opCountTarget);
            BigDecimal oc = new BigDecimal(opCount);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            
            //if we've not set a shapelet count, calculate one, based on the time set.
            numShapelets = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n,m,3,m);
            numShapelets *= prop.doubleValue();
             
             //we need to find atleast one shapelet in every series.
            searchBuilder.setSeed(seed);
            searchBuilder.setSearchType(MAGNIFY);
            searchBuilder.setNumShapeletsToEvaluate(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(ShapeletDistance.DistanceType.CACHED)
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
    
    public static ShapeletTransformFactoryOptions createIndependentShapeletSearch(Instances train){
        ShapeletSearchOptions sOps = new ShapeletSearchOptions.Builder()
                                    .setMin(3)
                                    .setMax(utilities.multivariate_tools.MultivariateInstanceTools.channelLength(train))
                                    .setSearchType(ShapeletSearch.SearchType.FULL)
                                    .setNumDimensions(utilities.multivariate_tools.MultivariateInstanceTools.numDimensions(train))
                                    .build();

        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setSearchOptions(sOps)
                                            .setDistanceType(ShapeletDistance.DistanceType.DIMENSION)
                                            .setKShapelets(Math.min(2000,train.numInstances()))
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
        
    public static ShapeletTransformFactoryOptions createSHAPELET_I(Instances train){
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setMinLength(3)
                                            .setMaxLength(utilities.multivariate_tools.MultivariateInstanceTools.channelLength(train))
                                            .setDistanceType(ShapeletDistance.DistanceType.INDEPENDENT)
                                            .setKShapelets(Math.min(2000,train.numInstances()))
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
    
    public static ShapeletTransformFactoryOptions createSHAPELET_D(Instances train){
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.ShapeletTransformOptions()
                                            .setMinLength(3)
                                            .setMaxLength(utilities.multivariate_tools.MultivariateInstanceTools.channelLength(train))
                                            .setDistanceType(ShapeletDistance.DistanceType.DEPENDENT)
                                            .setKShapelets(Math.min(2000,train.numInstances()))
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }


    public static void main(String[] args){


        Instances train = null;


        ShapeletTransformFactoryOptions options = TIMED_FACTORY_OPTIONS.get("RANDOM").apply(train, 100000l, 0l);
        ShapeletFilter st = new ShapeletTransformFactory(options).getFilter();

    }
}
