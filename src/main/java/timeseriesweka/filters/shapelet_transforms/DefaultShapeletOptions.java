/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import static timeseriesweka.filters.shapelet_transforms.ShapeletTransformTimingUtilities.nanoToOp;
import timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch;
import static timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch.SearchType.FULL;
import static timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch.SearchType.IMP_RANDOM;
import static timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch.SearchType.MAGNIFY;
import static timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch.SearchType.TABU;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearchOptions;
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
        searchBuilder.setNumDimensions(utilities.multivariate_tools.MultivariateInstanceTools.numChannels(train));
        
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
            searchBuilder.setSearchType(IMP_RANDOM);
            searchBuilder.setNumShapelets(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(SubSeqDistance.DistanceType.DIMENSION)
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
        opCount = opCount.multiply(BigInteger.valueOf(utilities.multivariate_tools.MultivariateInstanceTools.numChannels(train)));
        if(opCount.compareTo(opCountTarget) == 1){
            BigDecimal oct = new BigDecimal(opCountTarget);
            BigDecimal oc = new BigDecimal(opCount);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            
            //if we've not set a shapelet count, calculate one, based on the time set.
            numShapelets = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n,m,3,m);
            numShapelets *= prop.doubleValue();
             
             //we need to find atleast one shapelet in every series.
            searchBuilder.setSeed(seed);
            searchBuilder.setSearchType(IMP_RANDOM);
            searchBuilder.setNumShapelets(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(SubSeqDistance.DistanceType.INDEPENDENT)
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
        opCount = opCount.multiply(BigInteger.valueOf(utilities.multivariate_tools.MultivariateInstanceTools.numChannels(train)));
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
            searchBuilder.setSearchType(IMP_RANDOM);
            searchBuilder.setNumShapelets(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(SubSeqDistance.DistanceType.DEPENDENT)
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

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(SubSeqDistance.DistanceType.CACHED)
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
            searchBuilder.setNumShapelets(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(SubSeqDistance.DistanceType.CACHED)
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
            searchBuilder.setSearchType(IMP_RANDOM);
            searchBuilder.setNumShapelets(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(SubSeqDistance.DistanceType.CACHED)
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
            searchBuilder.setNumShapelets(numShapelets);
            
            // can't have more final shapelets than we actually search through.
            K =  numShapelets > K ? K : (int) numShapelets;
        }

        
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setKShapelets(K)
                                            .setSearchOptions(searchBuilder.build())
                                            .setDistanceType(SubSeqDistance.DistanceType.CACHED)
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
                                    .setNumDimensions(utilities.multivariate_tools.MultivariateInstanceTools.numChannels(train))
                                    .build();

        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setSearchOptions(sOps)
                                            .setDistanceType(SubSeqDistance.DistanceType.DIMENSION)
                                            .setKShapelets(Math.min(2000,train.numInstances()))
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
        
    public static ShapeletTransformFactoryOptions createSHAPELET_I(Instances train){
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setMinLength(3)
                                            .setMaxLength(utilities.multivariate_tools.MultivariateInstanceTools.channelLength(train))
                                            .setDistanceType(SubSeqDistance.DistanceType.INDEPENDENT)
                                            .setKShapelets(Math.min(2000,train.numInstances()))
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
    
    public static ShapeletTransformFactoryOptions createSHAPELET_D(Instances train){
        ShapeletTransformFactoryOptions options = new ShapeletTransformFactoryOptions.Builder()
                                            .setMinLength(3)
                                            .setMaxLength(utilities.multivariate_tools.MultivariateInstanceTools.channelLength(train))
                                            .setDistanceType(SubSeqDistance.DistanceType.DEPENDENT)
                                            .setKShapelets(Math.min(2000,train.numInstances()))
                                            .useBinaryClassValue()
                                            .useClassBalancing()
                                            .useCandidatePruning()
                                            .build();
        return options;
    }
}
