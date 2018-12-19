/*
 Shapelet Factory: determines shapelet type and parameters based on the input
 * 1. distance caching: used if there is enough memory. 
 *           Distance caching requires O(nm^2)  
 * 2. number of shapelets:  
 *           Set to n*m/10, with a max size max(1000,2*m,2*n) . Assume we will post process cluster
 * 3. shapelet length range: 3 to train.numAttributes()-1
 * 
 */
package timeseriesweka.filters.shapelet_transforms;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import timeseriesweka.classifiers.ShapeletTransformClassifier;
import utilities.InstanceTools;
import utilities.generic_storage.Pair;
import weka.core.Instances;
import timeseriesweka.filters.shapelet_transforms.class_value.BinaryClassValue;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearchOptions;
import timeseriesweka.filters.shapelet_transforms.distance_functions.CachedSubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.OnlineSubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearchFactory;

/**
 *
 * @author Aaron Bostrom
 */
public class ShapeletTransformTimingUtilities
{
    //this is an arbritary cutoff value for whether we should start subsampling. It's about 7 days(ish). TODO: test.
    public static final long opCountThreshold = 1000000000000000l; 
    
    public static final long dayNano = 86400000000000l;
    public static final long nanoToOp = 10l; //1 op takes 10 nanoseconds.
    
    
    
    //we create the Map using params jon found.
    //lazy way to avoid reading a text file. 
    //it is immutable.
    public static final Map<String, Pair<Integer, Integer>> shapeletParams;
    static{
        shapeletParams = new HashMap<>();
        shapeletParams.put("Adiac", new Pair(3,10));
        shapeletParams.put("ArrowHead", new Pair(17,90));
        shapeletParams.put("Beef", new Pair(8,30));
        shapeletParams.put("BeetleFly", new Pair(30,101));
        shapeletParams.put("BirdChicken", new Pair(30,101));
        shapeletParams.put("Car", new Pair(16,57));
        shapeletParams.put("CBF", new Pair(46,90));
        shapeletParams.put("ChlorineConcentration", new Pair(7,20));
        shapeletParams.put("CinCECGtorso", new Pair(697,814));
        shapeletParams.put("Coffee", new Pair(18,30));
        shapeletParams.put("Computers", new Pair(15,267));
        shapeletParams.put("CricketX", new Pair(120,255));
        shapeletParams.put("CricketY", new Pair(132,262));
        shapeletParams.put("CricketZ", new Pair(118,257));
        shapeletParams.put("DiatomSizeReduction", new Pair(7,16));
        shapeletParams.put("DistalPhalanxOutlineAgeGroup", new Pair(7,31));
        shapeletParams.put("DistalPhalanxOutlineCorrect", new Pair(6,16));
        shapeletParams.put("DistalPhalanxTW", new Pair(17,31));
        shapeletParams.put("Earthquakes", new Pair(24,112));
        shapeletParams.put("ECGFiveDays", new Pair(24,76));
        shapeletParams.put("FaceAll", new Pair(70,128));
        shapeletParams.put("FaceFour", new Pair(20,120));
        shapeletParams.put("FacesUCR", new Pair(47,128));
        shapeletParams.put("Fiftywords", new Pair(170,247));
        shapeletParams.put("Fish", new Pair(22,60));
        shapeletParams.put("FordA", new Pair(50,298));
        shapeletParams.put("FordB", new Pair(38,212));
        shapeletParams.put("GunPoint", new Pair(24,55));
        shapeletParams.put("Haptics", new Pair(21,103));
        shapeletParams.put("Herrings", new Pair(30,101));
        shapeletParams.put("InlineSkate", new Pair(750,896));
        shapeletParams.put("ItalyPowerDemand", new Pair(7,14));
        shapeletParams.put("LargeKitchenAppliances", new Pair(13,374));
        shapeletParams.put("Lightning2", new Pair(47,160));
        shapeletParams.put("Lightning7", new Pair(20,80));
        shapeletParams.put("Mallat", new Pair(52,154));
        shapeletParams.put("MedicalImages", new Pair(9,35));
        shapeletParams.put("MiddlePhalanxOutlineAgeGroup", new Pair(8,31));
        shapeletParams.put("MiddlePhalanxOutlineCorrect", new Pair(5,12));
        shapeletParams.put("MiddlePhalanxTW", new Pair(7,31));
        shapeletParams.put("MoteStrain", new Pair(16,31));
        shapeletParams.put("NonInvasiveFatalECGThorax1", new Pair(5,61));
        shapeletParams.put("NonInvasiveFatalECGThorax2", new Pair(12,58));
        shapeletParams.put("OliveOil", new Pair(8,27));
        shapeletParams.put("OSULeaf", new Pair(141,330));
        shapeletParams.put("PhalangesOutlinesCorrect", new Pair(5,14));
        shapeletParams.put("Plane", new Pair(18,109));
        shapeletParams.put("ProximalPhalanxOutlineAgeGroup", new Pair(7,31));
        shapeletParams.put("ProximalPhalanxOutlineCorrect", new Pair(5,12));
        shapeletParams.put("ProximalPhalanxTW", new Pair(9,31));
        shapeletParams.put("PtNDeviceGroups", new Pair(51,261));
        shapeletParams.put("PtNDevices", new Pair(100,310));
        shapeletParams.put("RefrigerationDevices", new Pair(13,65));
        shapeletParams.put("ScreenType", new Pair(11,131));
        shapeletParams.put("ShapeletSim", new Pair(25,35));
        shapeletParams.put("SmallKitchenAppliances", new Pair(31,443));
        shapeletParams.put("SonyAIBORobotSurface1", new Pair(15,36));
        shapeletParams.put("SonyAIBORobotSurface2", new Pair(22,57));
        shapeletParams.put("StarlightCurves", new Pair(68,650));
        shapeletParams.put("SwedishLeaf", new Pair(11,45));
        shapeletParams.put("Symbols", new Pair(52,155));
        shapeletParams.put("SyntheticControl", new Pair(20,56));
        shapeletParams.put("ToeSegmentation1", new Pair(39,153));
        shapeletParams.put("ToeSegmentation2", new Pair(100,248));
        shapeletParams.put("Trace", new Pair(62,232));
        shapeletParams.put("TwoLeadECG", new Pair(7,13));
        shapeletParams.put("TwoPatterns", new Pair(20,71));
        shapeletParams.put("UWaveGestureLibraryX", new Pair(113,263));
        shapeletParams.put("UWaveGestureLibraryY", new Pair(122,273));
        shapeletParams.put("UWaveGestureLibraryZ", new Pair(135,238));
        shapeletParams.put("Wafer", new Pair(29,152));
        shapeletParams.put("WordSynonyms", new Pair(137,238));
        shapeletParams.put("Worms", new Pair(93,382));
        shapeletParams.put("WormsTwoClass", new Pair(46,377));
        shapeletParams.put("Yoga", new Pair(12,132));
        Collections.unmodifiableMap(shapeletParams);
    }
    
    public static final double MEM_CUTOFF = 0.5;
    public static final int MAX_NOS_SHAPELETS = 1000;    
    
    public static ShapeletTransform createCachedTransform()
    {
        ShapeletTransform st = new ShapeletTransform();
        st.setSubSeqDistance(new CachedSubSeqDistance());
        return st;
    }
    
    public static ShapeletTransform createOnlineTransform()
    {
        ShapeletTransform st = new ShapeletTransform();
        st.setSubSeqDistance(new OnlineSubSeqDistance());
        return st;
    }
    
    public static ShapeletTransform createBasicTransform(int n, int m){
        ShapeletTransform fst = new ShapeletTransform();
        fst.setNumberOfShapelets(n);
        fst.setShapeletMinAndMax(3, m);
        fst.supressOutput();
        return fst;
    }
    
    public static ShapeletTransform createTransform(Instances train){
        int numClasses = train.numClasses();
        int numInstances = train.numInstances() <= 2000 ? train.numInstances() : 2000;
        int numAttributes = train.numAttributes()-1;
        
        ShapeletTransform transform;
        if(numClasses == 2){
            transform = new ShapeletTransform();
            System.out.println("2 class");
        }else{
            transform = new BalancedClassShapeletTransform();
            transform.setClassValue(new BinaryClassValue());
        }
        
        //transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
        transform.setShapeletMinAndMax(3, numAttributes);
        transform.setNumberOfShapelets(numInstances);
        transform.useCandidatePruning();
        transform.turnOffLog();
        transform.setRoundRobin(true);
        transform.supressOutput();
        
        return transform;
    }
    
    
    long getAvailableMemory()
    {
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long maxMemory = runtime.maxMemory();
        long usedMemory = totalMemory - freeMemory;
        long availableMemory = maxMemory - usedMemory;
        return availableMemory;
    }

    // Method to estimate min/max shapelet length for a given data
    public static int[] estimateMinAndMax(Instances data, ShapeletTransform st)
    {
        ShapeletTransform st1 = null;
        try {
            //we need to clone the FST.
            st1 = st.getClass().newInstance();
            st1.setClassValue(st.classValue.getClass().newInstance());
            st1.setSubSeqDistance(st.subseqDistance.getClass().newInstance());
        } catch (InstantiationException | IllegalAccessException ex) {
            System.out.println("Exception: ");
        }
        
        if(st1 == null)
            st1 = new ShapeletTransform();
        
        
        st1.supressOutput();
        
        ArrayList<Shapelet> shapelets = new ArrayList<>();
        st.supressOutput();
        st.turnOffLog();

        Instances randData = new Instances(data);
        Instances randSubset;

        for (int i = 0; i < 10; i++)
        {
            randData.randomize(new Random());
            randSubset = new Instances(randData, 0, 10);
            shapelets.addAll(st1.findBestKShapeletsCache(10, randSubset, 3, randSubset.numAttributes() - 1));
        }

        Collections.sort(shapelets, new ShapeletLengthComparator());
        int min = shapelets.get(24).getLength();
        int max = shapelets.get(74).getLength();

        return new int[]{min,max};
    }
    
    //bog standard min max estimation.
    public static int[] estimateMinAndMax(Instances data)
    {
        return estimateMinAndMax(data, new ShapeletTransform());
    }
    
    //Class implementing comparator which compares shapelets according to their length
    public static class ShapeletLengthComparator implements Comparator<Shapelet>{
   
        @Override
        public int compare(Shapelet shapelet1, Shapelet shapelet2){
            int shapelet1Length = shapelet1.getLength();        
            int shapelet2Length = shapelet2.getLength();

            return Integer.compare(shapelet1Length, shapelet2Length);  
        }
    }
    
    public static long calculateNumberOfShapelets(Instances train, int minShapeletLength, int maxShapeletLength){      
        return calculateNumberOfShapelets(train.numInstances(), train.numAttributes()-1, minShapeletLength, maxShapeletLength);
    }
    
    //Aaron
    //verified on Trace dataset from Ye2011 with 7,480,200 shapelets : page 158.
    //we assume as fixed length.
    public static long calculateNumberOfShapelets(int numInstances, int numAttributes, int minShapeletLength, int maxShapeletLength){
        long numShapelets=0;
        
        //calculate number of shapelets in a single instance.
        for (int length = minShapeletLength; length <= maxShapeletLength; length++) {
            numShapelets += numAttributes - length + 1;
        }
        
        numShapelets*=numInstances;
        
        return numShapelets;
    }
    
    
    public static long calculateShapeletsFromOpCount(int numInstances, int numAttributes, long opCount){
        return (long)((6.0* (double)opCount) / ( numAttributes * ((numAttributes*numAttributes) + (3*numAttributes) + 2)*(numInstances-1)));
    }
    
    public static long calculateOperations(Instances train, int minShapeletLength, int maxShapeletLength){      
        return calculateOperations(train.numInstances(), train.numAttributes()-1, minShapeletLength, maxShapeletLength);
    }
    
    //verified correct by counting ops in transform
    public static long calculateOperations(int numInstances, int numAttributes, int minShapeletLength, int maxShapeletLength){
        return calculateOperationsWithSkipping(numInstances, numAttributes, minShapeletLength, maxShapeletLength, 1,1, 1.0f);
    }
    
    public static long calculateOperationsWithProportion(Instances train, int minShapeletLength, int maxShapeletLength, float proportion){      
        return calculateOperationsWithSkipping(train.numInstances(), train.numAttributes()-1, minShapeletLength, maxShapeletLength,1,1,proportion);
    }
    
    
    //verified correct by counting ops in transform
    //not exact with shapelet proportion, because of nondeterministic nature.
    public static long calculateOperationsWithSkipping(int numInstances, int numAttributes, int minShapeletLength, int maxShapeletLength, int posSkip, int lengthSkip, float Shapeletproportion){
        long numOps=0;

        int shapelets =0;
        //calculate number of shapelets in a single instance.
        for (int length = minShapeletLength; length <= maxShapeletLength; length+=lengthSkip) {
            
            long shapeletsLength = (long) (((numAttributes - length + 1) / posSkip) * Shapeletproportion);
            shapelets+=shapeletsLength;
            
            //System.out.println(shapeletsLength);
            
            long shapeletsCompared = (numAttributes - length + 1);
            
            //each shapelet gets compared to all other subsequences, and they make l operations per comparison for every series..
            long comparisonPerSeries = shapeletsLength * shapeletsCompared * length * (numInstances-1);
            
            numOps +=comparisonPerSeries; 
        }

        //for every series.
        numOps *= numInstances;
        return numOps;
    }
    
    
    public static double calc(int n, int m, int min, int max, int pos, int len)
    {
        double numOps =0;
        
        //-1 from max because we index from 0.
        for(int length = 0; length <= ((max-min)/len); length++){
                        
            int currentLength = (len*length) + min;
            double shapeletsLength = Math.ceil((double)(m - currentLength + 1) / (double) pos); //shapelts found.
            
            double shapeletsCompared = (m - currentLength + 1);
            
            numOps += shapeletsLength*shapeletsCompared*currentLength;
        }

        numOps*= n * (n-1);
        return numOps;
    }    
 
    public static long calcShapelets(int n, int m, int min, int max, int pos, int len)
    {
        long numOps =0;
        
        //-1 from max because we index from 0.
        for(int length = 0; length <= ((max-min)/len); length++){
                        
            int currentLength = (len*length) + min;
            long shapeletsLength = (long) Math.ceil((double)(m - currentLength + 1) / (double) pos); //shapelts found.
            
            numOps += shapeletsLength;
        }

        numOps*= n;
        return numOps;
    }    

    
    
    
    public static BigInteger calculateOps(int n, int m, int posS, int lenS){
       
        BigInteger nSqd = new BigInteger(Long.toString(n));
        nSqd = nSqd.pow(2);
        
        long lenSqd = lenS*lenS;
        
        
        BigInteger mSqd = new BigInteger(Long.toString(m));
        mSqd = mSqd.pow(2);
        
        BigInteger temp1 = mSqd;
        temp1 = mSqd.multiply(new BigInteger(Long.toString(m)));
        BigInteger temp2 = mSqd.multiply(new BigInteger("7"));
        BigInteger temp3 = new BigInteger(Long.toString(m*(lenSqd - (18*lenS) + 27)));
        BigInteger temp4 = new BigInteger(Long.toString(lenS*((5*lenS) - 24) + 27));
        
        BigInteger bg = new BigInteger("0");
        bg = bg.add(temp1);
        bg = bg.add(temp2);
        bg = bg.subtract(temp3);
        bg = bg.add(temp4);
        bg = bg.multiply(nSqd.subtract(new BigInteger(Long.toString(n))));
        bg = bg.multiply(new BigInteger(Long.toString((m-3))));
        bg = bg.divide(new BigInteger(Long.toString((12 * posS * lenS))));
        return bg;
    }
    
    
    public static double calculateN(int n, int m, long time){
        long opCount = time / nanoToOp; 
        
        BigDecimal numerator = new BigDecimal(Long.toString(12*opCount));
        
        BigInteger temp1 = new BigInteger(Long.toString(m*m));
        temp1 = temp1.multiply(new BigInteger(Long.toString(m)));
        BigInteger temp2 = new BigInteger(Long.toString(7*m*m));
        BigInteger temp3 = new BigInteger(Long.toString(10*m));
        BigInteger temp4 = new BigInteger(Long.toString(8));
        
        temp1 = temp1.add(temp2);
        temp1 = temp1.subtract(temp3);
        temp1 = temp1.add(temp4);
        temp1 = temp1.multiply(new BigInteger(Long.toString(m-3)));
        
        BigDecimal denominator = new BigDecimal(temp1);

        BigDecimal result = utilities.StatisticalUtilities.sqrt(numerator.divide(denominator, MathContext.DECIMAL32), MathContext.DECIMAL32);
        
        //sqrt result.
        result = result.divide(new BigDecimal(n), MathContext.DECIMAL32);
        
        return Math.min(result.doubleValue(), 1.0); //return the proportion of n.
    }
    
    // added by JAL - both edited versions of ShapeletTransformClassifier.createTransformData
    // param changed to hours from nanos to make it human readable
    // seed included; set to 0 by default in ShapeletTransformClassifier, so same behaviour included unless specified
    public static ShapeletTransform createTransformWithTimeLimit(Instances train, double hours){
        return createTransformWithTimeLimit(train, hours, 0);
    }
    public static ShapeletTransform createTransformWithTimeLimit(Instances train, double hours, int seed){
        int minimumRepresentation = ShapeletTransformClassifier.minimumRepresentation;
        long nanoPerHour = 3600000000000l;
        long time = (long)(nanoPerHour*hours);
        
        int n = train.numInstances();
        int m = train.numAttributes()-1;

        ShapeletTransform transform;
        //construct shapelet classifiers from the factory.
        transform = ShapeletTransformTimingUtilities.createTransform(train);
        
        //Stop it printing everything
        transform.supressOutput();
        
        //at the moment this could be overrided.
        //transform.setSearchFunction(new LocalSearch(3, m, 10, seed));

        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        
        //we need to resample.
        if(opCount.compareTo(opCountTarget) == 1){
            
            double recommendedProportion = ShapeletTransformTimingUtilities.calculateN(n, m, time);
            
            //calculate n for minimum class rep of 25.
            int small_sf = InstanceTools.findSmallestClassAmount(train);           
            double proportion = 1.0;
            if (small_sf>minimumRepresentation){
                proportion = (double)minimumRepresentation/(double)small_sf;
            }
            
            //if recommended is smaller than our cutoff threshold set it to the cutoff.
            if(recommendedProportion < proportion){
                recommendedProportion = proportion;
            }
            
            //subsample out dataset.
            Instances subsample = utilities.InstanceTools.subSampleFixedProportion(train, recommendedProportion, seed);
            
            int i=1;
            //if we've properly resampled this should pass on first go. IF we haven't we'll try and reach our target. 
            //calculate N is an approximation, so the subsample might need to tweak q and p just to bring us under. 
            while(ShapeletTransformTimingUtilities.calculateOps(subsample.numInstances(), m, i, i).compareTo(opCountTarget) == 1){
                i++;
            }
            double percentageOfSeries = (double)i/(double)m * 100.0;
            
            //we should look for less shapelets if we've resampled. 
            //e.g. Eletric devices can be sampled to from 8000 for 2000 so we should be looking for 20,000 shapelets not 80,000
            transform.setNumberOfShapelets(subsample.numInstances());
            ShapeletSearchOptions sOptions = new ShapeletSearchOptions.Builder().setMin(3).setMax(m).setLengthInc(i).setPosInc(i).build();
            transform.setSearchFunction(new ShapeletSearchFactory(sOptions).getShapeletSearch());
            transform.process(subsample);
        }
        return transform;
    }
    
    
    public static void main(String[] args) throws IOException
    {     
        
       
        /*System.out.println(calculateOperationsWithSkipping(67, 23, 3, 23, 1, 1, 1.0f));
    
        System.out.println(calculateNumberOfShapelets(67,23,3,23));
        System.out.println((int) (calculateNumberOfShapelets(67,23,3,23) * 0.5f));
        
        System.out.println(calculateOperationsWithSkipping(67, 23, 3, 23, 1, 1, 0.5f));*/
        
        
        /*String dirPath = "F:\\Dropbox\\TSC Problems (1)\\";
        File dir  = new File(dirPath);
        for(File dataset : dir.listFiles()){
            if(!dataset.isDirectory()) continue;
            
            String f = dataset.getPath()+ File.separator + dataset.getName() + "_TRAIN.arff";
        
            Instances train = ClassifierTools.loadData(f);
            
            long shapelets = calculateNumberOfShapelets(train, 3, train.numAttributes()-1);
            //long ops = calculateOperations(train, 3, train.numAttributes()-1);
            
            System.out.print(dataset.getName() + ",");
            System.out.print(train.numAttributes()-1 + ",");
            System.out.print(train.numInstances() + ",");
            int min = 3;
            int max = train.numAttributes()-1;
            int pos = 1;
            int len = 1;
            
            FullShapeletTransform transform = new FullShapeletTransform();
            transform.setSearchFunction(new ShapeletSearch(min,max,len, pos));
            transform.setSubSeqDistance(new SubSeqDistance());
            transform.supressOutput();
            transform.process(train);
            long ops3 = transform.getCount();
            
            long ops4 = calc(train.numInstances(), train.numAttributes()-1, min, max,pos,len);
            
            double n = calculateN(train.numInstances(), train.numAttributes()-1, dayNano);

            
            //calculate n for minimum class rep of 25.
            int small_sf = InstanceTools.findSmallestClassAmount(train);           
            double proportion = 1.0;
            if (small_sf>25){
                proportion = (double)25/(double)small_sf;
            }
            
            
            System.out.print(ops4 + ",");
            System.out.print(n + ",");
            System.out.print(proportion + "\n");
        }*/
    }
}
