package timeseriesweka.filters;

import java.util.ArrayList;
import static timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance.ROUNDING_ERROR_CORRECTION;
import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 * 
 * Note: if desired, once a set of Instances are processed the accessor methods getDistances() and getIndices() can be used for manually using the values, rather than having to 
 *       extract the data back from the output Instances of process
 * 
 * To-do:
 *      - Cache distances that will be reused (is it worth it? Probably not since it's offline, but might be important for very large problems and small windows)
 *      - Implement 'stride' - not sure if this makes sense particularly, but we could allow it so the user can change the step between comparison subseries 
 *        that are evaluated when calculating the profile (e.g. not every 1 index, every 2, 3, ... etc.)
 * 
 */
public class MatrixProfile extends SimpleBatchFilter{
    
    private int windowSize = 3;
    private final int stride = 1; // to-do later (maybe!)
    private double[][] distances;
    private int[][] indices;
    
    public MatrixProfile(int windowSize){
        this.windowSize = windowSize;
    }

    @Override
    public Instances process(Instances instances) throws Exception {
        
        int seriesLength = instances.numAttributes()-(instances.classIndex()>=0?1:0);
        
        if(windowSize < 3){
            throw new Exception("Error: window must be at least 3. You have specified "+windowSize);
        }
        
        if(windowSize > seriesLength){
            throw new Exception("Error: window must be smaller than the number of attributes. Window length: "+windowSize+", series length: "+seriesLength);
        }
        
        if(seriesLength/4 < windowSize){
            throw new Exception("Error: the series length must be at least 4 times larger than the window size to satisfy the exclusion zone criteria for trivial matches. These instances have a series length of "+seriesLength+"; the maximum window size is therefore "+(seriesLength/4)+" and you have specified "+windowSize);
        }
        
        SingleInstanceMatrixProfile mpIns;
        Instances transformed = this.determineOutputFormat(instances);
        Instance out;
        
        this.distances = new double[instances.numInstances()][];
        this.indices = new int[instances.numInstances()][];
        
        for(int ins = 0; ins < instances.numInstances(); ins++){
            mpIns = new SingleInstanceMatrixProfile(instances.get(ins),this.windowSize, this.stride);
            out = new DenseInstance(transformed.numAttributes());
            
            distances[ins] = mpIns.distances;
            indices[ins] = mpIns.indices;
            
            for(int i = 0; i < mpIns.distances.length; i++){
                out.setValue(i, mpIns.distances[i]);
                
            }
            
            if(instances.classIndex() >=0){
                out.setValue(mpIns.distances.length, instances.instance(ins).classValue());
            }
            transformed.add(out);
        }
        return transformed;
    }
    
    /**
     * An alternate version of process() that returns two instances objects in an array. The first is the same output as process() (matrix profile distances) and the second is the indices of the closest match
     * 
     * Input: instances - instances to be transformed 
     * Output: instances[2] where: 
     *      [0] matrix profile distances
     *      [1] matrix profile indices
     */
    public Instances[] processDistancesAndIndices(Instances instances) throws Exception {
                
        int seriesLength = instances.numAttributes()-(instances.classIndex()>=0?1:0);
        
        if(windowSize < 3){
            throw new Exception("Error: window must be at least 3. You have specified "+windowSize);
        }
        
        if(windowSize > seriesLength){
            throw new Exception("Error: window must be smaller than the number of attributes. Window length: "+windowSize+", series length: "+seriesLength);
        }
        
        if(seriesLength/4 < windowSize){
            throw new Exception("Error: the series length must be at least 4 times larger than the window size to satisfy the exclusion zone criteria for trivial matches. These instances have a series length of "+seriesLength+"; the maximum window size is therefore "+(seriesLength/4)+" and you have specified "+windowSize);
        }
        
        SingleInstanceMatrixProfile mpIns;
        Instances outputDistances = this.determineOutputFormat(instances);
        Instances outputIndices = this.determineOutputFormat(instances);
        Instance outDist, outIdx;
        
        this.distances = new double[instances.numInstances()][];
        this.indices = new int[instances.numInstances()][];
        
        for(int a = 0; a < seriesLength-windowSize+1; a++){
            outputIndices.renameAttribute(a, "idx_"+a);
        }
        outputIndices.setRelationName(outputIndices.relationName()+"_indices");
        
        for(int ins = 0; ins < instances.numInstances(); ins++){
            mpIns = new SingleInstanceMatrixProfile(instances.get(ins),this.windowSize, this.stride);
            outDist = new DenseInstance(outputDistances.numAttributes());
            outIdx = new DenseInstance(outputIndices.numAttributes());
            
            distances[ins] = mpIns.distances;
            indices[ins] = mpIns.indices;
            
            for(int i = 0; i < mpIns.distances.length; i++){
                outDist.setValue(i, mpIns.distances[i]);
                outIdx.setValue(i, mpIns.indices[i]);
            }
            
            if(instances.classIndex() >=0){
                outDist.setValue(mpIns.distances.length, instances.instance(ins).classValue());
                outIdx.setValue(mpIns.indices.length, instances.instance(ins).classValue());
            }
            
            outputDistances.add(outDist);
            outputIndices.add(outIdx);
        }
        return new Instances[]{outputDistances,outputIndices};
    }
    
    public double[][] getDistances() throws Exception{
        if(this.distances == null){
            throw new Exception("Error: must process instances before accessing distances");
        }
        return this.distances;
    }
    
    public int[][] getIndices() throws Exception{
        if(this.indices == null){
            throw new Exception("Error: must process instances before accessing indices");
        }
        return this.indices;
    }
    
    private static class SingleInstanceMatrixProfile{
        private final double[] series;
        private final int windowSize;
        private final int stride;
        private final double[] distances;
        private final int[] indices;
        private final int seriesLength;
        
        public SingleInstanceMatrixProfile(Instance series, int windowSize, int stride){
            this.series = series.toDoubleArray();
            this.seriesLength = series.classIndex()>0 ? series.numAttributes()-1 : series.numAttributes();
            this.windowSize = windowSize;
            this.stride = stride;
            this.distances = new double[seriesLength+1-windowSize];
            this.indices = new int[seriesLength+1-windowSize];
            
            for(int a = 0; a <= seriesLength-windowSize; a++){
                this.locateBestMatch(a);
            }
        }
        
        public SingleInstanceMatrixProfile(double[] series, int windowSize, int stride){
            this.series = series;
            this.seriesLength = series.length;
            this.windowSize = windowSize;
            this.stride = stride;
            this.distances = new double[seriesLength+1-windowSize];
            this.indices = new int[seriesLength+1-windowSize];
            
            for(int a = 0; a <= seriesLength-windowSize; a++){
                this.locateBestMatch(a);
            }
        }
        
        // query is fixed, comparison is every possible match within the series
        private void locateBestMatch(int queryStartIdx){
            
            double dist;
            double bsfDist = Double.MAX_VALUE;
            int bsfIdx = -1;

            double[] query = zNormalise(series, queryStartIdx, this.windowSize, false);
            double[] comparison;

            for(int comparisonStartIdx = 0; comparisonStartIdx <= seriesLength-windowSize; comparisonStartIdx+=stride){
                
                // exclusion zone +/- windowSize/2 around the window
                if(comparisonStartIdx >= queryStartIdx-windowSize*1.5 && comparisonStartIdx <= queryStartIdx+windowSize*1.5){
                    continue;
                }
                
                // using a bespoke version of this, rather than the shapelet version, for efficiency - see notes with method
                comparison = zNormalise(series, comparisonStartIdx, windowSize, false);
                dist = 0;

                for(int j = 0; j < windowSize;j++){
                    dist += (query[j]-comparison[j])*(query[j]-comparison[j]);
                    if(dist > bsfDist){
                        dist = Double.MAX_VALUE;
                        break;
                    }
                }

                if(dist < bsfDist){
                    bsfDist = dist;
                    bsfIdx = comparisonStartIdx;
                }

            }
            
            this.distances[queryStartIdx] = bsfDist;
            this.indices[queryStartIdx] = bsfIdx;
        }
    }
    
    
   
    // adapted from shapelet code to avoid copying subsequences - logic is equivilent. In the shapelet version the input is the subsequence as double[] (i.e. the shapelet). 
    // In this case we don't already have the subsequence as a double[], so to avoid copying into one just to use this method, a bespoke version is used to process the subsequence
    // directly from the full series given a start idx. The logic that follows is consitent and taken directly from the shapelet code, however.    
    public static double[] zNormalise(double[] input, int startIdx, int subsequenceLength, boolean classValOn){
        double mean;
        double stdv;

        int classValPenalty = classValOn ? 1 : 0;
        int inputLength = subsequenceLength - classValPenalty;

        double[] output = new double[subsequenceLength+classValPenalty];
        double seriesTotal = 0;

        for (int i = startIdx; i < startIdx+inputLength; i++){
            seriesTotal += input[i];
        }

        mean = seriesTotal / (double) inputLength;
        stdv = 0;
        double temp;
        
        for (int i = startIdx; i < startIdx+inputLength; i++){
            temp = (input[i] - mean);
            stdv += temp * temp;
        }

        stdv /= (double) inputLength;

        // if the variance is less than the error correction, just set it to 0, else calc stdv.
        stdv = (stdv < ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(stdv);
        
        for (int i = startIdx; i < inputLength+startIdx; i++){
            //if the stdv is 0 then set to 0, else normalise.
            output[i-startIdx] = (stdv == 0.0) ? 0.0 : ((input[i] - mean) / stdv);
        }

        if (classValOn){
            output[output.length - 1] = input[input.length - 1];
        }

        return output;
    }
    
   
    public static void main(String[] args){
        try{
            
            short exampleOption = 1;
            
            switch (exampleOption) {
                case 0:
                    {
                        //<editor-fold defaultstate="collapsed" desc="An example of the code processing a single instance">
                        double[] exampleSeries = {
                            //<editor-fold defaultstate="collapsed" desc="hidden">
                            0.706958948,
                            0.750908517,
                            0.900082659,
                            0.392463961,
                            0.242465518,
                            0.612627784,
                            0.965461774,
                            0.511642268,
                            0.973824154,
                            0.765900772,
                            0.570131418,
                            0.978983617,
                            0.71732363,
                            0.694103358,
                            0.988679068,
                            0.516752819,
                            0.680371205,
                            0.041150128,
                            0.438617378,
                            0.962620183,
                            0.336994745,
                            0.109872653,
                            0.729607701,
                            0.553675396,
                            0.907678336,
                            0.296047233,
                            0.62139885,
                            0.047203274,
                            0.234199203,
                            0.507061681,
                            40.1775059,
                            40.92078108,
                            40.32998362,
                            40.24925086,
                            40.23031747,
                            40.2612678,
                            40.24958999,
                            0.206638348,
                            0.188084622,
                            0.435294443,
                            0.016919806,
                            0.488749443,
                            0.536798782,
                            0.604030646,
                            0.027743671,
                            0.475801082,
                            0.219379181,
                            0.197770558,
                            0.180549958,
                            0.424767962,
                            0.730424542,
                            0.050246332,
                            0.775454296,
                            0.598464994,
                            0.041599684,
                            0.678161584,
                            0.022935237,
                            0.572039895,
                            0.895840616,
                            0.430037881,
                            0.606246479,
                            0.595235683,
                            0.684102456,
                            0.876411514,
                            0.634496091,
                            0.583138615,
                            0.83459057,
                            0.604222487,
                            0.526759991,
                            0.796785741,
                            0.603588625,
                            0.78414503,
                            0.676148061,
                            0.631703028,
                            0.029891999,
                            0.66954295,
                            0.09326132,
                            0.324903263,
                            0.329370111,
                            0.349991934,
                            0.98813969,
                            0.212371375,
                            40.43175799,
                            40.64309996,
                            40.25703808,
                            40.68109205,
                            40.98675558,
                            40.67109108,
                            40.19057322,
                            0.547164791,
                            0.148980971,
                            0.657974529,
                            0.033686273,
                            0.925714876,
                            0.155158131,
                            0.562893421,
                            0.55974838,
                            0.067785579,
                            0.185605974,
                            0.056922816,
                            0.906773429,
                            0.108453764,
                            0.857711715,
                            0.054685775,
                            0.282340146,
                            0.356960824,
                            0.506107616,
                            0.682422972,
                            0.845058908,
                            0.825395344,
                            0.840462024,
                            0.452107774,
                            0.199188375,
                            0.745644811,
                            0.318544188,
                            0.437352361,
                            0.001509022,
                            0.325114368,
                            0.378086159,
                            0.510979193,
                            0.053430927,
                            0.134820265,
                            0.202091967,
                            0.365691307,
                            0.104942853,
                            0.444478755,
                            0.021250513,
                            40.93704671,
                            40.20245208,
                            40.87017417,
                            40.12272305,
                            40.79370847,
                            40.01667509,
                            40.29991657,
                            0.881084794,
                            0.015035975,
                            0.868897876,
                            0.00632042,
                            0.922354652,
                            0.601708676,
                            0.558058363,
                            0.11333862,
                            0.771422385,
                            0.52350838,
                            0.392103683,
                            0.42441807,
                            0.084006383,
                            0.810320086,
                            0.140575367,
                            0.592926995,
                            0.136968111,
                            0.86361884,
                            0.800409212,
                            0.361548663,
                            0.887355284,
                            0.520255152,
                            0.85104809,
                            0.350659883,
                            0.38558232,
                            0.963846112,
                            0.738944264,
                            0.177064402,
                            0.27604281,
                            0.173068454,
                            0.301392245,
                            0.327037831,
                            0.359321481,
                            40.4284963,
                            40.63135518,
                            40.0415674,
                            40.58448039,
                            40.7447205,
                            40.63741215,
                            40.38637122,
                            0.333073723,
                            0.835468558,
                            0.901962258,
                            0.661272244,
                            0.296970939,
                            0.604075557,
                            0.43405287,
                            0.517690996,
                            0.448041559,
                            0.100768093,
                            0.166518799,
                            0.59463445,
                            0.853616259,
                            0.617191115,
                            0.170413139,
                            0.46838602,
                            0.596948951,
                            0.634140074,
                            0.72695993,
                            0.407250642,
                            0.161077052,
                            0.017273795,
                            0.962110794,
                            0.531243218,
                            0.076041357,
                            0.516862396,
                            0.551316188,
                            0.854549962,
                            0.333861949,
                            0.381543776,
                            0.952493204,
                            0.626465371,
                            0.637232052,
                            0.918986949,
                            0.414714591,
                            0.028046619,
                            0.927337815,
                            0.730504031,
                            0.577524028,
                            0.738305301,
                            0.498088814,
                            0.030412342,
                            0.892963296,
                            0.158905784,
                            0.308830195,
                            0.001044088,
                            0.528515062,
                            0.770532238,
                            0.148622557,
                            0.564126052,
                            0.284379515,
                            0.912867459,
                            0.938835582,
                            0.634984824,
                            0.151908861,
                            0.45635823,
                            0.96444686,
                            0.773402777,
                            0.42544929,
                            0.540827843,
                            0.017940591,
                            0.710334758,
                            40.72691681,
                            40.51454394,
                            40.32967767,
                            40.84169443,
                            40.56804387,
                            40.15860845,
                            40.46282824,
                            0.451688955,
                            0.541844839,
                            0.841712041,
                            0.360199018,
                            0.058959662,
                            0.556940395,
                            0.632788865,
                            0.618176594,
                            0.794095294,
                            0.016679839,
                            0.274969095,
                            0.967616659,
                            0.50959933,
                            0.797046729,
                            0.960273369,
                            0.820735666,
                            0.446419259,
                            0.089017654,
                            0.192430069,
                            0.475741946,
                            0.280867131,
                            0.342160569,
                            0.837739216,
                            0.590364989,
                            0.82525571,
                            0.281604012,
                            0.167508301,
                            0.2274851,
                            0.543793246,
                            0.541841033,
                            0.002968891,
                            0.73975265,
                            0.710442853,
                            0.056361441,
                            0.494012768,
                            40.88467555,
                            40.32410117,
                            40.8734326,
                            40.62995568,
                            40.20315046,
                            40.10966019,
                            40.43205643,
                            0.600012785,
                            0.530871599,
                            0.879191584,
                            0.995522866,
                            0.840788635,
                            0.675717205,
                            0.708321268,
                            0.568667757,
                            0.471350118,
                            0.084376171,
                            0.64306147,
                            0.958564678,
                            0.851967071,
                            0.767858501,
                            0.377896405,
                            0.355198698,
                            0.347686315,
                            0.220245913,
                            0.66680921,
                            0.973033751,
                            0.379253749,
                            0.671333966,
                            0.811846499,
                            0.882194418,
                            0.906573824,
                            0.110526759,
                            0.528251005,
                            0.594284369,
                            0.450898007,
                            0.34381444,
                            0.078977677,
                            0.459869753,
                            0.637764085,
                            0.934354742,
                            0.686230405,
                            0.173913574,
                            0.837440544,
                            0.796244826,
                            0.482733612,
                            40.88451874,
                            40.9480389,
                            40.64721753,
                            40.36044806,
                            40.58392222,
                            40.16336175,
                            40.18406683,
                            0.098042009,
                            0.141072966,
                            0.485982161,
                            0.654872193,
                            0.975890136,
                            0.419647877,
                            0.39811089,
                            0.563886789,
                            0.968525154,
                            0.446994618,
                            0.32338217,
                            0.360504306,
                            0.876710065,
                            0.776301087,
                            0.444540067,
                            0.238726928,
                            0.114215352,
                            0.904878926,
                            0.102432501,
                            0.483207846,
                            0.271168946,
                            0.187234805,
                            0.70038387,
                            0.714664048,
                            0.165472715,
                            0.773527567,
                            0.072575256,
                            0.015110261,
                            0.832353831,
                            0.511964426,
                            0.510665358,
                            0.268989309,
                            0.386169927,
                            0.769298622,
                            0.482800936,
                            0.370094824,
                            0.980576579,
                            0.97774188,
                            40.1261874,
                            40.08808535,
                            40.83331539,
                            40.18693955,
                            40.75391492,
                            40.67115714,
                            40.97951474,
                            0.048625932,
                            0.300809154,
                            0.076411212,
                            0.285448268,
                            //</editor-fold>
                        };      int windowSize = 10;
                        SingleInstanceMatrixProfile simp = new SingleInstanceMatrixProfile(exampleSeries, windowSize, 1);
                        for(int i = 0; i < exampleSeries.length-windowSize+1; i++){
                            simp.locateBestMatch(i);
                        }       System.out.println("Example series:");
                        for(int a = 0; a < exampleSeries.length; a++){
                            System.out.print(exampleSeries[a]+",");
                        }       System.out.println("\n\nMatrix Profile");
                        for(int j = 0; j < exampleSeries.length-windowSize; j++){
                            System.out.print(simp.distances[j]+",");
                        }       System.out.println("\n\nIndices:");
                        for(int j = 0; j < exampleSeries.length-windowSize; j++){
                            System.out.print(simp.indices[j]+",");
                        }       System.out.println();
                        //</editor-fold>
                        break;
                    }
                case 1:
                    {
                        //<editor-fold defaultstate="collapsed" desc="An example using GunPoint"> 
                        String datapath = "C:/users/sjx07ngu/Dropbox/TSC Problems/";
                        String datasetName = "GunPoint";
                        Instances train = ClassifierTools.loadData(datapath+datasetName+"/"+datasetName+"_TRAIN");
                        int windowSize = 10;
                        MatrixProfile mp = new MatrixProfile(windowSize);
                        
                        Instances transformedToDistances = mp.process(train);
                        System.out.println(transformedToDistances);
                        
                        // additional use case: transform to indices too
                        /*
                        Instances[] transformedToDistancesAndIndices = mp.processDistancesAndIndices(train);  
                        System.out.println(transformedToDistancesAndIndices[0]+"\n");
                        System.out.println(transformedToDistancesAndIndices[1]);
                        */
                        break;
                        //</editor-fold>
                    }
                case 2:
                    {
                        //<editor-fold defaultstate="collapsed" desc="An example using GunPoint, but using a window that is bigger than m/4"> 
                        String datapath = "C:/users/sjx07ngu/Dropbox/TSC Problems/";
                        String datasetName = "GunPoint";
                        Instances train = ClassifierTools.loadData(datapath+datasetName+"/"+datasetName+"_TRAIN");
                        int windowSize = (train.numAttributes()-1)/4+1;
                        MatrixProfile mp = new MatrixProfile(windowSize);
                        Instances transformedToDistances = mp.process(train);
//                        Instances[] transformedToDistancesAndIndices = mp.process(train);
                        System.out.println(transformedToDistances);
                        break;
                        //</editor-fold>
                    }
                default:
                    System.out.println("I've run out of examples!");
                    break;
            }
        }catch(Exception e){
            e.printStackTrace();
        }
    }
    

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
   
        int seriesLength = inputFormat.classIndex() >= 0 ? inputFormat.numAttributes()-1 : inputFormat.numAttributes();        
        int numOutputAtts = seriesLength+1-windowSize;
        
        ArrayList<Attribute> atts = new ArrayList<>();
        for(int a = 0; a < numOutputAtts;a++){
            atts.add(new Attribute("dist_"+a));
        }
        
        if (inputFormat.classIndex() >= 0) {
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            ArrayList<String> vals = new ArrayList<>();
            for (int i = 0; i < target.numValues(); i++) {
                vals.add(target.value(i));
            }
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        
        Instances output = new Instances(inputFormat.relationName()+"_matrixProfile", atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            output.setClassIndex(output.numAttributes()-1);
        }
        return output;
    }

}
