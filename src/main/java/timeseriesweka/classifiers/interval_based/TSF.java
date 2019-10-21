
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
package timeseriesweka.classifiers.interval_based;
 
import java.util.ArrayList;
import utilities.ClassifierTools;
import evaluation.evaluators.CrossValidationEvaluator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import evaluation.tuning.ParameterSpace;
import experiments.data.DatasetLoading;
import java.io.File;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import timeseriesweka.classifiers.EnhancedAbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Randomizable;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import timeseriesweka.classifiers.Tuneable;
 
/** 
  <!-- globalinfo-start -->
* Implementation of Time Series Forest
 Time Series Forest (TimeSeriesForest) Deng 2013: 
* buildClassifier
 * Overview: Input n series length m
 * for each tree
 *      sample sqrt(m) intervals
 *      find three features on each interval: mean, standard deviation and slope
 *      concatenate to new feature set
 *      build tree on new feature set
* classifyInstance
*   ensemble the trees with majority vote
 
* This implementation may deviate from the original, as it is using the same
* structure as the weka random forest. In the paper the splitting criteria has a 
* tiny refinement. Ties in entropy gain are split with a further stat called margin 
* that measures the distance of the split point to the closest data. 
* So if the split value for feature 
* f=f_1,...f_n is v the margin is defined as
*   margin= min{ |f_i-v| } 
* for simplicity of implementation, and for the fact when we did try it and it made 
* no difference, we have not used this. Note also, the original R implementation 
* may do some sampling of cases
 
* Update 1:
* * A few changes made to enable testing refinements. 
*1. general baseClassifier rather than a hard coded RandomTree. We tested a few 
*  alternatives, the summary results NEED WRITING UP
*  Summary:
*  Base Classifier: 
*       a) C.45 (J48) significantly worse than random tree.  
*       b) CAWPE tbc
*       c) CART tbc
* 2. Added setOptions to allow parameter tuning. Tuning on parameters
*       #trees, #features 
 <!-- globalinfo-end -->
 <!-- technical-bibtex-start -->
* Bibtex
* <pre>
* article{deng13forest,
* author = {H. Deng and G. Runger and E. Tuv and M. Vladimir},
* title = {A time series forest for classification and feature extraction},
* journal = {Information Sciences},
* volume = {239},
* year = {2013}
*}
</pre>
<!-- technical-bibtex-end -->
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -T
 *  set number of trees in the ensemble.</pre>
 * 
 * <pre> -I
 *  set number of intervals to calculate.</pre>
 <!-- options-end -->
 
* author ajb
* date 7/10/15
* update1 14/2/19
* update2 13/9/19: Adjust to allow three methods for estimating test accuracy

**/
 
public class TSF extends EnhancedAbstractClassifier 
        implements TechnicalInformationHandler, Tuneable{
//Static defaults
     
    private final static int DEFAULT_NUM_CLASSIFIERS=500;
 
    /** Primary parameters potentially tunable*/   
    private int numClassifiers=DEFAULT_NUM_CLASSIFIERS;
 
    /** numIntervalsFinder sets numIntervals in buildClassifier. */   
    private int numIntervals=0;
    private Function<Integer,Integer> numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
    /** Secondary parameter, mainly there to avoid single item intervals, 
     which have no slope or std dev*/
    private int minIntervalLength=3;
 
    /** Ensemble members of base classifier, default to random forest RandomTree */
    private Classifier[] trees; 
    private Classifier base= new RandomTree();
 
    /** for each classifier [i]  interval j  starts at intervals[i][j][0] and 
     ends  at  intervals[i][j][1] */
    private int[][][] intervals;
     
    /**Holding variable for test classification in order to retain the header info*/    
    private Instances testHolder;
 
 
/** voteEnsemble determines whether to aggregate classifications or
     * probabilities when predicting */
    private boolean voteEnsemble=true;
     
 /** Flags and data required if Bagging **/  
    private boolean bagging=false; //Use if we want an OOB estimate
    private boolean[][] inBag;
    private int[] oobCounts;
    private double[][] trainDistributions;
 
   /** If trainAccuracy is required, there are three mechanisms to obtain it:
    * 1. bagging == true: use the OOB accuracy from the final model
    * 2. bagging == false,estimator=CV: do a 10x CV on the train set with a clone 
    * of this classifier
    * 3. bagging == false,estimator=OOB: build an OOB model just to get the OOB 
    * accuracy estimate
    */
//    boolean findTrainPredictions=false;  
    enum EstimatorMethod{CV,OOB}
    private EstimatorMethod estimator=EstimatorMethod.CV;
    private String trainFoldPath="";
/* If trainFoldPath is set, train results are overwritten with 
 each call to buildClassifier.*/    
     
     
    public TSF(){
//TSF Has the capability to form train estimates         
//In order to do this, 
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }
    public TSF(int s){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        setSeed(s);
    }
/**
 * 
 * @param c a base classifier constructed elsewhere and cloned into ensemble
 */   
    public void setBaseClassifier(Classifier c){
        base=c;
    }
    public void setBagging(boolean b){
        bagging=b;
    }
 
/**
 * ok,  two methods are a bit pointless, experimenting with ensemble method
 * @param b boolean to set vote ensemble
 */   
    public void setVoteEnsemble(boolean b){
        voteEnsemble=b;
    }
    public void setProbabilityEnsemble(boolean b){
        voteEnsemble=!b;
    }
     
/**
 * Perhaps make this coherent with setOptions(String[] ar)?
 * @return String written to results files
 */
    @Override
    public String getParameters() {
        String temp=super.getParameters()+",numTrees,"+numClassifiers+",numIntervals,"+numIntervals+",voting,"+voteEnsemble+",BaseClassifier,"+base.getClass().getSimpleName()+",Bagging,"+bagging;
        if(base instanceof RandomTree)
           temp+=",AttsConsideredPerNode,"+((RandomTree)base).getKValue();
        return temp;
 
    }
    public void setNumTrees(int t){
        numClassifiers=t;
    }
     
     
//<editor-fold defaultstate="collapsed" desc="results reported in Info Sciences paper">        
    static double[] reportedResults={
        0.2659,
        0.2302,
        0.2333,
        0.0256,
        0.2537,
        0.0391,
        0.0357,
        0.2897,
        0.2,
        0.2436,
        0.049,
        0.08,
        0.0557,
        0.2325,
        0.0227,
        0.101,
        0.1543,
        0.0467,
        0.552,
        0.6818,
        0.0301,
        0.1803,
        0.2603,
        0.0448,
        0.2237,
        0.119,
        0.0987,
        0.0865,
        0.0667,
        0.4339,
        0.233,
        0.1868,
        0.0357,
        0.1056,
        0.1116,
        0.0267,
        0.02,
        0.1177,
        0.0543,
        0.2102,
        0.2876,
        0.2624,
        0.0054,
        0.3793,
        0.1513
    };
      //</editor-fold>  
     
//<editor-fold defaultstate="collapsed" desc="problems used in Info Sciences paper">   
    static String[] problems={
            "FiftyWords",
            "Adiac",
            "Beef",
            "CBF",
            "ChlorineConcentration",
            "CinCECGtorso",
            "Coffee",
            "CricketX",
            "CricketY",
            "CricketZ",
            "DiatomSizeReduction",
            "ECG",
            "ECGFiveDays",
            "FaceAll",
            "FaceFour",
            "FacesUCR",
            "Fish",
            "GunPoint",
            "Haptics",
            "InlineSkate",
            "ItalyPowerDemand",
            "Lightning2",
            "Lightning7",
            "Mallat",
            "MedicalImages",
            "MoteStrain",
            "NonInvasiveFetalECGThorax1",
            "NonInvasiveFetalECGThorax2",
            "OliveOil",
            "OSULeaf",
            "SonyAIBORobotSurface1",
            "SonyAIBORobot Surface2",
            "StarLightCurves",
            "SwedishLeaf",
            "Symbols",
            "Synthetic Control",
            "Trace",
            "TwoLeadECG",
            "TwoPatterns",
            "UWaveGestureLibraryX",
            "UWaveGestureLibraryY",
            "UWaveGestureLibraryZ",
            "Wafer",
            "WordsSynonyms",
            "Yoga"
        };
      //</editor-fold>  
  
 /**
  * paper defining TSF
  * @return TechnicalInformation
  */  
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation    result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "H. Deng, G. Runger, E. Tuv and M. Vladimir");
        result.setValue(TechnicalInformation.Field.YEAR, "2013");
        result.setValue(TechnicalInformation.Field.TITLE, "A time series forest for classification and feature extraction");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Information Sciences");
        result.setValue(TechnicalInformation.Field.VOLUME, "239");
        result.setValue(TechnicalInformation.Field.PAGES, "142-153");
 
        return result;
  }


/**
 * main buildClassifier
 * @param data
 * @throws Exception 
 */     
    @Override
    public void buildClassifier(Instances data) throws Exception {
/** Build Stage: 
 *  Builds the final classifier with or without bagging.  
 */       
    // can classifier handle the data?
        getCapabilities().testWithFail(data);
        long t1=System.nanoTime();
        numIntervals=numIntervalsFinder.apply(data.numAttributes()-1);
//Set up instances size and format. 
        trees=new AbstractClassifier[numClassifiers];        
        ArrayList<Attribute> atts=new ArrayList<>();
        String name;
        for(int j=0;j<numIntervals*3;j++){
            name = "F"+j;
            atts.add(new Attribute(name));
        }
        //Get the class values as an array list     
        Attribute target =data.attribute(data.classIndex());
        ArrayList<String> vals=new ArrayList<>(target.numValues());
        for(int j=0;j<target.numValues();j++)
            vals.add(target.value(j));
        atts.add(new Attribute(data.attribute(data.classIndex()).name(),vals));
        //create blank instances with the correct class value                
        Instances result = new Instances("Tree",atts,data.numInstances());
        result.setClassIndex(result.numAttributes()-1);
        for(int i=0;i<data.numInstances();i++){
            DenseInstance in=new DenseInstance(result.numAttributes());
            in.setValue(result.numAttributes()-1,data.instance(i).classValue());
            result.add(in);
        }
         
        testHolder =new Instances(result,0);       
        DenseInstance in=new DenseInstance(result.numAttributes());
        testHolder.add(in);
//Need to hard code this because log(m)+1 is sig worse than sqrt(m) is worse than using all!
        if(base instanceof RandomTree){
            ((RandomTree) base).setKValue(result.numAttributes()-1);
//            ((RandomTree) base).setKValue((int)Math.sqrt(result.numAttributes()-1));
        }        
        /** Set up for Bagging if required **/
        if(bagging){
           inBag=new boolean[numClassifiers][];
           trainDistributions= new double[data.numInstances()][data.numClasses()];
           oobCounts=new int[data.numInstances()];
        }
         
        /** For each base classifier 
         *      generate random intervals
         *      do the transfrorms
         *      build the classifier
         * */
        intervals =new int[numClassifiers][][];
        for(int i=0;i<numClassifiers;i++){
        //1. Select random intervals for tree i
            intervals[i]=new int[numIntervals][2];  //Start and end
            if(data.numAttributes()-1<minIntervalLength)
                 minIntervalLength=data.numAttributes()-1;
            for(int j=0;j<numIntervals;j++){
               intervals[i][j][0]=rand.nextInt(data.numAttributes()-1-minIntervalLength);       //Start point
               int length=rand.nextInt(data.numAttributes()-1-intervals[i][j][0]);//Min length 3
               if(length<minIntervalLength)
                   length=minIntervalLength;
               intervals[i][j][1]=intervals[i][j][0]+length;
            }
        //2. Generate and store attributes            
            for(int j=0;j<numIntervals;j++){
                //For each instance
                for(int k=0;k<data.numInstances();k++){
                    //extract the interval
                    double[] series=data.instance(k).toDoubleArray();
                    FeatureSet f= new FeatureSet();
                    f.setFeatures(series, intervals[i][j][0], intervals[i][j][1]);
                    result.instance(k).setValue(j*3, f.mean);
                    result.instance(k).setValue(j*3+1, f.stDev);
                    result.instance(k).setValue(j*3+2, f.slope);
                }
            }
        //3. Create and build tree using all the features. Feature selection
            trees[i]=AbstractClassifier.makeCopy(base); 
            if(seedClassifier && trees[i] instanceof Randomizable)
                ((Randomizable)trees[i]).setSeed(seed*(i+1));
            if(bagging){
                inBag[i] = new boolean[result.numInstances()];
                Instances bagData = result.resampleWithWeights(rand, inBag[i]);
                trees[i].buildClassifier(bagData);
                if(getEstimateOwnPerformance()){
                    for(int j=0;j<result.numInstances();j++){
                        if(inBag[i][j])
                            continue;
                        double[] newProbs = trees[i].distributionForInstance(result.instance(j));
                        oobCounts[j]++;
                        for(int k=0;k<newProbs.length;k++)
                            trainDistributions[j][k]+=newProbs[k];
                         
                    }
                }
            }
            else
                trees[i].buildClassifier(result);
        }
        long t2=System.nanoTime();
/** Estimate accuracy stage: Three scenarios
 * 1. If we bagged the full build (bagging ==true), we estimate using the full build OOB
*  If we built on all data (bagging ==false) we estimate either 
*  2. with a 10xCV or (if 
*  3. Build a bagged model simply to get the estimate. 
 */       
        if(getEstimateOwnPerformance()){
             if(bagging){
            // Use bag data. Normalise probs
                long est1=System.nanoTime();
                double[] preds=new double[data.numInstances()];
                double[] actuals=new double[data.numInstances()];
                for(int j=0;j<data.numInstances();j++){
                    for(int k=0;k<trainDistributions[j].length;k++)
                        trainDistributions[j][k]/=oobCounts[j];
                    preds[j]=utilities.GenericTools.indexOfMax(trainDistributions[j]);
                    actuals[j]=data.instance(j).classValue();
                }
                long[] predTimes=new long[data.numInstances()];//Dummy variable, need something
                trainResults.addAllPredictions(actuals,preds, trainDistributions, predTimes, null);
                trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
                trainResults.setClassifierName("TSFBagging");
                trainResults.setDatasetName(data.relationName());
                trainResults.setSplit("train");
                trainResults.setFoldID(seed);
                trainResults.setParas(getParameters());
                trainResults.finaliseResults(actuals);
                long est2=System.nanoTime();
                trainResults.setErrorEstimateTime(est2-est1);
            }
//Either do a CV, or bag and get the estimates 
            else if(estimator==EstimatorMethod.CV){
                /** Defaults to 10 or numInstances, whichever is smaller. 
                 * Interface TrainAccuracyEstimate
                 * Could this be handled better? */
                long est1=System.nanoTime();
                int numFolds=setNumberOfFolds(data);
                CrossValidationEvaluator cv = new CrossValidationEvaluator();
                if (seedClassifier)
                  cv.setSeed(seed*5);
                cv.setNumFolds(numFolds);
                TSF tsf=new TSF();
                tsf.copyParameters(this);
                if (seedClassifier)
                   tsf.setSeed(seed*100);
                tsf.setEstimateOwnPerformance(false);
                trainResults=cv.evaluate(tsf,data);
                long est2=System.nanoTime();
                trainResults.setErrorEstimateTime(est2-est1);
                trainResults.setClassifierName("TSFCV");
                trainResults.setParas(getParameters());
                 
            }
            else if(estimator==EstimatorMethod.OOB){
               /** Build a single new TSF using Bagging, and extract the estimate from this
                */
                long est1=System.nanoTime();
                TSF tsf=new TSF();
                tsf.copyParameters(this);
                tsf.setSeed(seed);
                tsf.setEstimateOwnPerformance(true);
                tsf.bagging=true;
                tsf.buildClassifier(data);
                trainResults=tsf.trainResults;
                long est2=System.nanoTime();
                trainResults.setErrorEstimateTime(est2-est1);
                trainResults.setClassifierName("TSFOOB");
                trainResults.setParas(getParameters());
            }
              
            System.out.println("Build time ="+trainResults.getBuildTime());
            if(trainFoldPath!=""){
                trainResults.writeFullResultsToFile(trainFoldPath);
            }
        }
        trainResults.setBuildTime(t2-t1);
    }
     
    private void copyParameters(TSF other){
        this.numClassifiers=other.numClassifiers;
        this.numIntervalsFinder=other.numIntervalsFinder;
         
         
    }
    public void setEstimatorMethod(String str){
        String s=str.toUpperCase();
        if(s.equals("CV"))
            estimator=EstimatorMethod.CV;
        else if(s.equals("OOB"))
            estimator=EstimatorMethod.OOB;
        else
            throw new UnsupportedOperationException("Unknown estimator methof in TSF = "+str);
    }
/**
 * @param ins to classifier
 * @return array of doubles: probability of each class 
 * @throws Exception 
 */   
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception {
        double[] d=new double[ins.numClasses()];
        //Build transformed instance
        double[] series=ins.toDoubleArray();
        for(int i=0;i<trees.length;i++){
            for(int j=0;j<numIntervals;j++){
                //extract all intervals
                FeatureSet f= new FeatureSet();
                f.setFeatures(series, intervals[i][j][0], intervals[i][j][1]);
                testHolder.instance(0).setValue(j*3, f.mean);
                testHolder.instance(0).setValue(j*3+1, f.stDev);
                testHolder.instance(0).setValue(j*3+2, f.slope);
            }
            if(voteEnsemble){
                int c=(int)trees[i].classifyInstance(testHolder.instance(0));
                d[c]++;
            }else{
                double[] temp=trees[i].distributionForInstance(testHolder.instance(0));
                for(int j=0;j<temp.length;j++)
                    d[j]+=temp[j];
            }
        }
        double sum=0;
        for(double x:d)
            sum+=x;
        for(int i=0;i<d.length;i++)
            d[i]=d[i]/sum;
        return d;
    }
/**
 * @param ins
 * @return
 * @throws Exception 
 */
    @Override
    public double classifyInstance(Instance ins) throws Exception {
        double[] d=distributionForInstance(ins);
        int max=0;
        for(int i=1;i<d.length;i++)
            if(d[i]>d[max])
                max=i;
        return (double)max;
    }
  /**
   * Parses a given list of options to set the parameters of the classifier.
   * We use this for the tuning mechanism, setting parameters through setOptions 
   <!-- options-start -->
   * Valid options are: <p/>
   * <pre> -T
   * Number of trees.</pre>
   * 
   * <pre> -I
   * Number of intervals to fit.</pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
    @Override
    public void setOptions(String[] options) throws Exception{
//        System.out.print("TSF para sets ");
//        for (String str:options)
//             System.out.print(","+str);
//        System.out.print("\n");
        String numTreesString=Utils.getOption('T', options);
        if (numTreesString.length() != 0)
            numClassifiers = Integer.parseInt(numTreesString);
        else
            numClassifiers = DEFAULT_NUM_CLASSIFIERS;
         
        String numFeaturesString=Utils.getOption('I', options);
//Options here are a double between 0 and 1 (proportion of features), a text 
//string sqrt or log, or an integer number 
        if (numTreesString.length() != 0){
            try{
                if(numFeaturesString.equals("sqrt"))
                    numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
                else if(numFeaturesString.equals("log"))
                    numIntervalsFinder = (numAtts) -> (int) Utils.log2(numAtts) + 1;
                else{
                        double d=Double.parseDouble(numFeaturesString);
                        if(d<=0)
                            throw new Exception("proportion of features of of range 0 to 1");
                        if(d<=1)
                            numIntervalsFinder = (numAtts) -> (int)(d*numAtts);
                        else
                            numIntervalsFinder = (numAtts) -> (int)(d);
 
//                        System.out.println("Proportion/number of intervals = "+d);
                 }
            }catch(Exception e){
                System.err.print(" Error: invalid parameter passed to TSF setOptions for number of parameters. Setting to default");
                System.err.print("Value"+numIntervalsFinder+" Permissable values: sqrt, log, or a double range 0...1");
                numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
            }
        }
        else
            System.out.println("Unable to read number of intervals, not set");
    }
 
 
//Nested class to store three simple summary features used to construct train data
    public static class FeatureSet{
        public static boolean findSkew=false;
        public static boolean findKurtosis=false;
        double mean;
        double stDev;
        double slope;
        double skew;
        double kurtosis;
        public void setFeatures(double[] data, int start, int end){
            double sumX=0,sumYY=0;
            double sumY3=0,sumY4=0;
            double sumY=0,sumXY=0,sumXX=0;
            int length=end-start+1;
            for(int i=start;i<=end;i++){
                sumY+=data[i];
                sumYY+=data[i]*data[i];
                sumX+=(i-start);
                sumXX+=(i-start)*(i-start);
                sumXY+=data[i]*(i-start);
            }
            mean=sumY/length;
            stDev=sumYY-(sumY*sumY)/length;
            slope=(sumXY-(sumX*sumY)/length);
            double denom=sumXX-(sumX*sumX)/length;
            if(denom!=0)
                slope/=denom;
            else
                slope=0;
            stDev/=length;
            if(stDev==0)    //Flat line
                slope=0;
//            else //Why not doing this? Because not needed? 
//                stDev=Math.sqrt(stDev);
            if(slope==0)
                stDev=0;
            if(findSkew){
                if(stDev==0)
                    skew=1;
                else{
                    for(int i=start;i<=end;i++)
                        sumY3+=data[i]*data[i]*data[i];
                    skew=sumY3-3*sumY*sumYY+2*sumY*sumY;
                    skew/=length*stDev*stDev*stDev;
                }
            }
            if(findKurtosis){
                if(stDev==0)
                    kurtosis=1;
                else{
                    for(int i=start;i<=end;i++)
                        sumY4+=data[i]*data[i]*data[i]*data[i];
                    kurtosis=sumY4-4*sumY*sumY3+6*sumY*sumY*sumYY-3*sumY*sumY*sumY*sumY;
                    skew/=length*stDev*stDev*stDev*stDev;
                }
            }
             
        }
        public void setFeatures(double[] data){
            setFeatures(data,0,data.length-1);
        }
        @Override
        public String toString(){
            return "mean="+mean+" stdev = "+stDev+" slope ="+slope;
        }
    } 
     
    public static void main(String[] arg) throws Exception{
        
//        System.out.println(ClassifierTools.testUtils_getIPDAcc(new TSF(0)));
//        System.out.println(ClassifierTools.testUtils_confirmIPDReproduction(new TSF(0), 0.967930029154519, "2019/09/25"));
        
// Basic correctness tests, including setting paras through 
        String dataLocation="Z:\\Data\\TSCProblems2018\\";
        String resultsLocation="C:\\temp\\";
        String problem="ItalyPowerDemand";
        File f= new File(resultsLocation+problem);
        if(!f.isDirectory())
            f.mkdirs();
        Instances train=DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TRAIN");
        Instances test=DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TEST");
        TSF tsf = new TSF();
        tsf.setSeed(0);
//        tsf.writeTrainEstimatesToFile(resultsLocation+problem+"trainFold0.csv");
        double a;
        tsf.buildClassifier(train);
        System.out.println("build ok: original atts="+(train.numAttributes()-1)+" new atts ="+tsf.testHolder.numAttributes()+" num trees = "+tsf.numClassifiers+" num intervals = "+tsf.numIntervals);
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);
        String[] options=new String[4];
        options[0]="-T";
        options[1]="10";
        options[2]="-I";
        options[3]="1";
        tsf.setOptions(options);
        tsf.buildClassifier(train);
        System.out.println("build ok: original atts="+(train.numAttributes()-1)+" new atts ="+tsf.testHolder.numAttributes()+" num trees = "+tsf.numClassifiers+" num intervals = "+tsf.numIntervals);
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);
         
         
    }
     
    @Override
    public ParameterSpace getDefaultParameterSearchSpace(){
   //TUNED TSC Classifiers
  /* Valid options are: <p/>
   * <pre> -T Number of trees.</pre>
   * <pre> -I Number of intervals to fit.</pre>
   */
        ParameterSpace ps=new ParameterSpace();
        String[] numTrees={"100","200","300","400","500","600","700","800","900","1000"};
        ps.addParameter("-T", numTrees);
        String[] numInterv={"sqrt","log","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"}; 
        ps.addParameter("-I", numInterv);
        return ps;
    }
     
     
}
  