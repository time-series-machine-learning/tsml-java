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
package timeseriesweka.classifiers;

import fileIO.OutFile;
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import evaluation.evaluators.CrossValidationEvaluator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import utilities.TrainAccuracyEstimate;
import evaluation.storage.ClassifierResults;
import java.io.File;
import java.util.function.Function;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Utils;

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
*  alternatives, the summary results are available at
*  http://www.timeseriesclassification.com/Experiments/TSF.xlsx
*  Summary:
*  Base Classifier: 
*       a) C.45 (J48) significantly worse than random tree 
*       b) CAWPE tbc
*       c) CART tbc
* 2. Added setOptions to allow parameter tuning. Tuning on parameters
*       #trees, #features 
 <!-- globalinfo-end -->
 <!-- technical-bibtex-start -->
* Bibtex
* <pre>
* @article{deng13forest,
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
 *  set number of trees.</pre>
 * 
 * <pre> -F
 *  set number of features.</pre>
 <!-- options-end -->

* @author ajb
* @date 7/10/15
* @update1 14/2/19

**/ 

public class TSF extends AbstractClassifierWithTrainingInfo implements SaveParameterInfo, TrainAccuracyEstimate{
//Static defaults
    
    private final static int DEFAULT_NUM_CLASSIFIERS=500;

    /** Primary parameters potentially tunable*/    
    private int numClassifiers=DEFAULT_NUM_CLASSIFIERS;

    /** numIntervalsFinder sets numIntervals in buildClassifier. */    
    private int numIntervals=0;
    Function<Integer,Integer> numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));   
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

    /**Can seed for reproducability*/
    private Random rand;
    private boolean setSeed=false;
    private int seed=0;

   /** If trainCV is performed, a cross validation is done in buildClassifier
   If set, train results are overwritten with each call to buildClassifier
   File opened on this path.*/     
    boolean trainCV=false;  
    private String trainCVPath="";
    
    /** voteEnsemble determines whether to aggregate classifications or
     * probabilities when predicting */
    private boolean voteEnsemble=true;


    public TSF(){
        rand=new Random();
    }
    public TSF(int s){
        rand=new Random();
        seed=s;
        rand.setSeed(seed);
        setSeed=true;
    }
/**
 * 
 * @param c a base classifier constructed elsewhere and cloned into ensemble
 */    
    public void setBaseClassifier(Classifier c){
        base=c;
    }

/**
 * ok,  two methods are a bit pointless, experimenting with ensemble method
 * @param b 
 */    
    public void setVoteEnsemble(boolean b){
        voteEnsemble=b;
    }
    public void setProbabilityEnsemble(boolean b){
        voteEnsemble=!b;
    }
    
/**
 * Seed experiments for reproducability with the resample number
 * @param s 
 */    
    public void setSeed(int s){
        this.setSeed=true;
        seed=s;
        rand=new Random();
        rand.setSeed(seed);
    }
/**
 * Stores the classifier train CV results and writes them to file. 
 * boolean trainCV is a little redundant, but nicer than checking path for null
 * @param train 
 */    
    @Override
    public void writeCVTrainToFile(String train) {
        trainCVPath=train;
        trainCV=true;
    }
/**
 * We can perform trainCV without writing the results to file. The could be used,
 * for example, in an ensemble this TSF is a member of
 * @param setCV 
 */
    @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        trainCV=setCV;
    }
/** 
 * Maybe this method needs renaming?
 * @return boolean, whether trainCV happens or not
 */    
    @Override
    public boolean findsTrainAccuracyEstimate(){ return trainCV;}
    
/**
 * 
 * @return a ClassifierResults object, which will be null if tainCV is null
 */    
    @Override
     public ClassifierResults getTrainResults(){
         return trainResults;
     }        
/**
 * Perhaps make this coherent with setOptions(String[] ar)?
 * @return String written to results files
 */
    @Override
    public String getParameters() {
        String temp=super.getParameters()+",numTrees,"+numClassifiers+",numIntervals,"+numIntervals+",voting,"+voteEnsemble+",BaseClassifier,"+base.getClass().getSimpleName();
        if(base instanceof RandomTree)
           temp+=",k,"+((RandomTree)base).getKValue();
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
    public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
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
   * Returns default capabilities of the classifier. These are that the 
   * data must be numeric, with no missing and a nominal class
   * @return the capabilities of this classifier
**/    
    @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes must be numeric
    // Here add in relational when ready
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    // class
    result.enable(Capability.NOMINAL_CLASS);
    // instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }

/**
 * main buildClassifier
 * @param data
 * @throws Exception 
 */      
    @Override
    public void buildClassifier(Instances data) throws Exception {
    // can classifier handle the data?
        getCapabilities().testWithFail(data);
        long t1=System.currentTimeMillis();
        numIntervals=numIntervalsFinder.apply(data.numAttributes()-1);
    //Estimate train accuracy here if required
        if(trainCV){
            /** Defaults to 10 or numInstances, whichever is smaller. 
             * Interface TrainAccuracyEstimate
             * Could this be handled better? */
            int numFolds=setNumberOfFolds(data);
            CrossValidationEvaluator cv = new CrossValidationEvaluator();
            if (setSeed)
              cv.setSeed(seed);
            cv.setNumFolds(numFolds);
            TSF tsf=new TSF();
            tsf.setFindTrainAccuracyEstimate(false);
            trainResults=cv.crossValidateWithStats(tsf,data);
        }
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
            System.out.println("Base classifier num of features = "+((RandomTree) base).getKValue());
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
            trees[i].buildClassifier(result);
        }
        long t2=System.currentTimeMillis();
        //Store build time, this is always recorded
        trainResults.setBuildTime(t2-t1);
        //If trainCV ==true and we want to save results, write out object 
        if(trainCV && trainCVPath!=""){
             OutFile of=new OutFile(trainCVPath);
             of.writeLine(data.relationName()+",TSF,train");
             of.writeLine(getParameters());
            of.writeLine(trainResults.getAcc()+"");
            double[] trueClassVals,predClassVals;
            trueClassVals=trainResults.getTrueClassValsAsArray();
            predClassVals=trainResults.getPredClassValsAsArray();
            for(int i=0;i<data.numInstances();i++){
                //Basic sanity check
                if(data.instance(i).classValue()!=trueClassVals[i]){
                    throw new Exception("ERROR in TSF cross validation, class mismatch!");
                }
                of.writeString((int)trueClassVals[i]+","+(int)predClassVals[i]+",");
                for(double d:trainResults.getProbabilityDistribution(i))
                    of.writeString(","+d);
                of.writeString("\n");
            }
        }
        
    }
/**
 * Sums either the 
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
 * What about  
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
        String numTreesString=Utils.getOption('T', options);
        if (numTreesString.length() != 0)
            numClassifiers = Integer.parseInt(numTreesString);
        else
            numClassifiers = DEFAULT_NUM_CLASSIFIERS;
        String numFeaturesString=Utils.getOption('F', options);
//Options here are a double between 0 and 1 (proportion of features) or a text 
//string sqrt or log
        try{
        if(numFeaturesString.equals("sqrt"))
            numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
        else if(numFeaturesString.equals("log"))
            numIntervalsFinder = (numAtts) -> (int) Utils.log2(numAtts) + 1;
        else{
                double d=Double.parseDouble(numFeaturesString);
                if(d<=0 || d>1)
                    throw new Exception("proportion of features of of range 0 to 1");
                numIntervalsFinder = (numAtts) -> (int)(d*numAtts);
                System.out.println("Proportion of atts = "+d);
            }
        }catch(Exception e){
            System.err.print(" Error: invalid parameter passed to TSF setOptions for number of parameters. Setting to default");
            System.err.print("Value"+numIntervalsFinder+" Permissable values: sqrt, log, or a double range 0...1");
            numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
        }
    }

//Nested class to store three simple summary features used to construct train data
    public static class FeatureSet{
        double mean;
        double stDev;
        double slope;
        public void setFeatures(double[] data, int start, int end){
            double sumX=0,sumYY=0;
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
// Basic correctness tests, including setting paras through 
        String dataLocation="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
        String resultsLocation="C:\\temp\\";
        String problem="ItalyPowerDemand";
        File f= new File(resultsLocation+problem);
        if(!f.isDirectory())
            f.mkdirs();
        Instances train=ClassifierTools.loadData(dataLocation+problem+"\\"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(dataLocation+problem+"\\"+problem+"_TEST");
        TSF tsf = new TSF();
        tsf.writeCVTrainToFile(resultsLocation+problem+"trainFold0.csv");
        double a;
        tsf.buildClassifier(train);
        System.out.println("build ok: original atts="+(train.numAttributes()-1)+" new atts ="+tsf.testHolder.numAttributes()+" num trees = "+tsf.numClassifiers+" num intervals = "+tsf.numIntervals);
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);
        String[] options=new String[4];
        options[0]="-T";
        options[1]="10";
        options[2]="-F";
        options[3]="1";
        tsf.setOptions(options);
        tsf.buildClassifier(train);
        System.out.println("build ok: original atts="+(train.numAttributes()-1)+" new atts ="+tsf.testHolder.numAttributes()+" num trees = "+tsf.numClassifiers+" num intervals = "+tsf.numIntervals);
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);
        
        
    }
}
  
