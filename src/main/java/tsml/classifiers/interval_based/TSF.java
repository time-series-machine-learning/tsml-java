
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
package tsml.classifiers.interval_based;

import java.io.File;
import java.util.ArrayList;

import evaluation.storage.ClassifierResults;
import fileIO.OutFile;
import machine_learning.classifiers.TimeSeriesTree;
import tsml.classifiers.*;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import utilities.ClassifierTools;
import evaluation.evaluators.CrossValidationEvaluator;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import evaluation.tuning.ParameterSpace;
import experiments.data.DatasetLoading;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

import weka.classifiers.Classifier;
import weka.core.Randomizable;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/** 
  <!-- globalinfo-start -->
* Implementation of Time Series Forest
 * This classifier is Tunable, Contractable, Checkpointable and can estimate performance from the train data internally.
 *
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
 * Valid options are:
 * 
 * <pre> -T
 *  set number of trees in the ensemble.</pre>
 * 
 * <pre> -I
 *  set number of intervals to calculate.</pre>
 <!-- options-end -->
 
*       version1.0 author Tony Bagnall
* date 7/10/15  Tony Bagnall
* update 14/2/19 Tony Bagnall
 * A few changes made to enable testing refinements.
 * 1. general baseClassifier rather than a hard coded RandomTree. We tested a few
 *  alternatives, they did not improve things
 * 2. Added setOptions to allow parameter tuning. Tuning on parameters: #trees, #features
 * update2 13/9/19: Adjust to allow three methods for estimating test accuracy Tony Bagnall
*       version2.0 13/03/20 Matthew Middlehurst. contractable, checkpointable and tuneable,
 * This classifier is tested and deemed stable on 10/3/2020. It is unlikely to change again
 *  results for this classifier on 112 UCR data sets can be found at
 *  www.timeseriesclassification.com/results/ResultsByClassifier/TSF.csv. The first column of results  are on the default
 *  train/test split. The others are found through stratified resampling of the combined train/test
 *  individual results on each fold are
 *  timeseriesclassification.com/results/ResultsByClassifier/TSF/Predictions
 * update 1/7/2020: Tony Bagnall. Sort out correct recording of timing, and tidy up comments. The storage option for
 * either CV or OOB
*/
 
public class TSF extends EnhancedAbstractClassifier implements TechnicalInformationHandler,
        TrainTimeContractable, Checkpointable, Tuneable, Visualisable {
//Static defaults
    private final static int DEFAULT_NUM_CLASSIFIERS=500;
 
    /** Primary parameters potentially tunable*/   
    private int numClassifiers=DEFAULT_NUM_CLASSIFIERS;

    /** numIntervalsFinder sets numIntervals in buildClassifier. */
    private int numIntervals=0;
    private transient Function<Integer,Integer> numIntervalsFinder;
    /** Secondary parameter, mainly there to avoid single item intervals, 
     which have no slope or std dev*/
    private int minIntervalLength=3;
 
    /** Ensemble members of base classifier, default to random forest RandomTree */
    private ArrayList<Classifier> trees;
    private Classifier classifier = new TimeSeriesTree();
 
    /** for each classifier [i]  interval j  starts at intervals[i][j][0] and 
     ends  at  intervals[i][j][1] */
    private ArrayList<int[][]> intervals;

    /**Holding variable for test classification in order to retain the header info*/
    private Instances testHolder;
 
 
/** voteEnsemble determines whether to aggregate classifications or
     * probabilities when predicting */
    private boolean voteEnsemble=true;

    /** Flags and data required if Bagging **/
    private boolean bagging=false; //Use if we want an OOB estimate
    private ArrayList<boolean[]> inBag;
    private int[] oobCounts;
    private double[][] trainDistributions;



    /**** Checkpointing variables *****/
    private boolean checkpoint = false;
    private String checkpointPath=null;
    private long checkpointTime = 0;    //Time between checkpoints in nanosecs
    private long lastCheckpointTime = 0;    //Time since last checkpoint in nanos.


    private long checkpointTimeElapsed= 0;
    private boolean trainTimeContract = false;
    transient private long trainContractTimeNanos = 0;
    transient private long finalBuildtrainContractTimeNanos = 0;

    protected static final long serialVersionUID = 32554L;

    private int seriesLength;

    private String visSavePath;

    public TSF(){
        //TSF Has the capability to form train estimates
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
        classifier =c;
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
        String result=super.getParameters()+",numTrees,"+trees.size()+",numIntervals,"+numIntervals+",voting,"+voteEnsemble+",BaseClassifier,"+ classifier.getClass().getSimpleName()+",Bagging,"+bagging;

        if(trainTimeContract)
            result+= ",trainContractTimeNanos," +trainContractTimeNanos;
        else
            result+=",NoContract";
//Any other contract information here

        result+=",EstimateOwnPerformance,"+getEstimateOwnPerformance();
        if(getEstimateOwnPerformance())
            result+=",EstimateMethod,"+estimator;
        return result;
 
    }
    public void setNumTrees(int t){
        numClassifiers=t;
    }
     
     
//<editor-fold defaultstate="collapsed" desc="results reported in Info Sciences paper (errors)">
    static double[] reportedErrorResults ={
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
     * buildClassifier wrapper for TimeSeriesInstances
     * @param data
     * @throws Exception
     */
  @Override
  public void buildClassifier(TimeSeriesInstances data) throws Exception {
        Instances convertedData = Converter.toArff(data);
        convertedData.setClassIndex(convertedData.numAttributes() - 1);
        buildClassifier(convertedData);
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
        long startTime=System.nanoTime();
        File file = new File(checkpointPath + "TSF" + seed + ".ser");
        //Set up Checkpointing (saving to file)/ if checkpointing and serialised files exist load said file
        if (checkpoint && file.exists()){
            //path checkpoint files will be saved to
            printLineDebug("Loading from checkpoint file");
            loadFromFile(checkpointPath + "TSF" + seed + ".ser");
        }
        else {//else initialise variables
            seriesLength = data.numAttributes() - 1;
            if (numIntervalsFinder == null){
                numIntervals = (int)Math.sqrt(seriesLength);
            }
            else {
                numIntervals = numIntervalsFinder.apply(data.numAttributes() - 1);
            }
            printDebug("Building TSF: number of intervals = " + numIntervals+" number of trees ="+numClassifiers+"\n");
            trees = new ArrayList(numClassifiers);
            // Set up for train estimates
            if(getEstimateOwnPerformance()) {
                trainDistributions= new double[data.numInstances()][data.numClasses()];
            }
            //Set up for bagging
            if(bagging){
                inBag=new ArrayList();
                oobCounts=new int[data.numInstances()];
                printLineDebug("TSF is using Bagging");
            }
            intervals = new ArrayList();
            lastCheckpointTime=startTime;
        }
        finalBuildtrainContractTimeNanos=trainContractTimeNanos;
        //If contracted and estimating own performance, distribute the contract evenly between estimation and the final build
        if(trainTimeContract &&  !bagging && getEstimateOwnPerformance()){
            finalBuildtrainContractTimeNanos/=2;
            printLineDebug(" Setting final contract time to "+finalBuildtrainContractTimeNanos+" nanos");
        }

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
        Instances transformedData = new Instances("Tree",atts,data.numInstances());
        transformedData.setClassIndex(transformedData.numAttributes()-1);
        for(int i=0;i<data.numInstances();i++){
            DenseInstance in=new DenseInstance(transformedData.numAttributes());
            in.setValue(transformedData.numAttributes()-1,data.instance(i).classValue());
            transformedData.add(in);
        }
         
        testHolder =new Instances(transformedData,0);
        DenseInstance in=new DenseInstance(transformedData.numAttributes());
        testHolder.add(in);
        int classifiersBuilt = trees.size();



        /** MAIN BUILD LOOP
         *  For each base classifier
         *      generate random intervals
         *      do the transforms
         *      build the classifier
         * */
        while(withinTrainContract(startTime) && (classifiersBuilt < numClassifiers)){
            if(classifiersBuilt%100==0)
                printLineDebug("\t\t\t\t\tBuilding TSF tree "+classifiersBuilt+" time taken = "+(System.nanoTime()-startTime)+" contract ="+finalBuildtrainContractTimeNanos+" nanos");

            //1. Select random intervals for tree i
            int[][] interval =new int[numIntervals][2];  //Start and end

            if (data.numAttributes() - 1 < minIntervalLength)
                minIntervalLength = data.numAttributes() - 1;
            for (int j = 0; j < numIntervals; j++) {
                interval[j][0] = rand.nextInt(data.numAttributes() - 1 - minIntervalLength);       //Start point
                int length = rand.nextInt(data.numAttributes() - 1 - interval[j][0]);//Min length 3
                if (length < minIntervalLength)
                    length = minIntervalLength;
                interval[j][1] = interval[j][0] + length;
            }

            //2. Generate and store attributes
            for(int j=0;j<numIntervals;j++){
                for(int k=0;k<data.numInstances();k++){
                    //extract the interval, work out the features
                    double[] series=data.instance(k).toDoubleArray();
                    FeatureSet f= new FeatureSet();
                    f.setFeatures(series, interval[j][0], interval[j][1]);
                    transformedData.instance(k).setValue(j*3, f.mean);
                    transformedData.instance(k).setValue(j*3+1, f.stDev);
                    transformedData.instance(k).setValue(j*3+2, f.slope);
                }
            }
            //3. Create and build tree using all the features.
            Classifier tree = AbstractClassifier.makeCopy(classifier);
            if(seedClassifier && tree instanceof Randomizable)
                ((Randomizable)tree).setSeed(seed*(classifiersBuilt+1));

            if(bagging){
                long t1=System.nanoTime();
                boolean[] bag = new boolean[transformedData.numInstances()];
                Instances bagData = transformedData.resampleWithWeights(rand, bag);
                tree.buildClassifier(bagData);
                inBag.add(bag);
                if(getEstimateOwnPerformance()){
                    for(int j=0;j<transformedData.numInstances();j++){
                        if(bag[j])
                            continue;
                        double[] newProbs = tree.distributionForInstance(transformedData.instance(j));
                        oobCounts[j]++;
                        for(int k=0;k<newProbs.length;k++)
                            trainDistributions[j][k]+=newProbs[k];
                    }
                }
                long t2=System.nanoTime();
                if(getEstimateOwnPerformance())
                    trainResults.setErrorEstimateTime(t2-t1+trainResults.getErrorEstimateTime());
            }
            else
                tree.buildClassifier(transformedData);

            intervals.add(interval);
            trees.add(tree);
            classifiersBuilt++;

            if (checkpoint){
                if(checkpointTime>0)    //Timed checkpointing
                {
                    if(System.nanoTime()-lastCheckpointTime>checkpointTime){
                        saveToFile(checkpointPath);
                        lastCheckpointTime=System.nanoTime();
                    }
                }
                else {    //Default checkpoint every 100 trees
                    if(classifiersBuilt%100 == 0 && classifiersBuilt>0)
                        saveToFile(checkpointPath);
                }
            }
        }
        if(classifiersBuilt==0){//Not enough time to build a single classifier
            throw new Exception((" ERROR in TSF, no trees built, contract time probably too low. Contract time ="+trainContractTimeNanos));
        }
        if (checkpoint) {
            saveToFile(checkpointPath);
        }
        long endTime=System.nanoTime();
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(endTime-startTime-trainResults.getErrorEstimateTime());
        trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime());
        /** Estimate accuracy from Train data
         * distributions and predictions stored in trainResults */
        if(getEstimateOwnPerformance()){
            long est1=System.nanoTime();
            estimateOwnPerformance(data);
            long est2=System.nanoTime();
            if(bagging)
                trainResults.setErrorEstimateTime(est2-est1+trainResults.getErrorEstimateTime());
            else
                trainResults.setErrorEstimateTime(est2-est1);

            trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime()+trainResults.getErrorEstimateTime());
        }
        trainResults.setParas(getParameters());
        printLineDebug("*************** Finished TSF Build with "+classifiersBuilt+" Trees built in "+(System.nanoTime()-startTime)/1000000000+" Seconds  ***************");
    }

    /**
     * estimating own performance
     *  Three scenarios
     *          1. If we bagged the full build (bagging ==true), we estimate using the full build OOB. Assumes the final
     *          model has already been built
     *           If we built on all data (bagging ==false) we estimate either
     *              2. with a 10xCV if estimator==EstimatorMethod.CV
     *              3. Build a bagged model simply to get the estimate estimator==EstimatorMethod.OOB
     *    Note that all this needs to come out of any contract time we specify.
     * @param data
     * @throws Exception from distributionForInstance
     */
    private void estimateOwnPerformance(Instances data) throws Exception {
        if(bagging){
            // Use bag data, counts normalised to probabilities
            printLineDebug("Finding the OOB estimates");
            double[] preds=new double[data.numInstances()];
            double[] actuals=new double[data.numInstances()];
            long[] predTimes=new long[data.numInstances()];//Dummy variable, need something
            for(int j=0;j<data.numInstances();j++){
                long predTime = System.nanoTime();
                for(int k=0;k<trainDistributions[j].length;k++)
                    if(oobCounts[j]>0)
                        trainDistributions[j][k]/=oobCounts[j];
                preds[j]=findIndexOfMax(trainDistributions[j],rand);
                actuals[j]=data.instance(j).classValue();
                predTimes[j]=System.nanoTime()-predTime;
            }
            trainResults.addAllPredictions(actuals,preds, trainDistributions, predTimes, null);
            trainResults.setClassifierName("TSFBagging");
            trainResults.setDatasetName(data.relationName());
            trainResults.setSplit("train");
            trainResults.setFoldID(seed);
            trainResults.finaliseResults(actuals);
            trainResults.setErrorEstimateMethod("OOB");

        }
        //Either do a CV, or bag and get the estimates
        else if(estimator==EstimatorMethod.CV || estimator==EstimatorMethod.NONE){
            // Defaults to 10 or numInstances, whichever is smaller.
            int numFolds=setNumberOfFolds(data);
            CrossValidationEvaluator cv = new CrossValidationEvaluator();
            if (seedClassifier)
                cv.setSeed(seed*5);
            cv.setNumFolds(numFolds);
            TSF tsf=new TSF();
            tsf.copyParameters(this);
            tsf.setDebug(this.debug);
            if (seedClassifier)
                tsf.setSeed(seed*100);
            tsf.setEstimateOwnPerformance(false);
            if(trainTimeContract)//Need to split the contract time, will give time/(numFolds+2) to each fio
                tsf.setTrainTimeLimit(finalBuildtrainContractTimeNanos/numFolds);
            printLineDebug(" Doing CV evaluation estimate performance with  "+tsf.getTrainContractTimeNanos()/1000000000+" secs per fold.");
            long buildTime = trainResults.getBuildTime();
            trainResults=cv.evaluate(tsf,data);
            trainResults.setBuildTime(buildTime);
            trainResults.setClassifierName("TSFCV");
            trainResults.setErrorEstimateMethod("CV_"+numFolds);
        }
        else if(estimator==EstimatorMethod.OOB){
            // Build a single new TSF using Bagging, and extract the estimate from this
            TSF tsf=new TSF();
            tsf.copyParameters(this);
            tsf.setDebug(this.debug);
            tsf.setSeed(seed);
            tsf.setEstimateOwnPerformance(true);
            tsf.bagging=true;
            tsf.setTrainTimeLimit(finalBuildtrainContractTimeNanos);
            printLineDebug(" Doing Bagging estimate performance with "+tsf.getTrainContractTimeNanos()/1000000000+" secs per fold ");
            tsf.buildClassifier(data);
            long buildTime = trainResults.getBuildTime();
            trainResults=tsf.trainResults;
            trainResults.setBuildTime(buildTime);
            trainResults.setClassifierName("TSFOOB");
            trainResults.setErrorEstimateMethod("OOB");
        }
    }
     
    private void copyParameters(TSF other){
        this.numClassifiers=other.numClassifiers;
        this.numIntervalsFinder=other.numIntervalsFinder;
    }
    @Override
    public long getTrainContractTimeNanos(){
            return trainContractTimeNanos;
    }

    /**
     * @param ins TimeSeriesInstance to classifier
     * @return array of doubles: probability of each class
     */
    @Override
    public double[] distributionForInstance(TimeSeriesInstance ins) throws Exception {
        Instance convertedData = Converter.toArff(ins);
        return distributionForInstance(convertedData);
    }

    /**
     * @param ins Weka Instance to classifier
     * @return array of doubles: probability of each class
     * @throws Exception
     */
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception {
        double[] d=new double[ins.numClasses()];
        //Build transformed instance
        double[] series=ins.toDoubleArray();
        for(int i=0;i<trees.size();i++){
            for(int j=0;j<numIntervals;j++){
                //extract all intervals
                FeatureSet f= new FeatureSet();
                f.setFeatures(series, intervals.get(i)[j][0], intervals.get(i)[j][1]);
                testHolder.instance(0).setValue(j*3, f.mean);
                testHolder.instance(0).setValue(j*3+1, f.stDev);
                testHolder.instance(0).setValue(j*3+2, f.slope);
            }
            if(voteEnsemble){
                int c=(int)trees.get(i).classifyInstance(testHolder.instance(0));
                d[c]++;
            }else{
                double[] temp=trees.get(i).distributionForInstance(testHolder.instance(0));
                for(int j=0;j<temp.length;j++)
                    d[j]+=temp[j];
            }
        }
        double sum=0;
        for(double x:d)
            sum+=x;
        if(sum>0)
            for(int i=0;i<d.length;i++)
                d[i]=d[i]/sum;
        return d;
    }

    /**
     * @param ins TimeSeriesInstance
     */
    @Override
    public double classifyInstance(TimeSeriesInstance ins) throws Exception {
        Instance convertedData = Converter.toArff(ins);
        return classifyInstance(convertedData);
    }

/**
 * @param ins Weka Instance
 * @return
 * @throws Exception 
 */
    @Override
    public double classifyInstance(Instance ins) throws Exception {
        double[] d=distributionForInstance(ins);
        return findIndexOfMax(d, rand);
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
/*        System.out.print("TSF para sets ");
        for (String str:options)
            System.out.print(","+str);
        System.out.print("\n");
*/
        String numTreesString=Utils.getOption('T', options);

        if (numTreesString.length() != 0) {
            numClassifiers = Integer.parseInt(numTreesString);
        }

        String numFeaturesString=Utils.getOption('I', options);
//Options here are a double between 0 and 1 (proportion of features), a text 
//string sqrt or log, or an integer number 
        if (numFeaturesString.length() != 0){
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
 
                 }
            }catch(Exception e){
                System.err.print(" Error: invalid parameter passed to TSF setOptions for number of parameters. Setting to default");
                System.err.print("Value"+numIntervalsFinder+" Permissable values: sqrt, log, or a double range 0...1");
                numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
            }
        }
        else
            numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
    }

    @Override //Checkpointable
    public boolean setCheckpointPath(String path) {
        boolean validPath=Checkpointable.super.createDirectories(path);
        printLineDebug(" Writing checkpoint to "+path);
        if(validPath){
            checkpointPath = path;
            checkpoint = true;
        }
        return validPath;
    }
    @Override //Checkpointable
    public boolean setCheckpointTimeHours(int t){
        checkpointTime=TimeUnit.NANOSECONDS.convert(t,TimeUnit.HOURS);
        checkpoint = true;
        return true;
    }
    @Override //Checkpointable
    public void copyFromSerObject(Object obj) throws Exception {
        if(!(obj instanceof TSF))
            throw new Exception("The SER file is not an instance of TSF");
        TSF saved = ((TSF)obj);

        try{        printLineDebug("Loading TSF" + seed + ".ser");

            numClassifiers = saved.numClassifiers;
            numIntervals = saved.numIntervals;
            //numIntervalsFinder = saved.numIntervalsFinder;
            minIntervalLength = saved.minIntervalLength;
            trees = saved.trees;
            classifier = saved.classifier;
            intervals = saved.intervals;
            //testHolder = saved.testHolder;
            voteEnsemble = saved.voteEnsemble;
            bagging = saved.bagging;
            inBag = saved.inBag;
            oobCounts = saved.oobCounts;
            trainDistributions = saved.trainDistributions;
            estimator = saved.estimator;
            checkpoint = saved.checkpoint;
            checkpointPath = saved.checkpointPath;
            checkpointTime = saved.checkpointTime;
            checkpointTimeElapsed = saved.checkpointTime; //intentional, time spent building previously unchanged
//            trainTimeContract = saved.trainTimeContract;
//            trainContractTimeNanos = saved.trainContractTimeNanos;
            seriesLength = saved.seriesLength;

            rand = saved.rand;
            seedClassifier = saved.seedClassifier;
            seed = saved.seed;
            trainResults = saved.trainResults;
            estimateOwnPerformance = saved.estimateOwnPerformance;
        }catch(Exception ex){
            System.out.println("Unable to assign variables when loading serialised file");
        }
    }


    @Override//TrainTimeContractable
    public void setTrainTimeLimit(long amount) {
        printLineDebug(" TSF setting contract to "+amount);

        if(amount>0) {
            trainContractTimeNanos = amount;
            trainTimeContract = true;
        }
        else
            trainTimeContract = false;
    }
    @Override//TrainTimeContractable
    public boolean withinTrainContract(long start){
        if(trainContractTimeNanos<=0) return true; //Not contracted
        return System.nanoTime()-start < finalBuildtrainContractTimeNanos;
    }

    @Override // Checkpointable
    public void saveToFile(String filename) throws Exception{
        Checkpointable.super.saveToFile(checkpointPath + "TSF" + seed + "temp.ser");
        File file = new File(checkpointPath + "TSF" + seed + "temp.ser");
        File file2 = new File(checkpointPath + "TSF" + seed + ".ser");
        file2.delete();
        file.renameTo(file2);
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

    /**
     *TUNED TSF Classifiers. Method for interface Tuneable
     * Valid options are: <p/>
     * <pre> -T Number of trees.</pre>
     * <pre> -I Number of intervals to fit.</pre>
     *
     *
             * @return ParameterSpace object
     */
    @Override
    public ParameterSpace getDefaultParameterSearchSpace(){
        ParameterSpace ps=new ParameterSpace();
        String[] numTrees={"100","200","300","400","500","600","700","800","900","1000"};
        ps.addParameter("T", numTrees);
        String[] numInterv={"sqrt","log","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"};
        ps.addParameter("I", numInterv);
        return ps;
    }

    @Override
    public boolean setVisualisationSavePath(String path) {
        boolean validPath = Visualisable.super.createVisualisationDirectories(path);
        if(validPath){
            visSavePath = path;
        }
        return validPath;
    }

    @Override
    public boolean createVisualisation() throws Exception {
        if (!(classifier instanceof TimeSeriesTree)) {
            System.err.println("TSF temporal importance curve only available for time series tree.");
            return false;
        }

        if (visSavePath == null){
            System.err.println("TSF visualisation save path not set.");
            return false;
        }

        double[][] curves = new double[3][seriesLength];
        for (int i = 0; i < trees.size(); i++){
            TimeSeriesTree tree = (TimeSeriesTree)trees.get(i);
            ArrayList<Double>[] sg = tree.getTreeSplitsGain();

            for (int n = 0; n < sg[0].size(); n++){
                double split = sg[0].get(n);
                double gain = sg[1].get(n);
                int interval = (int)(split/3);
                int att = (int)(split%3);

                for (int j = intervals.get(i)[interval][0]; j <= intervals.get(i)[interval][1]; j++){
                    curves[att][j] += gain;
                }
            }
        }

        OutFile of = new OutFile(visSavePath + "/vis" + seed + ".txt");
        String[] atts = new String[]{"mean","stdev","slope"};
        for (int i = 0 ; i < 3; i++){
            of.writeLine(atts[i]);
            of.writeLine(Arrays.toString(curves[i]));
        }
        of.closeFile();

        Runtime.getRuntime().exec("py src/main/python/visCIF.py \"" +
                visSavePath.replace("\\", "/")+ "\" " + seed + " 3 3");

        return true;
    }

    public static void main(String[] arg) throws Exception{
        
//        System.out.println(ClassifierTools.testUtils_getIPDAcc(new TSF(0)));
//        System.out.println(ClassifierTools.testUtils_confirmIPDReproduction(new TSF(0), 0.967930029154519, "2019/09/25"));
        
// Basic correctness tests, including setting paras through 
        String dataLocation="Z:\\ArchiveData\\Univariate_arff\\";
        String resultsLocation="D:\\temp\\";
        String problem="ItalyPowerDemand";
        File f= new File(resultsLocation+problem);
        if(!f.isDirectory())
            f.mkdirs();
        Instances train=DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TRAIN");
        Instances test=DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TEST");
        TSF tsf = new TSF();
        tsf.setSeed(0);
        tsf.setTrainTimeLimit((long)1.5e+10);
        //tsf.setSavePath("D:\\temp\\");
        tsf.setEstimateOwnPerformance(true);
        double a;
        tsf.buildClassifier(train);
        ClassifierResults trainres = tsf.getTrainResults();
        trainres.writeFullResultsToFile(resultsLocation+problem+"trainFold0.csv");
        System.out.println("build ok: original atts="+(train.numAttributes()-1)+" new atts ="+tsf.testHolder.numAttributes()+" num trees = "+tsf.numClassifiers+" num intervals = "+tsf.numIntervals);
        System.out.println(tsf.trainResults.getBuildTime());
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);
        System.out.println();

        tsf = new TSF();
        tsf.setSeed(1);
        tsf.setTrainTimeLimit((long)1.5e+10);
        //tsf.setSavePath("D:\\temp\\");
        tsf.setEstimateOwnPerformance(true);
        tsf.setEstimatorMethod("OOB");
        String[] options=new String[4];
        options[0]="-T";
        options[1]="10";
        options[2]="-I";
        options[3]="1";
        tsf.setOptions(options);
        tsf.buildClassifier(train);
        System.out.println("build ok: original atts="+(train.numAttributes()-1)+" new atts ="+tsf.testHolder.numAttributes()+" num trees = "+tsf.numClassifiers+" num intervals = "+tsf.numIntervals);
        System.out.println(tsf.trainResults.getBuildTime());
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);
    }
}
  