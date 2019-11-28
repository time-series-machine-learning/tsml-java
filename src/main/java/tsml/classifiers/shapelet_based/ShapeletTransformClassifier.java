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
package tsml.classifiers.shapelet_based;

import experiments.data.DatasetLoading;
import tsml.filters.shapelet_transforms.ShapeletTransformFactory;
import tsml.filters.shapelet_transforms.ShapeletTransform;
import tsml.filters.shapelet_transforms.ShapeletTransformFactoryOptions;
import tsml.filters.shapelet_transforms.ShapeletTransformTimingUtilities;
import java.io.File;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import utilities.InstanceTools;
import machine_learning.classifiers.ensembles.CAWPE;
import weka.core.Instance;
import weka.core.Instances;
import tsml.filters.shapelet_transforms.distance_functions.SubSeqDistance;
import tsml.filters.shapelet_transforms.quality_measures.ShapeletQuality;
import tsml.filters.shapelet_transforms.search_functions.ShapeletSearch;
import tsml.filters.shapelet_transforms.search_functions.ShapeletSearch.SearchType;
import tsml.filters.shapelet_transforms.search_functions.ShapeletSearchOptions;
import machine_learning.classifiers.ensembles.voting.MajorityConfidence;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import fileIO.FullAccessOutFile;
import fileIO.OutFile;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import tsml.classifiers.EnhancedAbstractClassifier;

import machine_learning.classifiers.ensembles.voting.MajorityVote;
import machine_learning.classifiers.ensembles.weightings.EqualWeighting;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import tsml.classifiers.TrainTimeContractable;
import weka.core.TechnicalInformation;

/**
 * ShapeletTransformClassifier
 * Builds a time series classifier by first extracting the best numShapeletsInTransform
 *
 * By default, performs a shapelet transform through full enumeration (max 1000 shapelets selected)
 *  then classifies with rotation forest.
 * If can be contracted to a maximum run time for shapelets, and can be configured for a different base classifier
 *
 * //Subsequence distances in SubsSeqDistance : defines how to do sDist, outcomes should be the same, difference efficiency
 *
 * 
 */
public class ShapeletTransformClassifier  extends EnhancedAbstractClassifier implements TrainTimeContractable{
    //Basic pipeline is transform, then build classifier on transformed space
    private ShapeletTransform transform;    //Configurable ST
    private Instances shapeletData;         //Transformed shapelets header info stored here
    private Classifier classifier;          //Final classifier built on transformed shapelet

/*************** TRANSFORM STRUCTURE SETTINGS *************************************/
    /** Shapelet transform parameters that can be configured through the STC **/
//This is really just to avoid interfacing directly to the transform

    public static final int minimumRepresentation = 25; //If condensing the search set, this is the minimum number of instances per class to search
    public static int MAXTRANSFORMSIZE=1000;   //Default number in transform
    private int numShapeletsInTransform = MAXTRANSFORMSIZE;
    private ShapeletSearch.SearchType searchType = ShapeletSearch.SearchType.RANDOM;//FULL == enumeration, RANDOM =random sampled to train time cotnract
    private SubSeqDistance.DistanceType distType = SubSeqDistance.DistanceType.IMPROVED_ONLINE;
    private ShapeletQuality.ShapeletQualityChoice qualityMeasure=ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN;
    private SubSeqDistance.RescalerType rescaleType = SubSeqDistance.RescalerType.NORMALISATION;
    private boolean useRoundRobin=true;
    private boolean useCandidatePruning=true;
    private boolean useClassBalancing=false;
    private boolean useBinaryClassValue=false;
    private int minShapeletLength=3;
    private int maxShapeletlength;//Must be set, usually a function of series length, defaults to series length (m)
 //Dont need this   private Function<Integer,Integer> numIntervalsFinder = (seriesLength) -> seriesLength;


    /****************** CONTRACTING *************************************/
    /*The contracting is controlled by the number of shapelets to evaluate. This can either be explicitly set by the user
     * through setNumberOfShapeletsToEvaluate, or, if a contract time is set, it is estimated from the contract.
     * If this is zero and no contract time is set, a full evaluation is done.
      */
    private long numShapeletsInProblem = 0; //Number of shapelets in problem if we do a full enumeration
    private long transformBuildTime;
    private double proportionToEvaluate=1;// Proportion of total num shapelets to evaluate based on time contract
    private long numShapeletsToEvaluate = 0; //Total num shapelets to evaluate over all cases (NOT per case)
    private long totalTimeLimit = 0; //Time limit for transform + classifier, fixed by user
    private long transformTimeLimit = 0;//Time limit assigned to transform, based on totalTimeLimit, but fixed in buildClassifier

/** Shapelet saving options **/
    private String shapeletOutputPath;
    private String checkpointFullPath=""; //location to check point
    private boolean checkpoint=false;
    private boolean saveShapelets=false;
    private String shapeletPath="";

    //Can ditch this?
    private boolean setSeed=false;

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation    result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "authors");
        result.setValue(TechnicalInformation.Field.YEAR, "A shapelet transform for time series classification");
        result.setValue(TechnicalInformation.Field.TITLE, "stuff");
        result.setValue(TechnicalInformation.Field.JOURNAL, "places");
        result.setValue(TechnicalInformation.Field.VOLUME, "vol");
        result.setValue(TechnicalInformation.Field.PAGES, "pages");

        return result;
    }

    //Can be configured to multivariate
    enum TransformType{UNI,MULTI_D,MULTI_I}
    TransformType type=TransformType.UNI;
    
    public void setTransformType(TransformType t){
        type=t;
    }
    enum ShapeletConfig{BAKEOFF,DAWAK,LATEST}//Not sure I should do this like this. MAybe just have a default then methods to configure.
    ShapeletConfig sConfig=ShapeletConfig.LATEST;

    public void setConfiguration(ShapeletConfig s){
        sConfig=s;
    }
    public void setConfiguration(String s){
        s=s.toUpperCase();
        switch(s){
            case "BAKEOFF": case "BAKE OFF": case "BAKE-OFF":
                sConfig=ShapeletConfig.BAKEOFF;
                break;
            case "DAWAK": case "BINARY": case "AARON":
                sConfig=ShapeletConfig.DAWAK;
                break;
            case "LATEST": default:
                sConfig=ShapeletConfig.LATEST;
        }
    }

/** Redundant features in the shapelet space are removed **/
    int[] redundantFeatures;

    public void saveShapelets(String filePathForShapelets){
        shapeletPath=filePathForShapelets;
        saveShapelets=true;
    }
    public void setTransformType(String t){
        t=t.toLowerCase();
        switch(t){
            case "univariate": case "uni":
                type=TransformType.UNI;
                break;
            case "shapeletd": case "shapelet_d": case "dependent":
                type=TransformType.MULTI_D;
                break;
            case "shapeleti": case "shapelet_i":
                type=TransformType.MULTI_I;
                break;
                
        }
    }
    
    public ShapeletTransformClassifier(){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        
        RotationForest rf= new RotationForest();
        rf.setNumIterations(200);
        classifier=rf;
    }

    public void setClassifier(Classifier c){
        classifier=c;
    }

    /**
     * Set the search type, as defined in ShapeletSearch.SearchType
     *
     * Not configured to work with contracting yet.
     *
      * @param type: Search type with valid values  SearchType {FULL, FS, GENETIC, RANDOM, LOCAL, MAGNIFY, TIMED_RANDOM, SKIPPING, TABU, REFINED_RANDOM, IMP_RANDOM, SUBSAMPLE_RANDOM, SKEWED, BO_SEARCH};
     */
    public void setSearchType(ShapeletSearch.SearchType type) {
        searchType = type;
    }

    @Override
    public String getParameters(){
       String paras=transform.getShapeletCounts();
       String classifierParas="No Classifier Para Info";
       if(classifier instanceof EnhancedAbstractClassifier) 
            classifierParas=((EnhancedAbstractClassifier)classifier).getParameters();
        //Build time info
        String str= "TransformActualBuildTime,"+transformBuildTime+",totalTimeContract,"+ totalTimeLimit+",transformTimeContract,"+ transformTimeLimit;
        //Shapelet numbers and contract info
        str+=",numberOfShapeletsInProblem,"+numShapeletsInProblem+",proportionToEvaluate,"+proportionToEvaluate;
        //transform config
        str+=",SearchType,"+searchType+",DistType,"+distType+",QualityMeasure,"+qualityMeasure+",RescaleType,"+rescaleType;
        str+=",UseRoundRobin,"+useRoundRobin+",UseCandidatePruning,"+useCandidatePruning+",UseClassBalancing,"+useClassBalancing+",useBinaryClassValue,"+useBinaryClassValue;
        str+=",useBinaryClassValue,"+useBinaryClassValue+",minShapeletLength,"+minShapeletLength+",maxShapeletLength,"+maxShapeletlength+",ConfigSetup,"+sConfig;
        str+=","+paras+","+classifierParas;
        return str;
    }
    
    
    public long getTransformOpCount(){
        return transform.getCount();
    }
    
    
    public Instances transformDataset(Instances data){
        if(transform.isFirstBatchDone())
            return transform.process(data);
        return null;
    }
    
    //pass in an enum of hour, minute, day, and the amount of them.
    @Override
    public void setTrainTimeLimit(TimeUnit time, long amount) {
        //min,hour,day in longs.
        switch(time){
            case NANOSECONDS:
                totalTimeLimit = amount;
                break;
            case SECONDS:
                totalTimeLimit = (ShapeletTransformTimingUtilities.dayNano/24/60/60) * amount;
                break;
            case MINUTES:
                totalTimeLimit = (ShapeletTransformTimingUtilities.dayNano/24/60) * amount;
                break;
            case HOURS:
                totalTimeLimit = (ShapeletTransformTimingUtilities.dayNano/24) * amount;
                break;
            case DAYS:
                totalTimeLimit = ShapeletTransformTimingUtilities.dayNano * amount;
                break;
            default:
                throw new InvalidParameterException("Invalid time unit");
        }
    }
    
    public void setNumberOfShapeletsToEvaluate(long numS){
        numShapeletsToEvaluate = numS;
    }
    public void setNumberOfShapeletsInTransform(int numS){
        numShapeletsInTransform = numS;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
    // can classifier handle the data?
        getCapabilities().testWithFail(data);
        
        long startTime=System.nanoTime();
//Give 2/3 time for transform, 1/3 for classifier. Need to only do this if its set to have one.
        transformTimeLimit=(long)((((double) totalTimeLimit)*2.0)/3.0);
//        System.out.println("Time limit = "+timeLimit+"  transform time "+transformTimeLimit);
        switch(sConfig){
            case BAKEOFF:
    //To do, configure for bakeoff/hive-cote. Full enumeration, early abandon, CAWPE basic config
                configureBakeoffShapeletTransform(data);
                CAWPE base= new CAWPE();
                base.setupOriginalHESCASettings();
                base.setEstimateOwnPerformance(false);//Defaults to false anyway
                classifier=base;
                break;
            case DAWAK:
    //To do, configure for DAWAK.
                //To do, configure for bakeoff/hive-cote. Full enumeration, early abandon, CAWPE basic config
                configureDawakShapeletTransform(data);
                base= new CAWPE();
                base.setupOriginalHESCASettings();
                base.setEstimateOwnPerformance(false);//Defaults to false anyway
                classifier=base;
                break;
            case LATEST:  //Default config
                configureShapeletTransformTimeContract(data, transformTimeLimit);
                classifier=new RotationForest();
                ((RotationForest)classifier).setNumIterations(200);
        }
        transform=buildTransform(data);
        shapeletData = transform.process(data);
        transformBuildTime=System.nanoTime()-startTime; //Need to store this
        if(debug) {
            System.out.println("DATASET: num cases "+data.numInstances()+" series length "+(data.numAttributes()-1));
//            System.out.println("NANOS: Transform contract =" + transformTimeLimit + " Actual transform time = " + transformBuildTime+" Proportion of contract used ="+((double)transformBuildTime/transformTimeLimit));
            System.out.println("SECONDS:Transform contract =" +(transformTimeLimit/1000000000L)+" Actual transform time taken = " + (transformBuildTime / 1000000000L+" Proportion of contract used ="+((double)transformBuildTime/transformTimeLimit)));
            System.out.println(" Transform getParas  ="+transform.getParameters());
            //            System.out.println("MINUTES:Transform contract =" +(transformTimeLimit/60000000000L)+" Actual transform time = " + (transformBuildTime / 60000000000L));
        }
        redundantFeatures=InstanceTools.removeRedundantTrainAttributes(shapeletData);
        if(saveShapelets){
            System.out.println("Shapelet Saving  ....");
            //NO, do it properly.
            DatasetLoading.saveDataset(shapeletData,shapeletPath+"Transforms"+seed);
//            FullAccessOutFile of=new FullAccessOutFile(shapeletPath+"Transforms"+seed+".arff");
//            of.writeString(shapeletData.toString());
//            of.closeFile();
            FullAccessOutFile of=new FullAccessOutFile(shapeletPath+"Shaplelets"+seed+".csv");
            of.writeLine("BuildTime,"+(System.nanoTime()-startTime));
            of.writeLine("NumShapelets,"+transform.getNumberOfShapelets());
            of.writeLine("Count(not sure!),"+transform.getCount());
            of.writeString("ShapeletLengths");
            ArrayList<Integer> lengths=transform.getShapeletLengths();
            for(Integer i:lengths)
                of.writeString(","+i);
            of.writeString("\n");
            of.writeString(transform.toString());
/*            ArrayList<Shapelet>  shapelets= transform.getShapelets();
            of.writeLine("SHAPELETS:");
            for(Shapelet s:shapelets){
                double[] d=s.getUnivariateShapeletContent();
                for(double x:d)
                    of.writeString(x+",");
                of.writeString("\n");
*/
            of.closeFile();
        }
        long classifierTime= totalTimeLimit -transformBuildTime;
        if(classifier instanceof TrainTimeContractable)
            ((TrainTimeContractable)classifier).setTrainTimeLimit(classifierTime);
//Here get the train estimate directly from classifier using cv for now        
        if(debug)
            System.out.println("Starting build classifier ......");
        classifier.buildClassifier(shapeletData);
        shapeletData=new Instances(data,0);
        trainResults.setBuildTime(System.nanoTime()-startTime);
    }

    @Override
    public double classifyInstance(Instance ins) throws Exception{
        shapeletData.add(ins);
        
        Instances temp  = transform.process(shapeletData);
//Delete redundant
        for(int del:redundantFeatures)
            temp.deleteAttributeAt(del);
        
        Instance test  = temp.get(0);
        shapeletData.remove(0);
        return classifier.classifyInstance(test);
    }
     @Override
    public double[] distributionForInstance(Instance ins) throws Exception{
        shapeletData.add(ins);
        
        Instances temp  = transform.process(shapeletData);
//Delete redundant
        for(int del:redundantFeatures)
            temp.deleteAttributeAt(del);
        
        Instance test  = temp.get(0);
        shapeletData.remove(0);
        return classifier.distributionForInstance(test);
    }
    
    public void setShapeletOutputFilePath(String path){
        shapeletOutputPath = path;
    }

    public ShapeletTransform buildTransform(Instances data){
        //**** CONFIGURE TRANSFORM OPTIONS ****/
        ShapeletTransformFactoryOptions.Builder optionsBuilder = new ShapeletTransformFactoryOptions.Builder();
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        optionsBuilder.setDistanceType(distType);
        optionsBuilder.setQualityMeasure(qualityMeasure);
        optionsBuilder.setKShapelets(numShapeletsInTransform);
        if(useCandidatePruning)
            optionsBuilder.useCandidatePruning();
        if(useClassBalancing)
            optionsBuilder.useClassBalancing();
        if(useBinaryClassValue)
            optionsBuilder.useBinaryClassValue();
        if(useRoundRobin)
            optionsBuilder.useRoundRobin();
        if(setSeed)
            searchBuilder.setSeed(2*seed);
//NEED TO SET RESCALE TYPE HERE
//Set builder up with any time based constraints, defined by numShapeletsToEvaluate>0
        searchBuilder.setMin(minShapeletLength);
        searchBuilder.setMax(maxShapeletlength);
        searchBuilder.setSearchType(searchType);
        if(numShapeletsInProblem==0)
            numShapeletsInProblem=ShapeletTransformTimingUtilities.calculateNumberOfShapelets(data.numInstances(), data.numAttributes()-1, minShapeletLength, maxShapeletlength);

        optionsBuilder.setKShapelets(numShapeletsInTransform);
        searchBuilder.setNumShapeletsToEvaluate(numShapeletsToEvaluate/data.numInstances());//This is ignored if full search is performed

        optionsBuilder.setSearchOptions(searchBuilder.build());
//Finally, get the transform from a Factory with the options set by the builder
        ShapeletTransform st = new ShapeletTransformFactory(optionsBuilder.build()).getTransform();
        if(shapeletOutputPath != null)
            st.setLogOutputFile(shapeletOutputPath);
        return st;

    }
/**
 * Sets up the parameters for Dawak configuration
 * NO TIME CONTRACTING
  */
public void configureDawakShapeletTransform(Instances train){
    int n = train.numInstances();
    int m = train.numAttributes()-1;
//numShapeletsInTransform defaults to MAXTRANSFORMSIZE (500), can be set by user.
//  for very small problems, this number may be far to large, so we will reduce it here.
    distType=SubSeqDistance.DistanceType.IMPROVED_ONLINE;
    qualityMeasure=ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN;
    searchType=ShapeletSearch.SearchType.FULL;
    rescaleType=SubSeqDistance.RescalerType.NORMALISATION;
    useRoundRobin=true;
    useCandidatePruning=true;
    if(train.numClasses() > 2) {
        useBinaryClassValue = true;
        useClassBalancing = true;
    }else{
        useBinaryClassValue = false;
        useClassBalancing = false;
    }

    minShapeletLength=3;
    maxShapeletlength=m;
    numShapeletsInTransform=10*n;//Got to cap this surely!

}
    /**
     * configuring a ShapeletTransform involves configuring a ShapeletTransformFactoryOptions object (via a OptionsBuilder
     * and SearchBuilder)
     * NO Contracting.
     * @param train data set
  Work in progress */
    public void configureBakeoffShapeletTransform(Instances train){
        int n = train.numInstances();
        int m = train.numAttributes()-1;
        distType=SubSeqDistance.DistanceType.NORMAL;
        qualityMeasure=ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN;
        searchType=ShapeletSearch.SearchType.FULL;
        rescaleType=SubSeqDistance.RescalerType.NORMALISATION;
        useRoundRobin=true;
        if(train.numClasses() <10)
            useCandidatePruning=true;
        useBinaryClassValue = false;
        useClassBalancing = false;
        minShapeletLength=3;
        maxShapeletlength=m;
        numShapeletsInTransform=10*n;
        if(n*m<numShapeletsInTransform)
            numShapeletsInTransform=n*m;


    }



    /**
 * configuring a ShapeletTransform involves configuring a ShapeletTransformFactoryOptions object (via a OptionsBuilder
 * and SearchBuilder)
 *
 * @param train data set
 * @param time in nanoseconds that is allowed for the shapelet search
 */
    public void configureShapeletTransformTimeContract(Instances train, long time){
        int n = train.numInstances();
        int m = train.numAttributes()-1;
        distType=SubSeqDistance.DistanceType.IMPROVED_ONLINE;
        qualityMeasure=ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN;
        searchType=ShapeletSearch.SearchType.FULL;
        rescaleType=SubSeqDistance.RescalerType.NORMALISATION;
        useRoundRobin=true;
        useCandidatePruning=true;
        if(train.numClasses() > 2) {
            useBinaryClassValue = true;
            useClassBalancing = true;
        }else{
            useBinaryClassValue = true;
            useClassBalancing = true;
        }
        minShapeletLength=3;
        maxShapeletlength=m;
        if(n*m<numShapeletsInTransform)
            numShapeletsInTransform=n*m;
//Configure the search options if a contract has been ser
        if(debug)
            System.out.println("Number in transform ="+numShapeletsInTransform+" number to evaluate = "+numShapeletsToEvaluate);
        if(time>0) { //contract time in nanoseconds used to estimate the proportion and hence the number of shapelets to evaluate
            numShapeletsInProblem = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n, m, 3, m);
            proportionToEvaluate=estimatePropOfFullSearch(n,m,time);
            if(proportionToEvaluate==1.0) {
                searchType = SearchType.FULL;
                numShapeletsToEvaluate=numShapeletsInProblem;
            }
            else
                numShapeletsToEvaluate = (long)(numShapeletsInProblem*proportionToEvaluate);
            if(debug) {
                System.out.println(" Contract time (secs) = " + time/1000000000);
                System.out.println(" Total number of shapelets = " + numShapeletsInProblem);
                System.out.println(" Proportion to evaluate = " + proportionToEvaluate);
                System.out.println(" Number to evaluate = " + numShapeletsToEvaluate);
            }
        }
        else if(numShapeletsToEvaluate==0){ //User has not explicitly set time or number to evaluate, so we are doing a full search
            //Set to search for full. This is debatable, but seems sensible!
            numShapeletsToEvaluate = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n, m, 3, m);
            searchType=SearchType.FULL;
        }
        if(numShapeletsToEvaluate<n)//Got to do 1 per series. Really should reduce if we do this.
            numShapeletsToEvaluate=n;
        numShapeletsInTransform =  numShapeletsToEvaluate > numShapeletsInTransform ? numShapeletsInTransform : (int) numShapeletsToEvaluate;

    }

    private double estimatePropOfFullSearch(int n, int m, long time){
//nanoToOp is currently a hard coded to 10 nanosecs in ShapeletTransformTimingUtilities. This is a bit crap
//HERE we can estimate it for this run
        long nanoTimeForOp=ShapeletTransformTimingUtilities.nanoToOp;
// Operations contract
        BigInteger allowedNumberOfOperations = new BigInteger(Long.toString(time / nanoTimeForOp));
// Operations required
        BigInteger requiredNumberOfOperations = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
//Need more operations than we are allowed
        double p=1;
        if (requiredNumberOfOperations.compareTo(allowedNumberOfOperations) > 0) {
            BigDecimal oct = new BigDecimal(allowedNumberOfOperations);
            BigDecimal oc = new BigDecimal(requiredNumberOfOperations);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            p= prop.doubleValue();
        }
        return p;
    }
    public static void main(String[] args) throws Exception {
//        String dataLocation = "C:\\Temp\\TSC\\";
        String dataLocation = "E:\\Data\\TSCProblems2018\\";
        String saveLocation = "C:\\Temp\\TSC\\";
        String datasetName = "FordA";
        int fold = 0;
        
        Instances train= DatasetLoading.loadDataNullable(dataLocation+datasetName+File.separator+datasetName+"_TRAIN");
        Instances test= DatasetLoading.loadDataNullable(dataLocation+datasetName+File.separator+datasetName+"_TEST");
        String trainS= saveLocation+datasetName+File.separator+"TrainCV.csv";
        String testS=saveLocation+datasetName+File.separator+"TestPreds.csv";
        String preds=saveLocation+datasetName;
        System.out.println("Data Loaded");
        ShapeletTransformClassifier st= new ShapeletTransformClassifier();
        //st.saveResults(trainS, testS);
        st.setShapeletOutputFilePath(saveLocation+datasetName+"Shapelets.csv");
        st.setMinuteLimit(2);
        System.out.println("Start transform");
        
        long t1= System.currentTimeMillis();
        st.configureShapeletTransformTimeContract(train,st.totalTimeLimit);
        Instances stTrain=st.transform.process(train);
        long t2= System.currentTimeMillis();
        System.out.println("BUILD TIME "+((t2-t1)/1000)+" Secs");
        OutFile out=new OutFile(saveLocation+"ST_"+datasetName+".arff");
        out.writeString(stTrain.toString());
        
    }
/**
 * Checkpoint methods
 */
    public void setSavePath(String path){
        checkpointFullPath=path;
    }
    public void copyFromSerObject(Object obj) throws Exception{
        if(!(obj instanceof ShapeletTransformClassifier))
            throw new Exception("Not a ShapeletTransformClassifier object");
//Copy meta data
        ShapeletTransformClassifier st=(ShapeletTransformClassifier)obj;
//We assume the classifiers have not been built, so are basically copying over the set up
        classifier=st.classifier;
        shapeletOutputPath=st.shapeletOutputPath;
        transform=st.transform;
        shapeletData=st.shapeletData;
        int[] redundantFeatures=st.redundantFeatures;
        transformBuildTime=st.transformBuildTime;
        trainResults =st.trainResults;
        numShapeletsInTransform =st.numShapeletsInTransform;
        searchType =st.searchType;
        numShapeletsToEvaluate =st.numShapeletsToEvaluate;
        seed =st.seed;
        setSeed=st.setSeed;
        totalTimeLimit =st.totalTimeLimit;

        
    }

    
}
