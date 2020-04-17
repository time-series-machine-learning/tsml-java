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
package tsml.classifiers.legacy;


import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.tuning.ParameterSpace;
import experiments.data.DatasetLoading;
import fileIO.FullAccessOutFile;
import fileIO.OutFile;
import machine_learning.classifiers.ensembles.ContractRotationForest;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.Tuneable;
import tsml.transformers.ShapeletTransform;
import tsml.transformers.shapelet_tools.ShapeletTransformFactory;
import tsml.transformers.shapelet_tools.ShapeletTransformFactoryOptions.ShapeletTransformOptions;
import tsml.transformers.shapelet_tools.ShapeletTransformTimingUtilities;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch.SearchType;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;

import java.io.File;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.TimeUnit;


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
public class ShapeletTransformClassifierLegacy extends EnhancedAbstractClassifier
        implements TrainTimeContractable, Tuneable {
    //Basic pipeline is transform, then build classifier on transformed space
    private ShapeletTransform transform;    //Configurable ST
   //Default to one hour

    private Instances shapeletData;         //Transformed shapelets header info stored here
    private Classifier classifier;          //Final classifier built on transformed shapelet data

/*************** TRANSFORM STRUCTURE SETTINGS *************************************/
    /** Shapelet transform parameters that can be configured through the STC, stored here **/
    private ShapeletTransformOptions transformOptions=new ShapeletTransformOptions();
    private int numShapeletsInTransform = ShapeletTransform.MAXTRANSFORMSIZE;
    private SearchType searchType = SearchType.RANDOM;//FULL == enumeration, RANDOM =random sampled to train time cotnract
    /** Redundant features in the shapelet space are removed prior to building the classifier **/
    int[] redundantFeatures;

    /** PCA Option: not currently implemented, as it has not been debugged
    private boolean performPCA=false;
    private PCA pca;
    private int numPCAFeatures=100;
    */



    /****************** CONTRACTING *************************************/
    /*The contracting is controlled by the number of shapelets to evaluate. This can either be explicitly set by the user
     * through setNumberOfShapeletsToEvaluate, or, if a contract time is set, it is estimated from the contract.
     * If this is zero and no contract time is set, a full evaluation is done.
      */
    private boolean trainTimeContract =false;
    private long trainContractTimeNanos = 0; //Time limit for transform + classifier, fixed by user. If <=0, no contract


    private long transformContractTime = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);//Time limit assigned to transform, based on contractTime, but fixed in buildClassifier in an adhoc way


    private long classifierContractTime = 0;//Time limit assigned to classifier, based on contractTime, but fixed in buildClassifier in an adhoc way


    private long numShapeletsInProblem = 0; //Number of shapelets in problem if we do a full enumeration
    private double singleShapeletTime=0;    //Estimate of the time to evaluate a single shapelet
    private double proportionToEvaluate=1;// Proportion of total num shapelets to evaluate based on time contract
    private long numShapeletsToEvaluate = 0; //Total num shapelets to evaluate over all cases (NOT per case)
    private long transformBuildTime=0;
    public void setTransformTime(long t){
        transformContractTime=t;
    }
    public void setTransformTimeHours(long t){
        transformContractTime=TimeUnit.NANOSECONDS.convert(t, TimeUnit.HOURS);
    }

/************* CHECKPOINTING and SAVING ************ Could all  move to transformOptions */
//Check pointing is not fully debugged
    private String checkpointFullPath=""; //location to check point
    private boolean checkpoint=false;
//If these are set, the shapelet meta information is saved to <path>/Workspace/ and the transforms saved to <path>/Transforms
    private String shapeletOutputPath;
    private boolean saveShapelets=false;
    private boolean pruneMatchingShapelets=false;
    /**
     * @param pruneMatchingShapelets the pruneMatchingShapelets to set
     */
    public void setPruneMatchingShapelets(boolean pruneMatchingShapelets) {
        this.pruneMatchingShapelets = pruneMatchingShapelets;
    }

    //This is set up to allow the user to easily employ a default configuration. Needs a bit of work
    enum ShapeletConfig{BAKEOFF,BALANCED,DEFAULT}
    ShapeletConfig sConfig= ShapeletConfig.DEFAULT;//The default or user set up, to avoid overwriting user defined config

    /** If trainAccuracy is required, there are two mechanisms to obtain it:
     * 2. estimator=CV: do a 10x CV on the train set with a clone
     * of this classifier
     * 3. estimator=OOB: build an OOB model just to get the OOB
     * accuracy estimate
     */
    enum EstimatorMethod{CV,OOB}
    private EstimatorMethod estimator= EstimatorMethod.CV;
    public void setEstimatorMethod(String str){
        String s=str.toUpperCase();
        if(s.equals("CV"))
            estimator= EstimatorMethod.CV;
        else if(s.equals("OOB"))
            estimator= EstimatorMethod.OOB;
        else
            throw new UnsupportedOperationException("Unknown estimator method in TSF = "+str);
    }



    public ShapeletTransformClassifierLegacy(){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        configureDefaultShapeletTransform();
        ContractRotationForest rotf=new ContractRotationForest();
        rotf.setMaxNumTrees(200);
        classifier=rotf;


/*        CAWPE base= new CAWPE();//Change to RotF
        base.setupOriginalHESCASettings();
        base.setEstimateOwnPerformance(false);//Defaults to false anyway
        classifier=base;
*/
    }
/* Not debugged, doesnt currently work
    public void usePCA(){
        setPCA(true);
    }
    public void setPCA(boolean b) {
        setPCA(b,numPCAFeatures);
    }
    public void setPCA(boolean b, int numberEigenvectorsToRetain) {
        performPCA = b;
        pca=new PCA();
        numPCAFeatures=numberEigenvectorsToRetain;
        pca.setNumAttributesToKeep(numPCAFeatures);
    }
*/
    @Override
    public void buildClassifier(Instances data) throws Exception {
    // can classifier handle the data?
        getCapabilities().testWithFail(data);
    //Add the requirement to test if there are at least one of each class
        long startTime=System.nanoTime();
//Give 2/3 time for transform, 1/3 for classifier. Need to only do this if its set to have one.
//All in nanos
        System.out.println("Are we contracting? "+trainTimeContract+" transform contract time ="+trainContractTimeNanos);
        if(trainTimeContract) {

            transformContractTime = trainContractTimeNanos * 2 / 3;
            classifierContractTime = trainContractTimeNanos - transformContractTime;
        }
        else{
            classifierContractTime=0;
        }
//       Full set up of configs to match published. Note these will reset other parameters to their default
//      to the default, so user set parameters prior to this point will be overwritten,
        switch(sConfig){
            case BAKEOFF:
    //Full enumeration, early abandon, CAWPE basic config, 10n shapelets in transform, capped at n*m
                configureBakeoffShapeletTransform(data);
                break;
            case BALANCED:
    //As with bakeoff, but with binary shapelets and class balancing when #classes > 2, 10n shapelets in transform, uncapped!
                configureDawakShapeletTransform(data);
                break;
            default:
                configureDataDependentShapeletTransform(data);
        }
//Contracting with the shapelet transform is handled by setting the number of shapelets per series to evaluate.
//This is done by estimating the time to evaluate a single shapelet then extrapolating (not in aarons way)
        if(transformContractTime >0) {
            printLineDebug(" Contract time limit = "+ transformContractTime);
            configureTrainTimeContract(data, transformContractTime);
        }
        //This is hacked to build a cShapeletTransform
        transform= constructShapeletTransform(data);
        transform.setSuppressOutput(debug);

//The cConfig CONTRACT option is currently hacked into buildTransfom. here for now
//        if(transform instanceof cShapeletFilter)
//            ((cShapeletFilter)transform).setContractTime(transformTimeLimit);
        if(transformContractTime >0) {
//            long numberOfShapeletsPerSeries=numShapeletsInProblem/data.numInstances();
            double timePerShapelet= transformContractTime /numShapeletsToEvaluate;
            printLineDebug("Total shapelets per series "+numShapeletsInProblem/data.numInstances()+" num to eval = "+numShapeletsToEvaluate/data.numInstances());
            transform.setContractTime(transformContractTime);
            transform.setAdaptiveTiming(true);
            transform.setTimePerShapelet(timePerShapelet);
            printLineDebug(" Time per shapelet = "+timePerShapelet);
//            transform.setProportionToEvaluate(proportionToEvaluate);
        }
//Put this in the options rather than here
        transform.setPruneMatchingShapelets(pruneMatchingShapelets);

        shapeletData = transform.fitTransform(data);
        transformBuildTime=System.nanoTime()-startTime; //Need to store this
        printLineDebug("SECONDS:Transform contract =" +(transformContractTime /1000000000L)+" Actual transform time taken = " + (transformBuildTime / 1000000000L+" Proportion of contract used ="+((double)transformBuildTime/ transformContractTime)));
        printLineDebug(" Transform getParas  ="+transform.getParameters());
        redundantFeatures=InstanceTools.removeRedundantTrainAttributes(shapeletData);
        if(saveShapelets)
            saveShapeletData(data);


        printLineDebug("Starting STC build classifier after "+(System.nanoTime()-startTime)/1000000000+" ......");
        if(getEstimateOwnPerformance()){
// if the classifier can estimate its own performance, do that. This is not yet in the time contract!
            boolean doExternalCV=false;
            doExternalCV=!((classifier instanceof EnhancedAbstractClassifier)&&((EnhancedAbstractClassifier)classifier).ableToEstimateOwnPerformance());
            if(doExternalCV) {
                printLineDebug("Doing a CV with base to estimate accuracy");
                int numFolds = setNumberOfFolds(data);
                CrossValidationEvaluator cv = new CrossValidationEvaluator();
                cv.setSeed(seed * 12);
                cv.setNumFolds(numFolds);
                trainResults = cv.crossValidateWithStats(classifier, shapeletData);
            }
            else{//The classifier can handler it internally
                throw new RuntimeException(("ERROR: internal estimates not sorted out yet"));

            }
        }

        if(classifierContractTime>0 && classifier instanceof TrainTimeContractable) {
            ((TrainTimeContractable) classifier).setTrainTimeLimit(classifierContractTime);
        }



        //Optionally do a PCA to reduce dimensionality. Not an option currently, it is broken
/*
        if(performPCA){
            pca.setNumAttributesToKeep(shapeletData.numAttributes()-1);
            pca.fit(shapeletData);
            shapeletData=pca.transform(shapeletData);
            System.out.println(shapeletData.toString());
        }
*/
//Here get the train estimate directly from classifier using cv for now
        classifier.buildClassifier(shapeletData);
        shapeletData=new Instances(data,0);
        trainResults.setBuildTime(System.nanoTime()-startTime);
        trainResults.setParas(getParameters());
//HERE: If the base classifier can estimate its own performance, then lets do it here

    }

    @Override
    public double classifyInstance(Instance ins) throws Exception{
        shapeletData.add(ins);

        Instances temp  = transform.transform(shapeletData);
//Delete redundant
        for(int del:redundantFeatures)
            temp.deleteAttributeAt(del);
/*        if(performPCA){
            temp=pca.transform(temp);
        }
*/
        Instance test  = temp.get(0);
        shapeletData.remove(0);
        return classifier.classifyInstance(test);
    }
     @Override
    public double[] distributionForInstance(Instance ins) throws Exception{
        shapeletData.add(ins);

        Instances temp  = transform.transform(shapeletData);
//Delete redundant
        for(int del:redundantFeatures)
            temp.deleteAttributeAt(del);
/*         if(performPCA){
             temp=pca.transform(temp);
         }
*/
        Instance test  = temp.get(0);
        shapeletData.remove(0);

        return classifier.distributionForInstance(test);
    }




    public void setShapeletOutputFilePath(String path){
        shapeletOutputPath = path;
        saveShapelets=true;
    }
    private void saveShapeletData(Instances data){
        System.out.println("Saving the transform as an arff file and the transform data in different files. The shapelets will also be saved by the transform in the same location.");
        //Write shapelet transform to arff file
        File f= new File(shapeletOutputPath+"ShapeletTransforms/"+data.relationName());
        if(!f.exists())
            f.mkdirs();
        shapeletData.setRelationName(data.relationName());
        DatasetLoading.saveDataset(shapeletData,shapeletOutputPath+"ShapeletTransforms/"+data.relationName()+"/"+data.relationName()+seed+"_TRAIN");
        f= new File(shapeletOutputPath+"Workspace/"+data.relationName());
        if(!f.exists())
            f.mkdirs();
        FullAccessOutFile of=new FullAccessOutFile(shapeletOutputPath+"Workspace/"+data.relationName()+"/shapleletInformation"+seed+".csv");
        String str= getTransformParameters();
        Date date = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        of.writeLine("Generated by ShapeletTransformClassifier.java on " + formatter.format(date));
        of.writeLine(str);
        of.writeLine("NumShapelets,"+transform.getNumberOfShapelets());
        of.writeLine("Operations count(not sure!),"+transform.getCount());
        of.writeString("ShapeletLengths");
        ArrayList<Integer> lengths=transform.getShapeletLengths();
        for(Integer i:lengths)
            of.writeString(","+i);
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

    public ShapeletTransform constructShapeletTransform(Instances data){
        //**** Builds the transform using transformOptions and a search builder ****/
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        if(seedClassifier)
            searchBuilder.setSeed(2*seed);
//For some reason stored twice in the transform options and the search builder.
        searchBuilder.setMin(transformOptions.getMinLength());
        searchBuilder.setMax(transformOptions.getMaxLength());
        searchBuilder.setSearchType(searchType);
        if(numShapeletsInProblem==0)
            numShapeletsInProblem=ShapeletTransformTimingUtilities.calculateNumberOfShapelets(data.numInstances(), data.numAttributes()-1, transformOptions.getMinLength(), transformOptions.getMaxLength());
        transformOptions.setKShapelets(numShapeletsInTransform);
        searchBuilder.setNumShapeletsToEvaluate(numShapeletsToEvaluate/data.numInstances());//This is ignored if full search is performed
        transformOptions.setSearchOptions(searchBuilder.build());
        //Finally, get the transform from a Factory with the options set by the builder
        ShapeletTransform st = new ShapeletTransformFactory(transformOptions.build()).getTransform();
        if(saveShapelets && shapeletOutputPath != null)
            st.setLogOutputFile(shapeletOutputPath+"Workspace/"+data.relationName()+"/shapelets"+seed+".csv");
        return st;

    }


    /*********** METHODS TO CONFIGURE TRANSFORM
     * Note there are two types of parameters: data independent and data dependent. The
     * former are set in the constructor, the latter in buildClassifier. We could tidy this up with lambdas


     * Sets up this default parameters that are not data dependent. This is called in the constructor
     * and the user can reconfigure these prior to classifier build. These could also be tuned.
      */
    public void configureDefaultShapeletTransform(){
    searchType= SearchType.FULL;
    transformOptions.setDistanceType(ShapeletDistance.DistanceType.IMPROVED_ONLINE);
    transformOptions.setQualityMeasure(ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN);
    transformOptions.setRescalerType(ShapeletDistance.RescalerType.NORMALISATION);
    transformOptions.setRoundRobin(true);
    transformOptions.setCandidatePruning(true);
    transformOptions.setBinaryClassValue(false);
    transformOptions.setClassBalancing(false);
    transformOptions.setKShapelets(ShapeletTransform.MAXTRANSFORMSIZE);
}
    /**
     * Sets up the parameters that require the data characteristics (series length, number of classes and number of cases
     */
    public void configureDataDependentShapeletTransform(Instances train){
        int n = train.numInstances();
        int m = train.numAttributes()-1;
        transformOptions.setMinLength(3);
        transformOptions.setMaxLength(m);
//DEtermine balanced or not,
        if(train.numClasses() > 2) {
            transformOptions.setBinaryClassValue(true);
            transformOptions.setClassBalancing(true);
        }else{
            transformOptions.setBinaryClassValue(false);
            transformOptions.setClassBalancing(false);
        }
        if(numShapeletsInTransform==ShapeletTransform.MAXTRANSFORMSIZE)//It has not then been set by the user
            numShapeletsInTransform=  10*train.numInstances() < ShapeletTransform.MAXTRANSFORMSIZE ? 10*train.numInstances(): ShapeletTransform.MAXTRANSFORMSIZE;        //Got to cap this surely!
    }

    /**
     * Specific set up for the DAWAK version (rename) described in
     * @param train
     */
    public void configureDawakShapeletTransform(Instances train) {
        configureDefaultShapeletTransform();
        if(train.numClasses() > 2) {
        transformOptions.setBinaryClassValue(true);
        transformOptions.setClassBalancing(true);
    }else{
        transformOptions.setBinaryClassValue(false);
        transformOptions.setClassBalancing(false);
    }
    if(numShapeletsInTransform==ShapeletTransform.MAXTRANSFORMSIZE)//It has not then been set by the user
        numShapeletsInTransform=  10*train.numInstances() < ShapeletTransform.MAXTRANSFORMSIZE ? 10*train.numInstances(): ShapeletTransform.MAXTRANSFORMSIZE;        //Got to cap this surely!
    transformOptions.setKShapelets(numShapeletsInTransform);

}
   /**
     * configuring a ShapeletTransform to the original ST format used in the bakeoff
     *
     * @param train data set
  Work in progress */
    public void configureBakeoffShapeletTransform(Instances train){
        transformOptions.setDistanceType(ShapeletDistance.DistanceType.NORMAL);
        if(train.numClasses() <10)
            transformOptions.setCandidatePruning(true);
        else
            transformOptions.setCandidatePruning(false);
        transformOptions.setBinaryClassValue(false);
        transformOptions.setClassBalancing(false);
        if(numShapeletsInTransform==ShapeletTransform.MAXTRANSFORMSIZE)//It has not then been set by the user
            numShapeletsInTransform=  10*train.numInstances() < ShapeletTransform.MAXTRANSFORMSIZE ? 10*train.numInstances(): ShapeletTransform.MAXTRANSFORMSIZE;        //Got to cap this surely!
        transformOptions.setKShapelets(numShapeletsInTransform);
    }




    /**
     * This method estimates how many shapelets per series (numShapeletsToEvaluate) can be evaluated given a specific time contract.
     * It should just return this value
     * It also calculates numShapeletsInTransform and proportionToEvaluate, both stored by the classifier. It can set searchType to FULL, if the proportion
     * is estimated to be full.
     * Note the user can set numShapeletsToEvaluate explicitly. The user can also set the contract time explicitly, thus invoking
     * this method in buildClassifier. If both numShapeletsToEvaluate and time have been set, we have a contradiction from the user.
     * We assume time take precedence, and overwrite numShapeletsToEvaluate
     *
     *NEED TO RECONFIGURE FOR USER SET numShapeletToEvaluate
      * @param train train data
     * @param time contract time in nanoseconds
     */
    public void configureTrainTimeContract(Instances train, long time){
        //Configure the search options if a contract has been ser
        // else
        int n = train.numInstances();
        int m = train.numAttributes() - 1;
        if(time>0){
            searchType = SearchType.RANDOM;
            if(debug)
                System.out.println("Number in transform ="+numShapeletsInTransform+" number to evaluate = "+numShapeletsToEvaluate+" contract time (secs) = "+ time/1000000000);
            numShapeletsInProblem = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n, m, 3, m);
//This is aarons way of doing it based on hard coded estimate of the time for a single operation
            proportionToEvaluate= estimatePropOfFullSearchAaron(n,m,time);


            if(proportionToEvaluate==1.0) {
                searchType = SearchType.FULL;
                numShapeletsToEvaluate=numShapeletsInProblem;
            }
            else
                numShapeletsToEvaluate = (long)(numShapeletsInProblem*proportionToEvaluate);
            if(debug) {
                System.out.println(" Total number of shapelets = " + numShapeletsInProblem);
                System.out.println(" Proportion to evaluate = " + proportionToEvaluate);
                System.out.println(" Number to evaluate = " + numShapeletsToEvaluate);
            }
            if(numShapeletsToEvaluate<n)//Got to do 1 per series. Really should reduce if we do this.
                numShapeletsToEvaluate=n;
            numShapeletsInTransform =  numShapeletsToEvaluate > numShapeletsInTransform ? numShapeletsInTransform : (int) numShapeletsToEvaluate;
        }

    }

    // Tony's way of doing it based on a timing model for predicting for a single shapelet
//Point estimate to set prop, could use a hard coded
//This is a bit unintuitive, should move full towards a time per shapelet model
    private double estimatePropOfFullSearchTony(int n, int m, int totalNumShapelets, long time){
        double nPower=1.2;
        double mPower=1.3;
        double scaleFactor=Math.pow(2,26);
        singleShapeletTime=Math.pow(n,nPower)*Math.pow(m,mPower)/scaleFactor;
        long timeRequired=(long)(singleShapeletTime*totalNumShapelets);
        double p=1;
        if(timeRequired>time)
            p=timeRequired/(double)time;
        return p;
    }


// Aarons way of doing it based on time for a single operation
    private double estimatePropOfFullSearchAaron(int n, int m, long time){
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

    /**
     * @return String, comma separated relevant variables, used in Experiment.java to write line 2 of results
     */
    @Override
    public String getParameters(){

        String paras=transform.getShapeletCounts();
        //Build time info
        String result=super.getParameters();
        //Shapelet numbers and contract info
        result+=",numberOfShapeletsInProblem,"+numShapeletsInProblem+",proportionToEvaluate,"+proportionToEvaluate;
        //transform config
        result+=",SearchType,"+searchType;
        result+=","+transformOptions.toString();
        result+=",ConfigSetup,"+sConfig;
        result+=","+paras;
        result+=",Classifier,"+classifier.getClass().getSimpleName();
        String classifierParas="No Classifier Para Info";
        if(classifier instanceof EnhancedAbstractClassifier)
            classifierParas=((EnhancedAbstractClassifier)classifier).getParameters();
        result+=","+classifierParas;
        if(trainTimeContract)
            result+= ",TimeContract(ns), " +trainContractTimeNanos;
        else
            result+=",NoContract";
        result+= ",TransformActualBuildTime,"+transformBuildTime+",trainContractTimeNanos,"+ trainContractTimeNanos +",transformContractTime,"+ transformContractTime;


        result+=",EstimateOwnPerformance,"+getEstimateOwnPerformance();
        if(getEstimateOwnPerformance()) {
            result += ",trainEstimateMethod," + estimator;
        }

        return result;
    }

    /**
     *
     * @return a string containing just the transform parameters
     */
    public String getTransformParameters(){
        String paras=transform.getShapeletCounts();
        String str= "TransformActualBuildTime,"+transformBuildTime+",totalTimeContract,"+ trainContractTimeNanos +",transformTimeContract,"+ transformContractTime;
        //Shapelet numbers and contract info
        str+=",numberOfShapeletsInProblem,"+numShapeletsInProblem+",proportionToEvaluate,"+proportionToEvaluate;
        //transform config
        str+=",SearchType,"+searchType;
        str+=","+transformOptions.toString();
        str+=",ConfigSetup,"+sConfig;
        str+=","+paras;
        return str;
    }

    public long getTransformOpCount(){
        return transform.getCount();
    }


    public void setTrainTimeLimit(long amount) {
        trainTimeContract=true;
        trainContractTimeNanos = amount;
    }
/*    I think this is now redundant, default method in TrainTimeContractable does this
    //pass in an enum of hour, minute, day, and the amount of them.
    @Override
    public void setTrainTimeLimit(TimeUnit time, long amount) {
        //min,hour,day converted to nanosecs.
        switch(time){
            case NANOSECONDS:
                contractTime = amount;
                break;
            case SECONDS:
                contractTime = (ShapeletTransformTimingUtilities.dayNano/24/60/60) * amount;
                break;
            case MINUTES:
                contractTime = (ShapeletTransformTimingUtilities.dayNano/24/60) * amount;
                break;
            case HOURS:
                contractTime = (ShapeletTransformTimingUtilities.dayNano/24) * amount;
                break;
            case DAYS:
                contractTime = ShapeletTransformTimingUtilities.dayNano * amount;
                break;
            default:
                throw new InvalidParameterException("Invalid time unit");
        }
    }
*/
    public void setNumberOfShapeletsToEvaluate(long numS){
        numShapeletsToEvaluate = numS;
    }
    public void setNumberOfShapeletsInTransform(int numS){
        numShapeletsInTransform = numS;
    }
    public void setConfiguration(ShapeletConfig s){
        sConfig=s;
    }
    public final void setConfiguration(String s){
        String temp=s.toUpperCase();
        switch(temp){
            case "BAKEOFF": case "BAKE OFF": case "BAKE-OFF": case "FULL":
                sConfig= ShapeletConfig.BAKEOFF;
                break;
            case "DAWAK": case "BINARY": case "AARON": case "BALANCED":
                sConfig= ShapeletConfig.BALANCED;
                break;
            default:
                sConfig= ShapeletConfig.DEFAULT;
        }
    }

    /**
     * Checkpoint methods
     */
    public void setSavePath(String path){
        checkpointFullPath=path;
    }
    public void copyFromSerObject(Object obj) throws Exception{
        if(!(obj instanceof ShapeletTransformClassifierLegacy))
            throw new Exception("Not a ShapeletTransformClassifier object");
//Copy meta data
        ShapeletTransformClassifierLegacy st=(ShapeletTransformClassifierLegacy)obj;
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
        seedClassifier=st.seedClassifier;
        trainContractTimeNanos =st.trainContractTimeNanos;


    }
/*********** SETTERS AND GETTERS :  Methods for manual configuration  **********/
 /**
     * Set how shapelets are assessed
     * @param qual Quality measure type, options are         INFORMATION_GAIN,F_STAT,KRUSKALL_WALLIS,MOODS_MEDIAN
     */
    public void setQualityMeasure(ShapeletQuality.ShapeletQualityChoice qual){
        transformOptions.setQualityMeasure(qual);
    }
    public void setRescalerType(ShapeletDistance.RescalerType r){
        transformOptions.setRescalerType(r);
    }

    /**
     * Set how shapelets are searched for in a given series.
     * @param type: Search type with valid values
     *            SearchType {FULL, FS, GENETIC, RANDOM, LOCAL, MAGNIFY, TIMED_RANDOM, SKIPPING, TABU,
     *             REFINED_RANDOM, IMP_RANDOM, SUBSAMPLE_RANDOM, SKEWED, BO_SEARCH};
     */
    public void setSearchType(SearchType type) {
        searchType = type;
    }
    public void setClassifier(Classifier c){
        classifier=c;
    }
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

    /**
     * From the interface Tuneable
     * @return the range of parameters to tune over
     */
    @Override
    public ParameterSpace getDefaultParameterSearchSpace(){
        ParameterSpace ps=new ParameterSpace();
        String[] maxNumShapelets={"100","200","300","400","500","600","700","800","900","1000"};
        ps.addParameter("T", maxNumShapelets);

        return ps;
    }
    /**
     * Parses a given list of options to set the parameters of the classifier.
     * We use this for the tuning mechanism, setting parameters through setOptions
     <!-- options-start -->
     * Valid options are: <p/>
     * <pre> -S
     * Number of shapelets kept in the transform.</pre>
     * More to follow
     <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception{
        String numShapeletsString= Utils.getOption('S', options);
        if (numShapeletsString.length() != 0)
            numShapeletsInTransform = Integer.parseInt(numShapeletsString);
        else
            throw new Exception("in setOptions Unable to read number of intervals, -T flag is not set");
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
        ShapeletTransformClassifierLegacy st= new ShapeletTransformClassifierLegacy();
        //st.saveResults(trainS, testS);
        st.setShapeletOutputFilePath(saveLocation+datasetName+"Shapelets.csv");
        st.setMinuteLimit(2);
        System.out.println("Start transform");

        long t1= System.currentTimeMillis();

        st.configureDefaultShapeletTransform();
        st.configureTrainTimeContract(train,st.trainContractTimeNanos);
        Instances stTrain=st.transform.fitTransform(train);
        long t2= System.currentTimeMillis();
        System.out.println("BUILD TIME "+((t2-t1)/1000)+" Secs");
        OutFile out=new OutFile(saveLocation+"ST_"+datasetName+".arff");
        out.writeString(stTrain.toString());

    }
}
