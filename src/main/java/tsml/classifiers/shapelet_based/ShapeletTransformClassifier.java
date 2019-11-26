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
    enum ShapeletConfig{BAKEOFF,DAWAK,LATEST}

    ShapeletConfig sConfig=ShapeletConfig.DAWAK;
    //Basic pipeline is transform, then build classifier on transformed space
    private ShapeletTransform transform;
//Transformed shapelets header info stored here
    private Instances shapeletData;
//Final classifier built on transformed shapelet
    private Classifier classifier;

/** Shapelet transform parameters that can be configured through the STC ***/
//This is really just to avoid interfacing directly to the transform
//If condensing the search set, this is the minimum number of instances per class to search
  public static final int minimumRepresentation = 25;
  //Default number in transform
    public static int MAXTRANSFORMSIZE=1000;
    private long transformBuildTime;
    private int numShapeletsInTransform = MAXTRANSFORMSIZE;
    private SearchType searchType = SearchType.RANDOM;
    private SubSeqDistance.DistanceType distType = SubSeqDistance.DistanceType.IMPROVED_ONLINE;

    private long numShapeletsInProblem = 0; //Number of shapelets in problem if we do a full enumeration
    /** The contracting is controlled by the number of shapelets to evaluate. This can either be explicitly set by the user
     * through setNumberOfShapeletsToEvaluate, or, if a contract time is set, it is estimated from the contract.
     * If this is zero and no contract time is set, a full evaluation is done.
      */
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
    enum TransformType{UNI,MULTI_D,MULTI_I};
    TransformType type=TransformType.UNI;
    
    public void setTransformType(TransformType t){
        type=t;
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
       String paras=transform.getParameters();
       String classifierParas="No Classifier Para Info";
       if(classifier instanceof EnhancedAbstractClassifier) 
            classifierParas=((EnhancedAbstractClassifier)classifier).getParameters();
        return "TransformActualBuildTime,"+transformBuildTime+",totalTimeContract,"+ totalTimeLimit+",transformTimeContract,"+ transformTimeLimit+",numberOfShapeletsInProblem,"+numShapeletsInProblem+",proportionToEvaluate,"+proportionToEvaluate+",TransformParas,"+paras+",ClassifierParas,"+classifierParas;
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
//Give 2/3 time for transform, 1/3 for classifier.
        transformTimeLimit=(long)((((double) totalTimeLimit)*2.0)/3.0);
//        System.out.println("Time limit = "+timeLimit+"  transform time "+transformTimeLimit);
        switch(sConfig){
            case BAKEOFF:
    //To do, configure for bakeoff/hive-cote
            case DAWAK:
    //To do, configure for DAWAK.
            case LATEST://Default config
                transform=configureShapeletTransform(data, transformTimeLimit);
                break;
            default:
                transform=configureShapeletTransform(data, transformTimeLimit);

        }

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
/**
 * Classifiers used in the HIVE COTE paper
 */    
    public void configureCAWPEEnsemble(){
//HIVE_SHAPELET_SVMQ    HIVE_SHAPELET_RandF    HIVE_SHAPELET_RotF    
//HIVE_SHAPELET_NN    HIVE_SHAPELET_NB    HIVE_SHAPELET_C45    HIVE_SHAPELET_SVML   
        classifier=new CAWPE();
        ((CAWPE)classifier).setWeightingScheme(new TrainAcc(1));
        ((CAWPE)classifier).setVotingScheme(new MajorityConfidence());
        Classifier[] classifiers = new Classifier[7];
        String[] classifierNames = new String[7];        
        SMO smo = new SMO();
        smo.turnChecksOff();
        smo.setBuildLogisticModels(true);
        PolyKernel kl = new PolyKernel();
        kl.setExponent(2);
        smo.setKernel(kl);
        if (setSeed)
            smo.setRandomSeed((int)seed);
        classifiers[0] = smo;
        classifierNames[0] = "SVMQ";

        RandomForest r=new RandomForest();
        r.setNumTrees(500);
        if(setSeed)
           r.setSeed((int)seed);            
        classifiers[1] = r;
        classifierNames[1] = "RandF";
            
            
        RotationForest rf=new RotationForest();
        rf.setNumIterations(100);
        if(setSeed)
           rf.setSeed((int)seed);
        classifiers[2] = rf;
        classifierNames[2] = "RotF";
        IBk nn=new IBk();
        classifiers[3] = nn;
        classifierNames[3] = "NN";
        NaiveBayes nb=new NaiveBayes();
        classifiers[4] = nb;
        classifierNames[4] = "NB";
        J48 c45=new J48();
        classifiers[5] = c45;
        classifierNames[5] = "C45";
        SMO svml = new SMO();
        svml.turnChecksOff();
        svml.setBuildLogisticModels(true);
        PolyKernel k2 = new PolyKernel();
        k2.setExponent(1);
        smo.setKernel(k2);
        classifiers[6] = svml;
        classifierNames[6] = "SVML";
        ((CAWPE)classifier).setClassifiers(classifiers, classifierNames, null);
    }
//This sets up the ensemble to work within the time constraints of the problem    
    public void configureEnsemble(){
        ((CAWPE)classifier).setWeightingScheme(new TrainAcc(4));
        ((CAWPE)classifier).setVotingScheme(new MajorityConfidence());
        
        Classifier[] classifiers = new Classifier[3];
        String[] classifierNames = new String[3];
        SMO smo = new SMO();
        smo.turnChecksOff();
        smo.setBuildLogisticModels(true);
        PolyKernel kl = new PolyKernel();
        kl.setExponent(2);
        smo.setKernel(kl);
        if (setSeed)
            smo.setRandomSeed((int)seed);
        classifiers[0] = smo;
        classifierNames[0] = "SVMQ";

        RandomForest r=new RandomForest();
        r.setNumTrees(500);
        if(setSeed)
           r.setSeed((int)seed);            
        classifiers[1] = r;
        classifierNames[1] = "RandF";
            
            
        RotationForest rf=new RotationForest();
        rf.setNumIterations(100);
        if(setSeed)
           rf.setSeed((int)seed);
        classifiers[2] = rf;
        classifierNames[2] = "RotF";
       ((CAWPE)classifier).setClassifiers(classifiers, classifierNames, null);        
    }
    

    
    public void configureBasicEnsemble(){
// Random forest only
        classifier=new CAWPE();
        Classifier[] classifiers = new Classifier[1];
        String[] classifierNames = new String[1];
        RandomForest r=new RandomForest();
        r.setNumTrees(500);
        if(setSeed)
           r.setSeed((int)seed);            
        classifiers[0] = r;
        classifierNames[0] = "RandF";


        ((CAWPE)classifier).setWeightingScheme(new EqualWeighting());
        ((CAWPE)classifier).setVotingScheme(new MajorityVote());
        RotationForest rf=new RotationForest();
        rf.setNumIterations(100);
        if(setSeed)
           rf.setSeed((int)seed);
        ((CAWPE)classifier).setClassifiers(classifiers, classifierNames, null);
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



    /**
     * configuring a ShapeletTransform involves configuring a ShapeletTransformFactoryOptions object (via a OptionsBuilder
     * and SearchBuilder)
     *
     * @param train data set
     * @param time in nanoseconds that is allowed for the shapelet search
  Work in progress NOT TO USE YET*/
    public ShapeletTransform configureBakeoffShapeletTransform(Instances train, long time){
        int n = train.numInstances();
        int m = train.numAttributes()-1;
//numShapeletsInTransform defaults to MAXTRANSFORMSIZE (500), can be set by user.
//  for very small problems, this number may be far to large, so we will reduce it here.

        if(n*m<numShapeletsInTransform)
            numShapeletsInTransform=n*m;

        //**** CONFIGURE TRANSFORM OPTIONS ****/
        ShapeletTransformFactoryOptions.Builder optionsBuilder = new ShapeletTransformFactoryOptions.Builder();
        //Distance type options are in package distance_functions: {NORMAL, ONLINE, IMPROVED_ONLINE, CACHED, ONLINE_CACHED,: See class for info,
        // and three options for multivariate: DEPENDENT, INDEPENDENT, DIMENSION};
        optionsBuilder.setDistanceType(distType);
//Quality measure options {INFORMATION_GAIN, F_STAT, KRUSKALL_WALLIS, MOODS_MEDIAN: change to an ST Parameter
        optionsBuilder.setQualityMeasure(ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN);

//Method of selecting series to search. Round Robin takes one of each class in turn
//Defaults to just sequentially scanning by series
        optionsBuilder.useRoundRobin();
//Candidate pruning:   from the original Ye paper, abandons the whole shapelet based on the order line
//this is only really sensible if a two class problem: the threshold computation rises quickly with number of classes
        if(train.numClasses() <4){
            optionsBuilder.useCandidatePruning();
        }

/*** DETERMINE THE SEARCH OPTIONS . Also sets numShapeletsToEvaluate, numShapeletsInTransform, proportionToEvaluate and can
 * force full search if contract greater than time required for full search ***/
        ShapeletSearchOptions.Builder searchBuilder =configureSearchBuilder(n,m, time);

//if we are evaluating fewer than are in the transform we must change this
        numShapeletsInTransform =  numShapeletsToEvaluate > numShapeletsInTransform ? numShapeletsInTransform : (int) numShapeletsToEvaluate;
        if(debug)
            System.out.println("Number in transform ="+numShapeletsInTransform+" number to evaluate = "+numShapeletsToEvaluate+" number per series = "+numShapeletsToEvaluate/n);
        optionsBuilder.setKShapelets(numShapeletsInTransform);
        optionsBuilder.setSearchOptions(searchBuilder.build());

//Finally, get the transform from a Factory with the options set by the builder
        ShapeletTransform st = new ShapeletTransformFactory(optionsBuilder.build()).getTransform();
        st.setPrintDebug(true);

        if(shapeletOutputPath != null)
            st.setLogOutputFile(shapeletOutputPath);
        return st;

    }



    /**
 * configuring a ShapeletTransform involves configuring a ShapeletTransformFactoryOptions object (via a OptionsBuilder
 * and SearchBuilder)
 *
 * @param train data set
 * @param time in nanoseconds that is allowed for the shapelet search
 */
    public ShapeletTransform configureShapeletTransform(Instances train, long time){
        int n = train.numInstances();
        int m = train.numAttributes()-1;
//numShapeletsInTransform defaults to MAXTRANSFORMSIZE (500), can be set by user.
//  for very small problems, this number may be far to large, so we will reduce it here.

        if(n*m<numShapeletsInTransform)
            numShapeletsInTransform=n*m;

        //**** CONFIGURE TRANSFORM OPTIONS ****/
        ShapeletTransformFactoryOptions.Builder optionsBuilder = new ShapeletTransformFactoryOptions.Builder();
        //Distance type options: {NORMAL, ONLINE, IMP_ONLINE, CACHED, ONLINE_CACHED,: See class for info, refactor IMP (improved)
        // and three options for multivariate: DEPENDENT, INDEPENDENT, DIMENSION};
        optionsBuilder.setDistanceType(distType);
//Quality measure options {INFORMATION_GAIN, F_STAT, KRUSKALL_WALLIS, MOODS_MEDIAN: change to an ST Parameter
        optionsBuilder.setQualityMeasure(ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN);

//These make shapelets binary in terms of quality measure (one vs all)
//Balancing on two class problems generally makes things worse, check out DAWAK
        if(train.numClasses() > 2){
            optionsBuilder.useBinaryClassValue();
            optionsBuilder.useClassBalancing();
        }
//Method of selecting series to search. Round Robin takes one of each class in turn
//Defaults to just sequentially scanning by series
        optionsBuilder.useRoundRobin();
//Candidate pruning:   from the original Ye paper, abandons the whole shapelet based on the order line
        optionsBuilder.useCandidatePruning();

/*** DETERMINE THE SEARCH OPTIONS . Also sets numShapeletsToEvaluate, numShapeletsInTransform, proportionToEvaluate and can
 * force full search if contract greater than time required for full search ***/
        ShapeletSearchOptions.Builder searchBuilder =configureSearchBuilder(n,m, time);

//if we are evaluating fewer than are in the transform we must change this
        numShapeletsInTransform =  numShapeletsToEvaluate > numShapeletsInTransform ? numShapeletsInTransform : (int) numShapeletsToEvaluate;
        if(debug)
            System.out.println("Number in transform ="+numShapeletsInTransform+" number to evaluate = "+numShapeletsToEvaluate);
        optionsBuilder.setKShapelets(numShapeletsInTransform);
        optionsBuilder.setSearchOptions(searchBuilder.build());

//Finally, get the transform from a Factory with the options set by the builder
        ShapeletTransform st = new ShapeletTransformFactory(optionsBuilder.build()).getTransform();
        st.setPrintDebug(true);

        if(shapeletOutputPath != null)
            st.setLogOutputFile(shapeletOutputPath);
        return st;

    }

    /**
     * * Configure the search algorithm.
     * This is done by having a Builder static nested within ShapeletSearchOptions
     *  which is just a clone of the outer class. This is a style choice in order to make sure the options are immutable
     *  once it has been configured and built.
      * @return Builder to configure shapelet search
     */
    private ShapeletSearchOptions.Builder configureSearchBuilder(int n, int m, long time){
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);
// How many operations can we perform based on time contract.
//if both time and numShapeletsToEvaluate have been set, time trumps numShapeletsToEvaluate
        if(time>0) { //contract time in nanoseconds used to estimate the proportion and hence the number of shapelets to evaluate
            numShapeletsInProblem = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n, m, 3, m);
            proportionToEvaluate=estimatePropOfFullSearch(m,n,time);
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
    //      else: user has set numShapeletsToEvaluate
        if(setSeed)
            searchBuilder.setSeed(2*seed);
//Set builder up with any time based constraints, defined by numShapeletsToEvaluate>0
        searchBuilder.setSearchType(searchType);
        //Searching requires the number of shapelets PER SERIES, not the total number.
        searchBuilder.setNumShapeletsToEvaluate(numShapeletsToEvaluate/n);//This is ignored if full search is performed
        return searchBuilder;
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
        st.configureBasicEnsemble();
        //st.saveResults(trainS, testS);
        st.setShapeletOutputFilePath(saveLocation+datasetName+"Shapelets.csv");
        st.setMinuteLimit(2);
        System.out.println("Start transform");
        
        long t1= System.currentTimeMillis();
        st.configureShapeletTransform(train,st.totalTimeLimit);
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
