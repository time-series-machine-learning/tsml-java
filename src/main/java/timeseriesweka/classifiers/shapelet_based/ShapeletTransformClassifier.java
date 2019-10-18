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
package timeseriesweka.classifiers.shapelet_based;

import experiments.data.DatasetLoading;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransformFactory;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransform;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransformFactoryOptions;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransformTimingUtilities;
import java.io.File;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import utilities.InstanceTools;
import weka_extras.classifiers.ensembles.CAWPE;
import weka.core.Instance;
import weka.core.Instances;
import static timeseriesweka.filters.shapelet_transforms.ShapeletTransformTimingUtilities.nanoToOp;
import timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQuality;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch.SearchType;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearchOptions;
import weka_extras.classifiers.ensembles.voting.MajorityConfidence;
import weka_extras.classifiers.ensembles.weightings.TrainAcc;
import fileIO.FullAccessOutFile;
import fileIO.OutFile;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import timeseriesweka.classifiers.EnhancedAbstractClassifier;

import weka_extras.classifiers.ensembles.voting.MajorityVote;
import weka_extras.classifiers.ensembles.weightings.EqualWeighting;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import timeseriesweka.classifiers.TrainTimeContractable;

/**
 *
 * By default, performs a shapelet transform through full enumeration (max 1000 shapelets selected)
 *  then classifies with rotation forest.
 * If can be contracted to a maximum run time for shapelets, and can be configured for a different base classifier
 * 
 * 
 */
public class ShapeletTransformClassifier  extends EnhancedAbstractClassifier implements TrainTimeContractable{
//Basic pipeline is transform, then build classifier on transformed space
    private ShapeletTransform transform;
//Transformed shapelets header info stored here
    private Instances shapeletData;
//Final classifier built on transformed shapelet
    private Classifier classifier;
//Minimum number of instances per class in the train set
    public static final int minimumRepresentation = 25;
//Default number in transform
    private static int MAXTRANSFORMSIZE=1000; 

    private boolean preferShortShapelets = false;
    private String shapeletOutputPath;
    int[] redundantFeatures;
    private long transformBuildTime;
    private int numShapeletsInTransform = MAXTRANSFORMSIZE;
    private SearchType searchType = SearchType.IMP_RANDOM;
    private long numShapelets = 0;
    private long seed = 0;
    private boolean setSeed=false;
    private long timeLimit = Long.MAX_VALUE;
    private String checkpointFullPath=""; //location to check point 
    private boolean checkpoint=false;
    private boolean saveShapelets=false;
    private String shapeletPath="";
//Can be configured to multivariate     
    enum TransformType{UNI,MULTI_D,MULTI_I};
    TransformType type=TransformType.UNI;
    
    public void setTransformType(TransformType t){
        type=t;
    }
    public void saveShapelets(String str){
        shapeletPath=str;
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
        
//        configureCAWPEEnsemble();
    }

    public void setSeed(long sd){
        setSeed=true;
        seed = sd;
    }
    public void setClassifier(Classifier c){
        classifier=c;
    }
    
    //careful when setting search type as you could set a type that violates the contract.
    public void setSearchType(ShapeletSearch.SearchType type) {
        searchType = type;
    }


    /*//if you want CAWPE to perform CV.
    public void setEstimateEnsemblePerformance(boolean b) {
        ensemble.setEstimateEnsemblePerformance(b);
    }*/
 
        
    @Override
    public String getParameters(){
       String paras=transform.getParameters();
       String classifierParas="No Classifier Para Info";
       if(classifier instanceof EnhancedAbstractClassifier) 
            classifierParas=((EnhancedAbstractClassifier)classifier).getParameters();
        return "BuildTime,"+trainResults.getBuildTime()+",CVAcc,"+trainResults.getAcc()+",TransformBuildTime,"+transformBuildTime+",timeLimit,"+timeLimit+",TransformParas,"+paras+",ClassifierParas,"+classifierParas;
    }
    
    
    public long getTransformOpCount(){
        return transform.getCount();
    }
    
    
    public Instances transformDataset(Instances data){
        if(transform.isFirstBatchDone())
            return transform.process(data);
        return null;
    }
    
    //pass in an enum of hour, minut, day, and the amount of them. 
    @Override
    public void setTrainTimeLimit(TimeUnit time, long amount) {
        //min,hour,day in longs.
        switch(time){
            case NANOSECONDS:
                timeLimit = amount;
                break;
            case MINUTES:
                timeLimit = (ShapeletTransformTimingUtilities.dayNano/24/60) * amount;
                break;
            case HOURS:
                timeLimit = (ShapeletTransformTimingUtilities.dayNano/24) * amount;
                break;
            case DAYS:
                timeLimit = ShapeletTransformTimingUtilities.dayNano * amount;
                break;
            default:
                throw new InvalidParameterException("Invalid time unit");
        }
    }
    
    public void setNumberOfShapelets(long numS){
        numShapelets = numS;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
    // can classifier handle the data?
        getCapabilities().testWithFail(data);
        
        long startTime=System.nanoTime(); 
        long transformTime=(long)((((double)timeLimit)*2.0)/3.0);
//        System.out.println("Time limit = "+timeLimit+"  transform time "+transformTime);
        shapeletData =  createTransformData(data, transformTime);
        transformBuildTime=System.nanoTime()-startTime;
//        if(setSeed)
//            classifier.setRandSeed((int) seed);
//        if(debug)
        redundantFeatures=InstanceTools.removeRedundantTrainAttributes(shapeletData);
        if(saveShapelets){
            System.out.println("Shapelet Saving  ....");
            FullAccessOutFile of=new FullAccessOutFile(shapeletPath+"Transforms"+seed+".arff");
            of.writeString(shapeletData.toString());
            of.closeFile();
            of=new FullAccessOutFile(shapeletPath+"Shaplelets"+seed+".csv");
            of.writeLine("BuildTime,"+(System.nanoTime()-startTime));
            of.writeLine("NumShapelets,"+transform.getNumberOfShapelets());
            of.writeLine("Count(not sure!),"+transform.getCount());
            of.writeString("ShapeletLengths");
            ArrayList<Integer> lengths=transform.getShapeletLengths();
            for(Integer i:lengths)
                of.writeString(","+i);
            of.writeString("\n");
            of.writeString(transform.toString());
            of.closeFile();
        }
        long classifierTime=timeLimit-transformBuildTime;
        if(classifier instanceof TrainTimeContractable)
            ((TrainTimeContractable)classifier).setTrainTimeLimit(classifierTime);
//Here get the train estimate directly from classifier using cv for now        
        
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
    
    public void preferShortShapelets(){
        preferShortShapelets = true;
    }
/**
 * ADAPT FOR MTSC
 * @param train
 * @param time
 * @return 
 */
    public Instances createTransformData(Instances train, long time){
        int n = train.numInstances();
        int m = train.numAttributes()-1;
//Set the number of shapelets to keep, max is MAXTRANSFORMSIZE (500)
//numShapeletsInTransform
//    n*m < 1000 ? n*m: 1000;   
        if(n*m<numShapeletsInTransform)
            numShapeletsInTransform=n*m;
//construct the options for the transform.
        ShapeletTransformFactoryOptions.Builder optionsBuilder = new ShapeletTransformFactoryOptions.Builder();
        optionsBuilder.setDistanceType(SubSeqDistance.DistanceType.IMP_ONLINE);
        optionsBuilder.setQualityMeasure(ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN);
        if(train.numClasses() > 2){
            optionsBuilder.useBinaryClassValue();
            optionsBuilder.useClassBalancing();
        }
        optionsBuilder.useRoundRobin();
        optionsBuilder.useCandidatePruning();
        
        //create our search options.
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);

        
//how much time do we have vs. how long our algorithm will take.
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        if(opCount.compareTo(opCountTarget) == 1){
            BigDecimal oct = new BigDecimal(opCountTarget);
            BigDecimal oc = new BigDecimal(opCount);
            BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
            
            //if we've not set a shapelet count, calculate one, based on the time set.
            if(numShapelets == 0){
                numShapelets = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n,m,3,m);
                numShapelets *= prop.doubleValue();
            }
             
             //we need to find atleast one shapelet in every series.
            if(setSeed)
                searchBuilder.setSeed(seed);
            searchBuilder.setSearchType(searchType);
            searchBuilder.setNumShapelets(numShapelets);
            // can't have more final shapelets than we actually search through.
            numShapeletsInTransform =  numShapelets > numShapeletsInTransform ? numShapeletsInTransform : (int) numShapelets;
        }
        optionsBuilder.setKShapelets(numShapeletsInTransform);
        optionsBuilder.setSearchOptions(searchBuilder.build());
        transform = new ShapeletTransformFactory(optionsBuilder.build()).getTransform();
        transform.supressOutput();
        
        if(shapeletOutputPath != null)
            transform.setLogOutputFile(shapeletOutputPath);
        
        if(preferShortShapelets)
            transform.setShapeletComparator(new Shapelet.ShortOrder());
        transform.setNumberOfShapelets((int)numShapeletsInTransform);
        return transform.process(train);
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
        Instances stTrain=st.createTransformData(train,st.timeLimit);
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
        preferShortShapelets = st.preferShortShapelets;
        shapeletOutputPath=st.shapeletOutputPath;
        transform=st.transform;
        shapeletData=st.shapeletData;
        int[] redundantFeatures=st.redundantFeatures;
        transformBuildTime=st.transformBuildTime;
        trainResults =st.trainResults;
        numShapeletsInTransform =st.numShapeletsInTransform;
        searchType =st.searchType;
        numShapelets  =st.numShapelets;
        seed =st.seed;
        setSeed=st.setSeed;
        timeLimit =st.timeLimit;

        
    }

    
}
