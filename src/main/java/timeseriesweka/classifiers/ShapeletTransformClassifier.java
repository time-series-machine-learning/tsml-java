/*
Shaplet transform with the weighted ensemble
 */
package timeseriesweka.classifiers;

import timeseriesweka.filters.shapelet_transforms.ShapeletTransformFactory;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransform;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransformFactoryOptions;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransformTimingUtilities;
import java.io.File;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.classifiers.AbstractClassifier;
import vector_classifiers.CAWPE;
import weka.core.Instance;
import weka.core.Instances;
import static timeseriesweka.filters.shapelet_transforms.ShapeletTransformTimingUtilities.nanoToOp;
import timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQuality;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch.SearchType;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearchOptions;
import timeseriesweka.classifiers.cote.HiveCoteModule;
import timeseriesweka.classifiers.ensembles.voting.MajorityConfidence;
import timeseriesweka.classifiers.ensembles.weightings.TrainAcc;
import timeseriesweka.filters.shapelet_transforms.DefaultShapeletOptions;
import utilities.ClassifierResults;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

/**
 *
 * @author raj09hxu
 * By default, performs a shapelet transform through full enumeration (max 2000 shapelets selected)
 *  then classifies with the heterogeneous ensemble CAWPE, using randF, rotF and SVMQ.
 * If can be contracted to a maximum run time for shapelets, and can be configured for a different 
 * 
 */
public class ShapeletTransformClassifier  extends AbstractClassifier implements HiveCoteModule, SaveParameterInfo, TrainAccuracyEstimate, ContractClassifier, CheckpointClassifier{

    //Minimum number of instances per class in the train set
    public static final int minimumRepresentation = 25;
    private static int MAXTRANSFORMSIZE=1000; //Default number in transform

    private boolean preferShortShapelets = false;
    private String shapeletOutputPath;
    private CAWPE ensemble;
    private ShapeletTransform transform;
    private Instances format;
    int[] redundantFeatures;
    private boolean doTransform=true;
    private long transformBuildTime;
    protected ClassifierResults res =new ClassifierResults();
    int numShapeletsInTransform = MAXTRANSFORMSIZE;
    private SearchType searchType = SearchType.IMP_RANDOM;
    private long numShapelets = 0;
    private long seed = 0;
    private boolean setSeed=false;
    private long timeLimit = Long.MAX_VALUE;
    private String checkpointFullPath; //location to check point 
    private boolean checkpoint=false;
    enum TransformType{UNI,MULTI_D,MULTI_I};
    TransformType type=TransformType.UNI;
    
    public void setTransformType(TransformType t){
        type=t;
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
        configureDefaultEnsemble();
    }

    
    public void setSeed(long sd){
        setSeed=true;
        seed = sd;
    }
    
    //careful when setting search type as you could set a type that violates the contract.
    public void setSearchType(ShapeletSearch.SearchType type) {
        searchType = type;
    }

    @Override
    public void writeCVTrainToFile(String train) {
        ensemble.writeCVTrainToFile(train);
    }
    @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        ensemble.setFindTrainAccuracyEstimate(setCV);

    }

    /*//if you want CAWPE to perform CV.
    public void setPerformCV(boolean b) {
        ensemble.setPerformCV(b);
    }*/
    
    @Override
    public ClassifierResults getTrainResults() {
        return  ensemble.getTrainResults();
    }
        
    @Override
    public String getParameters(){
        String paras=transform.getParameters();
        String ensemble=this.ensemble.getParameters();
        return "BuildTime,"+res.buildTime+",CVAcc,"+res.acc+",TransformBuildTime,"+transformBuildTime+",timeLimit,"+timeLimit+",TransformParas,"+paras+",EnsembleParas,"+ensemble;
    }
    
    @Override
    public double getEnsembleCvAcc() {
        return ensemble.getEnsembleCvAcc();
    }

    @Override
    public double[] getEnsembleCvPreds() {
        return ensemble.getEnsembleCvPreds();
    }
    
    public void doSTransform(boolean b){
        doTransform=b;
    }
    
    public long getTransformOpCount(){
        return transform.getCount();
    }
    
    
    public Instances transformDataset(Instances data){
        if(transform.isFirstBatchDone())
            return transform.process(data);
        return null;
    }
    
    //set any value in nanoseconds you like.
    @Override
    public void setTimeLimit(long time){
        timeLimit = time;
    }
    
    //pass in an enum of hour, minut, day, and the amount of them. 
    @Override
    public void setTimeLimit(TimeLimit time, int amount){
        //min,hour,day in longs.
        long[] times = {ShapeletTransformTimingUtilities.dayNano/24/60, ShapeletTransformTimingUtilities.dayNano/24, ShapeletTransformTimingUtilities.dayNano};
        
        timeLimit = times[time.ordinal()] * amount;
    }
    
    public void setNumberOfShapelets(long numS){
        numShapelets = numS;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if(checkpoint){
            buildCheckpointClassifier(data);
        }
        else{
            long startTime=System.currentTimeMillis(); 
            format = doTransform ? createTransformData(data, timeLimit) : data;
            transformBuildTime=System.currentTimeMillis()-startTime;
            if(setSeed)
                ensemble.setRandSeed((int) seed);

            redundantFeatures=InstanceTools.removeRedundantTrainAttributes(format);

            ensemble.buildClassifier(format);
            format=new Instances(data,0);
            res.buildTime=System.currentTimeMillis()-startTime;
        }
    }
    private void  buildCheckpointClassifier(Instances data) throws Exception {   
//Load file if one exists

//Set timer options

//Sample shapelets until checkpoint time

//Save to file

//When finished, build classifier
            ensemble.buildClassifier(format);
            format=new Instances(data,0);
//            res.buildTime=System.currentTimeMillis()-startTime;

    }
/**
 * Classifiers used in the HIVE COTE paper
 */    
    public void configureDefaultEnsemble(){
//HIVE_SHAPELET_SVMQ    HIVE_SHAPELET_RandF    HIVE_SHAPELET_RotF    
//HIVE_SHAPELET_NN    HIVE_SHAPELET_NB    HIVE_SHAPELET_C45    HIVE_SHAPELET_SVML   
        ensemble=new CAWPE();
        ensemble.setWeightingScheme(new TrainAcc(4));
        ensemble.setVotingScheme(new MajorityConfidence());
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
        ensemble.setClassifiers(classifiers, classifierNames, null);
    }
//This sets up the ensemble to work within the time constraints of the problem    
    public void configureEnsemble(){
        ensemble.setWeightingScheme(new TrainAcc(4));
        ensemble.setVotingScheme(new MajorityConfidence());
        
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
        
        
       ensemble.setClassifiers(classifiers, classifierNames, null);        
        
    }
    
     @Override
    public double classifyInstance(Instance ins) throws Exception{
        format.add(ins);
        
        Instances temp  = doTransform ? transform.process(format) : format;
//Delete redundant
        for(int del:redundantFeatures)
            temp.deleteAttributeAt(del);
        
        Instance test  = temp.get(0);
        format.remove(0);
        return ensemble.classifyInstance(test);
    }
     @Override
    public double[] distributionForInstance(Instance ins) throws Exception{
        format.add(ins);
        
        Instances temp  = doTransform ? transform.process(format) : format;
//Delete redundant
        for(int del:redundantFeatures)
            temp.deleteAttributeAt(del);
        
        Instance test  = temp.get(0);
        format.remove(0);
        return ensemble.distributionForInstance(test);
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
//    n > 2000 ? 2000 : n;   
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
        
        return transform.process(train);
    }
    
    public static void main(String[] args) throws Exception {
        String dataLocation = "C:\\Temp\\TSC\\";
        //String dataLocation = "..\\..\\resampled transforms\\BalancedClassShapeletTransform\\";
        String saveLocation = "C:\\Temp\\TSC\\";
        String datasetName = "ItalyPowerDemand";
        int fold = 0;
        
        Instances train= ClassifierTools.loadData(dataLocation+datasetName+File.separator+datasetName+"_TRAIN");
        Instances test= ClassifierTools.loadData(dataLocation+datasetName+File.separator+datasetName+"_TEST");
        String trainS= saveLocation+datasetName+File.separator+"TrainCV.csv";
        String testS=saveLocation+datasetName+File.separator+"TestPreds.csv";
        String preds=saveLocation+datasetName;

        ShapeletTransformClassifier st= new ShapeletTransformClassifier();
        //st.saveResults(trainS, testS);
        st.doSTransform(true);
        st.setShapeletOutputFilePath(saveLocation+datasetName+"Shapelets.csv");
        st.setOneMinuteLimit();
        st.buildClassifier(train);

        double accuracy = utilities.ClassifierTools.accuracy(test, st);
        
        System.out.println("accuracy: " + accuracy);    }
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
        ensemble=st.ensemble;
        preferShortShapelets = st.preferShortShapelets;
        shapeletOutputPath=st.shapeletOutputPath;
        transform=st.transform;
        format=st.format;
        int[] redundantFeatures=st.redundantFeatures;
        doTransform=st.doTransform;
        transformBuildTime=st.transformBuildTime;
        res =st.res;
        numShapeletsInTransform =st.numShapeletsInTransform;
        searchType =st.searchType;
        numShapelets  =st.numShapelets;
        seed =st.seed;
        setSeed=st.setSeed;
        timeLimit =st.timeLimit;

        
    }

    
}
