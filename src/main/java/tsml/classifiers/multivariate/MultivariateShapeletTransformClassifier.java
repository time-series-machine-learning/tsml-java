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
package tsml.classifiers.multivariate;

import tsml.classifiers.*;
import tsml.transformers.shapelet_tools.ShapeletTransformFactory;
import tsml.filters.shapelet_filters.ShapeletFilter;
import tsml.transformers.shapelet_tools.ShapeletTransformFactoryOptions;
import tsml.transformers.shapelet_tools.ShapeletTransformTimingUtilities;
import java.io.File;
import java.security.InvalidParameterException;
import java.util.concurrent.TimeUnit;

import utilities.InstanceTools;
import machine_learning.classifiers.ensembles.CAWPE;
import weka.core.Instance;
import weka.core.Instances;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch.SearchType;
import machine_learning.classifiers.ensembles.voting.MajorityConfidence;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.transformers.shapelet_tools.DefaultShapeletOptions;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import tsml.classifiers.TrainTimeContractable;

/**
 *
 * @author raj09hxu
 * By default, performs a shapelet transform through full enumeration (max 2000 shapelets selected)
 *  then classifies with the heterogeneous ensemble CAWPE, using randF, rotF and SVMQ.
 * If can be contracted to a maximum run time for shapelets, and can be configured for a different 
 * 
 */
public class MultivariateShapeletTransformClassifier  extends EnhancedAbstractClassifier implements TrainTimeContractable, Checkpointable{

    //Minimum number of instances per class in the train set
    public static final int minimumRepresentation = 25;
    private static int MAXTRANSFORMSIZE=1000; //Default number in transform

    private boolean preferShortShapelets = false;
    private String shapeletOutputPath;
    private CAWPE ensemble;
    private ShapeletFilter transform;
    private Instances format;
    int[] redundantFeatures;
    private boolean doTransform=true;
    private long transformBuildTime;
    protected ClassifierResults res =new ClassifierResults();
    int numShapeletsInTransform = MAXTRANSFORMSIZE;
    private SearchType searchType = SearchType.IMPROVED_RANDOM;
    private long numShapelets = 0;

    private long timeLimit = Long.MAX_VALUE;
    private boolean trainTimeContract = false;


    private String checkpointFullPath; //location to check point 
    private boolean checkpoint=false;
    enum TransformType{INDEP,MULTI_D,MULTI_I};
    TransformType type=TransformType.MULTI_D;
    
    public void setTransformType(TransformType t){
        type=t;
    }
    public void setTransformType(String t){
        t=t.toLowerCase();
        switch(t){
            case "shapeletd": case "shapelet_d": case "dependent":
                type=TransformType.MULTI_D;
                break;
            case "shapeleti": case "shapelet_i":
                type=TransformType.MULTI_I;
                break;
            case "indep": case "shapelet_indep": case "shapeletindep":
                type=TransformType.INDEP;
                break;
                
        }
    }
    
    public MultivariateShapeletTransformClassifier(){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        configureDefaultEnsemble();
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
    public ClassifierResults getTrainResults() {
        return  ensemble.getTrainResults();
    }
        
    @Override
    public String getParameters(){
        String paras=transform.getParameters();
        String ens=this.ensemble.getParameters();
        return super.getParameters()+",CVAcc,"+res.getAcc()+",TransformBuildTime,"+transformBuildTime+",timeLimit,"+timeLimit+",TransformParas,"+paras+",EnsembleParas,"+ens;
    }
    

    public double getTrainAcc() {
        return ensemble.getTrainResults().getAcc();
    }


    public double[] getTrainPreds() {
        return ensemble.getTrainResults().getPredClassValsAsArray();
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

    @Override
    public void setTrainTimeLimit(long amount) {
        timeLimit=amount;
        trainTimeContract = false;

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
            long startTime=System.nanoTime();
            format = doTransform ? createTransformData(data, timeLimit) : data;
            transformBuildTime=System.nanoTime()-startTime;
            if(seedClassifier)
                ensemble.setSeed((int) seed);

            redundantFeatures=InstanceTools.removeRedundantTrainAttributes(format);

            ensemble.buildClassifier(format);
            format=new Instances(data,0);

            res.setTimeUnit(TimeUnit.NANOSECONDS);
            res.setBuildTime(System.currentTimeMillis()-startTime);
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
        if (seedClassifier)
            smo.setRandomSeed((int)seed);
        classifiers[0] = smo;
        classifierNames[0] = "SVMQ";

        RandomForest r=new RandomForest();
        r.setNumTrees(500);
        if(seedClassifier)
           r.setSeed((int)seed);            
        classifiers[1] = r;
        classifierNames[1] = "RandF";
            
            
        RotationForest rf=new RotationForest();
        rf.setNumIterations(100);
        if(seedClassifier)
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
        if (seedClassifier)
            smo.setRandomSeed((int)seed);
        classifiers[0] = smo;
        classifierNames[0] = "SVMQ";

        RandomForest r=new RandomForest();
        r.setNumTrees(500);
        if(seedClassifier)
           r.setSeed((int)seed);            
        classifiers[1] = r;
        classifierNames[1] = "RandF";
            
            
        RotationForest rf=new RotationForest();
        rf.setNumIterations(100);
        if(seedClassifier)
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
//All hard coded for now to 1 day and whatever Aaron's defaults are!
        ShapeletTransformFactoryOptions options;
        switch(type){
            case INDEP:
                options = DefaultShapeletOptions.TIMED_FACTORY_OPTIONS.get("INDEPENDENT").apply(train, ShapeletTransformTimingUtilities.dayNano,(long)seed);
                break;                
            case MULTI_D:
                options = DefaultShapeletOptions.TIMED_FACTORY_OPTIONS.get("SHAPELET_D").apply(train, ShapeletTransformTimingUtilities.dayNano,(long)seed);
                break;
            case MULTI_I: default:
                options = DefaultShapeletOptions.TIMED_FACTORY_OPTIONS.get("SHAPELET_I").apply(train, ShapeletTransformTimingUtilities.dayNano,(long)seed);
                break;
        }
        
        transform = new ShapeletTransformFactory(options).getFilter();
        if(shapeletOutputPath != null)
            transform.setLogOutputFile(shapeletOutputPath);
        
        return transform.process(train);
    }
    
    public static void main(String[] args) throws Exception {
        String dataLocation = "C:\\Temp\\MTSC\\";
        //String dataLocation = "..\\..\\resampled transforms\\BalancedClassShapeletTransform\\";
        String saveLocation = "C:\\Temp\\MTSC\\";
        String datasetName = "ERing";
        int fold = 0;

        Instances train= DatasetLoading.loadDataNullable(dataLocation+datasetName+File.separator+datasetName+"_TRAIN");
        Instances test= DatasetLoading.loadDataNullable(dataLocation+datasetName+File.separator+datasetName+"_TEST");
        String trainS= saveLocation+datasetName+File.separator+"TrainCV.csv";
        String testS=saveLocation+datasetName+File.separator+"TestPreds.csv";
        String preds=saveLocation+datasetName;

        MultivariateShapeletTransformClassifier st= new MultivariateShapeletTransformClassifier();
        //st.saveResults(trainS, testS);
        st.doSTransform(true);
        st.setOneMinuteLimit();
        st.buildClassifier(train);

        double accuracy = utilities.ClassifierTools.accuracy(test, st);

        System.out.println("accuracy: " + accuracy);
    }
/**
 * Checkpoint methods
 */
    public boolean setCheckpointPath(String path) {
        boolean validPath=Checkpointable.super.createDirectories(path);
        if(validPath){
            this.checkpointFullPath=path;
        }
        return validPath;
    }
    public void copyFromSerObject(Object obj) throws Exception{
        if(!(obj instanceof MultivariateShapeletTransformClassifier))
            throw new Exception("Not a ShapeletTransformClassifier object");
//Copy meta data
        MultivariateShapeletTransformClassifier st=(MultivariateShapeletTransformClassifier)obj;
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
        seedClassifier=st.seedClassifier;
        timeLimit =st.timeLimit;

        
    }

    
}
