/*
 * A new Elastic Ensemble for sharing with others
@article{lines15elastic,
  title={Time Series Classification with Ensembles of Elastic Distance Measures},
  author={J. Lines and A. Bagnall},
  journal={Data Mining and Knowledge Discovery},
  volume={29},
  issue={3},
  pages={565--592},
  year={2015}
}

 */
package timeseriesweka.classifiers;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import timeseriesweka.classifiers.cote.HiveCoteModule;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ED1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ERP1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.Efficient1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.LCSS1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.MSM1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.TWE1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.WDTW1NN;
import utilities.ClassifierTools;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.filters.DerivativeFilter;
import utilities.SaveParameterInfo;
import utilities.WritableTestResults;
import utilities.ClassifierResults;
import weka.core.TechnicalInformation;

/**
 *
 * @author sjx07ngu
 */
public class ElasticEnsemble extends AbstractClassifierWithTrainingData implements HiveCoteModule, WritableTestResults,TrainAccuracyEstimate{

    
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "J. Lines and A. Bagnall");
        result.setValue(TechnicalInformation.Field.TITLE, "Time Series Classification with Ensembles of Elastic Distance Measures");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "29");
        result.setValue(TechnicalInformation.Field.NUMBER, "3");
        
        result.setValue(TechnicalInformation.Field.PAGES, "565-592");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        return result;
    }    
        
    
    public enum ConstituentClassifiers{ 
        Euclidean_1NN, 
        DTW_R1_1NN, 
        DTW_Rn_1NN, 
        WDTW_1NN, 
        DDTW_R1_1NN, 
        DDTW_Rn_1NN, 
        WDDTW_1NN, 
        LCSS_1NN, 
        MSM_1NN, 
        TWE_1NN, 
        ERP_1NN
    };
    
    public static boolean isDerivative(ConstituentClassifiers classifier){
        return (classifier==ConstituentClassifiers.DDTW_R1_1NN || classifier==ConstituentClassifiers.DDTW_Rn_1NN || classifier==ConstituentClassifiers.WDDTW_1NN);
    }
    
    public static boolean isFixedParam(ConstituentClassifiers classifier){
        return (classifier==ConstituentClassifiers.DDTW_R1_1NN || classifier==ConstituentClassifiers.DTW_R1_1NN || classifier==ConstituentClassifiers.Euclidean_1NN);
    }
    
    
    private final ConstituentClassifiers[] classifiersToUse;
    private String datasetName;
    private int resampleId;
    private String resultsDir;
    private double[] cvAccs;
    private double[][] cvPreds;
    
    private boolean buildFromFile = false;
    private boolean writeToFile = false;
    private Instances train;
    private Instances derTrain;
    private Efficient1NN[] classifiers = null;
    
    private boolean writeEnsembleTrainingFile = false;
    private String ensembleTrainFilePathAndName = null;
    
    
    private boolean usesDer = false;
    private static DerivativeFilter df = new DerivativeFilter();
    
    // utility to enable AJBs COTE 
    double[] previousPredictions = null;
    
    double ensembleCvAcc =-1;
    double[] ensembleCvPreds = null;
    
    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    
    public String[] getIndividualClassifierNames() {
        String[] names= new String[this.classifiersToUse.length];
        for(int i  = 0; i < classifiersToUse.length; i++){
            names[i] = classifiersToUse[i].toString();
        }
        return names;
    }

    
    public double[] getIndividualCVAccs() {
        return this.cvAccs;
    }

    @Override
    public double getEnsembleCvAcc() {
        if(this.ensembleCvAcc != -1 && this.ensembleCvPreds!=null){
            return this.ensembleCvAcc;
        }
        
        this.getEnsembleCvPreds();
        return this.ensembleCvAcc;
    }

//    @Override
    public double[] getEnsembleCvPreds() {
        if(this.ensembleCvPreds!=null){
            return this.ensembleCvPreds;
        }
        
        this.ensembleCvPreds = new double[train.numInstances()];
        
        double actual, pred;
        double bsfWeight;
        int correct = 0;
        ArrayList<Double> bsfClassVals;
        double[] weightByClass;
        for(int i = 0; i < train.numInstances(); i++){
            actual = train.instance(i).classValue();
            bsfClassVals = null;
            bsfWeight = -1;
            weightByClass = new double[train.numClasses()];

            for(int c = 0; c < classifiers.length; c++){
                weightByClass[(int)this.cvPreds[c][i]]+=this.cvAccs[c];
                
                if(weightByClass[(int)this.cvPreds[c][i]] > bsfWeight){
                    bsfWeight = weightByClass[(int)this.cvPreds[c][i]];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(this.cvPreds[c][i]);
                }else if(weightByClass[(int)this.cvPreds[c][i]] == bsfWeight){
                    bsfClassVals.add(this.cvPreds[c][i]);
                }
            }
            
            if(bsfClassVals.size()>1){
                pred = bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
            }else{
                pred = bsfClassVals.get(0);
            }
            
            if(pred==actual){
                correct++;
            }
            this.ensembleCvPreds[i]=pred;
        }
        
        this.ensembleCvAcc = (double)correct/train.numInstances();
        return this.ensembleCvPreds;
    }
    


//    @Override
    public double[] getIndividualCvAccs() {
        return this.cvAccs;
    }

//    @Override
    public double[][] getIndividualCvPredictions() {
        return this.cvPreds;
    }
    
    /**
     * Default constructor; includes all constituent classifiers
     */
    public ElasticEnsemble(){
        this.classifiersToUse = ConstituentClassifiers.values();
    }
    
    /**
     * Constructor allowing specific constituent classifier types to be passed
     * @param classifiersToUse ConstituentClassifiers[] list of classifiers to use as enums
     */
    public ElasticEnsemble(ConstituentClassifiers[] classifiersToUse){
        this.classifiersToUse = classifiersToUse;
    }
    
    /**
     * Constructor that builds an EE from existing training output. By default includes all constituent classifier types.
     * NOTE: this DOES NOT resample data; data must be resampled independently of the classifier. This just ensures the correct naming convention of output files
     * 
     * @param resultsDir path to the top-level of the stored training output 
     * @param datasetName name of the dataset to be loaded
     * @param resampleId  resampleId of the dataset to be loaded
     */
    public ElasticEnsemble(String resultsDir, String datasetName, int resampleId){
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.classifiersToUse = ConstituentClassifiers.values();
        this.buildFromFile = true;
    }
    
    /**
     * Constructor that builds an EE from existing training output. Includes the classifier types passed in as an array of enums
     * 
     * @param resultsDir path to the top-level of the stored training output 
     * @param datasetName name of the dataset to be loaded
     * @param resampleId  resampleId of the dataset to be loaded
     * @param classifiersToUse the classifiers to load
     */
    public ElasticEnsemble(String resultsDir, String datasetName, int resampleId, ConstituentClassifiers[] classifiersToUse){
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.classifiersToUse = classifiersToUse;
        this.buildFromFile = true;
    }
    
    /** 
     * Turns on file writing to store training output. NOTE: this doesn't resample the data; data needs to be resampled independently of the classifier. This just ensures the correct naming convention for output files.
     * 
     * @param resultsDir path to the top-level of the training output store (makes dir if it doesn't exist)
     * @param datasetName identifier in the written files for this dataset
     * @param resampleId  resample id of the dataset
     */
    public void setInternalFileWritingOn(String resultsDir, String datasetName, int resampleId){
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.writeToFile = true;
    }
    
    @Override
    public void writeCVTrainToFile(String outputPathAndName){
        this.writeEnsembleTrainingFile = true;
        ensembleTrainFilePathAndName = outputPathAndName;
    }
    @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        this.writeEnsembleTrainingFile =setCV;
    }
    
    @Override
    public boolean findsTrainAccuracyEstimate(){ return writeEnsembleTrainingFile;}
    
    @Override
    public ClassifierResults getTrainResults(){
//Temporary : copy stuff into trainResults.acc here
        trainResults.acc=ensembleCvAcc;
//TO DO: Write the other stats        
        return trainResults;
    }        
    
    /**
     * Builds classifier. If building from file, cv weights and predictions will be loaded from file. If running from scratch, training cv will be performed for constituents to find best params, cv accs, and cv preds
     * @param train The training data
     * @throws Exception if building from file and results not found, or if there is an issue with the training data
     */
    @Override
    public void buildClassifier(Instances train) throws Exception{
        trainResults.buildTime=System.currentTimeMillis();
        this.train = train;
        this.derTrain = null;
        usesDer = false;
        
        this.classifiers = new Efficient1NN[this.classifiersToUse.length];
        this.cvAccs = new double[classifiers.length];
        this.cvPreds = new double[classifiers.length][this.train.numInstances()];
        
        for(int c = 0; c < classifiers.length; c++){
            classifiers[c] = getClassifier(this.classifiersToUse[c]);
            if(isDerivative(this.classifiersToUse[c])){
                usesDer = true;
            }
        }
        
        if(usesDer){
            this.derTrain = df.process(train);
        }
        
        if(buildFromFile){
            File existingTrainOut;
            Scanner scan;
            int paramId;
            double cvAcc;
            for(int c = 0; c < classifiers.length; c++){
                existingTrainOut = new File(this.resultsDir+classifiersToUse[c]+"/Predictions/"+datasetName+"/trainFold"+this.resampleId+".csv");
                if(!existingTrainOut.exists()){
                    throw new Exception("Error: training file doesn't exist for "+existingTrainOut.getAbsolutePath());
                }
                scan = new Scanner(existingTrainOut);
                scan.useDelimiter("\n");
                scan.next();//header
                paramId = Integer.parseInt(scan.next().trim().split(",")[0]);
                cvAcc = Double.parseDouble(scan.next().trim().split(",")[0]);
                
                for(int i = 0; i < train.numInstances(); i++){
                    this.cvPreds[c][i] = Double.parseDouble(scan.next().split(",")[1]);
                }
                
                scan.close();
                if(isDerivative(classifiersToUse[c])){
                    if(!isFixedParam(classifiersToUse[c])){
                        classifiers[c].setParamsFromParamId(derTrain, paramId);
                    }
                    classifiers[c].buildClassifier(derTrain);
                }else{
                    if(!isFixedParam(classifiersToUse[c])){
                        classifiers[c].setParamsFromParamId(train, paramId);
                    }
                    classifiers[c].buildClassifier(train);
                }
                cvAccs[c] = cvAcc;
            }
        }else{
            double[] cvAccAndPreds;
            for(int c = 0; c < classifiers.length; c++){
                if(writeToFile){
                    classifiers[c].setFileWritingOn(this.resultsDir, this.datasetName, this.resampleId);
                }
                if(isDerivative(classifiersToUse[c])){
                    cvAccAndPreds = classifiers[c].loocv(derTrain);
                }else{
                    cvAccAndPreds = classifiers[c].loocv(train);
                }
                
                cvAccs[c] = cvAccAndPreds[0];
                for(int i = 0; i < train.numInstances(); i++){
                    this.cvPreds[c][i] = cvAccAndPreds[i+1];
                }
            }
            
            
            if(this.writeEnsembleTrainingFile){
                StringBuilder output = new StringBuilder();
                
                double[] ensembleCvPreds = this.getEnsembleCvPreds();
                
                output.append(train.relationName()).append(",EE,train\n");
                output.append(this.getParameters()).append("\n");
                output.append(this.getEnsembleCvAcc()).append("\n");
                
                for(int i = 0; i < train.numInstances(); i++){
                    output.append(train.instance(i).classValue()).append(",").append(ensembleCvPreds[i]).append("\n");
                }
                
                FileWriter fullTrain = new FileWriter(this.ensembleTrainFilePathAndName);
                fullTrain.append(output);
                fullTrain.close();
            }
        }
        trainResults.buildTime=System.currentTimeMillis()-trainResults.buildTime;
        
    }
    
    /**
     * Returns an Efficient1NN object corresponding to the input enum. Output classifier includes the correct internal information for handling LOOCV/param tuning.
     * @param classifier
     * @return
     * @throws Exception 
     */
    public static Efficient1NN getClassifier(ConstituentClassifiers classifier) throws Exception{
        Efficient1NN knn = null;
        switch(classifier){
            case Euclidean_1NN:
                return new ED1NN();
            case DTW_R1_1NN:
                return new DTW1NN(1);
            case DDTW_R1_1NN:
                knn = new DTW1NN(1);
                knn.setClassifierIdentifier(classifier.toString());
                return knn;
            case DTW_Rn_1NN:
                return new DTW1NN();
            case DDTW_Rn_1NN:
                knn = new DTW1NN();
                knn.setClassifierIdentifier(classifier.toString());;
                return knn;
            case WDTW_1NN:
                return new WDTW1NN();
            case WDDTW_1NN:
                knn = new WDTW1NN();
                knn.setClassifierIdentifier(classifier.toString());
                return knn;
            case LCSS_1NN:
                return new LCSS1NN();
            case ERP_1NN:
                return new ERP1NN();
            case MSM_1NN:
                return new MSM1NN();
            case TWE_1NN:
                return new TWE1NN();
            default: 
                throw new Exception("Unsupported classifier type");
        }
            
    }
    
    /**
     * Classify a test instance. Each constituent classifier makes a prediction, votes are weighted by CV accs, and the majority weighted class value vote is returned
     * @param instance test instance
     * @return predicted class value of instance
     * @throws Exception 
     */
    public double classifyInstance(Instance instance) throws Exception{
        if(classifiers==null){
            throw new Exception("Error: classifier not built");
        }
        Instance derIns = null;
        if(this.usesDer){
            Instances temp = new Instances(derTrain,1);
            temp.add(instance);
            temp = df.process(temp);
            derIns = temp.instance(0);
        }
        
        double bsfVote = -1;
        double[] classTotals = new double[train.numClasses()];
        ArrayList<Double> bsfClassVal = null;
        
        double pred;
        this.previousPredictions = new double[this.classifiers.length];
        
        for(int c = 0; c < classifiers.length; c++){
            if(isDerivative(classifiersToUse[c])){
                pred = classifiers[c].classifyInstance(derIns);
            }else{
                pred = classifiers[c].classifyInstance(instance);
            }
            previousPredictions[c] = pred;
            
            try{
                classTotals[(int)pred] += cvAccs[c];
            }catch(Exception e){
                System.out.println("cv accs "+cvAccs.length);
                System.out.println(pred);
                throw e;
            }
            
            if(classTotals[(int)pred] > bsfVote){
                bsfClassVal = new ArrayList<>();
                bsfClassVal.add(pred);
                bsfVote = classTotals[(int)pred];
            }else if(classTotals[(int)pred] == bsfVote){
                bsfClassVal.add(pred);
            }
        }
        
        if(bsfClassVal.size()>1){
            return bsfClassVal.get(new Random(46).nextInt(bsfClassVal.size()));
        }
        return bsfClassVal.get(0);
    }
    
    public double[] classifyInstanceByConstituents(Instance instance) throws Exception{
        Instance ins = instance;
        
        double[] predsByClassifier = new double[this.classifiers.length];
                
        for(int i=0;i<classifiers.length;i++){
            predsByClassifier[i] = classifiers[i].classifyInstance(ins);
        }
        
        return predsByClassifier;
    }
    
    public double[] getPreviousPredictions() throws Exception{
        if(this.previousPredictions == null){
            throw new Exception("Error: no previous instance found");
        }
        return this.previousPredictions;
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        if(classifiers==null){
            throw new Exception("Error: classifier not built");
        }
        Instance derIns = null;
        if(this.usesDer){
            Instances temp = new Instances(derTrain,1);
            temp.add(instance);
            temp = df.process(temp);
            derIns = temp.instance(0);
        }
        
        double[] classTotals = new double[train.numClasses()];
        double cvSum = 0;
        double pred;
        
        for(int c = 0; c < classifiers.length; c++){
            if(isDerivative(classifiersToUse[c])){
                pred = classifiers[c].classifyInstance(derIns);
            }else{
                pred = classifiers[c].classifyInstance(instance);
            }
            try{
                classTotals[(int)pred] += cvAccs[c];
            }catch(Exception e){
                System.out.println("cv accs "+cvAccs.length);
                System.out.println(pred);
                throw e;
            }
            cvSum+=cvAccs[c];
        }
        
        for(int c = 0; c < classTotals.length; c++){
            classTotals[c]/=cvSum;
        }
        
        return classTotals;
    }
    
    public double[] getCVAccs() throws Exception{
        if(this.cvAccs==null){
            throw new Exception("Error: classifier not built yet");
        }
        return this.cvAccs;
    }
    
    
    private String getClassifierInfo(){
        StringBuilder st = new StringBuilder();
        st.append("EE using:\n");
        st.append("=====================\n");
        for(int c = 0; c < classifiers.length; c++){
            st.append(classifiersToUse[c]).append(" ").append(classifiers[c].getClassifierIdentifier()).append(" ").append(cvAccs[c]).append("\n");
        }
        return st.toString();
    }
    
    public String getParameters(){
        StringBuilder params = new StringBuilder();
        params.append(super.getParameters()).append(",");
        for(int c = 0; c < classifiers.length; c++){
            params.append(classifiers[c].getClassifierIdentifier()).append(",").append(classifiers[c].getParamInformationString()).append(",");
        }
        return params.toString();
    }
    

    @Override
    public String toString(){
        return super.toString()+"\n"+this.getClassifierInfo();
    }
 
    
    public static void exampleUsage(String datasetName, int resampeId, String outputResultsDirName) throws Exception{
        
        System.out.println("to do");
        
    }
    
    public static void main(String[] args) throws Exception{

        ElasticEnsemble ee = new ElasticEnsemble();
        Instances train = ClassifierTools.loadData("C:/users/sjx07ngu/dropbox/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
        Instances test = ClassifierTools.loadData("C:/users/sjx07ngu/dropbox/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TEST");
        ee.buildClassifier(train);
        
        int correct = 0;
        for(int i = 0; i < test.numInstances(); i++){
            if(test.instance(i).classValue()==ee.classifyInstance(test.instance(i))){
                correct++;
            }
        }
        System.out.println("correct: "+correct+"/"+test.numInstances());
        System.out.println((double)correct/test.numInstances());
        System.out.println(ee.getEnsembleCvAcc());
    }
}

