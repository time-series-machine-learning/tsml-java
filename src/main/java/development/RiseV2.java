/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import fileIO.OutFile;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.ContractClassifier;
import timeseriesweka.classifiers.RISE;
import timeseriesweka.filters.ACF;
import timeseriesweka.filters.FFT;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author cjr13geu
 */
public class RiseV2 implements Classifier, SaveParameterInfo, ContractClassifier{
    
    private Random random = null;
    private Classifier baseClassifier = null;
    private ArrayList<Classifier> baseClassifiers = null;
    private ArrayList<int[]> startEndArray = null;
    private int maxNumClassifiers = 0;
    private int interval = 0;
    private String relationName = null;
    private Filter filter;
    private Timer timer;

    private enum Filter{PS,ACF,FFT,PS_ACF};
    private FFT fft;
    private long seed = 0;
    private Boolean buildFromSavedData;
    private Instances testInstances = null;
    private int testClassificationIndex = 0;
    private int minimumIntervalLength = 2;
    private int maximumIntervalLength = 50;
    private String modelOutPath = null;
    
    public RiseV2(Long seed){
        this.seed = seed;
        random = new Random(seed);
        initialise();
    }
    
    public RiseV2(){
        random = new Random();
        initialise();
    }
    
    private void initialise(){
        maxNumClassifiers = 50;
        setBaseClassifier();
        setTransformType("PS");
        setTimerType("NAIVE");
        fft = new FFT();
        startEndArray = new ArrayList<>();
        baseClassifiers = new ArrayList<>();
        buildFromSavedData = false;
    }
    
    private void setBaseClassifier(){
        baseClassifier = new RandomTree();
    }
    
    public void setBaseClassifier(Classifier classifier){
        baseClassifier = classifier;
    }
    
    public void setMinimumIntervalLength(int length){
        minimumIntervalLength = length;
    }
    
    public void setMaximumIntervalLength(int length){
        maximumIntervalLength = length;
    }
    
    public void setNumClassifiers(int numClassifiers){
        this.maxNumClassifiers = numClassifiers;
    }
    
    public void setModelOutPath(String path){
        modelOutPath = path+"/Adaptive_Timings";
        new File(modelOutPath).mkdirs();
    }
    
    public void buildFromSavedData(Boolean buildFromSavedData){
        this.buildFromSavedData = buildFromSavedData;
    }
    
    public boolean getBuildFromSavedData(){
        return buildFromSavedData;
    }
    
    public void setTransformType(String s){
        
        String str=s.toUpperCase();
        switch(str){
            case "ACF": case "AFC": case "AUTOCORRELATION":
                filter = Filter.ACF;                
                break;
            case "PS": case "POWERSPECTRUM":
                filter = Filter.PS;
                break;
            case "PS_ACF": case "ACF_PS": case "BOTH":
                filter = Filter.PS_ACF;
                break;       
        }
    }
    
    public void setTimerType(String s){
        String str = s.toUpperCase();
        switch(str){
            case "NAIVE":
                timer = new Timer();
                break;
            case "ADAPTIVE":
                timer = new AdaptiveTimer();
                break;
        }
    }
    
    private void setTimeLimit(double hours){
        timer.setTimeLimit(hours);
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        timer.resetModel();
        baseClassifiers.clear();
 
        if(maximumIntervalLength > data.get(0).numAttributes()-1 || maximumIntervalLength <= 0){
            maximumIntervalLength = data.get(0).numAttributes()-1;
        }
        if(minimumIntervalLength >= data.get(0).numAttributes()-1 || minimumIntervalLength <= 0){
            minimumIntervalLength = 2;
        }
        if (buildFromSavedData)
            testClassificationIndex = 0; 

        timer.startForestTimer();
        
        for (int i = 0; i < maxNumClassifiers && timer.queryForestDuration(); i++) {
            
            if (!buildFromSavedData)
                incrementStartEndArray(data);
            
            
            
            if(timer instanceof AdaptiveTimer && i >= 2){
                ((AdaptiveTimer)timer).makePrediciton(startEndArray.get(i)[1] - startEndArray.get(i)[0]);
            }
            
            if(timer instanceof AdaptiveTimer){
                ((AdaptiveTimer)timer).startTreeTimer();
                ((AdaptiveTimer)timer).addIndependantVar(startEndArray.get(i)[1] - startEndArray.get(i)[0]);
            }
            
            Instances intervalInstances = null;
            
            if(!buildFromSavedData)
                intervalInstances = produceIntervalInstances(data, i);
            else
                intervalInstances = ClassifierTools.loadData("RISE/Training Data/Fold " + (int)seed + "/Classifier " + i);
            
            //TRAIN CLASSIFIERS.
            baseClassifiers.add(AbstractClassifier.makeCopy(baseClassifier));
            baseClassifiers.get(baseClassifiers.size()-1).buildClassifier(intervalInstances);
            
            if(timer instanceof AdaptiveTimer){
                ((AdaptiveTimer)timer).addDependantVar();
            }
        }   
        System.out.println("Classifier built");
        if(timer instanceof AdaptiveTimer && modelOutPath != null){
            ((AdaptiveTimer)timer).saveModelToCSV(modelOutPath);
        }
        System.out.println("Model saved");
    }
    
    private void incrementStartEndArray(Instances instances){
        //Produce start and end values for interval, can include whole series.
        startEndArray.add(new int[2]);
        do{
            startEndArray.get(startEndArray.size()-1)[0] = random.nextInt(instances.numAttributes()-1); 
            startEndArray.get(startEndArray.size()-1)[1] = random.nextInt((instances.numAttributes()) - startEndArray.get(startEndArray.size()-1)[0]) + startEndArray.get(startEndArray.size()-1)[0];
            interval = startEndArray.get(startEndArray.size()-1)[1] - startEndArray.get(startEndArray.size()-1)[0]; 
        }while(interval < minimumIntervalLength || interval > maximumIntervalLength || (interval & (interval - 1)) != 0);
        //System.out.println(interval);
    }
    
    private Instances produceTransform(Instances instances){
        
        Instances temp = null;
        
        switch(filter){
            case ACF:
                temp = ACF.formChangeCombo(instances);
                break;
            case PS: 
                try {
                    fft.useFFT();
                    
                    temp = fft.process(instances);
                } catch (Exception ex) {
                    Logger.getLogger(RiseV2.class.getName()).log(Level.SEVERE, null, ex);
                }
                break;
            case PS_ACF: 
                temp = combinedPSACF(instances);
                break;
        }
        return temp;
    }
    
    private Instances combinedPSACF(Instances instances){
        
        Instances combo=ACF.formChangeCombo(instances);
        Instances temp = null;
        try {
            temp = fft.process(instances);
        } catch (Exception ex) {
            Logger.getLogger(RiseV2.class.getName()).log(Level.SEVERE, null, ex);
        }
        combo.setClassIndex(-1);
        combo.deleteAttributeAt(combo.numAttributes()-1); 
        combo = Instances.mergeInstances(combo, temp);
        combo.setClassIndex(combo.numAttributes()-1);
        
        return combo;        
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {

        double[]distribution = distributionForInstance(instance);

        int maxVote=0;
        for(int i = 1; i < distribution.length; i++)
            if(distribution[i] > distribution[maxVote])
                maxVote = i;
        return maxVote;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        
        double[]distribution = new double[instance.numClasses()];
        
        for (int i = 0; i < baseClassifiers.size(); i++) {
            int classVal = 0;
            
            if (!buildFromSavedData) {
                classVal = (int) baseClassifiers.get(i).classifyInstance(produceIntervalInstance(instance, i));
            }else{
                testInstances = ClassifierTools.loadData("RISE/Test Data/Fold " + (int)seed + "/Classifier " + i);
                classVal = (int) baseClassifiers.get(i).classifyInstance(testInstances.get(testClassificationIndex));
            }        
            distribution[classVal]++;    
        }
        
        for(int i = 0 ; i < distribution.length; i++)
            distribution[i] /= baseClassifiers.size();
        
        if (buildFromSavedData)
            testClassificationIndex++;
        
        return distribution;
    }
    
    private Instance produceIntervalInstance(Instance instance, int classifierNum){
        
        ArrayList<Attribute>attributes = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            attributes.add(instance.attribute(i));
        }
        Instances intervalInstances = new Instances(relationName, attributes, 1);
        intervalInstances.add(instance);
        intervalInstances.setClassIndex(instance.numAttributes()-1);
        intervalInstances = produceIntervalInstances(intervalInstances, classifierNum);
        
        return intervalInstances.firstInstance();
    }
    
    private Instances produceIntervalInstances(Instances instances, int classifierNum){

        //POPULATE INTERVAL INSTANCES. 
        //Create and populate attribute information based on interval, class attribute is an addition.
        ArrayList<Attribute>attributes = new ArrayList<>();
        for (int i = startEndArray.get(classifierNum)[0]; i < startEndArray.get(classifierNum)[1]; i++) {
            attributes.add(instances.attribute(i));
        }
        attributes.add(instances.attribute(instances.numAttributes()-1));

        //Create new Instances to hold intervals.
        relationName = instances.relationName();
        Instances intervalInstances = new Instances(relationName, attributes, instances.size());

        for (int i = 0; i < instances.size(); i++) {
            //Produce intervals from input instances, additional attribute needed to accomidate class value.
            double[] temp = Arrays.copyOfRange(instances.get(i).toDoubleArray(), startEndArray.get(classifierNum)[0], startEndArray.get(classifierNum)[1] + 1);
            DenseInstance instance = new DenseInstance(temp.length);
            instance.replaceMissingValues(temp);
            instance.setValue(temp.length-1, instances.get(i).classValue());
            intervalInstances.add(instance);     
        }
        intervalInstances.setClassIndex(intervalInstances.numAttributes()-1);
        
        intervalInstances = produceTransform(intervalInstances);
        
        return intervalInstances;
    }

    public void createIntervalInstancesARFF(Instances training, Instances test){
        incrementStartEndArray(training);
        
        for (int i = 0; i < maxNumClassifiers; i++) {
            if (!(new File("RISE/Training Data/Fold " + (int)seed + "/Classifier " + i + ".arff").isFile())) {
                ClassifierTools.saveDataset(produceIntervalInstances(training, i), "RISE/Training Data/Fold " + (int)seed + "/Classifier " + i);
            }
            if (!(new File("RISE/Test Data/Fold " + (int)seed + "/Classifier " + i + ".arff").isFile())) {
                ClassifierTools.saveDataset(produceIntervalInstances(test, i), "RISE/Test Data/Fold " + (int)seed + "/Classifier " + i);
            }  
        }   
    }
    
    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public String getParameters() {
        String result = "BuildTime," + this.timer.currentTime + ", MaxNumTrees," + this.maxNumClassifiers +
                ", MaxIntervalLeangth," + this.maximumIntervalLength + ", MinIntervalLength," + this.minimumIntervalLength;
        return result;
    }
    
    private class Timer{
        
        protected final String name = "Naive";
        protected long forestTimeLimit = 0;
        protected long forestStartTime = 0;
        protected long currentTime = 0;
        
        public String getName(){
            return name;
        }
        
        public void resetModel(){}
        
        public void setTimeLimit(double timeLimit){
            this.forestTimeLimit = (long) (timeLimit * 3600000000000L);
        }
        
        public void startForestTimer(){
            forestStartTime = System.nanoTime();
        }
        
        public boolean queryForestDuration(){
            if(forestTimeLimit == 0){
                return true;
            }else{
                return (System.nanoTime() - forestStartTime) < forestTimeLimit;
            }  
        }
        
    };
    
    private class AdaptiveTimer extends Timer{

        protected final String name = "ADAPTIVE";
        protected long treeStartTime = 0;
        protected ArrayList<Double> predictionValues = null;
        protected ArrayList<Integer> xValues = null;
        protected ArrayList<Long> yValues = null;
        protected double a = 0.0;
        protected double b = 0.0;
        protected double c = 0.0;
        
        
        AdaptiveTimer(){
            initialise();
        }
        
        private void initialise(){
            xValues = new ArrayList<>();
            yValues = new ArrayList<>();
            predictionValues = new ArrayList<>();
            predictionValues.add(0.0);
            predictionValues.add(0.0);
        }
        
        @Override
        public void resetModel(){
            initialise();
        }
        
        @Override
        public String getName(){
            return name;
        }
        
        private void buildModel(){
            a = 0.0;
            b = 0.0;
            c = 0.0;
            
            double numberOfVals = (double)xValues.size();
            double smFrstScrs = 0.0;
            double smScndScrs = 0.0;
            double smSqrFrstScrs = 0.0;
            double smCbFrstScrs = 0.0;
            double smPwrFrFrstScrs = 0.0;
            double smPrdtFrstScndScrs = 0.0;
            double smSqrFrstScrsScndScrs = 0.0;
            
            for (int i = 0; i < xValues.size(); i++) {
                smFrstScrs += xValues.get(i);
                smScndScrs += yValues.get(i);
                smSqrFrstScrs += Math.pow(xValues.get(i), 2);
                smCbFrstScrs += Math.pow(xValues.get(i), 3);
                smPwrFrFrstScrs += Math.pow(xValues.get(i), 4);
                smPrdtFrstScndScrs += xValues.get(i) * yValues.get(i);
                smSqrFrstScrsScndScrs += Math.pow(xValues.get(i), 2) * yValues.get(i);
            }
            
            double valOne = smSqrFrstScrs - (Math.pow(smFrstScrs, 2) / numberOfVals);
            double valTwo = smPrdtFrstScndScrs - ((smFrstScrs * smScndScrs) / numberOfVals);
            double valThree = smCbFrstScrs - ((smSqrFrstScrs * smFrstScrs) / numberOfVals);
            double valFour = smSqrFrstScrsScndScrs - ((smSqrFrstScrs * smScndScrs) / numberOfVals);
            double valFive = smPwrFrFrstScrs - (Math.pow(smSqrFrstScrs, 2) / numberOfVals);
            
            a = ((valFour * valOne) - (valTwo * valThree)) / ((valOne * valFive) - Math.pow(valThree, 2));
            b = ((valTwo * valFive) - (valFour * valThree)) / ((valOne * valFive) - Math.pow(valThree, 2));
            c = (smScndScrs / numberOfVals) - (b * (smFrstScrs / numberOfVals)) - (a * (smSqrFrstScrs / numberOfVals));
        }
        
        public double makePrediciton(int x){  
            
            buildModel();
            
            predictionValues.add(a * Math.pow(x, 2) + b * x + c);
            
            return predictionValues.get(predictionValues.size()-1);
        }
        
        public double getFeatureSpace(){
            double y = forestTimeLimit - (System.nanoTime() - forestStartTime);
            //System.out.println(y);
            double x = ((-b) + (Math.sqrt((b * b) - (4 * a * (c - y))))) / (2 * a);
            //System.out.println("y: " + y + "    x:" + x);
            //System.out.println("a: " +  a + "     b:" + b + "     c:" + c);
            if (x > maximumIntervalLength || Double.isNaN(x)) {
               x = RiseV2.this.maximumIntervalLength;
            }
            if(x < RiseV2.this.minimumIntervalLength){
                x = RiseV2.this.minimumIntervalLength;
            }
            //System.out.println("y: " + y + "    x:" + x);
            //System.out.println("");
            return x;
        }
        
        public void startTreeTimer(){
            treeStartTime = System.nanoTime();
        }
        
        public void addDependantVar() {
            yValues.add(System.nanoTime() - treeStartTime);
        }
        
        public void addIndependantVar(int x){
            xValues.add(x);
        } 
        
        public void saveModelToCSV(String filePath){
            
            OutFile outFile = new OutFile(filePath + "\\testModel"+ RiseV2.this.seed +".csv");
            for (int i = 0; i < xValues.size(); i++) {
                outFile.writeLine(Double.toString(xValues.get(i)) + "," + Double.toString(yValues.get(i)) + "," + Double.toString(predictionValues.get(i)));
            }
            outFile.closeFile();
        }
    };
    
    @Override
    public void setTimeLimit(long time) {
        setTimeLimit(time * 0.00000000000027778);
    }

    @Override
    public void setTimeLimit(TimeLimit time, int amount) {
        double conversion = 0.0;
        switch(time){
            case MINUTE:
                conversion = 0.0166667;
                break;
            case HOUR:
                conversion = 1;
                break;
            case DAY:
                conversion = 24;
                break;    
        }        
        setTimeLimit(amount * conversion);
    }
    
    public static void main(String[] args) throws Exception {
        
        //Instances all = ClassifierTools.loadData("/gpfs/home/cjr13geu/Datasets/MosquitoDatasets/Unaltered/Time/Time.arff");
        
        //LOCAL TESTING PATH.
        //Instances all = ClassifierTools.loadData("C:\\PhD\\Data\\Cancer\\Cancer.arff");
        Instances all = ClassifierTools.loadData("C:\\PhD\\Data\\Mosquito\\Truncated_5441-10440\\MiddleTrunc_5000Att_Time.arff");
        
        Instances[] instances;
        
        RiseV2 rise = new RiseV2((long)0);
        instances = InstanceTools.resampleInstances(all, 0, .5);
        
        rise.setMaximumIntervalLength(100);
        
        rise.setNumClassifiers(600);
        
        rise.setTimerType("Adaptive");
        
        rise.setTimeLimit(3600000000000L);
        
        rise.setModelOutPath("C:\\PhD\\Experiments\\InsectProblem\\Analysis\\RISE_Timing_Experiments\\Middle5000att_FFT_PowersOf2");
       
        rise.buildClassifier(instances[0]);      
        
        
        //System.out.println(Arrays.toString(rise.distributionForInstance(instances[1].firstInstance())));       
        
        
        
        //rise.buildFromSavedData(false);
        
        //rise.buildClassifier(train);
        
        //ArrayList<double[]> testDistributions = new ArrayList<>();
        //ArrayList<Double> testClassifications = new ArrayList<>();
        
        //for (int i = 0; i < test.size(); i++) {
        //    testDistributions.add(rise.distributionForInstance(test.get(i)));
            //testClassifications.add(rise.classifyInstance(test.get(i)));
        //}
        
        //double accuracy = 0;
        //for (int i = 0; i < test.size(); i++) {
            
            //if(testClassifications.get(i) == test.get(i).classValue())
            //    accuracy++;
            
        //    System.out.println(testDistributions.get(i)[0] + " - " + testDistributions.get(i)[1]);
        //}
       
        //System.out.println("Accuracy: " + accuracy/test.size());
    }
    
}