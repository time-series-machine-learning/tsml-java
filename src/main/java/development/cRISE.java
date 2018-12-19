/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import fileIO.OutFile;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.ContractClassifier;
import timeseriesweka.filters.ACF;
import timeseriesweka.filters.FFT;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.multivariate_tools.MultivariateInstanceTools;
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
public class cRISE implements Classifier, SaveParameterInfo, ContractClassifier, Serializable{
  
    private int maxIntervalLength = 0;
    private int minIntervalLength = 2;
    private int numTrees = 500;
    private int treeCount = 0;
    private int minNumTrees = 200;
    private boolean downsample = false;
    private boolean loadedFromFile = false;
    private int stabilise = 0;
    private long seed = 0;
    
    private Timer timer = null;
    private Random random = null;
    private Classifier classifier = new RandomTree();
    private ArrayList<Classifier> baseClassifiers = null;
    private ArrayList<int[]> intervalsInfo = null;
    private ArrayList<ArrayList<Integer>> intervalsAttIndexes = null;
    private ArrayList<Integer> rawIntervalIndexes = null;
    private FFT fft;
    private String transformType = null;
    private String serialisePath = null;
    private Instances data = null;
    
    public cRISE(long seed){
        this.seed = seed;
        random = new Random(seed);
        timer = new Timer();
    }
    
    private void initialise(){
        timer.reset();
        baseClassifiers = new ArrayList<>();
        intervalsInfo = new ArrayList<>();
        intervalsAttIndexes = new ArrayList<>();
        rawIntervalIndexes = new ArrayList<>();
        fft = new FFT();
    }
    
    public void setNumTrees(int numTrees){
        this.numTrees = numTrees;
    }
    
    public void setMinNumTrees(int minNumTrees){
        this.minNumTrees = minNumTrees;
    }
    
    public void setDownsample(boolean bool){
        this.downsample = bool;
    }
    
    public void setStabilise(int width){
        this.stabilise = width;
    }
    
    public void setModelOutPath(String modelOutPath){   
        timer.modelOutPath = modelOutPath;
    }
    
    public void setTransformType(String transformType){
        if (!transformType.isEmpty()) {
            this.transformType = transformType.toUpperCase();
        } 
    }
    
    public void setBaseClassifier(Classifier classifier){
        this.classifier = classifier;
    }
    
    public void setSerializePath(String serializePath){
        this.serialisePath = serializePath;
        System.out.println("Attempting to load from file location: " 
                + serialisePath 
                + "\\SERIALISE_cRISE_" 
                + seed 
                + ".txt");
        cRISE temp = readSerialise(seed);
        asignVaribles(temp);
    }
    
    private int[] selectIntervalAttributes(int maxIntervalLength, int instanceLength){
        
        //rawIntervalLength[0], startIndex[1], downSampleFactor[2];
        int[] intervalInfo = new int[3];
        
        //Produce powers of 2 ArrayList for interval selection.
        ArrayList<Integer> powersOf2 = new ArrayList<>();
        for (int j = maxIntervalLength; j >= 1; j--) { 
            // If i is a power of 2 
            if ((j & (j - 1)) == 0){ 
                powersOf2.add(j); 
            } 
        }       
        
        Collections.reverse(powersOf2);
        int index = 0;
        if(stabilise > 0 && !rawIntervalIndexes.isEmpty()){
            
            if(stabilise > powersOf2.size()-1){
                stabilise = powersOf2.size()-1;
                while(stabilise % 2 == 0){
                    stabilise --;
                }
            }else if(stabilise < 2){
                stabilise = 2;
                while(stabilise % 2 == 0){
                    stabilise ++;
                }
            }else{
                while(stabilise % 2 == 0){
                    stabilise ++;
                } 
            }
            
            int option = random.nextInt(stabilise - 1);
            if(rawIntervalIndexes.get(rawIntervalIndexes.size()-1) - ((stabilise - 1)/2) <= 2){
                //index = rawIntervalIndexes.get(rawIntervalIndexes.size()-1) + option;
                index = option + 2;
            }
            if (rawIntervalIndexes.get(rawIntervalIndexes.size()-1) - ((stabilise - 1)/2) > 2 && rawIntervalIndexes.get(rawIntervalIndexes.size()-1) + ((stabilise - 1)/2) < powersOf2.size() - 1) {
                option = option - ((stabilise - 1)/2);
                index = rawIntervalIndexes.get(rawIntervalIndexes.size()-1) + option; 
            }
            if(rawIntervalIndexes.get(rawIntervalIndexes.size()-1) + ((stabilise - 1)/2) >= powersOf2.size() - 1){
                //index = rawIntervalIndexes.get(rawIntervalIndexes.size()-1) - option;
                index = (powersOf2.size() - 1) - option;
            }
        }else{
            index = random.nextInt(powersOf2.size() - 1) + 1;
        }
        
        try{
            //intervalInfo[0] = powersOf2.get(index);
            if (treeCount == 0) {
                intervalInfo[0] = powersOf2.get(2);
            }
            if (treeCount == 1) {
                intervalInfo[0] = powersOf2.get(3);
            }
            if (treeCount == 2) {
                intervalInfo[0] = powersOf2.get(4);
            }
            if (treeCount == 3) {
                intervalInfo[0] = powersOf2.get(5);
            }
            if (treeCount > 3) {
                intervalInfo[0] = powersOf2.get(index);
            }
            
        }catch(Exception e){
            System.out.println(e);
        }
        
        if ((instanceLength - intervalInfo[0]) != 0 ) {
            intervalInfo[1] = random.nextInt(instanceLength - intervalInfo[0]);
        }else{
            intervalInfo[1] = 0;
        }
        
        if (downsample) {
            try{
                 intervalInfo[2] = powersOf2.get(random.nextInt(index) + 1);
            }catch(Exception e){
                 intervalInfo[2] = powersOf2.get(random.nextInt(index) + 1);
            }
            //intervalInfo[2] = powersOf2.get(random.nextInt(index) + 1);
        }else{
            intervalInfo[2] = intervalInfo[0];
        }

        this.intervalsInfo.add(intervalInfo);
        this.rawIntervalIndexes.add(index);
        return intervalInfo;
    }
    
    private Instances produceIntervalInstances(int maxIntervalLength, Instances trainingData){
        
        Instances intervalInstances;
        ArrayList<Attribute>attributes = new ArrayList<>();
        int[] intervalInfo = selectIntervalAttributes(maxIntervalLength, trainingData.numAttributes() - 1);
        ArrayList<Integer> intervalAttIndexes = new ArrayList<>();
        
        for (int i = intervalInfo[1]; i < (intervalInfo[1] + intervalInfo[0]); i += (intervalInfo[0] / intervalInfo[2])) {
            attributes.add(trainingData.attribute(i));
            intervalAttIndexes.add(i);
        }
        
        intervalsAttIndexes.add(intervalAttIndexes);
        attributes.add(trainingData.attribute(trainingData.numAttributes()-1));
        intervalInstances = new Instances(trainingData.relationName(), attributes, trainingData.size());
        double[] intervalInstanceValues = new double[intervalInfo[2] + 1];
        
        for (int i = 0; i < trainingData.size(); i++) {
            
            for (int j = 0; j < intervalInfo[2]; j++) {
                intervalInstanceValues[j] = trainingData.get(i).value(intervalAttIndexes.get(j));
            }
            
            DenseInstance intervalInstance = new DenseInstance(intervalInstanceValues.length);
            intervalInstance.replaceMissingValues(intervalInstanceValues);
            intervalInstance.setValue(intervalInstanceValues.length-1, trainingData.get(i).classValue());
            intervalInstances.add(intervalInstance);
        }
        
        intervalInstances.setClassIndex(intervalInstances.numAttributes() - 1);
        
        return intervalInstances;
    }
    
    private Instance produceIntervalInstance(Instance testInstance, int classifierNum){
        double[] instanceValues = new double[intervalsAttIndexes.get(classifierNum).size() + 1];
        ArrayList<Attribute>attributes = new ArrayList<>();
        
        for (int i = 0; i < intervalsAttIndexes.get(classifierNum).size(); i++) {
            attributes.add(testInstance.attribute(i));
            instanceValues[i] = testInstance.value(intervalsAttIndexes.get(classifierNum).get(i));
        }
        
        Instances testInstances = new Instances("relationName", attributes, 1);
        instanceValues[instanceValues.length - 1] = testInstance.value(testInstance.numAttributes() - 1);
        Instance temp = new DenseInstance(instanceValues.length);
        temp.replaceMissingValues(instanceValues);
        testInstances.add(temp);
        testInstances.setClassIndex(testInstances.numAttributes() - 1);
        
        return testInstances.firstInstance();
    }
    
    private Instances transformInstances(Instances instances){ 
        Instances temp = null;
        
        switch(transformType){
            case "ACF":
                temp = ACF.formChangeCombo(instances);
                break;
            case "PS": 
                try {
                    fft.useFFT();
                    temp = fft.process(instances);
                } catch (Exception ex) {
                    System.out.println("FFT failed (could be build or classify) \n" + ex);
                }
                break;
            default: 
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
            Logger.getLogger("Combined PS-ACF failed (could be build or classify) \n" + ex);
        }
        combo.setClassIndex(-1);
        combo.deleteAttributeAt(combo.numAttributes()-1); 
        combo = Instances.mergeInstances(combo, temp);
        combo.setClassIndex(combo.numAttributes()-1);
        
        return combo;        
    }
    
    private void serialise(long seed){
        try{
            System.out.println("Serialising classifier.");
            FileOutputStream f = new FileOutputStream(new File(serialisePath
                    + (serialisePath.isEmpty()? "SERIALISE_cRISE_" : "\\SERIALISE_cRISE_")
                    + seed 
                    + ".txt"));
            ObjectOutputStream o = new ObjectOutputStream(f);
            this.timer.forestElapsedTime = System.nanoTime() - this.timer.forestStartTime;
            o.writeObject(this);
            o.close();
            f.close();
            System.out.println("Serialisation completed: " + treeCount + " trees");
        } catch (IOException ex) {
            System.out.println("Serialisation failed: " + ex);
        }
    }
    
    private cRISE readSerialise(long seed){
        ObjectInputStream oi = null;
        cRISE temp = null;
        try {
            FileInputStream fi = new FileInputStream(new File(
                    serialisePath
                    + (serialisePath.isEmpty()? "SERIALISE_cRISE_" : "\\SERIALISE_cRISE_")
                    + seed 
                    + ".txt"));
            oi = new ObjectInputStream(fi);
            temp = (cRISE)oi.readObject();
            oi.close();
            fi.close();
            System.out.println("File load successful: " + ((cRISE)temp).treeCount + " trees.");
        } catch (IOException | ClassNotFoundException ex) {
            System.out.println("File load: failed.");
        }   
        return temp;
    }
    
    private void asignVaribles(cRISE temp){
        try{
            this.baseClassifiers = temp.baseClassifiers;
            this.classifier = temp.classifier;
            this.data = temp.data;
            this.downsample = temp.downsample;
            this.fft = temp.fft;
            this.intervalsAttIndexes = temp.intervalsAttIndexes;
            this.intervalsInfo = temp.intervalsInfo;
            this.maxIntervalLength = temp.maxIntervalLength;
            this.minIntervalLength = temp.minIntervalLength;
            this.numTrees = temp.numTrees;
            this.random = temp.random;
            this.rawIntervalIndexes = temp.rawIntervalIndexes;
            this.serialisePath = temp.serialisePath;
            this.stabilise = temp.stabilise;
            this.timer = temp.timer;
            this.transformType = temp.transformType;
            this.treeCount = temp.treeCount;
            this.loadedFromFile = true;
            System.out.println("Varible assignment: successful.");
        }catch(Exception ex){
            System.out.println("Varible assignment: unsuccessful.");
        }
        
    }
    
    private long getTime(){
        long time = 0;
        if(loadedFromFile){
            time = timer.forestElapsedTime;
        }else{
            time = 0;
        }
        return time;
    }
    
    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        
        //If serialsed cRISE found.
        if (!loadedFromFile) {
            //Just used for getParameters.
            data = trainingData;
            //(re)Initailse all variables to account for mutiple calls of buildClassifier.
            initialise();

            //Check min & max interval lengths are valid.
            if(maxIntervalLength > trainingData.numAttributes()-1 || maxIntervalLength <= 0){
                maxIntervalLength = trainingData.numAttributes()-1;
            }
            if(minIntervalLength >= trainingData.numAttributes()-1 || minIntervalLength <= (int)Math.sqrt(trainingData.numAttributes()-1)){
                minIntervalLength = (int)Math.sqrt(trainingData.numAttributes()-1);
            }

        }
        
        //Start forest timer.
        timer.forestStartTime = System.nanoTime();
        
        for (; treeCount < numTrees && (System.nanoTime() - timer.forestStartTime) < (timer.forestTimeLimit - getTime()); treeCount++) {
            
            //Start tree timer.
            timer.treeStartTime = System.nanoTime();
            
            //Compute maximum interval length given time remaining.
            timer.buildModel();
            maxIntervalLength = (int)timer.getFeatureSpace((timer.forestTimeLimit) - (System.nanoTime() - (timer.forestStartTime - getTime())));
            
            
            //Produce intervalInstances from trainingData using interval attributes.
            Instances intervalInstances;
            intervalInstances = produceIntervalInstances(maxIntervalLength, trainingData);
            
            //Transform instances.
            if (transformType != null) {
                intervalInstances = transformInstances(intervalInstances);
            }
            
            //Add independant varible to model (length of interval).
            timer.makePrediciton(intervalInstances.numAttributes() - 1);
            timer.independantVaribles.add(intervalInstances.numAttributes() - 1);
            
            //Build classifier with intervalInstances.
            baseClassifiers.add(AbstractClassifier.makeCopy(classifier));
            baseClassifiers.get(baseClassifiers.size()-1).buildClassifier(intervalInstances);
            
            //Add dependant varible to model (time taken).
            timer.dependantVaribles.add(System.nanoTime() - timer.treeStartTime);
            
            if(treeCount % 100 == 0 && treeCount != 0 && serialisePath != null){
                serialise(seed);
                System.out.print("");
            } 
        }
        if (serialisePath != null) {
            serialise(seed);
        }
        
        if (timer.modelOutPath != null) {
            timer.saveModelToCSV(trainingData.relationName());    
        }  
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
    public double[] distributionForInstance(Instance testInstance) throws Exception {
        double[]distribution = new double[testInstance.numClasses()];
        ArrayList<Attribute> attributes;
        
        for (int i = 0; i < baseClassifiers.size(); i++) {
            attributes = new ArrayList<>();
            for (int j = 0; j < intervalsAttIndexes.get(i).size(); j++) {
                attributes.add(testInstance.attribute(intervalsAttIndexes.get(i).get(j)));
            }
            attributes.add(testInstance.attribute(testInstance.numAttributes() - 1));
            Instances instances = new Instances("relationName", attributes, 1);
            Instance intervalInstance = produceIntervalInstance(testInstance, i);
            instances.add(intervalInstance);
            instances.setClassIndex(instances.numAttributes() - 1);
            if (transformType != null) {
                intervalInstance = transformInstances(instances).firstInstance();
            }
            for (int j = 0; j < testInstance.numClasses(); j++) {
                distribution[j] += baseClassifiers.get(i).distributionForInstance(intervalInstance)[j];
            }
        }
        for (int j = 0; j < testInstance.numClasses(); j++) {
                distribution[j] /= baseClassifiers.size();
            }
        return distribution;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String getParameters() {
        
        long buildClassifierTime = 0;
        for (int i = 0; i < timer.independantVaribles.size(); i++) {
            buildClassifierTime += timer.independantVaribles.get(i);
        }
        
        String result = "Total Time Taken," + (System.nanoTime() - timer.forestStartTime)
                + ", Contract Length (ns), " + timer.forestTimeLimit
                + ", Build Classifier (ns)," + buildClassifierTime
                + ", NumAtts," + data.numAttributes()
                + ", MaxNumTrees," + numTrees  
                + ", MinIntervalLength," + minIntervalLength
                + ", Final Coefficients (time = a * x^2 + b * x + c)"
                + ", a, " + timer.a
                + ", b, " + timer.b
                + ", c, " + timer.c;
        return result;
    }

    @Override
    public void setTimeLimit(long time) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setTimeLimit(TimeLimit time, int amount) {
        double conversion;
        switch(time){
            case MINUTE:
                conversion = 0.0166667;
                break;
            case DAY:
                conversion = 24;
                break;
            default:
                case HOUR:
                conversion = 1;
                break;
        }        
        timer.setTimeLimit(amount * conversion);
    }

    private class Timer implements Serializable{
        
        protected long forestTimeLimit = Long.MAX_VALUE;
        protected long forestStartTime = 0;
        protected long treeStartTime = 0;
        protected long forestElapsedTime = 0;
        
        protected ArrayList<Integer> independantVaribles = null;
        protected ArrayList<Long> dependantVaribles = null;
        protected ArrayList<Double> predictions = null;
        private ArrayList<Double> aValues = null;
        private ArrayList<Double> bValues = null;
        private ArrayList<Double> cValues = null;
        
        protected double a = 0.0;
        protected double b = 0.0;
        protected double c = 0.0;
        
        protected String modelOutPath = null;
        
        protected void reset(){
            independantVaribles = new ArrayList<>();
            dependantVaribles = new ArrayList<>();
            predictions = new ArrayList<>();
            aValues = new ArrayList<>();
            bValues = new ArrayList<>();
            cValues = new ArrayList<>();
        }
        
        protected void buildModel(){
            
            a = 0.0;
            b = 0.0;
            c = 0.0;
            double numberOfVals = (double)independantVaribles.size();
            double smFrstScrs = 0.0;
            double smScndScrs = 0.0;
            double smSqrFrstScrs = 0.0;
            double smCbFrstScrs = 0.0;
            double smPwrFrFrstScrs = 0.0;
            double smPrdtFrstScndScrs = 0.0;
            double smSqrFrstScrsScndScrs = 0.0;
            
            for (int i = 0; i < independantVaribles.size(); i++) {
                smFrstScrs += independantVaribles.get(i);
                smScndScrs += dependantVaribles.get(i);
                smSqrFrstScrs += Math.pow(independantVaribles.get(i), 2);
                smCbFrstScrs += Math.pow(independantVaribles.get(i), 3);
                smPwrFrFrstScrs += Math.pow(independantVaribles.get(i), 4);
                smPrdtFrstScndScrs += independantVaribles.get(i) * dependantVaribles.get(i);
                smSqrFrstScrsScndScrs += Math.pow(independantVaribles.get(i), 2) * dependantVaribles.get(i);
            }
            
            double valOne = smSqrFrstScrs - (Math.pow(smFrstScrs, 2) / numberOfVals);
            double valTwo = smPrdtFrstScndScrs - ((smFrstScrs * smScndScrs) / numberOfVals);
            double valThree = smCbFrstScrs - ((smSqrFrstScrs * smFrstScrs) / numberOfVals);
            double valFour = smSqrFrstScrsScndScrs - ((smSqrFrstScrs * smScndScrs) / numberOfVals);
            double valFive = smPwrFrFrstScrs - (Math.pow(smSqrFrstScrs, 2) / numberOfVals);
            
            a = ((valFour * valOne) - (valTwo * valThree)) / ((valOne * valFive) - Math.pow(valThree, 2));
            b = ((valTwo * valFive) - (valFour * valThree)) / ((valOne * valFive) - Math.pow(valThree, 2));
            c = (smScndScrs / numberOfVals) - (b * (smFrstScrs / numberOfVals)) - (a * (smSqrFrstScrs / numberOfVals));
            
            aValues.add(a);
            bValues.add(b);
            cValues.add(c);
        }
        
        protected void makePrediciton(int x){  
            predictions.add(a * Math.pow(x, 2) + b * x + c);    
        }
        
        protected double getFeatureSpace(long timeRemaining){
            double y = timeRemaining;
            double x = ((-b) + (Math.sqrt((b * b) - (4 * a * (c - y))))) / (2 * a);
            
            if (treeCount < minNumTrees) {
                x = x / (minNumTrees - treeCount);
            }
            if(treeCount == minNumTrees){
                maxIntervalLength = data.numAttributes()-1;
            }
            
            if (x > maxIntervalLength || Double.isNaN(x)) {
               x = maxIntervalLength;
            }
            if(x < minIntervalLength){
                x = minIntervalLength;
            }
            System.out.println("y: " + y + "    x:" + x);
            return x;
        }
        
        protected void setTimeLimit(double timeLimit){
            this.forestTimeLimit = (long) (timeLimit * 3600000000000L);
        }
        
        protected void printModel(){
            
            for (int i = 0; i < independantVaribles.size(); i++) {
                System.out.println(Double.toString(independantVaribles.get(i)) + "," + Double.toString(dependantVaribles.get(i)) + "," + Double.toString(predictions.get(i)));
            }
        }
        
        protected void saveModelToCSV(String problemName){
            try{
                OutFile outFile = new OutFile((modelOutPath.isEmpty() ? "timingModel" + (int) seed + ".csv" : modelOutPath + "/" + problemName + "/" + "/timingModel" + (int) seed + ".csv"));
                for (int i = 0; i < independantVaribles.size(); i++) {
                    outFile.writeLine(Double.toString(independantVaribles.get(i)) + "," 
                            + Double.toString(dependantVaribles.get(i)) + "," 
                            + Double.toString(predictions.get(i)) + "," 
                            + Double.toString(timer.aValues.get(i)) + ","
                            + Double.toString(timer.bValues.get(i)) + ","
                            + Double.toString(timer.cValues.get(i)));
                }
                outFile.closeFile();
            }catch(Exception e){
                System.out.println("Mismatch between relation name and name of results folder: " + e);
            }
            
        }
    }    
    
    public static void main(String[] args){

        Instances train;
        Instances test;
        Instances instances;
         
        train = ClassifierTools.loadData("Z:\\Data\\TSCProblems2018\\StarLightCurves\\StarLightCurves_TRAIN.arff");
        test = ClassifierTools.loadData("Z:\\Data\\TSCProblems2018\\StarLightCurves\\StarLightCurves_TEST.arff");
        //instances = ClassifierTools.loadData("C:\\PhD\\Data\\CatsDogs\\ARFF\\CatsDogs\\CatsDogs.arff");
        //Instances[] data = InstanceTools.resampleInstances(instances, 0, 0.5);
        //train = data[0];
        //test = data[1];
        
        double[] dist = null;
        double acc = 0.0;
        double classification = 0.0;
        
        cRISE c = new cRISE(0);
        c.setDownsample(false);
        c.setTransformType("PS");
        //c.setStabilise((int)Math.floor(Math.sqrt(instances.numAttributes()-1)));
        c.setModelOutPath("C:\\PhD\\Experiments\\cRISE\\Predictions\\StarLightCurves\\");
        c.setSerializePath("C:\\PhD\\Experiments\\cRISE\\Predictions\\StarLightCurves\\");
        c.setTimeLimit(TimeLimit.MINUTE,60);
        
        //Train
        try {
            c.buildClassifier(train);
        } catch (Exception ex) {
            System.out.println("Build failed: " + ex);
        }
        
        //Test
        for (int i = 0; i < test.size(); i++) {
            
            try {
                dist = c.distributionForInstance(test.get(i));
                classification = c.classifyInstance(test.get(i));
            } catch (Exception ex) {
                System.out.println("distributionForInstance | classifyInstance failed: " + ex);
            }

            //print dist
            for (int j = 0; j < dist.length; j++) {
                //System.out.print(dist[j] + ", ");
            }
            //System.out.println();

            if(test.get(i).classValue() == classification)
                acc++;
        }
        
        acc /= test.size();
        System.out.println(acc);
        
    }
}
