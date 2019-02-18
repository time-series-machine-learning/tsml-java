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
package evaluation;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import utilities.DebugPrinting;
import utilities.GenericTools;
import utilities.InstanceTools;
import utilities.generic_storage.Pair;

/**
 * This class has morphed over time. At it's base form, it's a simple container class for the 
 * predictions and results of a classifier on a single set of instances (for example, the test 
 * set of a particular resample of a particular dataset) 
 * 
 * It can be used in batch mode (add all predictions at once) or online (add in one at a time).
 * The former is sensible for storing training set results, the latter for test results.
 * 
 * Supports reading/writing of results from/to file, in the 'classifierResults file-format'
 * 
 * Also supports the calculation of various evaluative performance metrics  based on the results (accuracy, 
 * auroc, nll etc.) which are used in the MultipleClassifierEvaluation pipeline
 * 
 * @author James Large (james.large@uea.ac.uk) + edits from just about everybody
 */
public class ClassifierResults implements DebugPrinting, Serializable{
        
    //LINE 1: meta info, set by user
    private String classifierName = "";
    private String datasetName = "";
    private int foldID = -1;
    private String split = ""; //e.g train or test
    private String description= ""; //optional extra info if wanted. 
    
    //LINE 2: classifier setup/info, parameters. precise format is up to user. 
    //e.g maybe this line includes the accuracy of each parameter set searched for in a tuning process, etc
    //old versions of file format also include build time.
    public String paras = "No parameter info";
    
    //LINE 3: acc, buildTime, testTime, memory
    //simple summarative performance stats. 
    public double acc = -1; 
    public long buildTime = -1; //buildClassifier(Instances) timing. might be cumulative time over many parameter set builds, etc
    public long testTime = -1; //total testtime for all predictions
    public long memory = -1; //user dependent on exactly what this means, typically mem used classifier is built
    

    //REMAINDER OF THE FILE - 1 prediction per line
    //raw performance data. currently just four parallel arrays
    private ArrayList<Double> trueClassValues;
    private ArrayList<Double> predictedClassValues;
    private ArrayList<double[]> predictedClassProbabilities;
    public ArrayList<Long> predictionTimes;
    
    //inferred/supplied dataset meta info
    private int numClasses; 
    private int numInstances;
    
    //calculated performance metrics
        //accuracy can be re-calced, as well as stored on line three in files
    public double balancedAcc; 
    public double sensitivity;
    public double specificity;
    public double precision;
    public double recall;
    public double f1; 
    public double mcc; //mathews correlation coefficient
    public double nll; 
    public double meanAUROC;
    public double stddev; //across cv folds, where applicable
    public double[][] confusionMatrix; //[actual class][predicted class]
    public double[] countPerClass;
    
    
    //Used to avoid infinite NLL scores when prob of true class =0 or 
    public static double NLL_PENALTY=-6.64; //Log_2(0.01)
    
    //Consistent time unit assumed across build times, test times, individual prediction times. 
    //Before considering different timeunits, all timing were in milliseconds, via
    //System.currentTimeMillis(). Some classifiers on some datasets may run in < 1 millisecond 
    //however, so sometimes nanseconds is wanted. Conversely, in the case of e.g large 
    //meta-ensembles on large datasets, the cumulative time may be massive. 
    TimeUnit timeUnit = TimeUnit.MILLISECONDS;
    
    //self-management flags
    /**
     * essentially controls whether a classifierresults object can have finaliseResults(trueClassVals)
     * called upon it. In theory, every class using the classifierresults object should make new 
     * instantiations of it each time a set of results is being computed, and so this is not needed
     * 
     * this was relevant expecially prior to on-line prediction storage being supported, and effectively
     * the intention was to turn the results into a const object after all the results were stored
     * 
     * todo: verify that this can be removed, or update to be more relevant. 
     */
    private boolean finalised = false;
    private boolean allStatsFound = false;
    
    //functional getters to retrieve info from a classifierresults object, initialised/stored here for conveniance 
    public static final Function<ClassifierResults, Double> GETTER_Accuracy = (ClassifierResults cr) -> {return cr.acc;};
    public static final Function<ClassifierResults, Double> GETTER_BalancedAccuracy = (ClassifierResults cr) -> {return cr.balancedAcc;};
    public static final Function<ClassifierResults, Double> GETTER_AUROC = (ClassifierResults cr) -> {return cr.meanAUROC;};
    public static final Function<ClassifierResults, Double> GETTER_NLL = (ClassifierResults cr) -> {return cr.nll;};
    public static final Function<ClassifierResults, Double> GETTER_F1 = (ClassifierResults cr) -> {return cr.f1;};
    public static final Function<ClassifierResults, Double> GETTER_MCC = (ClassifierResults cr) -> {return cr.mcc;};
    public static final Function<ClassifierResults, Double> GETTER_Precision = (ClassifierResults cr) -> {return cr.precision;};
    public static final Function<ClassifierResults, Double> GETTER_Recall = (ClassifierResults cr) -> {return cr.recall;};
    public static final Function<ClassifierResults, Double> GETTER_Sensitivity = (ClassifierResults cr) -> {return cr.sensitivity;};
    public static final Function<ClassifierResults, Double> GETTER_Specificity = (ClassifierResults cr) -> {return cr.specificity;};
    
    
    /*********************************
     * 
     *       CONSTRUCTORS
     * 
     */
    
    public ClassifierResults() {
        trueClassValues= new ArrayList<>();
        predictedClassValues = new ArrayList<>();
        predictedClassProbabilities = new ArrayList<>();
        predictionTimes = new ArrayList<>();
        
        finalised = false;
    }
    
    public ClassifierResults(String filePathAndName) throws FileNotFoundException {
        loadFromFile(filePathAndName);
    }
    
    public ClassifierResults(int numClasses) {
        trueClassValues= new ArrayList<>();
        predictedClassValues = new ArrayList<>();
        predictedClassProbabilities = new ArrayList<>();
        predictionTimes = new ArrayList<>();
        
        this.numClasses = numClasses;
        finalised = false;
    }
    
    /**
     * for if we are only storing the cv accuracy in the context of SaveCVAccuracy
     * 
     */
    public ClassifierResults(double cvacc, int numClasses) {
        this();
        this.acc = cvacc;
        this.numClasses = numClasses;
        finalised = false;
    }
    
    public ClassifierResults(double acc, double[] classVals, double[] preds, double[][] distsForInsts, int numClasses) {        
        this();
        
        for(double d:preds)
            predictedClassValues.add(d);
        this.acc = acc;
        for(double[] d:distsForInsts)
            predictedClassProbabilities.add(d);
 
        this.numClasses = numClasses;
        for(double d:classVals)
           trueClassValues.add(d);
        this.confusionMatrix = buildConfusionMatrix();
        
        this.stddev = -1; //not defined 
        finalised = true;
    }
    
    /**
     * for storing the stddev over cv folds as well 
     */
    public ClassifierResults(double acc, double[] classVals, double[] preds, double[][] distsForInsts, double stddev, int numClasses) { 
        this(acc,classVals,preds,distsForInsts,numClasses);
        this.stddev = stddev; 
        
        finalised = true;
    }

    
    
    /***********************
     * 
     *      DATASET META INFO
     * 
     * 
     */
    
    public int numClasses() { 
        if (numClasses <= 0)
            inferNumClasses();
        return numClasses; 
    }
    public void setNumClasses(int numClasses) { 
        this.numClasses = numClasses; 
    }
    private void inferNumClasses() {
        if (predictedClassProbabilities.isEmpty())
            this.numClasses = 0;
        else
            this.numClasses = predictedClassProbabilities.get(0).length;
    }
    
    
    public int numInstances() { 
        if (numInstances <= 0)
            inferNumInstances();
        return numInstances; 
    }
    public void setNumInstances(int numInstances) { 
        this.numInstances = numInstances; 
    }
    private void inferNumInstances() {
        this.numInstances = predictedClassValues.size();
    }
    
    
    
    
    /***************************
     * 
     *   LINE 1 GETS/SETS
     *  
     *  Just basic descriptive stuff, nothing fancy goign on here
     * 
     */
    
    public String getClassifierName() { return classifierName; }
    public void setClassifierName(String classifierName) { this.classifierName = classifierName; }

    public String getDatasetName() { return datasetName; }
    public void setDatasetName(String datasetName) { this.datasetName = datasetName; }

    public int getFoldID() { return foldID; }
    public void setFoldID(int foldID) { this.foldID = foldID; }

    public String getSplit() { return split; }
    public void setSplit(String split) { this.split = split; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    
    
    
    
    /*****************************
     * 
     *     LINE 2 GETS/SETS
     * 
     */
    public String getParas() { return paras; }
    public void setParas(String paras) { this.paras = paras; }
    
    


    
    
    
    
    /****************************
     *   
     *    PREDICTION STORAGE
     * 
     */
    
    /**
     * The true class is missing, however can be added in one go later with the 
     * method finaliseResults(double[] trueClassVals)
     */
    public void storeSingleResult(double[] dist, double predictedClass, long predictionTime) {      
        
        predictedClassProbabilities.add(dist);
        predictedClassValues.add(predictedClass);
        
        //allowing 0 in case user was unaware and doesnt care about a classifier taking e.g
        //0 milliseconds. todo revisit at some point when time units implemented/enforced 
        if (predictionTime < 0)
            //add a null placeholder, in case later predictions have timings? todo revisit
            predictionTimes.add(null);
        else {
            predictionTimes.add(predictionTime);

            if (testTime == -1)
                testTime = predictionTime;
            else 
                testTime += predictionTime;
        }
    }
    
    
    public void storeSingleResult(double trueClassVal, double[] dist, double predictedClass, long predictionTime) {        
        storeSingleResult(dist,predictedClass,predictionTime);
        trueClassValues.add(trueClassVal);
    }
    
    
    /**
     * Prediction shall be inferred as the class with the max probability. The true class is 
     * missing, however can be added in one go later with the method finaliseResults(double[] trueClassVals)
     * 
     * TIES ARE RESOLVED BY TAKING THE FIRST CLASS WITH THE TIED MAX PROBABILITY FOR REPRODUCABILITY REASONS.  
     * IF A DIFFERENT TIE-RESOLVING MECHANISM IS WANTED, USE THE METHOD THAT ALLOWS 
     * THE PASSING OF THE PREDICTED CLASS VAL AS WELL (e.g resolve randomly, take modal class)
     */
    public void storeSingleResult(double[] dist, long predictionTime) {        
        double max = dist[0];
        double maxInd = 0;
        for (int i = 0; i < dist.length; i++) {
            if (dist[i] > max) {
                max = dist[i];
                maxInd = i;
            }
        }
        
        storeSingleResult(dist, maxInd, predictionTime);
    }
    
    /**
     * Prediction shall be inferred as the class with the max probability. 
     * 
     * TIES ARE RESOLVED BY TAKING THE FIRST CLASS WITH THE TIED MAX PROBABILITY FOR REPRODUCABILITY REASONS.  
     * IF A DIFFERENT TIE-RESOLVING MECHANISM IS WANTED, USE THE METHOD THAT ALLOWS 
     * THE PASSING OF THE PREDICTED CLASS VAL AS WELL (e.g resolve randomly, take modal class)
     */
    public void storeSingleResult(double trueClassVal, double[] dist, long predictionTime) {        
        storeSingleResult(dist,predictionTime);
        trueClassValues.add(trueClassVal);
    }

        
    /**
     * Will perform some basic validation to make sure that everything is here 
     * that is expected, and compute the accuracy etc ready for file writing. 
     * 
     * Typical usage: results.finaliseResults(instances.attributeToDoubleArray(instances.classIndex()))
     */
    public void finaliseResults(double[] testClassVals) throws Exception {
        if (finalised) {
            System.out.println("Results already finalised, skipping re-finalisation");
            return;
        }
        
        if (predictedClassProbabilities == null || predictedClassValues == null ||
                predictedClassProbabilities.isEmpty() || predictedClassValues.isEmpty())
            throw new Exception("finaliseTestResults(): no test predictions stored for this module");
        
        if (testClassVals.length != predictedClassValues.size())
            throw new Exception("finaliseTestResults(): Number of test predictions made and number of test cases do not match");
        
        for(double d:testClassVals)
            trueClassValues.add(d);
        
        
        double correct = .0;
        for (int inst = 0; inst < predictedClassValues.size(); inst++) {
            if (testClassVals[inst] == predictedClassValues.get(inst))
                ++correct;
        }
        acc = correct/testClassVals.length;
        
        finalised = true;
    }
    
    
    
    
    
    
    
    
    /******************************
    *
    *          RAW DATA ACCESSORS
    * 
    *     getAsList, getAsArray, and getSingleElement of the four lists describing predictions
    * 
    */
    
    public ArrayList<Double> getTrueClassVals() {
        return trueClassValues;
    }
    
    public double[] getTrueClassValsAsArray(){
        double[] d=new double[trueClassValues.size()];
        int i=0;
        for(double x:trueClassValues)
            d[i++]=x;
        return d;
    }
    
    public double getTrueClassValue(int index){
        return trueClassValues.get(index);
    }
    
    
    public ArrayList<Double> getPredClassVals(){
        return predictedClassValues;
    }
    
    public double[] getPredClassValsAsArray(){
        double[] d=new double[predictedClassValues.size()];
        int i=0;
        for(double x:predictedClassValues)
            d[i++]=x;
        return d;
    }
    
    public double getPredClassValue(int index){
        return predictedClassValues.get(index);
    }
    

    public ArrayList<double[]> getProbabilityDistributions() { 
        return predictedClassProbabilities;
    }
    
    public double[][] getProbabilityDistributionsAsArray() { 
        return predictedClassProbabilities.toArray(new double[][] {});
    }
    
    public double[] getDistributionForInstance(int i){
       if(i<predictedClassProbabilities.size())
            return predictedClassProbabilities.get(i);
       return null;
    }
    
    
    public ArrayList<Long> getPredictionTimes() {
        return predictionTimes;
    }
    
    public long[] getPredictionTimesAsArray() {
        long[] l=new long[predictionTimes.size()];
        int i=0;
        for(long x:predictionTimes)
            l[i++]=x;
        return l;
    }

    public long getPredictionTime(int index) {
        return predictionTimes.get(index);
    }
    
    public void cleanPredictionInfo() {
        predictedClassProbabilities = null;
        predictedClassValues = null;
        trueClassValues = null;
        predictionTimes = null;
    }
        
        
    
    
    /********************************
    *
    *     FILE READ/WRITING
    *
    */
    
    public static boolean exists(File file) {
       return file.exists() && file.length()>0;
       //todo and is valid, maybe
    }
    public static boolean exists(String path) {
        return exists(new File(path));
    }
    
    /**
     * reads and STORES the prediction in this clasifierresults object
     * returns true if the prediction described by this string was correct (i.e. truclass==predclass) 
     * 
     * does NOT increment numInstances.
     * if numClasses is still < 0, WILL set numclasses if distribution info is present. 
     */
    private boolean instancePredictionFromString(String predLine) { 
        String[] split=predLine.split(",");

        //collect actual/predicted class
        double trueClassVal=Double.valueOf(split[0].trim());
        double predClassVal=Double.valueOf(split[1].trim());
        
        if(split.length==2) //no probabilities, no timing. VERY old files will not have them
            return true;
        
        //split[2] should be empty (if we're still here), separator before probs
        assert(split[2].equals(""));
        
        //collect probabilities
        double[] dist = null;
        if (numClasses < 2) {
            List<Double> distL = new ArrayList<>();
            for(int i = 3; i < split.length; i++) {
                if (split[i].equals(""))
                    break; //we're at the empty-space-separator between probs and timing 
                else 
                    distL.add(Double.valueOf(split[i].trim()));
            }
                  
            numClasses = distL.size();
            assert(numClasses >= 2);
            
            dist = new double[numClasses];
            for (int i = 0; i < numClasses; i++)
                dist[i] = distL.get(i);
        }
        else {
            //we know how many classes there should be, use this as implicit
            //file verification
            dist = new double[numClasses];
            for (int i = 0; i < numClasses; i++) {
                //now need to offset by 3.
                dist[i] = Double.valueOf(split[i+3].trim());
            }
        }
        
        //collect timings
        long predTime = -1;
        //      act/pred, space, dist, space, *timing*
        int numParts = 2 + 1 + numClasses + 1 + 1; 
        if (split.length >= numParts)
            predTime = Long.parseLong(split[numParts-1].trim());
        
        storeSingleResult(trueClassVal, dist, predClassVal, predTime);
        return trueClassVal==predClassVal;
    }
    
    
    private String instancePredictionToString(int i) { 
        StringBuilder sb = new StringBuilder();
        
        sb.append(trueClassValues.get(i).intValue()).append(",");
        sb.append(predictedClassValues.get(i).intValue());
        
        //probs
        sb.append(","); //<empty space>
        double[] probs=predictedClassProbabilities.get(i);
        for(double d:probs)
            sb.append(",").append(GenericTools.RESULTS_DECIMAL_FORMAT.format(d));
        
        //timing 
        sb.append(",,").append(predictionTimes.get(i)); //<empty space>, timing
        
        return sb.toString();
    }
    
    public String instancePredictionsToString(){
        if(numInstances()>0 &&(predictedClassProbabilities.size()==trueClassValues.size()&& predictedClassProbabilities.size()==predictedClassValues.size())){
            StringBuilder sb=new StringBuilder("");
            
            for(int i=0;i<numInstances();i++){
                sb.append(instancePredictionToString(i));

                if(i<numInstances()-1)
                    sb.append("\n");
            }
            
            return sb.toString();
        }
        else
           return "No Instance Prediction Information";
    }
   
    @Override
    public String toString() {                
        return generateFirstLine();
    }
    
    public String writeResultsFileToString() {                
        StringBuilder st = new StringBuilder();
        st.append(generateFirstLine()).append("\n");
        st.append(generateSecondLine()).append("\n");
        st.append(generateThirdLine()).append("\n");

        st.append(instancePredictionsToString());
        return st.toString();
    }
   
    public void writeResultsFile(String path) throws Exception {
        OutFile out = new OutFile(path);
        try {
            out.writeString(writeResultsFileToString());
        } catch (Exception e) { 
             throw new Exception("TODO stop using or update outfile... : "
                     + "Outfile most likely didnt open successfully, probably directory doesnt exist yet.\n" + e);
        }
        out.closeFile();
    }
    
    private void parseFirstLine(String line) {
        String[] parts = line.split(",");
        if (parts.length == 0)
            return;
        
        //old tuned classifiers (and maybe others) just wrote a classifier name identifier
        //covering for backward compatability, otherwise datasetname is first
        if (parts.length == 1)
            classifierName = parts[0];
        else {
            datasetName = parts[0];
            classifierName = parts[1];
        }
        
        if (parts.length > 2)
            split = parts[2];
        if (parts.length > 3)
            foldID = Integer.parseInt(parts[3]);
        if (parts.length > 4)
            description = parts[4];

        //in case the description had commas in it? ...
        for (int i = 5; i < parts.length; i++)
            description += parts[i];
    }
    private String generateFirstLine() { 
        return datasetName + "," + classifierName + "," + split + "," + foldID + "," + description;
    }
   
    private void parseSecondLine(String line) { 
        paras = line;
       
        //handle buildtime if it's on this line like older files may have, 
        //taking it out of the generic paras string and putting the value into the actual field
        String[] parts = paras.split(",");
        if (parts.length > 0 && parts[0].contains("BuildTime")) {
            buildTime = (long)Double.parseDouble(parts[1].trim());
            
            if (parts.length > 2) { //this has actual paras too, rebuild this string without buildtime
                paras = parts[2];
                for (int i = 3; i < parts.length; i++) {
                    paras += "," + parts[i];
                }
            }
        }
    }
    private String generateSecondLine() {
        //todo decide what to do with this
        return paras;
    }
    
    /**
     * Returns the test acc reported on this line, for comparison with acc 
     * computed later to assert they align. Accuracy has always been reported 
     * on this line in this file format, so fair to assume if this fails 
     * then the file is simply malformed
     */
    private double parseThirdLine(String line) { 
        String[] parts = line.split(",");
        
        acc = Double.parseDouble(parts[0]);
       
        //if buildtime is here, it shouldn't be on the paras line too. 
        //if it is, likely an old SaveParameterInfo implementation put it there
        //for now, overwriting that buildtime with this one, but printing warning 
        if (parts.length > 1)  {
            if (buildTime != -1)  //the default
                System.out.println("CLASSIFIERRESULTS READ WARNING: build time reported on both "
                        + "second and third line. Using the value reported on the third line");
            
            buildTime = Long.parseLong(parts[1]); 
        }
        if (parts.length > 2) 
            testTime = Long.parseLong(parts[2]);
        if (parts.length > 3) 
            memory = Long.parseLong(parts[3]);
        
        return acc;
    }
    private String generateThirdLine() {
        //todo revisit: printing all 4, since e.g what if testTime was calced but not buildtime ?
        //would have been printed where we're expecting buildtime to be, etc       
        String res = acc+"";
        res += "," + buildTime;
        res += "," + testTime;
        res += "," + memory;
        return res;
    }

    public void loadFromFile(String path) throws FileNotFoundException {
        //init
        trueClassValues = new ArrayList<>();
        predictedClassValues = new ArrayList<>();
        predictedClassProbabilities = new ArrayList<>();
        predictionTimes = new ArrayList<>();
        numInstances = 0;
        acc = -1;
        buildTime = -1;
        testTime = -1; 
        memory = -1;

        //check file exists
        File f = new File(path);
        if (!(f.exists() && f.length() > 0)) 
            throw new FileNotFoundException("File " + path + " NOT FOUND");

        Scanner inf = new Scanner(f);

        //parse meta infos
        parseFirstLine(inf.nextLine());
        parseSecondLine(inf.nextLine());
        double reportedTestAcc = parseThirdLine(inf.nextLine());

        //have all meta info, start reading predictions
        double correct = 0;
        while (inf.hasNext()) {
            String line = inf.nextLine();
            //may be trailing empty lines at the end of the file
            if (line == null || line.equals(""))
                break;
            
            if (instancePredictionFromString(line))
                correct++;
            numInstances++;
        }

        //acts as a basic form of verification
        acc = correct / numInstances;
        double eps = 1.e-8;
        if (Math.abs(reportedTestAcc - acc) > eps) {
            throw new ArithmeticException("Calculated accuracy (" + acc + ") differs from written accuracy (" + reportedTestAcc + ") "
                    + "by more than eps (" + eps + ")");
        }

        finalised = true;
        inf.close();
    }
   
    
    
    
    
    
    
    
    
    
    /******************************************
     * 
     *   METRIC CALCULATIONS 
     *
     */
    
    
    
    /**
     * Find: Accuracy, Balanced Accuracy, F1 (1 vs All averaged?), 
     * Sensitivity, Specificity, AUROC, negative log likelihood, MCC
     */   
    public void findAllStats(){
       if (numInstances <= 0)
           inferNumInstances();
        
       confusionMatrix=buildConfusionMatrix();
       
       countPerClass=new double[confusionMatrix.length];
       for(int i=0;i<trueClassValues.size();i++)
           countPerClass[trueClassValues.get(i).intValue()]++;
       
       acc=0;
       for(int i=0;i<numClasses;i++)
           acc+=confusionMatrix[i][i];
       acc/=numInstances;
       
       balancedAcc=findBalancedAcc(confusionMatrix);
       nll=findNLL();
       meanAUROC=findMeanAUROC();
       mcc = computeMCC(confusionMatrix);
       
       f1=findF1(confusionMatrix); //also handles spec/sens/prec/recall in the process of finding f1
       
       allStatsFound = true;
    }
   
    public void findAllStatsOnce(){
        if (finalised && allStatsFound) {
            System.out.println("Stats already found, ignoring findAllStatsOnce()");
            return;
        } 
        else {
            findAllStats();
        }
    }
      
       
    /**
    * @return [actual class][predicted class]
    */
    private double[][] buildConfusionMatrix() {
        double[][] matrix = new double[numClasses][numClasses];
        for (int i = 0; i < predictedClassValues.size(); ++i){
            double actual=trueClassValues.get(i);
            double predicted=predictedClassValues.get(i);
            ++matrix[(int)actual][(int)predicted];
        }
        return matrix;
    }
    
    
    /**
     * uses only the probability of the true class
     */
    public double findNLL(){
        double nll=0;
        for(int i=0;i<trueClassValues.size();i++){
            double[] dist=getDistributionForInstance(i);
            int trueClass = trueClassValues.get(i).intValue();
            
            if(dist[trueClass]==0)
                nll+=NLL_PENALTY;
            else
                nll+=Math.log(dist[trueClass])/Math.log(2);//Log 2
        }
        return -nll/trueClassValues.size();
    }
           
    public double findMeanAUROC(){
        double a=0;
        if(numClasses==2){
            a=findAUROC(1);
/*            if(countPerClass[0]<countPerClass[1])
            else
                a=findAUROC(1);
 */       }
        else{
            double[] classDist = InstanceTools.findClassDistributions(trueClassValues, numClasses);
            for(int i=0;i<numClasses;i++){
                a+=findAUROC(i) * classDist[i];
            }
            
            //original, unweighted
//            for(int i=0;i<numClasses;i++){
//                a+=findAUROC(i);
//            }
//            a/=numClasses;
        }
        return a;
    }
   
    /**
     * todo could easily be optimised further if really wanted
     */
    public double computeMCC(double[][] confusionMatrix) {
        
        double num=0.0;
        for (int k = 0; k < confusionMatrix.length; ++k)
            for (int l = 0; l < confusionMatrix.length; ++l)
                for (int m = 0; m < confusionMatrix.length; ++m) 
                    num += (confusionMatrix[k][k]*confusionMatrix[m][l])-
                            (confusionMatrix[l][k]*confusionMatrix[k][m]);

        if (num == 0.0)
            return 0;
        
        double den1 = 0.0; 
        double den2 = 0.0;
        for (int k = 0; k < confusionMatrix.length; ++k) {
            
            double den1Part1=0.0;
            double den2Part1=0.0;
            for (int l = 0; l < confusionMatrix.length; ++l) {
                den1Part1 += confusionMatrix[l][k];
                den2Part1 += confusionMatrix[k][l];
            }

            double den1Part2=0.0;
            double den2Part2=0.0;
            for (int kp = 0; kp < confusionMatrix.length; ++kp)
                if (kp!=k) {
                    for (int lp = 0; lp < confusionMatrix.length; ++lp) {
                        den1Part2 += confusionMatrix[lp][kp];
                        den2Part2 += confusionMatrix[kp][lp];
                    }
                }
                      
            den1 += den1Part1 * den1Part2;
            den2 += den2Part1 * den2Part2;
        }
        
        return num / (Math.sqrt(den1)*Math.sqrt(den2));
    }
   
    /**
     * Balanced accuracy: average of the accuracy for each class
     * @param cm
     * @return 
     */   
    public double findBalancedAcc(double[][] cm){
        double[] accPerClass=new double[cm.length];
        for(int i=0;i<cm.length;i++)
            accPerClass[i]=cm[i][i]/countPerClass[i];
        double b=accPerClass[0];
        for(int i=1;i<cm.length;i++)
            b+=accPerClass[i]; 
        b/=cm.length;
        return b;
    }
    
    /**
     * F1: If it is a two class problem we use the minority class
     * if it is multiclass we average over all classes. 
     * @param cm
     * @return 
     */   
    public double findF1(double[][] cm){
        double f=0;
        if(numClasses==2){
            if(countPerClass[0]<countPerClass[1])
                f=findConfusionMatrixMetrics(cm,0,1);
            else
                f=findConfusionMatrixMetrics(cm,1,1);
        }
        else{//Average over all of them
            for(int i=0;i<numClasses;i++)
                f+=findConfusionMatrixMetrics(cm,i,1);
            f/=numClasses;
        }
        return f;
    }
   
    protected double findConfusionMatrixMetrics(double[][] confMat, int c,double beta) {
        double tp = confMat[c][c]; //[actual class][predicted class]
        //some very small non-zero value, in the extreme case that no cases of 
        //this class were correctly classified
        if (tp == .0)
            return .0000001; 
        
        double fp = 0.0, fn = 0.0,tn=0.0;
        
        for (int i = 0; i < confMat.length; i++) {
            if (i!=c) {
                fp += confMat[i][c];
                fn += confMat[c][i];
                tn+=confMat[i][i];
            }
        }
         
        precision = tp / (tp+fp);
        recall = tp / (tp+fn);
        sensitivity=recall;
        specificity=tn/(fp+tn);
        
        //jamesl
        //one in a million case on very small AND unbalanced datasets (lenses...) that particular train/test splits and their cv splits
        //lead to a divide by zero on one of these stats (C4.5, lenses, trainFold7 (and a couple others), specificity in the case i ran into)
        //as a little work around, if this case pops up, will simply set the stat to 0
        if (Double.compare(precision, Double.NaN) == 0)
            precision = 0;
        if (Double.compare(recall, Double.NaN) == 0)
            recall = 0;
        if (Double.compare(sensitivity, Double.NaN) == 0)
            sensitivity = 0;
        if (Double.compare(specificity, Double.NaN) == 0)
            specificity = 0;
        
        return (1+beta*beta) * (precision*recall) / ((beta*beta)*precision + recall);
    }
    
    protected double findAUROC(int c){
        class Pair implements Comparable<Pair>{
            Double x;
            Double y;
            public Pair(Double a, Double b){
                x=a;
                y=b;
            }
            @Override
            public int compareTo(Pair p) {
                return p.x.compareTo(x);
            }
            public String toString(){ return "("+x+","+y+")";}
        }
        
        ArrayList<Pair> p=new ArrayList<>();
        double nosPositive=0,nosNegative;
        for(int i=0;i<numInstances;i++){
            Pair temp=new Pair(predictedClassProbabilities.get(i)[c],trueClassValues.get(i));
            if(c==trueClassValues.get(i))
                nosPositive++;
            p.add(temp);
        }
        nosNegative=trueClassValues.size()-nosPositive;
        Collections.sort(p);
        
        /* http://www.cs.waikato.ac.nz/~remco/roc.pdf
                Determine points on ROC curve as follows; 
                starts in the origin and goes one unit up, for every
        negative outcome the curve goes one unit to the right. Units on the x-axis
        are 1
        #TN and on the y-axis 1
        #TP where #TP (#TN) is the total number
        of true positives (true negatives). This gives the points on the ROC curve
        (0; 0); (x1; y1); : : : ; (xn; yn); (1; 1).
        */
        ArrayList<Pair> roc=new ArrayList<>();
        double x=0;
        double oldX=0;
        double y=0;
        int xAdd=0, yAdd=0;
        boolean xLast=false,yLast=false;
        roc.add(new Pair(x,y));
        for(int i=0;i<numInstances;i++){
            if(p.get(i).y==c){
                if(yLast)
                    roc.add(new Pair(x,y));
                xLast=true;
                yLast=false;
                x+=1/nosPositive;
                xAdd++;
                if(xAdd==nosPositive)
                    x=1.0;
                
            }
            else{ 
                if(xLast)
                    roc.add(new Pair(x,y));
                yLast=true;
                xLast=false;
                y+=1/nosNegative;
                yAdd++;
                if(yAdd==nosNegative)
                    y=1.0;
            }
        }
        roc.add(new Pair(1.0,1.0));
        
        //Calculate the area under the ROC curve, as the sum over all trapezoids with
        //base xi+1 to xi , that is, A

        double auroc=0;
        for(int i=0;i<roc.size()-1;i++){
            auroc+=(roc.get(i+1).y-roc.get(i).y)*(roc.get(i+1).x);
        }
        return auroc;
    } 
    
    public String allPerformanceMetricsToString(){
        String str="Acc,"+acc+"\n";
        str+="BalancedAcc,"+balancedAcc+"\n"; 
        str+="sensitivity,"+sensitivity+"\n"; 
        str+="precision,"+precision+"\n"; 
        str+="recall,"+recall+"\n"; 
        str+="specificity,"+specificity+"\n";         
        str+="f1,"+f1+"\n"; 
        str+="mcc,"+mcc+"\n"; 
        str+="nll,"+nll+"\n"; 
        str+="meanAUROC,"+meanAUROC+"\n"; 
        str+="stddev,"+stddev+"\n"; 
        str+="Count per class:\n";
        for(int i=0;i<countPerClass.length;i++)
            str+="Class "+i+","+countPerClass[i]+"\n";
        str+="Confusion Matrix:\n";
        for(int i=0;i<confusionMatrix.length;i++){
            for(int j=0;j<confusionMatrix[i].length;j++)
                str+=confusionMatrix[i][j]+",";
            str+="\n";
        }
        return str;
    }

    public static ArrayList<Pair<String, Function<ClassifierResults, Double>>> getDefaultStatistics() { 
        ArrayList<Pair<String, Function<ClassifierResults, Double>>> stats = new ArrayList<>();
        stats.add(new Pair<>("ACC", GETTER_Accuracy));
        stats.add(new Pair<>("BALACC", GETTER_BalancedAccuracy));
        stats.add(new Pair<>("AUROC", GETTER_AUROC));
        stats.add(new Pair<>("NLL", GETTER_NLL));
        return stats;
    }
        
    public static ArrayList<Pair<String, Function<ClassifierResults, Double>>> getAllStatistics() { 
        ArrayList<Pair<String, Function<ClassifierResults, Double>>> stats = new ArrayList<>();
        stats.add(new Pair<>("ACC", GETTER_Accuracy));
        stats.add(new Pair<>("BALACC", GETTER_BalancedAccuracy));
        stats.add(new Pair<>("AUROC", GETTER_AUROC));
        stats.add(new Pair<>("NLL", GETTER_NLL));
        stats.add(new Pair<>("F1", GETTER_F1));
        stats.add(new Pair<>("MCC", GETTER_MCC));
        stats.add(new Pair<>("Prec", GETTER_Precision));
        stats.add(new Pair<>("Recall", GETTER_Recall));
        stats.add(new Pair<>("Sens", GETTER_Sensitivity));
        stats.add(new Pair<>("Spec", GETTER_Specificity));
        return stats;
    }
    
    public static ArrayList<Pair<String, Function<ClassifierResults, Double>>> getAccuracyStatistic() { 
        ArrayList<Pair<String, Function<ClassifierResults, Double>>> stats = new ArrayList<>();
        stats.add(new Pair<>("ACC", GETTER_Accuracy));
        return stats;
    }
    

    
}
