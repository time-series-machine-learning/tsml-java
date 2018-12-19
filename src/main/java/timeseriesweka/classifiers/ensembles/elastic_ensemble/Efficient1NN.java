package timeseriesweka.classifiers.ensembles.elastic_ensemble;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Scanner;
import utilities.ClassifierResults;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * An abstract class to allow for distance-specific optimisations of DTW for use 
 * in the ElasticEnsemble. This class allows for univariate and multivariate
 * time series to be used; the multivariate version calculates distances as the
 * sum of individual distance calculations between common dimensions of two
 * instances (using the same parameter setting on all channels).
 * 
 * E.G. a DTW implementation with window = 0.5 (50%) for two instances with 10 
 * channels would calculate the DTW distance separately for each channel, and 
 * sum the 10 distances together. 
 * 
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public abstract class Efficient1NN extends AbstractClassifier implements SaveParameterInfo{
    
    protected Instances train;
    protected Instances[] trainGroup;
    protected String classifierIdentifier;
    protected boolean allowLoocv = true;
    protected boolean singleParamCv = false; 
    
    private boolean fileWriting = false;
    private boolean individualCvParamFileWriting = false;
    private String outputDir;
    private String datasetName;
    private int resampleId;
    private ClassifierResults res =new ClassifierResults();
    
    /**
     * Abstract method to calculates the distance between two Instance objects
     * @param first 
     * @param second
     * @param cutOffValue a best-so-far value to allow early abandons
     * @return the distance between first and second. If early abandon occurs, it will return Double.MAX_VALUE.
     */
    public abstract double distance(Instance first, Instance second, double cutOffValue);
    
    /**
     * Multi-dimensional equivalent of the univariate distance method. Iterates 
     * through channels calculating distances independently using the same param
     * options, summing together at the end to return a single distance. 
     * 
     * @param first
     * @param second
     * @param cutOffValue
     * @return 
     */
    public double distance(Instance[] first, Instance[] second, double cutOffValue){
        double sum = 0;
        double decliningCutoff = cutOffValue;
        double thisDist;
        for(int d = 0; d < first.length; d++){
            thisDist = this.distance(first[d], second[d], decliningCutoff);
            sum += thisDist;
            if(sum > cutOffValue){
                return Double.MAX_VALUE;
            }
            decliningCutoff -= thisDist;
        }
        
        return sum;
    }
    
    /**
     * Utility method for easy cross-validation experiments. Each inheriting 
     * class has 100 param options to select from (some dependent on information
     * for the training data). Passing in the training data and a param
     * 
     * 
     * @param train
     * @param paramId 
     */
    public abstract void setParamsFromParamId(Instances train, int paramId);
    
    public void buildClassifier(Instances train) throws Exception{
        this.train = train;
        this.trainGroup = null;
    }
    
    public void buildClassifier(Instances[] trainGroup) throws Exception{
        this.train = null;
        this.trainGroup = trainGroup;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        
        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.train.numClasses()];
        
        double thisDist;
                
        for(Instance i:this.train){
            thisDist = distance(instance, i, bsfDistance); 
            if(thisDist < bsfDistance){
                bsfDistance = thisDist;
                classCounts = new int[train.numClasses()];
                classCounts[(int)i.classValue()]++;
            }else if(thisDist==bsfDistance){
                classCounts[(int)i.classValue()]++;
            }
        }
        
        double bsfClass = -1;
        double bsfCount = -1;
        for(int c = 0; c < classCounts.length; c++){
            if(classCounts[c]>bsfCount){
                bsfCount = classCounts[c];
                bsfClass = c;
            }
        }
        
        return bsfClass;
    }
    
    public double classifyInstanceMultivariate(Instance[] instance) throws Exception {
    
        if(this.trainGroup==null){
            throw new Exception("Error: this configuration is for multivariate data");
        }
        
        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.trainGroup[0].numClasses()];
        
        double thisDist;
            
        Instance[] trainInstancesByDimension;
        for(int i = 0; i < this.trainGroup[0].numInstances(); i++){
            trainInstancesByDimension = new Instance[this.trainGroup.length];
            for(int j = 0; j < trainInstancesByDimension.length; j++){
                trainInstancesByDimension[j] = this.trainGroup[j].instance(i);
            }
            
            thisDist = distance(instance, trainInstancesByDimension, bsfDistance);
            if(thisDist < bsfDistance){
                bsfDistance = thisDist;
                classCounts = new int[trainGroup[0].numClasses()];
                classCounts[(int)trainGroup[0].instance(i).classValue()]++;
            }else if(thisDist==bsfDistance){
                classCounts[(int)trainGroup[0].instance(i).classValue()]++;
            }
        }        
       
        double bsfClass = -1;
        double bsfCount = -1;
        for(int c = 0; c < classCounts.length; c++){
            if(classCounts[c]>bsfCount){
                bsfCount = classCounts[c];
                bsfClass = c;
            }
        }
        
        return bsfClass;
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        
        double bsfDistance = Double.MAX_VALUE;
        // for tie splitting
        int[] classCounts = new int[this.train.numClasses()];
        
        double thisDist;
        int sumOfBest = 0;
                
        for(Instance i:this.train){
            thisDist = distance(instance, i, bsfDistance); 
            if(thisDist < bsfDistance){
                bsfDistance = thisDist;
                classCounts = new int[train.numClasses()];
                classCounts[(int)i.classValue()]++;
                sumOfBest = 1;
            }else if(thisDist==bsfDistance){
                classCounts[(int)i.classValue()]++;
                sumOfBest++;
            }
        }
        
        double[] classDistributions = new double[this.train.numClasses()];
        for(int c = 0; c < classCounts.length; c++){
            classDistributions[c] = (double)classCounts[c]/sumOfBest;
        }
 
        return classDistributions;
    }
    
    public void setClassifierIdentifier(String classifierIdentifier){
        this.classifierIdentifier = classifierIdentifier;
    }
    
    public String getClassifierIdentifier(){
        return classifierIdentifier;
    } 
    
    @Override
    public String getParameters(){
        String paras="BuildTime,"+res.buildTime;
        return paras;
        
    }    
    // could parallelise here
//    public void writeLOOCVOutput(String tscProblemDir, String datasetName, int resampleId, String outputResultsDir, boolean tidyUp) throws Exception{    
//        for(int paramId = 0; paramId < 100; paramId++){
//            writeLOOCVOutput(tscProblemDir, datasetName, resampleId, outputResultsDir, paramId);
//        }    
//        parseLOOCVResults(tscProblemDir, datasetName, resampleId, outputResultsDir, tidyUp);
//    }
//    
//    public double writeLOOCVOutput(String tscProblemDir, String datasetName, int resampleId, String outputResultsDir, int paramId) throws Exception{
//        new File(outputResultsDir+classifierIdentifier+"/Predictions/"+datasetName+"/loocvForParamOptions/").mkdirs();
//        
//        Instances train = ClassifierTools.loadData(tscProblemDir+datasetName+"/"+datasetName+"_TRAIN");
//        Instances test = ClassifierTools.loadData(tscProblemDir+datasetName+"/"+datasetName+"_TEST");
//        
//        if(resampleId!=0){
//            Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, resampleId);
//            train = temp[0];
//            test = temp[1];
//        }
//        
//        this.setParamsFromParamId(paramId);
//        
//        Instances trainLoocv;
//        Instance testLoocv;
//        
//        int correct = 0;
//        double pred, actual;
//        for(int i = 0; i < train.numInstances(); i++){
//            trainLoocv = new Instances(train);
//            testLoocv = trainLoocv.remove(i);
//            actual = testLoocv.classValue();
//            this.buildClassifier(train);
//            pred = this.classifyInstance(testLoocv);
//            if(pred==actual){
//                correct++;
//            }
//        }
//        
//        return (double)correct/train.numInstances();
//    }
    
    
    public void setFileWritingOn(String outputDir, String datasetName, int resampleId){
        this.fileWriting = true;
        this.outputDir = outputDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
    }
    public void setIndividualCvFileWritingOn(String outputDir, String datasetName, int resampleId){
        this.individualCvParamFileWriting = true;
        this.outputDir = outputDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
    }
    
    
    
    public double[] loocv(Instances train) throws Exception{
        double[] accAndPreds = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";
        
//        System.out.println(parsedFileName);
        
        if(fileWriting){
            File existing = new File(parsedFileName);
            if(existing.exists()){
//                throw new Exception("Parsed results already exist for this measure: "+ parsedFileName);
                Scanner scan = new Scanner(existing);
                scan.useDelimiter("\n");
                scan.next(); // skip header line
                int paramId = Integer.parseInt(scan.next().trim().split(",")[0]);
                if(this.allowLoocv){
                    this.setParamsFromParamId(train, paramId);
                }
                this.buildClassifier(train);
                accAndPreds = new double[train.numInstances()+1];
                accAndPreds[0] = Double.parseDouble(scan.next().trim().split(",")[0]);
                int correct = 0;
                String[] temp;
                for(int i = 0; i < train.numInstances(); i++){
                    temp = scan.next().split(",");
                    accAndPreds[i+1] = Double.parseDouble(temp[1]);
                    if(accAndPreds[i+1]==Double.parseDouble(temp[0])){
                        correct++;
                    }
                }
                // commented out for now as this breaks the new EE loocv thing we're doing for the competition. Basically, if we try and load for train-1 ins for test in loocv, the number of train instances doesn't match so the acc is slightly off. should be an edge case, but can leave this check out so long as we trust the code
//                if(((double)correct/train.numInstances())!=accAndPreds[0]){
//                    System.err.println(existing.getAbsolutePath());
//                    System.err.println(((double)correct/train.numInstances())+" "+accAndPreds[0]);
//                    throw new Exception("Attempted file loading, but accuracy doesn't match itself?!");
//                }
                return accAndPreds;
            }else{
                new File(this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/").mkdirs();
            }
        }
        
//        write output
//        maybe a different version which looks for missing files and runs them?
        

        double bsfAcc = -1;
        int bsfParamId = -1;
        double[] bsfaccAndPreds = null;

        for(int paramId = 0; paramId < 100; paramId++){
//            System.out.print(paramId+" ");
            accAndPreds = loocvAccAndPreds(train,paramId);
//            System.out.println(this.allowLoocv);
//            System.out.println(accAndPreds[0]);
            if(accAndPreds[0]>bsfAcc){
                bsfAcc = accAndPreds[0];
                bsfParamId = paramId;
                bsfaccAndPreds = accAndPreds;
            }
            if(!this.allowLoocv){
                paramId = 100;
            }
        }
//        System.out.println(this.classifierIdentifier+", bsfParamId "+bsfParamId);
        this.buildClassifier(train);
        if(this.allowLoocv){
            this.setParamsFromParamId(train, bsfParamId);
        }    
        if(fileWriting){
            FileWriter out = new FileWriter(parsedFileName);
            out.append(this.classifierIdentifier+","+datasetName+",parsedTrain\n");
            out.append(bsfParamId+"\n");
            out.append(bsfAcc+"\n");
            for(int i = 1; i < bsfaccAndPreds.length; i++){
                out.append(train.instance(i-1).classValue()+","+bsfaccAndPreds[i]+"\n");
            }
            out.close();
        }
        
        return bsfaccAndPreds;
    }
    
    DecimalFormat df = new DecimalFormat("##.###");
    
    public double[] loocv(Instances[] trainGroup) throws Exception{
        double[] accAndPreds = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";
        
        Instances concatenated = concatenate(trainGroup);
        
        
        if(fileWriting){
            File existing = new File(parsedFileName);
            if(existing.exists()){
//                throw new Exception("Parsed results already exist for this measure: "+ parsedFileName);
                Scanner scan = new Scanner(existing);
                scan.useDelimiter("\n");
                scan.next(); // skip header line
                int paramId = Integer.parseInt(scan.next().trim().split(",")[0]);
                if(this.allowLoocv){
                    this.setParamsFromParamId(concatenated, paramId);
                }
                this.buildClassifier(trainGroup);
                accAndPreds = new double[trainGroup[0].numInstances()+1];
                accAndPreds[0] = Double.parseDouble(scan.next().trim().split(",")[0]);
                int correct = 0;
                String[] temp;
                for(int i = 0; i < trainGroup[0].numInstances(); i++){
                    temp = scan.next().split(",");
                    accAndPreds[i+1] = Double.parseDouble(temp[1]);
                    if(accAndPreds[i+1]==Double.parseDouble(temp[0])){
                        correct++;
                    }
                }
                // commented out for now as this breaks the new EE loocv thing we're doing for the competition. Basically, if we try and load for train-1 ins for test in loocv, the number of train instances doesn't match so the acc is slightly off. should be an edge case, but can leave this check out so long as we trust the code
//                if(((double)correct/train.numInstances())!=accAndPreds[0]){
//                    System.err.println(existing.getAbsolutePath());
//                    System.err.println(((double)correct/train.numInstances())+" "+accAndPreds[0]);
//                    throw new Exception("Attempted file loading, but accuracy doesn't match itself?!");
//                }
                return accAndPreds;
            }else{
                new File(this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/").mkdirs();
            }
        }
        
//        write output
//        maybe a different version which looks for missing files and runs them?
        

        double bsfAcc = -1;
        int bsfParamId = -1;
        double[] bsfaccAndPreds = null;

        for(int paramId = 0; paramId < 100; paramId++){
//            System.out.print(paramId+" ");
            accAndPreds = loocvAccAndPreds(trainGroup,concatenated,paramId);
//            System.out.println(this.allowLoocv);
//            System.out.println(accAndPreds[0]);
            if(accAndPreds[0]>bsfAcc){
                bsfAcc = accAndPreds[0];
                bsfParamId = paramId;
                bsfaccAndPreds = accAndPreds;
            }
            System.out.println("\t"+paramId+": "+df.format(accAndPreds[0]*100)+" ("+df.format(bsfAcc*100)+")");
            if(!this.allowLoocv){
                paramId = 100;
            }
        }
//        System.out.println(this.classifierIdentifier+", bsfParamId "+bsfParamId);
        this.buildClassifier(trainGroup);
        if(this.allowLoocv){
            this.setParamsFromParamId(concatenated, bsfParamId);
        }    
        if(fileWriting){
            FileWriter out = new FileWriter(parsedFileName);
            out.append(this.classifierIdentifier+","+datasetName+",parsedTrain\n");
            out.append(bsfParamId+"\n");
            out.append(bsfAcc+"\n");
            for(int i = 1; i < bsfaccAndPreds.length; i++){
                out.append(trainGroup[0].instance(i-1).classValue()+","+bsfaccAndPreds[i]+"\n");
            }
            out.close();
        }
        
        return bsfaccAndPreds;
    }
    
    public double[] loocvAccAndPreds(Instances train, int paramId) throws Exception{
        if(this.allowLoocv){
            this.setParamsFromParamId(train, paramId);
        }
        FileWriter out = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";
        String singleFileName = this.outputDir+this.classifierIdentifier+"/cv/"+datasetName+"/trainFold"+resampleId+"/pid"+paramId+".csv";
        if(this.individualCvParamFileWriting){
            if(new File(parsedFileName).exists()){//|| new File(singleFileName).exists()){
                throw new Exception("Error: Full parsed training results already exist - "+parsedFileName);
            }else if(new File(singleFileName).exists()){
                throw new Exception("Error: CV training results already exist for this pid - "+singleFileName);
            }
        }
        
        
        // else we already know what the params are, so don't need to set
        
        Instances trainLoocv;
        Instance testLoocv;
        
        int correct = 0;
        double pred, actual;
        
        double[] accAndPreds = new double[train.numInstances()+1];
        for(int i = 0; i < train.numInstances(); i++){
            trainLoocv = new Instances(train);
            testLoocv = trainLoocv.remove(i);
            actual = testLoocv.classValue();
            this.buildClassifier(trainLoocv);
            pred = this.classifyInstance(testLoocv);
            if(pred==actual){
                correct++;
            }
            accAndPreds[i+1]= pred;
        }
        accAndPreds[0] = (double)correct/train.numInstances();
//        System.out.println(accAndPreds[0]);
        
        if(individualCvParamFileWriting){
            new File(this.outputDir+this.classifierIdentifier+"/cv/"+datasetName+"/trainFold"+resampleId+"/").mkdirs();
            out = new FileWriter(singleFileName);
            out.append(this.classifierIdentifier+","+datasetName+",cv\n");
            out.append(paramId+"\n");
            out.append(accAndPreds[0]+"\n");
            for(int i = 1; i < accAndPreds.length;i++){
                out.append(train.instance(i-1).classValue()+","+accAndPreds[i]+"\n");
            }
            out.close();
        }
        
        return accAndPreds;
    }
    
    public double[] loocvAccAndPreds(Instances[] trainGroup, Instances concatenated, int paramId) throws Exception{
        
        FileWriter out = null;
        String parsedFileName = this.outputDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv";
        String singleFileName = this.outputDir+this.classifierIdentifier+"/cv/"+datasetName+"/trainFold"+resampleId+"/pid"+paramId+".csv";
//        if(this.fileWriting){
//            if(new File(parsedFileName).exists()){//|| new File(singleFileName).exists()){
//                throw new Exception("Error: Full parsed training results already exist - "+parsedFileName);
//            }else if(new File(singleFileName).exists()){
//                throw new Exception("Error: CV training results already exist for this pid - "+singleFileName);
//            }
//        }
        
        if(this.allowLoocv){
//            System.out.println("allowed");
            this.setParamsFromParamId(concatenated, paramId);
//            System.out.println(this.toString());
        }
        // else we already know what the params are, so don't need to set
        
        Instances[] trainLoocv;
        Instance[] testLoocv;
        
        int correct = 0;
        double pred, actual;
        
        double[] accAndPreds = new double[trainGroup[0].numInstances()+1];
        for(int i = 0; i < trainGroup[0].numInstances(); i++){
            trainLoocv = new Instances[trainGroup.length];
            testLoocv = new Instance[trainGroup.length];
            
            for(int d = 0; d < trainGroup.length; d++){
                trainLoocv[d] = new Instances(trainGroup[d]);
                testLoocv[d] = trainLoocv[d].remove(i);
            }
            
//            trainLoocv = new Instances(train);
//            testLoocv = trainLoocv.remove(i);
            actual = testLoocv[0].classValue();
            this.buildClassifier(trainLoocv);
            pred = this.classifyInstanceMultivariate(testLoocv);
            if(pred==actual){
                correct++;
            }
            accAndPreds[i+1]= pred;
        }
        accAndPreds[0] = (double)correct/trainGroup[0].numInstances();
//        System.out.println(accAndPreds[0]);
        
//        if(fileWriting){
//            out = new FileWriter(singleFileName);
//            out.append(this.classifierIdentifier+","+datasetName+",cv\n");
//            out.append(paramId+"\n");
//            out.append(accAndPreds[0]+"\n");
//            for(int i = 1; i < accAndPreds.length;i++){
//                out.append(train.instance(i-1).classValue()+","+accAndPreds[i]+"\n");
//            }
//            out.close();
//        }
        
        return accAndPreds;
    }
    
    public void writeTrainTestOutput(String tscProblemDir, String datasetName, int resampleId, String outputResultsDir) throws Exception{
        
        // load in param id from training results
        File cvResults = new File(outputResultsDir+classifierIdentifier+"/Predictions/"+datasetName+"/trainFold"+resampleId+".csv");
        if(!cvResults.exists()){
            throw new Exception("Error loading file "+cvResults.getAbsolutePath());
        }
        Scanner scan = new Scanner(cvResults);
        scan.useDelimiter(System.lineSeparator());
        scan.next();
        int paramId = Integer.parseInt(scan.next().trim());
        this.setParamsFromParamId(train, paramId);
        
        // Now classifier is set up, make the associated files and do the test classification
        
        new File(outputResultsDir+classifierIdentifier+"/Predictions/"+datasetName+"/").mkdirs();
        StringBuilder headerInfo = new StringBuilder();
        
        headerInfo.append(classifierIdentifier).append(System.lineSeparator());
        headerInfo.append(this.getParamInformationString()).append(System.lineSeparator());
        
        Instances train = ClassifierTools.loadData(tscProblemDir+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData(tscProblemDir+datasetName+"/"+datasetName+"_TEST");
        
        if(resampleId!=0){
            Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, resampleId);
            train = temp[0];
            test = temp[1];
        }
        
        this.buildClassifier(train);
        StringBuilder classificationInfo = new StringBuilder();
        int correct = 0;
        double pred, actual;
        for(int i = 0; i < test.numInstances(); i++){
            actual = test.instance(i).classValue();
            pred = this.classifyInstance(test.instance(i));
            classificationInfo.append(actual).append(",").append(pred).append(System.lineSeparator());
            if(actual==pred){
                correct++;
            }
        }
        
        FileWriter outWriter = new FileWriter(outputResultsDir+this.classifierIdentifier+"/Predictions/"+datasetName+"/testFold"+resampleId+".csv");
        outWriter.append(headerInfo);
        outWriter.append(((double)correct/test.numInstances())+System.lineSeparator());
        outWriter.append(classificationInfo);
        outWriter.close();
        
        
    }
    
//    public static void parseLOOCVResults(String tscProblemDir, String datasetName, int resampleId, String outputResultsDir, boolean tidyUp){
//        
//    }
    
    public abstract String getParamInformationString();
    
    public Instances getTrainingData(){  
        return this.train;
    } 
    
    
    
    public static Instances concatenate(Instances[] train){
    // make a super arff for finding params that need stdev etc
        Instances temp = new Instances(train[0],0);
        for(int i = 1; i < train.length; i++){
            for(int j = 0; j < train[i].numAttributes()-1;j++){
                temp.insertAttributeAt(train[i].attribute(j), temp.numAttributes()-1);
            }
        }

        int dataset, attFromData;

        for(int insId = 0; insId < train[0].numInstances(); insId++){
            DenseInstance dense = new DenseInstance(temp.numAttributes());
            for(int attId = 0; attId < temp.numAttributes()-1; attId++){
            
                dataset = attId/(train[0].numAttributes()-1);
                attFromData = attId%(train[0].numAttributes()-1);
                dense.setValue(attId,train[dataset].instance(insId).value(attFromData));
                
            }
            dense.setValue(temp.numAttributes()-1, train[0].instance(insId).classValue());
            temp.add(dense);
        }
        return temp;
    }
}
