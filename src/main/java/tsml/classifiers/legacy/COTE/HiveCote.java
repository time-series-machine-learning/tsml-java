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
package tsml.classifiers.legacy.COTE;


import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.interval_based.TSF;
import tsml.classifiers.legacy.RISE;
import tsml.classifiers.dictionary_based.BOSS;
import tsml.classifiers.distance_based.ElasticEnsemble;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;
import tsml.classifiers.EnhancedAbstractClassifier;

import tsml.filters.shapelet_filters.ShapeletFilter;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.*;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
/**
 * Note this is here for legacy reasons only, it has been replaced by tsml.classifiers.hybrids.HIVE_COTE
 * @author Jason Lines (j.lines@uea.ac.uk)
 * 
 * Basic use case
 *  HiveCote hc=new HiveCote();
 *  hc.buildClassifier(data);

DEVELOPMENT NOTES for any users added by ajb on 23/7/18: 
* 1. Hive COTE defaults to shouty mode (prints out stuff). To turn off, call 
*   hc.setShouty(false)
* 2. Hive COTE can be used with existing results as a post processor, or built all in one. 
* For larger problems we advise building the components individually as it makes it easier to control
* The mechanism for doing this is to use HiveCotePostProcessed in the cote package (NOTE could be tidied).
* 3. Full Hive COTE is very slow, but a huge amount of the computation is unnecessary. 
* The slowest module is the shapelet transform, when
* set to do a full enumeration of the shapelet space. However, this is never necessary.
* You can contract ST to only search for a fixed time. We are making all the components contract classifiers
* By default, we set the sequential build time to 7 days. This is APPROXIMATE and OPTIMISTIC. So we advise set lower to start then increase
* change hours amount with, e.g.
* hc.setDayLimit(int), hc.setHourLimit(int), hc.setMinuteLimit(int)
* or by default, to set hours,
* hc.setTrainTimeLimit(long) //breaking aarons interface, soz
* to remove any limits, call
* hc.setContract(false)
*
* 
* To review: whole file writing thing. 

*/
public class HiveCote extends EnhancedAbstractClassifier implements TrainTimeContractable,TechnicalInformationHandler{


    private ArrayList<Classifier> classifiers;
    private ArrayList<String> names;
    private ConstituentHiveEnsemble[] modules;
    private boolean verbose = true;
    private int maxCvFolds = 10;// note: this only affects manual CVs from this class using the crossvalidate method. This will not affect internal classifier cv's if they are set within those classes
    
//    private boolean writeEnsembleTrainingPredictions = false;
//    private String ensembleTrainingPredictionsPathAndName = null;
    
    private boolean fileWriting = false;
    private String fileOutputDir;
    private String fileOutputDataset;
    private String fileOutputResampleId;

    private static int MAXCONTRACTHOURS=7*24;
    private int contractHours=MAXCONTRACTHOURS;  //Default to maximum 7 days run time
    private long contractTimeNanos = 0;
    private boolean trainTimeContract =false;

    public HiveCote(){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.setDefaultEnsembles();
    }
    
    public HiveCote(ArrayList<Classifier> classifiers, ArrayList<String> classifierNames){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        setClassifiers(classifiers,classifierNames);
        this.classifiers = classifiers;
        this.names = classifierNames;
        if(trainTimeContract){
            setTrainTimeLimit(TimeUnit.HOURS,contractHours);
        }

    }
    public final void setClassifiers(ArrayList<Classifier> classifiers, ArrayList<String> classifierNames){
        for(Classifier c: classifiers) {
            if (c instanceof EnhancedAbstractClassifier)
                if (((EnhancedAbstractClassifier) c).ableToEstimateOwnPerformance())
                    ((EnhancedAbstractClassifier) c).setEstimateOwnPerformance(true);
        }

    }
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "J. Lines, S. Taylor and A. Bagnall");
        result.setValue(TechnicalInformation.Field.TITLE, "Time Series Classification with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based Ensembles");
        result.setValue(TechnicalInformation.Field.JOURNAL, "ACM Transactions on Knowledge Discovery from Data");
        result.setValue(TechnicalInformation.Field.VOLUME, "12");
        result.setValue(TechnicalInformation.Field.NUMBER, "5");
        
        result.setValue(TechnicalInformation.Field.PAGES, "52");
        result.setValue(TechnicalInformation.Field.YEAR, "2018");
        return result;
    }    
    public void setContract(boolean b){
        trainTimeContract =b;
        contractHours=MAXCONTRACTHOURS;
        setHourLimit(contractHours);
    }
    public void setContract(int hours){
        trainTimeContract =true;
        contractHours=hours;
        setHourLimit(contractHours);
    }
    @Override
    public void setSeed(int seed) { 
        seedClassifier=true;
        this.seed = seed;
        rand=new Random(seed);
        int count =2;
        for(Classifier c:classifiers){
            if(c instanceof Randomizable)
                ((Randomizable)c).setSeed(seed+count++);
        }
    }    
    
    private void setDefaultEnsembles(){
        
        classifiers = new ArrayList<>();
        names = new ArrayList<>();
        
        ElasticEnsemble ee = new ElasticEnsemble();
        ee.setEstimateOwnPerformance(true);
        classifiers.add(ee);
        
        ShapeletTransformClassifier stc = new ShapeletTransformClassifier();
//        CAWPE h = new CAWPE();
//        DefaultShapeletTransformPlaceholder st= new DefaultShapeletTransformPlaceholder();
        if(trainTimeContract){
            stc.setHourLimit(contractHours);
        }
        
        classifiers.add(stc); // to get around the issue of needing training data 
        RISE rise = new RISE();
        classifiers.add(rise);
        
        BOSS boss = new BOSS();
        boss.setEstimateOwnPerformance(true);
        classifiers.add(boss);
        
        TSF tsf=new TSF();
        tsf.setEstimatorMethod("CV");
        tsf.setEstimateOwnPerformance(true);
        classifiers.add(tsf);
        
        names.add("EE");
        names.add("ST");
        names.add("RISE");
        names.add("BOSS");
        names.add("TSF");
    }
    
    public void turnOnFileWriting(String outputDir, String datasetName){
        turnOnFileWriting(outputDir, datasetName, "0");
    }
    public void turnOnFileWriting(String outputDir, String datasetName, String resampleFoldIdentifier){
        this.fileWriting = true;
        this.fileOutputDir = outputDir;
        this.fileOutputDataset = datasetName;
        this.fileOutputResampleId = resampleFoldIdentifier;
    }
    
    @Override
    public void buildClassifier(Instances train) throws Exception{
        getCapabilities().testWithFail(train);
        long t1=System.nanoTime();
        optionalOutputLine("Start of training");
                
        modules = new ConstituentHiveEnsemble[classifiers.size()];
        
        System.out.println("modules include:");
        for(int i = 0; i < classifiers.size();i++){
            System.out.println(names.get(i));
        }
        
        double ensembleAcc;
        String outputFilePathAndName;
        
        for(int i = 0; i < classifiers.size(); i++){
            
            
// if classifier is an implementation of TrainAccuracyEstimator, no need to cv for ensemble accuracy as it can self-report
// e.g. of the default modules, EE, CAWPE, and BOSS should all have this functionality (group a); RISE and TSF do not currently (group b) so must manualy cv

            if(EnhancedAbstractClassifier.classifierIsEstimatingOwnPerformance(classifiers.get(i))){
                optionalOutputLine("training (group a): "+this.names.get(i));
                classifiers.get(i).buildClassifier(train);
                ClassifierResults res= ((EnhancedAbstractClassifier)classifiers.get(i)).getTrainResults();
                modules[i] = new ConstituentHiveEnsemble(this.names.get(i), this.classifiers.get(i), res.getAcc());
                
                if(this.fileWriting){    
                    outputFilePathAndName = fileOutputDir+names.get(i)+"/Predictions/"+this.fileOutputDataset+"/trainFold"+this.fileOutputResampleId+".csv";    
                    genericCvResultsFileWriter(outputFilePathAndName, train, res.getPredClassValsAsArray(), this.fileOutputDataset, modules[i].classifierName, ((EnhancedAbstractClassifier)(modules[i].classifier)).getParameters(), modules[i].ensembleCvAcc);
                }
                
                
            // else we must do a manual cross validation to get the module's encapsulated cv acc
            // note this isn't optimal; would be better to change constituent ensembles to self-record cv acc during training, rather than cv-ing and then building
            // however, this is effectively a wrapper so we can add any classifier to the collective without worrying about implementation support
            }else{
                optionalOutputLine("crossval (group b): "+this.names.get(i));
                ensembleAcc = crossValidateWithFileWriting(classifiers.get(i), train, maxCvFolds,this.names.get(i));
                optionalOutputLine("training (group b): "+this.names.get(i));

                classifiers.get(i).buildClassifier(train);                
                modules[i] = new ConstituentHiveEnsemble(this.names.get(i), this.classifiers.get(i), ensembleAcc);
                
                
                
            }
            optionalOutputLine("done "+modules[i].classifierName);
        }        

        if(verbose){
            printModuleCvAccs();
        }
        long t2=System.nanoTime();
        trainResults.setBuildTime(t2-t1);
    }
    

    
    
    private static void genericCvResultsFileWriter(String outFilePathAndName, Instances instances, String classifierName, double[] preds, double cvAcc) throws Exception{
        genericCvResultsFileWriter(outFilePathAndName, instances, preds, instances.relationName(), classifierName, "noParamInfo", cvAcc);
    }
    private static void genericCvResultsFileWriter(String outFilePathAndName, Instances instances, double[] preds, String datasetName, String classifierName, String paramInfo, double cvAcc) throws Exception{
        
        if(instances.numInstances()!=preds.length){
            throw new Exception("Error: num instances doesn't match num preds.");
        }
        
        File outPath = new File(outFilePathAndName);
        outPath.getParentFile().mkdirs();
        FileWriter out = new FileWriter(outFilePathAndName);
        
        out.append(datasetName+","+classifierName+",train\n");
        out.append(paramInfo+"\n");
        out.append(cvAcc+"\n");
        for(int i =0; i < instances.numInstances(); i++){
            out.append(instances.instance(i).classValue()+","+preds[i]+"\n");
        }
        out.close();
        
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        return distributionForInstance(instance, null);
    }
    
    private double[] distributionForInstance(Instance instance, StringBuilder[] outputFileBuilders) throws Exception{
        
        if(outputFileBuilders!=null && outputFileBuilders.length!=(modules.length+1)){
            throw new Exception("Error: to write test files, there must be m+1 output StringBuilders (where m is the number of modules)");
        }
        
        double[] hiveDists = new double[instance.numClasses()];
        double[] moduleDists;
        double moduleWeight;
        
        double bsfClassVal,bsfClassWeight;
        StringBuilder moduleString;
        
        double cvAccSum = 0;
        for(int m = 0; m < modules.length; m++){
            moduleDists = modules[m].classifier.distributionForInstance(instance);
            moduleString = new StringBuilder();
            moduleWeight = modules[m].ensembleCvAcc;
                        
            bsfClassVal = -1;
            bsfClassWeight = -1;

            for(int c = 0; c < hiveDists.length; c++){
                hiveDists[c] += moduleDists[c]*moduleWeight;
                if(outputFileBuilders!=null){
                    if(moduleDists[c] > bsfClassWeight){
                        bsfClassWeight = moduleDists[c];
                        bsfClassVal = c;
                    }
                    moduleString.append(",").append(moduleDists[c]);
                    
                    
                }
            }
            if(outputFileBuilders!=null){
                outputFileBuilders[m].append(instance.classValue()).append(",").append(bsfClassVal).append(",").append(moduleString.toString()+"\n");
            }
            cvAccSum+=modules[m].ensembleCvAcc;
        }
        
        for(int h = 0; h < hiveDists.length; h++){
            hiveDists[h]/=cvAccSum;
        }
        
        if(outputFileBuilders!=null){
            
            bsfClassVal = -1;
            bsfClassWeight = -1;
            moduleString = new StringBuilder();
            for(int c = 0; c < hiveDists.length; c++){
                if(hiveDists[c] > bsfClassWeight){
                    bsfClassWeight = hiveDists[c];
                    bsfClassVal = c;
                }
                moduleString.append(",").append(hiveDists[c]);
            }
            outputFileBuilders[outputFileBuilders.length-1].append(instance.classValue()).append(",").append(bsfClassVal).append(",").append(moduleString.toString()+"\n");
        }
                   
        return hiveDists;
    }
    
    
    public double[] classifyInstanceByEnsemble(Instance instance) throws Exception{
        
        double[] output = new double[modules.length];
        
        for(int m = 0; m < modules.length; m++){
            output[m] = modules[m].classifier.classifyInstance(instance);
        }
        return output;
    }
    
    public void printModuleCvAccs() throws Exception{
        if(this.modules==null){
            throw new Exception("Error: modules don't exist. Train classifier first.");
        }
        System.out.println("CV accs by module:");
        System.out.println("------------------");
        StringBuilder line1 = new StringBuilder();
        StringBuilder line2 = new StringBuilder();
        for (ConstituentHiveEnsemble module : modules) {
            line1.append(module.classifierName).append(",");
            line2.append(module.ensembleCvAcc).append(",");
        }
        System.out.println(line1);
        System.out.println(line2);
        System.out.println();
    }
    
    public void makeShouty(){
        this.verbose = true;
    }
    public void setShouty(boolean b){
        this.verbose = b;
    }
    
    private void optionalOutputLine(String message){
        if(this.verbose){
            System.out.println(message);
        }
    }
    
    public void setMaxCvFolds(int maxFolds){
        this.maxCvFolds = maxFolds;
    }
    
    
    public void writeTestPredictionsToFile(Instances test, String outputDir, String datasetName) throws Exception{
        writeTestPredictionsToFile(test, outputDir, datasetName, "0");
    }
    public void writeTestPredictionsToFile(Instances test, String outputDir, String datasetName, String datasetResampleIdentifier) throws Exception{
        
        this.fileOutputDir = outputDir;
        this.fileOutputDataset = datasetName;
        this.fileOutputResampleId = datasetResampleIdentifier;
        
        
        StringBuilder[] outputs = new StringBuilder[this.modules.length+1];
        for(int m = 0; m < outputs.length; m++){
            outputs[m] = new StringBuilder();
        }
        
        for(int i = 0; i < test.numInstances(); i++){
            this.distributionForInstance(test.instance(i), outputs);
        }
        
        FileWriter out;
        File dir;
        Scanner scan;
        int correct;
        String lineParts[];
        for(int m = 0; m < modules.length; m++){
            dir  = new File(this.fileOutputDir+modules[m].classifierName+"/Predictions/"+this.fileOutputDataset+"/");
            if(dir.exists()==false){
                dir.mkdirs();
            }
            correct = 0;
            scan = new Scanner(outputs[m].toString());
            scan.useDelimiter("\n");
            while(scan.hasNext()){
                lineParts = scan.next().split(",");
                if(lineParts[0].trim().equalsIgnoreCase(lineParts[1].trim())){
                    correct++;
                }
            }
            scan.close();
            out = new FileWriter(this.fileOutputDir+modules[m].classifierName+"/Predictions/"+this.fileOutputDataset+"/testFold"+this.fileOutputResampleId+".csv");
            out.append(this.fileOutputDataset+","+this.modules[m].classifierName+",test\n");
            out.append("builtInHive\n");
            out.append(((double)correct/test.numInstances())+"\n");
            out.append(outputs[m]);
            out.close();
        }

        correct = 0;
        scan = new Scanner(outputs[outputs.length-1].toString());
        scan.useDelimiter("\n");
        while(scan.hasNext()){
            lineParts = scan.next().split(",");
            if(lineParts[0].trim().equalsIgnoreCase(lineParts[1].trim())){
                correct++;
            }
        }
        scan.close();
        
        
        dir  = new File(this.fileOutputDir+"HIVE-COTE/Predictions/"+this.fileOutputDataset+"/");
        if(!dir.exists()){
            dir.mkdirs();
        }
        out = new FileWriter(this.fileOutputDir+"HIVE-COTE/Predictions/"+this.fileOutputDataset+"/testFold"+this.fileOutputResampleId+".csv");
        out.append(this.fileOutputDataset+",HIVE-COTE,test\nconstituentCvAccs,");
        
        for(int m = 0; m < modules.length; m++){
            out.append(modules[m].classifierName+","+modules[m].ensembleCvAcc+",");
        }
        out.append("\n"+((double)correct/test.numInstances())+"\n");
        out.append("\n"+outputs[outputs.length-1]);
        out.close();
        
    }
    
    
    
    public double crossValidate(Classifier classifier, Instances train, int maxFolds) throws Exception{
        return crossValidateWithFileWriting(classifier, train, maxFolds, null);
    }
    public double crossValidateWithFileWriting(Classifier classifier, Instances train, int maxFolds, String classifierName) throws Exception{
        
        int numFolds = maxFolds;
        if(numFolds <= 1 || numFolds > train.numInstances()){
            numFolds = train.numInstances();
        }

        ArrayList<Instances> folds = new ArrayList<>();
        ArrayList<ArrayList<Integer>> foldIndexing = new ArrayList<>();

        for(int i = 0; i < numFolds; i++){
            folds.add(new Instances(train,0));
            foldIndexing.add(new ArrayList<>());
        }

        ArrayList<Integer> instanceIds = new ArrayList<>();
        for(int i = 0; i < train.numInstances(); i++){
            instanceIds.add(i);
        }
        Collections.shuffle(instanceIds, rand);

        ArrayList<Instances> byClass = new ArrayList<>();
        ArrayList<ArrayList<Integer>> byClassIndices = new ArrayList<>();
        for(int i = 0; i < train.numClasses(); i++){
            byClass.add(new Instances(train,0));
            byClassIndices.add(new ArrayList<>());
        }

        int thisInstanceId;
        double thisClassVal;
        for(int i = 0; i < train.numInstances(); i++){
            thisInstanceId = instanceIds.get(i);
            thisClassVal = train.instance(thisInstanceId).classValue();

            byClass.get((int)thisClassVal).add(train.instance(thisInstanceId));
            byClassIndices.get((int)thisClassVal).add(thisInstanceId);
        }

         // now stratify        
        Instances strat = new Instances(train,0);
        ArrayList<Integer> stratIndices = new ArrayList<>();
        int stratCount = 0;
        int[] classCounters = new int[train.numClasses()];

        while(stratCount < train.numInstances()){

            for(int c = 0; c < train.numClasses(); c++){
                if(classCounters[c] < byClass.get(c).size()){
                    strat.add(byClass.get(c).instance(classCounters[c]));
                    stratIndices.add(byClassIndices.get(c).get(classCounters[c]));
                    classCounters[c]++;
                    stratCount++;
                }
            }
        }


        train = strat;
        instanceIds = stratIndices;

        double foldSize = (double)train.numInstances()/numFolds;

        double thisSum = 0;
        double lastSum = 0;
        int floor;
        int foldSum = 0;


        int currentStart = 0;
        for(int f = 0; f < numFolds; f++){


            thisSum = lastSum+foldSize+0.000000000001;  // to try and avoid double imprecision errors (shouldn't ever be big enough to effect folds when double imprecision isn't an issue)
            floor = (int)thisSum;

            if(f==numFolds-1){
                floor = train.numInstances(); // to make sure all instances are allocated in case of double imprecision causing one to go missing
            }

            for(int i = currentStart; i < floor; i++){
                folds.get(f).add(train.instance(i));
                foldIndexing.get(f).add(instanceIds.get(i));
            }

            foldSum+=(floor-currentStart);
            currentStart = floor;
            lastSum = thisSum;
        }

        if(foldSum!=train.numInstances()){
            throw new Exception("Error! Some instances got lost file creating folds (maybe a double precision bug). Training instances contains "+train.numInstances()+", but the sum of the training folds is "+foldSum);
        }


        Instances trainLoocv;
        Instances testLoocv;

        double pred, actual;
        double[] predictions = new double[train.numInstances()];

        int correct = 0;
        Instances temp; // had to add in redundant instance storage so we don't keep killing the base set of Instances by mistake

        for(int testFold = 0; testFold < numFolds; testFold++){

            trainLoocv = null;
            testLoocv = new Instances(folds.get(testFold));

            for(int f = 0; f < numFolds; f++){
                if(f==testFold){
                    continue;
                }
                temp = new Instances(folds.get(f));
                if(trainLoocv==null){
                    trainLoocv = temp;
                }else{
                    trainLoocv.addAll(temp);
                }
            }

            classifier.buildClassifier(trainLoocv);
            for(int i = 0; i < testLoocv.numInstances(); i++){
                pred = classifier.classifyInstance(testLoocv.instance(i));
                actual = testLoocv.instance(i).classValue();
                predictions[foldIndexing.get(testFold).get(i)] = pred;
                if(pred==actual){
                    correct++;
                }
            }
        }
        
        double cvAcc = (double)correct/train.numInstances();
        if(this.fileWriting){   
            String outputFilePathAndName = fileOutputDir+classifierName+"/Predictions/"+this.fileOutputDataset+"/trainFold"+this.fileOutputResampleId+".csv"; 
            genericCvResultsFileWriter(outputFilePathAndName, train, predictions, this.fileOutputDataset, classifierName, "genericInternalCv,numFolds,"+numFolds, cvAcc);
        }
    
        return cvAcc;
//        return predictions;

  


    }

    @Override
    public String getParameters() {
        String str=super.getParameters();
        str+=",NumModules,"+classifiers.size();
        for(String s:names)
            str+=","+s;
        str+=",trainAccEstimate";
        for(ConstituentHiveEnsemble m:modules)
            str+=","+m.ensembleCvAcc;
        return str;
    }
    @Override
    public void setTrainTimeLimit(long amount) {
//Split the time up equally if contracted, if not we have no control    
        trainTimeContract =true;
        contractTimeNanos=amount;
        long used=0;
        for(Classifier c:classifiers){
            if(c instanceof TrainTimeContractable)
                ((TrainTimeContractable) c).setTrainTimeLimit(amount/classifiers.size());
            used+=amount/classifiers.size();    
        }
        long remaining = amount-used;
//Give any extra to first contracted, 
//for no real reason othe than simplicity and to avoid hidden randomization.       
        if(remaining>0){
            for(Classifier c:classifiers){
                if(c instanceof TrainTimeContractable){
                    ((TrainTimeContractable) c).setTrainTimeLimit(amount/classifiers.size()+remaining);
                    break;
                }
            }
        }
    }

    /**
     * Parses a given list of options to set the parameters of the classifier.
     * We use this for the tuning mechanism, setting parameters through setOptions
     <!-- options-start -->
     * Valid options are: <p/>
     * <pre> -T
     * Number of trees.</pre>
     *
     * <pre> -I
     * Number of intervals to fit.</pre>
     *
     <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */

        private class ConstituentHiveEnsemble{

        public final Classifier classifier;
        public final double ensembleCvAcc;
        public final String classifierName;

        public ConstituentHiveEnsemble(String classifierName, Classifier classifier, double ensembleCvAcc){
            this.classifierName = classifierName;
            this.classifier = classifier;
            this.ensembleCvAcc = ensembleCvAcc;
        }
    }
    
  
    
    public static class DefaultShapeletFilterPlaceholder extends ShapeletFilter {}
    
    public static void main(String[] args) throws Exception{
       
//        System.out.println(ClassifierTools.testUtils_getIPDAcc(new HiveCote()));
        
        String dataDir = "C:/users/ajb/dropbox/Code2019/tsml/src/main/java/experiments/data/tsc/";
        String datasetName = "Chinatown";
        Instances train = DatasetLoading.loadDataNullable(dataDir+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = DatasetLoading.loadDataNullable(dataDir+datasetName+"/"+datasetName+"_TEST");
        
        HiveCote hive = new HiveCote();
        System.out.println("Example usage of HiveCote: this is the code used in the paper");
        System.out.println(hive.getTechnicalInformation().toString());
        System.out.println("Evaluated on "+datasetName);
        hive.makeShouty();
        hive.buildClassifier(train);
        System.out.println("Classifier built: Parameter info ="+hive.getParameters());
        double a=ClassifierTools.accuracy(test, hive);
        System.out.println("Test acc for "+datasetName+" = "+a);


        System.out.println("This is exiting now. After here in main is legacy code. Ask Jason Lines!");
        System.exit(0);
        hive.writeTestPredictionsToFile(test, "prototypeSheets/", datasetName, "0");
        
        int correct = 0;
        double[] predByEnsemble;
        int[] correctByEnsemble = new int[hive.modules.length];
        for(int i = 0; i < test.numInstances(); i++){
            if(hive.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                correct++;
            }
            predByEnsemble = hive.classifyInstanceByEnsemble(test.instance(i)); // not efficient, just informative. can add this in to the classifyInstance in a hacky way later if need be
            for(int m = 0; m < predByEnsemble.length; m++){
                if(predByEnsemble[m]==test.instance(i).classValue()){
                    correctByEnsemble[m]++;
                }
            }
        }
        System.out.println("Overall Acc: "+(double)correct/test.numInstances());
        System.out.println("Acc by Module:");
    
        StringBuilder line1 = new StringBuilder();
        StringBuilder line2 = new StringBuilder();
        for(int m = 0; m < hive.modules.length; m++){
            line1.append(hive.modules[m].classifierName).append(",");
            line2.append((double)correctByEnsemble[m]/test.numInstances()).append(",");
        }
        System.out.println(line1);
        System.out.println(line2);
    }
    
}
