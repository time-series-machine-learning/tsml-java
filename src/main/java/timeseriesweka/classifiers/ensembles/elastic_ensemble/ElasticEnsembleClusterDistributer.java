package timeseriesweka.classifiers.ensembles.elastic_ensemble;

import timeseriesweka.classifiers.ElasticEnsemble;
import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.core.Instances;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 * 
 * A class to assist with distributing cross-validation experiments for the 
 * Elastic Ensemble class on an LSF HPC. Methods inclue script making, cv 
 * running, and cv parsing. 
 * 
 * Also includes a clusterMaster method for managing the workflow when passed
 * args remotely. 
 * 
 * Note: class is experimental and not robustly tested (e.g. cluster master 
 */
public class ElasticEnsembleClusterDistributer {
    
    /**
     * A method to create bsub submission scripts for running CV experiments in a distributed environment. 
     * Scripts create array jobs with 100 subjobs, creating a single script to do all 100 param options of 
     * a single classifier on a single resample of a dataset.
     * 
     * @param datasetName the name of the dataset
     * @param resample the resample id to run
     * @param classifier the relevant enum corresponding to the classifier
     * @param instructionBuilder a StringBuilder to concatenate sh instructions. If null, method will just ignore this step
     * @throws Exception 
     */
    public static void scriptMaker_runCv(String datasetName, int resample, ElasticEnsemble.ConstituentClassifiers classifier, StringBuilder instructionBuilder) throws Exception{
        
        String theWholeMess = 
            "#!/bin/csh\n" +
            "\n" +
            "#BSUB -q long-eth\n" +
            "#BSUB -J runCv_"+datasetName+"_"+resample+"_"+classifier+"[1-100]\n" +
            "#BSUB -oo output/runCv_"+datasetName+"_"+resample+"_"+classifier+"_%I.out\n" +
            "#BSUB -eo error/runCv_"+datasetName+"_"+resample+"_"+classifier+"_%I.err\n" +
            "#BSUB -R \"rusage[mem=4000]\"\n" +
            "#BSUB -M 4000\n" +
            "\n" +
            "module add java/jdk1.8.0_51\n" +
            "\n" +
            "java -jar -Xmx4000m TimeSeriesClassification.jar runCv "+datasetName+" "+resample+" "+classifier+" $LSB_JOBINDEX";
        
        File outputDir = new File("scripts_eeCv/");
        outputDir.mkdirs();
        FileWriter out = new FileWriter("scripts_eeCv/"+datasetName+"_"+resample+"_"+classifier+".bsub");
        out.append(theWholeMess);
        out.close();
        if(instructionBuilder!=null){
            instructionBuilder.append("bsub < scripts_eeCv/").append(datasetName).append("_").append(resample).append("_").append(classifier).append(".bsub\n");
        }
    }
    
      
    /**
     * A method to run the CV experiment for a single param id of a measure on a dataset. 
     * NOTE: method does not resample data; this should be done independently of the method
     * (access to test data is necessary for repartitioning the data). resampleId param is
     * purely for file writing purposes
     * 
     * @param train 
     * @param dataName 
     * @param resampleIdentifier
     * @param classifier
     * @param paramId
     * @throws Exception 
     */
    private static void runCv(Instances train, String dataName, int resampleIdentifier, ElasticEnsemble.ConstituentClassifiers classifier, int paramId) throws Exception{
        String resultsDir = "eeResults/";
        if(classifier==ElasticEnsemble.ConstituentClassifiers.DDTW_R1_1NN || classifier == ElasticEnsemble.ConstituentClassifiers.DTW_R1_1NN || classifier == ElasticEnsemble.ConstituentClassifiers.Euclidean_1NN){
            if(paramId > 0){
                return;
            }
        }
        Efficient1NN oneNN = ElasticEnsemble.getClassifier(classifier);
        oneNN.setIndividualCvFileWritingOn(resultsDir, dataName, resampleIdentifier);
        oneNN.loocvAccAndPreds(train, paramId);
    }
    
    /**
     * A method to parse the 100 output files for a dataset/resample/measure 
     * combination. Results in a single file with the CV results of the best
     * paramId for this classifier. Also includes the option to delete old cv 
     * files after parsing to help storage management
     * 
     * @param resultsDir
     * @param dataName
     * @param resampleId
     * @param measureType
     * @param tidyUp boolean to delete the now-redundant 100 param cv files once the best param id file has been written
     * @throws Exception 
     */
    private static void runCv_parseIndividualCvsForBest(String resultsDir, String dataName, int resampleId, ElasticEnsemble.ConstituentClassifiers measureType, boolean tidyUp) throws Exception{

        String cvPath = resultsDir+measureType+"/cv/"+dataName+"/trainFold"+resampleId+"/";
        String parsedPath = resultsDir+measureType+"/Predictions/"+dataName+"/";
        String parsedName = parsedPath+"trainFold"+resampleId+".csv";
        File existingParsed = new File(parsedName);

        if(existingParsed.exists() && existingParsed.length() > 0){
            if(tidyUp){
                deleteDir(new File(resultsDir+measureType+"/cv/"));
            }
            return;
        }

        int expectedParams;
        if(measureType.equals(ElasticEnsemble.ConstituentClassifiers.Euclidean_1NN)||measureType.equals(ElasticEnsemble.ConstituentClassifiers.DTW_R1_1NN)||measureType.equals(ElasticEnsemble.ConstituentClassifiers.DDTW_R1_1NN)){
            expectedParams = 1;
        }else{
            expectedParams =100;
        }

        double acc;
        double bsfAcc = -1;

        Scanner scan;
        File individualCv;
        File bsfParsed = null;
        
        for(int p = 0; p < expectedParams; p++){ // check accuracy of each parameter
            
            individualCv = new File(cvPath+"pid"+p+".csv");
            if(individualCv.exists()==false){
                throw new Exception("error: cv file does not exist - "+individualCv.getAbsolutePath());
            }
            scan = new Scanner(individualCv);
            scan.useDelimiter("\n");
            scan.next();
            scan.next();
            acc = Double.parseDouble(scan.next().trim());
            scan.close();
            if(acc > bsfAcc){
                bsfAcc = acc;
                bsfParsed = new File(cvPath+"pid"+p+".csv");
            }
        }
        
        new File(parsedPath).mkdirs();
        scan = new Scanner(bsfParsed);
        scan.useDelimiter("\n");
        FileWriter out = new FileWriter(parsedName);
        while(scan.hasNext()){
            out.append(scan.next()+"\n");
        }
        out.close();

        if(tidyUp){
            deleteDir(new File(resultsDir+measureType+"/cv/"));
        }
    }
    
    /**
     * Alternative main method written for remote computation that is triggered 
     * when args.length > 0. 
     * @param args
     * @throws Exception 
     */
    public static void clusterMaster(String[] args)throws Exception{
        String arffDir = "Problems/";
        
        if(args[0].equalsIgnoreCase("makeScripts")){
            // do locally for now
        }else if(args[0].equalsIgnoreCase("runCv")){
            String datasetName = args[1].trim();
            int resampleId = Integer.parseInt(args[2].trim());
            String classifier = args[3].trim();
            int paramId = Integer.parseInt(args[4].trim())-1;
            
            Instances train = ClassifierTools.loadData(arffDir+datasetName+"_TRAIN");
            runCv(train, datasetName, resampleId, ElasticEnsemble.ConstituentClassifiers.valueOf(classifier), paramId);
            
        }else if(args[0].equalsIgnoreCase("parseCv")){ 
            String datasetName = args[1].trim();
            String resultsDirName = args[2].trim();
            int resampleId = Integer.parseInt(args[3].trim());

            for(ElasticEnsemble.ConstituentClassifiers c: ElasticEnsemble.ConstituentClassifiers.values()){
                runCv_parseIndividualCvsForBest(resultsDirName, datasetName, resampleId, c, true);
            }
                
        }else if(args[0].equalsIgnoreCase("buildEEandRunTest")){
            String datasetName = args[1].trim();
            String resultsDirName = args[2].trim();
            String arffPath = args[3].trim();
            int resampleId = 0;
            
            
            Instances train = ClassifierTools.loadData(arffPath+datasetName+"/"+datasetName+"_TRAIN");
            Instances test = ClassifierTools.loadData(arffPath+datasetName+"/"+datasetName+"_TEST");
            if(args.length > 4){
                resampleId = Integer.parseInt(args[4].trim());
                Instances temp[] = InstanceTools.resampleTrainAndTestInstances(train, test, resampleId);
                train = temp[0];
                test = temp[1];
            }
            
            ElasticEnsemble ee = new ElasticEnsemble(resultsDirName, datasetName, resampleId);
            ee.buildClassifier(train);
            ee.writeTestResultsToFile(test, datasetName, "EE", ee.getParameters(), resultsDirName+"EE/Predictions/"+datasetName+"/testFold"+resampleId+".csv");
            
        }else{
            throw new Exception("Error: Unexpected operation - " + args[0]);
        }
                           
    }
    
    
    
    /**
     * Utility method to recursively remove a directory and a contents
     * @param dir File object of the directory to be deleted
     */
    private static void deleteDir(File dir){
        if(dir.exists()==false){
            return;
        }
        if(dir.isDirectory()){
            File[] files = dir.listFiles();
            for (File file: files) {
                deleteDir(file);
            }
        }
        dir.delete();
    }
    
    
    /**
     * Main method. When args.length > 0, clusterMaster method is triggered with 
     * args instead of the local main method. 
     * 
     * @param args
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception{
        
        if(args.length>0){
            clusterMaster(args);
            return;
        }
        // else, local:
//        String problemName = "alphabet_raw_26_sampled_10";
        String problemName = "vowel_raw_sampled_10";
        
        StringBuilder instructionBuilder = new StringBuilder();
        for(ElasticEnsemble.ConstituentClassifiers c:ElasticEnsemble.ConstituentClassifiers.values()){
            scriptMaker_runCv(problemName, 0, c, instructionBuilder);
        }
        FileWriter out = new FileWriter("instructions_"+problemName+".txt");
        out.append(instructionBuilder);
        out.close();

    }
    
}
