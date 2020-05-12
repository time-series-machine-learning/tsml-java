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
package evaluation.tuning;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.storage.ClassifierResults;
import evaluation.evaluators.Evaluator;
import evaluation.tuning.searchers.GridSearcher;
import evaluation.tuning.searchers.ParameterSearcher;
import experiments.data.DatasetLoading;
import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import utilities.FileHandlingTools;
import utilities.InstanceTools;
import machine_learning.classifiers.SaveEachParameter;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.TrainTimeContractable;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class Tuner 
        implements SaveEachParameter,Checkpointable, TrainTimeContractable {
    
    //Main 3 design choices.
    private ParameterSearcher searcher;                      //default = new GridSearcher();
    private Evaluator evaluator;                             //default = new CrossValidationEvaluator();
    private Function<ClassifierResults, Double> evalMetric;  //default = ClassifierResults.GETTER_Accuracy;
    
    private ParameterResults bestParaSetAndResults = null;
    
    
    private int seed;
    private String classifierName; //interpreted from simpleClassName(), maybe have getter setter later
    private String datasetName; //interpreted from relationName(), maybe have getter setter later
    
    
    
    ////////// start interface variables
    
    //we're implementing CheckpointClassifier AND SaveEachParameter for now, however for this class checkpointing is 
    //identical to SaveEachParamter, we just implicitely checkpoint after each parameterSet evaluation
    private String parameterSavingPath = null; //SaveEachParameter //CheckpointClassifier
    private boolean saveParameters = false; //SaveEachParameter //CheckpointClassifier
    
    long trainContractTimeNanos; //TrainTimeContractClassifier  //note, leaving in nanos for max fidelity, max val of long = 2^64-1 = 586 years in nanoseconds
    boolean trainTimeContract = false; //TrainTimeContractClassifier
    
    ////////// end interface variables
    
    private boolean includeMarkersInParaLine = true;
    
    /**
     * if true, the base classifier will be cloned in order to evaluate each parameter set 
     * this will prevent any potentially un-handled changes to the classifiers' state after 
     * the previous parameter build/eval affecting the next one. 
     * 
     * if you know that the classifier either has no or correctly re-instantiates any data 
     * that would effect consecutive builds on the same classifier instance, just leave this as 
     * false to save mem/time
     */
    boolean cloneClassifierForEachParameterEval = false;
    
    
    /**
     * if true, the dataset will be cloned in order to evaluate each parameter set 
     * this will prevent any potentially un-handled changes to the dataset caused by the classifier 
     * after each parameter build/eval
     * 
     * if you know that the classifier does not edit the original data (as every classifier should not...) 
     * just leave this as false to save mem/time
     */
    boolean cloneTrainSetForEachParameterEval = false;

    public Tuner() { 
        this(new CrossValidationEvaluator());
    }

    public Tuner(Evaluator evaluator) {
        this.searcher = new GridSearcher();
        this.evaluator = evaluator;
        this.evalMetric = ClassifierResults.GETTER_Accuracy;

        setSeed(0);
    }

    
    //handled by the interface methods now
//    public String getParameterSavingPath() {
//        return parameterSavingPath;
//    }
//
//    public void setParameterSavingPath(String parameterSavingPath) {
//        this.parameterSavingPath = parameterSavingPath;
//    }
    
    public boolean getCloneTrainSetForEachParameterEval() {
        return cloneTrainSetForEachParameterEval;
    }

    public void setCloneTrainSetForEachParameterEval(boolean cloneTrainSetForEachParameterEval) {
        this.cloneTrainSetForEachParameterEval = cloneTrainSetForEachParameterEval;
    }

    public Instances cloneDataIfNeeded(Instances data) {
        if (cloneTrainSetForEachParameterEval)
            return new Instances(data);
        else 
            return data;
    }
    
    public boolean getCloneClassifierForEachParameterEval() {
        return cloneClassifierForEachParameterEval;
    }

    public void setCloneClassifierForEachParameterEval(boolean cloneClassifierForEachParameterEval) {
        this.cloneClassifierForEachParameterEval = cloneClassifierForEachParameterEval;
    }
    
    public AbstractClassifier cloneClassifierIfNeeded(AbstractClassifier classifier) throws Exception {
        if (cloneClassifierForEachParameterEval) {
            //for some reason, the (abstract classifiers)' copy method returns a (classifier interface) reference...
            return (AbstractClassifier)AbstractClassifier.makeCopy(classifier); 
        }
        else {
            //just reuse the same instance of the classifier, assume that no info 
            //that from the previous build/eval affects this one.
            //potentially saves a lot of memory/time etc.
            return classifier;
        }
    }
    
    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
        
        searcher.setSeed(seed);
        evaluator.setSeed(seed);
    }

    public boolean getSaveParameters() {
        return saveParameters;
    }

    public void setSaveParameters(boolean saveParameters) {
        this.saveParameters = saveParameters;
    }

    public boolean getIncludeMarkersInParaLine() {
        return includeMarkersInParaLine;
    }

    public void setIncludeMarkersInParaLine(boolean includeMarkersInParaLine) {
        this.includeMarkersInParaLine = includeMarkersInParaLine;
    }
    
    public ParameterSearcher getSearcher() {
        return searcher;
    }

    public void setSearcher(ParameterSearcher searcher) {
        this.searcher = searcher;
    }

    public Evaluator getEvaluator() {
        return evaluator;
    }

    public void setEvaluator(Evaluator evaluator) {
        this.evaluator = evaluator;
    }

    public Function<ClassifierResults, Double> getEvalMetric() {
        return evalMetric;
    }

    public void setEvalMetric(Function<ClassifierResults, Double> evalMetric) {
        this.evalMetric = evalMetric;
    }
    
    public ClassifierResults evaluateParameterSetByIndex(AbstractClassifier baseClassifier, Instances trainSet, ParameterSpace parameterSpace, int parameterIndex) throws Exception { 
        classifierName = baseClassifier.getClass().getSimpleName();
        datasetName = trainSet.relationName();
        
        searcher.setParameterSpace(parameterSpace);
        Iterator<ParameterSet> iter = searcher.iterator();
        
        //iterate up to the specified parameter
        int id = 0;
        while (iter.hasNext()) {
            ParameterSet pset = iter.next();
            if (id++ == parameterIndex) {
                //para found, evaluate it and return the results
                ClassifierResults results = evaluateParameterSet(baseClassifier, trainSet, pset);
                return results;
            }
        }
        
        return null; //todo, this should probs be an exception throw instead, tbd
    }
    
    public ClassifierResults evaluateParameterSet(AbstractClassifier baseClassifier, Instances trainSet, ParameterSet parameterSet) throws Exception { 
        Instances data = cloneDataIfNeeded(trainSet);
        AbstractClassifier classifier = cloneClassifierIfNeeded(baseClassifier); 
            
        String[] options = parameterSet.toOptionsList();
        classifier.setOptions(options);

        ClassifierResults results = evaluator.evaluate(classifier, data);
        results.setClassifierName("TunedClassifier:"+classifierName);
        results.setDatasetName(datasetName);
        results.setFoldID(seed);
        results.setSplit("train");
        results.setParas(parameterSet.toClassifierResultsParaLine(includeMarkersInParaLine));
        
        return results;
    }
    
    public ParameterResults tune(AbstractClassifier baseClassifier, Instances trainSet, ParameterSpace parameterSpace) throws Exception {
        //System.out.println("Evaluating para space: " + parameterSpace);
        
        //for contracting
        long startTime = System.nanoTime();
        long maxParaEvalTime = 0;
        
        //meta info in case we're saving para files
        classifierName = baseClassifier.getClass().getSimpleName();
        datasetName = trainSet.relationName();

        //init the space searcher
        searcher.setParameterSpace(parameterSpace);
        Iterator<ParameterSet> iter = searcher.iterator();
        
        //for resolving ties for the best paraset
        List<ParameterResults> tiesBestSoFar = new ArrayList<>();
        
        //iterate over the space
        int parameterSetID = -1;
        while (iter.hasNext()) {
            parameterSetID++;
            ParameterSet pset = iter.next();
            long thisParaStartTime = System.nanoTime();
            if (saveParameters && parametersAlreadyEvaluated(parameterSetID))
                continue;
            
            // THE WORK
            ClassifierResults results = evaluateParameterSet(baseClassifier, trainSet, pset);
            
            if (saveParameters)
                saveParaResults(parameterSetID, results);
            else 
                storeParaResult(pset, results, tiesBestSoFar);
            
            if (trainTimeContract) {
                long thisParaTime = System.nanoTime() - thisParaStartTime;
                if (thisParaTime > maxParaEvalTime) 
                    maxParaEvalTime = thisParaTime;
                
                long totalTimeSoFar = System.nanoTime() - startTime;
                
//                int numParasEvald = parameterSetID + 1; 
//                long avgTimePerPara = totalTimeSoFar / numParasEvald;
                
                if (!canWeEvaluateAnotherParaSet(maxParaEvalTime, totalTimeSoFar))
                    break;
            }
            
            //System.out.println("Score: " + String.format("%5f", score) + "\tParas: " + pset);
        }
        
        
        if (saveParameters) {
            // if we're contracting, (but also saving parasets)
            // we might not have had time to eval ALL the psets, justfind the best so far
            // if we're contracting but not saving each paraset, we'll have been using 
            // storeParaResult() and have them in memory currently anyway
            if (trainTimeContract)
                tiesBestSoFar = loadBestOfSavedParas_SoFar();
            else
                tiesBestSoFar = loadBestOfSavedParas_All(parameterSpace.numUniqueParameterSets());
            //conversely if we're NOT contracting, we have the strict requirement that
            //the entire space has been evaluated (or at least has been fully iterated over as defined by the 
            //searcher, e.g RandomSearcher has searched it's full 1000 times etc)
        }
        
        bestParaSetAndResults = resolveTies(tiesBestSoFar);
        //System.out.println("Best parameter set was: " + bestSet);
        
        return bestParaSetAndResults;
    }
    
    private boolean canWeEvaluateAnotherParaSet(long maxParaEvalTime, long totalTimeSoFar) {
        return trainContractTimeNanos - totalTimeSoFar > maxParaEvalTime;
    }
    
    private boolean parametersAlreadyEvaluated(int paraID) {
        return ClassifierResults.exists(parameterSavingPath + buildParaFilename(paraID));
    }   
    
    private String buildParaFilename(int paraID) {
//        return "fold" + seed + "_" +paraID + ".csv";
        //experiments paasses us /path/[classifier]/predictions/[dataset]/fold[seed]_
        return paraID + ".csv";
    }
    
    private void storeParaResult(ParameterSet pset, ClassifierResults results, List<ParameterResults> tiesBestSoFar) {
        double score = evalMetric.apply(results);
            
        ParameterResults paraScore = new ParameterResults(pset, results, score);
        
        if (tiesBestSoFar.isEmpty()) //first time around loop
            tiesBestSoFar.add(paraScore);
        else {
            if (score == tiesBestSoFar.get(0).score) {
                //another tie 
                tiesBestSoFar.add(paraScore);
            }
            else if (score > tiesBestSoFar.get(0).score) {
                //new best so far
                tiesBestSoFar.clear();
                tiesBestSoFar.add(paraScore);
            }
        }
    }
    
    public void saveParaResults(int paraID, ClassifierResults results) throws Exception {
//        File f = new File(parameterSavingPath);
//        if (!f.exists()){ 
//            System.out.println("Creating directory " + parameterSavingPath);
//            f.mkdirs();
//        }
        //experiments paasses us /path/[classifier]/predictions/[dataset]/fold[seed]_
        //so no need to make dir, just add on para id and write
        
        results.writeFullResultsToFile(parameterSavingPath + buildParaFilename(paraID));
    }
    
    /**
     * Loads all the saved parameter results files with the expectation that every parameter set 
     * up to the id# passed has been evaluated (intended usage being that numParasExpected = parameterSpace.numUniqueParameterSets())
     * 
     * populates and returns a list of the ties for best parameterSet
     */
    private List<ParameterResults> loadBestOfSavedParas_All(int numParasExpected) throws Exception {
        List<ParameterResults> tiesBestSoFar = new ArrayList<>();
        
        for (int paraID = 0; paraID < numParasExpected; paraID++) {
            String path = parameterSavingPath + buildParaFilename(paraID);
            
            if (ClassifierResults.exists(path)) {
                ClassifierResults tempResults = new ClassifierResults(path);
                ParameterSet pset = new ParameterSet();
                pset.readClassifierResultsParaLine(tempResults.getParas(), includeMarkersInParaLine);
                storeParaResult(pset, tempResults, tiesBestSoFar);
            } else {
                throw new Exception("Trying to load paras back in, but missing expected parameter set ID: " + paraID + ", numParasExpected: " + numParasExpected);
            }
        }
        
        return tiesBestSoFar;
    }
    
    /**
     * Loads all the saved parameter results files that have been written 'so far',
     * using parameterSavingPath as a search term to look for saved files
     * 
     * populates and returns a list of the ties for best parameterSet
     */
    private List<ParameterResults> loadBestOfSavedParas_SoFar() throws Exception {
        List<ParameterResults> tiesBestSoFar = new ArrayList<>();
        
        //assumption, parameterSavingPath is of form some/long/path/fold[seed]_
        File f = new File(parameterSavingPath);
        String filenamePrefix = f.getName(); // fold[seed]_
        String dir = f.getParent();          // some/long/path/
        
        File[] files = FileHandlingTools.listFilesContaining(dir, filenamePrefix);
        
        for (File file : files) {
            ClassifierResults tempResults = new ClassifierResults(file.getAbsolutePath());
            ParameterSet pset = new ParameterSet();
            pset.readClassifierResultsParaLine(tempResults.getParas(), includeMarkersInParaLine);
            storeParaResult(pset, tempResults, tiesBestSoFar);
        }
        
        return tiesBestSoFar;
    }
    
    private ParameterResults resolveTies(List<ParameterResults> tiesBestSoFar) {
        if (tiesBestSoFar.size() == 1) {
            //clear winner
            return tiesBestSoFar.get(0);
        }
        else { 
            //resolve ties randomly: todo future, maybe allow for some other method of resolving ties, 
            //e.g choose 'least complex' parameter set of the ties
            Random rand = new Random(seed);
            return tiesBestSoFar.get(rand.nextInt(tiesBestSoFar.size()));
        }
    }
    
    public static void main(String[] args) throws Exception {
        int seed = 0;
        
        SMO svm = new SMO();
        PolyKernel p=new PolyKernel();
        p.setExponent(2);
        svm.setKernel(p);
        svm.setRandomSeed(seed);
        
        int size = 5;
        double[] cs = new double[size];
        for (int i = 0; i < cs.length; i++)
            cs[i] = Math.pow(10.0, (i-size/2));
        
        ParameterSpace space = new ParameterSpace();
        space.addParameter("C", cs);
        
        Tuner tuner = new Tuner();
        tuner.setPathToSaveParameters("C:/Temp/TunerTests/first/");
        tuner.setSeed(seed);
        
        String dataset = "hayes-roth";
        Instances all = DatasetLoading.loadDataNullable("Z:\\Data\\UCIDelgado\\"+dataset+"\\"+dataset+".arff");
        Instances[] data = InstanceTools.resampleInstances(all, seed, 0.5);
        
        System.out.println(tuner.tune(svm, data[0], space));
    }

    
    
    
    
    
    
    
    
    
    // METHODS FOR:        SaveEachParameter,CheckpointClassifier,TrainTimeContractClassifier
    

    @Override //SaveEachParameter
    public void setPathToSaveParameters(String r) {
        this.parameterSavingPath = r;
        this.saveParameters = true; 
    }

    @Override //SaveEachParameter
    public void setSaveEachParaAcc(boolean b) {
        this.saveParameters = b;
        //does anywhere set this to true but not give the path?. part of interface cleanup tests
    }

    @Override //Checkpointable
    public boolean setCheckpointPath(String path) {
        boolean validPath=Checkpointable.super.createDirectories(path);
        if(validPath){
            this.parameterSavingPath = path;
            this.saveParameters = true;
        }
        return validPath;
    }

    @Override //CheckpointClassifier
    public void copyFromSerObject(Object obj) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public void setTrainTimeLimit(long amount) {
        trainTimeContract = true;
        trainContractTimeNanos =amount;
    }

}
