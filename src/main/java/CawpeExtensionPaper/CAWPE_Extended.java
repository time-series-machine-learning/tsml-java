/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package CawpeExtensionPaper;

import CawpeExtensionPaper.CAWPEClassifierList;
import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.Experiments;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import utilities.TrainAccuracyEstimate;
import vector_classifiers.CAWPE;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Class for testing CAWPE while keeping the models trained on the cross validation 
 * folds of the error estimates of each member
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPE_Extended extends CAWPE {
    
    /**
     * intended to be a copy of the raw modules (not expanded from modules built on each cv fold) 
     * such that if cawpe is rebuilt on subsequent datasets, the first step of build classifier
     * can be to replace the list of expanded modules with the core set again
     */
    private EnsembleModule[] coreModules = null;
    
    /**
     * if true, the core modules shall be retrained on the full train set and be part of 
     * the final ensemble. otherwise, ONLY the 'foldClassifiers' will be in the ensemble
     * e.g. if modules.length = 5 and numCVFolds = 10, final ensemble size would be (5*10) + 5 = 55 
     * if retrainOnFullTrainSet = true, else just (5*10) = 50 if false
     */
    private boolean retrainOnFullTrainSet = true;

    public static final Function<Double, Double> priorScheme_none = (numCVFolds) -> 1.0; //input ignored
    public static final Function<Double, Double> priorScheme_oneOverNumFolds = (numCVFolds) -> { return 1.0 / numCVFolds; };
    
    /**
     * While the core modules trained on the full train set will maintain a 
     * prior weighting of 1.0, sub modules will be given a prior weighting according 
     * to this function
     */
    private Function<Double, Double> subModulePriorWeightingScheme;

    public CAWPE_Extended() {
        this.ensembleIdentifier = "CAWPE_Extended";
        this.transform = null;
        this.setDefaultCAWPESettings();
      
        //modules set in setDefaultCAWPESettings()
        coreModules = Arrays.copyOf(modules, modules.length);
        subModulePriorWeightingScheme = CAWPE_Extended.priorScheme_oneOverNumFolds;
    }
    
    public boolean getRetrainOnFullTrainSet() {
        return retrainOnFullTrainSet;
    }

    public void setRetrainOnFullTrainSet(boolean retrainOnFullTrainSet) {
        this.retrainOnFullTrainSet = retrainOnFullTrainSet;
    }

    public Function<Double, Double> getSubModulePriorWeightingScheme() {
        return subModulePriorWeightingScheme;
    }

    public void setSubModulePriorWeightingScheme(Function<Double, Double> subModulePriorWeightingScheme) {
        this.subModulePriorWeightingScheme = subModulePriorWeightingScheme;
    }
        
    @Override
    public void setClassifiers(Classifier[] classifiers, String[] classifierNames, String[] classifierParameters) {
        super.setClassifiers(classifiers, classifierNames, classifierParameters);
        coreModules = Arrays.copyOf(modules, modules.length);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        modules = coreModules;
        
        long startTime = System.nanoTime();
        super.buildClassifier(data);
        buildTime = System.nanoTime() - startTime;
        
        //mega hack, since experiments expects us to make a train results object 
        //and for us to record our build time, going to record it here instead of 
        //editting experiments to record the buildtime at that level
        ensembleTrainResults = new ClassifierResults();
        ensembleTrainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        ensembleTrainResults.turnOffZeroTimingsErrors();
        ensembleTrainResults.setBuildTime(buildTime);
        ensembleTrainResults.turnOnZeroTimingsErrors();
    }
    
    @Override
    protected void initialiseModules() throws Exception {
        //prep cv
        if (willNeedToDoCV()) {
            //int numFolds = setNumberOfFolds(train); //through TrainAccuracyEstimate interface

            cv = new CrossValidationEvaluator(seed, true, false, true, true); //new in _extended. keep fold classifiers/clone everything
            cv.setNumFolds(numCVFolds);
            cv.buildFolds(trainInsts);
        }

        //currently will only have file reading ON or OFF (not load some files, train the rest)
        //having that creates many, many, many annoying issues, especially when classifying test cases
        if (readIndividualsResults) {
            if (!resultsFilesParametersInitialised)
                throw new Exception("Trying to load CAWPE modules from file, but parameters for results file reading have not been initialised");
            loadModules(); //will throw exception if a module cannot be loaded (rather than e.g training that individual instead)
        }
        else
            trainModules();

        
        for (int m = 0; m < modules.length; m++) { 
            //see javadoc for this bool, hacky insert for handling old results. NOT to be used/turned on by default
            //remove all this garbage when possible
            if (fillMissingDistsWithOneHotVectors) {
                ClassifierResults origres =  modules[m].trainResults;
                
                List<double[]> dists = origres.getProbabilityDistributions();
                if (dists == null || dists.isEmpty() || dists.get(0) == null) { 
                    
                    double[][] newdists = new double[numTrainInsts][];
                    for (int i = 0; i < numTrainInsts; i++) {
                        double[] dist = new double[numClasses];
                        dist[(int) origres.getPredClassValue(i)] = 1.0;
                        newdists[i] = dist;
                    }
                    
                    ClassifierResults replacementRes = new ClassifierResults(
                            origres.getTrueClassValsAsArray(), 
                            origres.getPredClassValsAsArray(), 
                            newdists, 
                            origres.getPredictionTimesAsArray(), 
                            origres.getPredDescriptionsAsArray());
                    
                    replacementRes.setClassifierName(origres.getClassifierName());
                    replacementRes.setDatasetName(origres.getDatasetName());
                    replacementRes.setFoldID(origres.getFoldID());
                    replacementRes.setSplit(origres.getSplit());
                    replacementRes.setTimeUnit(origres.getTimeUnit());
                    replacementRes.setDescription(origres.getDescription());
                    
                    replacementRes.setParas(origres.getParas());
                    
                    replacementRes.setBuildTime(origres.getBuildTime());
                    replacementRes.setTestTime(origres.getTestTime());
                    replacementRes.setBenchmarkTime(origres.getBenchmarkTime());
                    replacementRes.setMemory(origres.getMemory());
                    
                    replacementRes.finaliseResults();
                    
                    modules[m].trainResults = replacementRes;
                }
            }
                        
            //in case train results didnt have probability distributions, hack for old hive cote results tony todo clean
            modules[m].trainResults.setNumClasses(trainInsts.numClasses());
            modules[m].trainResults.findAllStatsOnce();
        }
    }

    @Override
    protected void trainModules() throws Exception {

        EnsembleModule[][] newSubModules = new EnsembleModule[modules.length][];
//        StratifiedResamplesEvaluator cv = new StratifiedResamplesEvaluator(seed, true, false, true, true);
        
        for (int m = 0; m < modules.length; m++) {
            EnsembleModule module = modules[m];
            
            if (module.getClassifier() instanceof TrainAccuracyEstimate) {
                //currently not implemented. 
                //some classifiers, especially those that perform some kind of internal 
                //tuning, will generate their own error estimate, which previously we 
                //were using instead of performing another level of nested cv, since 
                //we weren't getting the estimate for strict statistical correctness, 
                //rather to use as an informal weighting for the classifier's future 
                //predictions

                //since for this application we are only going to be using the simpler base
                //classifiers, this particular functionality is not needed and so I
                //am not goign to update the advanced classifiers to have the functionality
                //to decide to keep their models made on the cv folds.
                
                //i.e/tl;dr : only using the else case
            }
            else {
                printlnDebug(module.getModuleName() + " performing cv...");
                
                module.trainResults = cv.evaluate(module.getClassifier(), trainInsts);

                Classifier[] foldClassifiers = cv.getFoldClassifiers();
                ClassifierResults[] foldResults = cv.getFoldResults();
                
                EnsembleModule[] subModules = new EnsembleModule[cv.getNumFolds()];
                assert(subModules.length == foldClassifiers.length && subModules.length == foldResults.length);
                
                for (int i = 0; i < subModules.length; i++) {
                    subModules[i] = new EnsembleModule(module.getModuleName()+"_cvFold"+i, foldClassifiers[i], module.getParameters());
                    subModules[i].trainResults = foldResults[i];
                    subModules[i].priorWeight = subModulePriorWeightingScheme.apply((double) cv.getNumFolds());
                    
                    if (writeIndividualsResults) { //if we're doing trainFold# file writing
                        writeResultsFile(subModules[i].getModuleName(), subModules[i].getParameters(), subModules[i].trainResults, "train"); //write results out
                        printlnDebug(subModules[i].getModuleName() + " writing submodule train file with full preds from scratch...");
                    }
                }
                
                newSubModules[m] = subModules;
                
                if (retrainOnFullTrainSet) { 
                    long startTime = System.nanoTime();
                    module.getClassifier().buildClassifier(trainInsts);
                    module.trainResults.setBuildTime(System.nanoTime() - startTime);
                    module.trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
                    
                    if (writeIndividualsResults) { //if we're doing trainFold# file writing
                        writeResultsFile(module.getModuleName(), module.getParameters(), module.trainResults, "train"); //write results out
                        printlnDebug(module.getModuleName() + " writing train file with full preds from scratch...");
                    }
                }
            }
        }
        
        //copy across the modules into complete list
        int finalisedLength = modules.length * cv.getNumFolds() + (retrainOnFullTrainSet ? modules.length : 0);
        EnsembleModule[] expandedModules = new EnsembleModule[finalisedLength];
        
        //to index into the new list while taking from various sources, 
        //while the loop vars stay within 0 to the respective source's length
        int globalModuleIndex = 0;

        //pull in the 'foldclassifiers'
        for (int j = 0; j < newSubModules.length; j++)
            for (int k = 0; k < newSubModules[j].length; k++, globalModuleIndex++)
                expandedModules[globalModuleIndex] = newSubModules[j][k];
                
        if (retrainOnFullTrainSet) //the core modules will have been rebuilt on the full train set, add them on
            for (int j = 0; j < modules.length; j++, globalModuleIndex++)
                expandedModules[globalModuleIndex] = modules[j];
        //else ignore them
        
        this.modules = expandedModules;
    }
    
    
    
    public static void main(String[] args) throws Exception {
        test_basic();
//        test_smallComparison();
    }
    
    public static void test_basic() throws Exception { 
        String resLoc = "C:/Temp/cawpeExtensionTests/";
        String dataLoc = "C:/TSC Problems/";
        String dset = "ItalyPowerDemand";
        
        CAWPE_Extended[] classifiers = { new CAWPE_Extended()  }; //, new CAWPE(),
        classifiers[0].performEnsembleCV = false;
        classifiers[0].retrainOnFullTrainSet = false;
//        classifiers[1].performEnsembleCV = false;
        
        
//        Experiments.main(new String[] { "-dp="+dataLoc, "-rp="+resLoc, "-cn=CAWPE", "-dn="+dset, "-f=0" }); //_noRetrain

        int numResamples = 30;
        for (Classifier classifier : classifiers) { 
            for (int resample = 0; resample < numResamples; resample++) {
                Instances[] data = Experiments.sampleDataset(dataLoc, dset, resample);
                classifier.buildClassifier(data[0]);
                SingleTestSetEvaluator testeval = new SingleTestSetEvaluator(resample, true, false);
                System.out.println("\t" + testeval.evaluate(classifier, data[1]).getAcc());
            }
        }
    }
    
    
    public static void test_smallComparison() throws Exception { 
        String resLoc = "C:/Temp/cawpeExtensionTests/";
        String dataLoc = "C:/UCI Problems/";
        String[] dsets = { "hayes-roth", "fertility", "blood", "hepatitis" };
        String[] classifierNames = CAWPEClassifierList.all_ensembles;
        int numResamples = 5;
        
        double[] accs = new double[classifierNames.length];
        
        for (int c = 0; c < classifierNames.length; c++) {
            String classifierName = classifierNames[c];
            System.out.println(classifierName);
            
            for (String dset : dsets) {
                System.out.print("\t"+dset+"\n\t\t");
                
                for (int resample = 0; resample < numResamples; resample++) {
                    Classifier classifier = CAWPEClassifierList.setClassifier(classifierName, resample);
                    Instances[] data = Experiments.sampleDataset(dataLoc, dset, resample);
                    
                    classifier.buildClassifier(data[0]);
                    SingleTestSetEvaluator testeval = new SingleTestSetEvaluator(resample, true, false);
                    
                    double acc = testeval.evaluate(classifier, data[1]).getAcc();
                    accs[c] += acc;
                    System.out.print(acc + ",");
                }
                
                System.out.println("");
            }
            
            accs[c] /= dsets.length * numResamples;
        }
        
        System.out.println("\n\n");
        
        for (int c = 0; c < accs.length; c++) {
            System.out.println(String.format("%-30s %f", classifierNames[c], accs[c]));
        }
    }
}

