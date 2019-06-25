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

package vector_classifiers;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.storage.ClassifierResults;
import java.util.List;
import java.util.concurrent.TimeUnit;
import timeseriesweka.classifiers.SaveParameterInfo;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;

/**
 * Class for testing CAWPE while keeping the models trained on the cross validation 
 * folds of the error estimates of each member
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPE_Extended extends CAWPE {
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

    protected void trainModules() throws Exception {

        EnsembleModule[][] newSubModules = new EnsembleModule[modules.length][];
        
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
                module.trainResults = cv.crossValidateWithStats(module.getClassifier(), trainInsts);
                module.trainResults.finaliseResults();
                
                // START MAIN NEW STUFF FOR _EXTENDED
                Classifier[] foldClassifiers = cv.getFoldClassifiers();
                ClassifierResults[] foldResults = cv.getFoldResults();
                
                EnsembleModule[] subModules = new EnsembleModule[cv.getNumFolds()];
                assert(subModules.length == foldClassifiers.length && subModules.length == foldResults.length);
                
                for (int i = 0; i < subModules.length; i++) {
                    subModules[i] = new EnsembleModule(module.getModuleName()+"_cvFold"+i, foldClassifiers[i], module.getParameters());
                    subModules[i].trainResults = foldResults[i];
                }
                
                newSubModules[m] = subModules;
                // END   MAIN NEW STUFF FOR _EXTENDED
                //todo timings, look for anything else that needs to be done for module init, fundtionality for prior weightings
                //  test, finalise results? 
                
                
                //assumption: classifiers that maintain a classifierResults object, which may be the same object that module.trainResults refers to,
                //and which this subsequent building of the final classifier would tamper with, would have been handled as an instanceof TrainAccuracyEstimate above
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
        
        //copy across the modules into complete list
        EnsembleModule[] expandedModules = new EnsembleModule[modules.length + newSubModules.length * newSubModules[0].length];
        
        //to index into the new list while taking from various sources, 
        //while the loop vars stay within 0 to the respective source's length
        int globalModuleIndex = 0;
        for (int j = 0; j < modules.length; j++, globalModuleIndex++)
            expandedModules[globalModuleIndex] = modules[j];
        
        
        for (int j = 0; j < newSubModules.length; j++)
            for (int k = 0; k < newSubModules[j].length; k++, globalModuleIndex++)
                expandedModules[globalModuleIndex] = newSubModules[j][k];
        
        //done
        this.modules = expandedModules;
    }
}
