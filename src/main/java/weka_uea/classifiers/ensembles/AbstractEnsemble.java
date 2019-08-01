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

package weka_uea.classifiers.ensembles;

import evaluation.evaluators.Evaluator;
import evaluation.evaluators.SamplingEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.MultiThreadable;
import timeseriesweka.classifiers.SaveParameterInfo;
import timeseriesweka.classifiers.TrainAccuracyEstimator;
import timeseriesweka.classifiers.TrainTimeContractable;
import utilities.DebugPrinting;
import utilities.ErrorReport;
import static utilities.GenericTools.indexOfMax;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka_uea.classifiers.ensembles.voting.ModuleVotingScheme;
import weka_uea.classifiers.ensembles.weightings.ModuleWeightingScheme;

/**
 * This class defines the base functionality for an ensemble of Classifiers. 
 * 
 * Given a 
 *      - classifier list (Classifiers); 
 *      - a method of estimating error on a dataset (SamplingEvaluator);
 *      - a method of weighting classifier outputs if needed (ModuleWeightingScheme, EqualWeighting if not needed);
 *      = and a method of combining classifier outputs (ModuleVotingScheme);
 * 
 * Extensions of this class will form an ensemble with all standard Classifier functionality,
 * as well as the following: 
 * 
 *      Current functionality
 *          - TrainAccuracyEstimator
 *          - Optional filewriting for individuals' and ensemble's results
 *          - Can train from scratch, or build on results saved to file in ClassifierResults format
 * 
 *      Planned functionality
 *          - Threading the ensemble/components
 *          - Contracting the ensemble/components
 *          - Checkpointing the ensemble/components
 * 
 * TODO Expand javadoc
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public abstract class AbstractEnsemble extends AbstractClassifier implements SaveParameterInfo, DebugPrinting, TrainAccuracyEstimator, MultiThreadable, TrainTimeContractable {

    //Main ensemble design decisions/variables
    protected String ensembleName;
    protected ModuleWeightingScheme weightingScheme;
    protected ModuleVotingScheme votingScheme;
    protected EnsembleModule[] modules;
    protected SamplingEvaluator trainEstimator;
    protected SimpleBatchFilter transform;

    protected int seed = 0;
    protected Instances trainInsts;
    protected boolean estimateEnsemblePerformance = true;

    protected ClassifierResults ensembleTrainResults;//data generated during buildclassifier if above = true
    protected ClassifierResults ensembleTestResults;//data generated during testing

    //saved after building so that it can be added to our test results, even if for some reason 
    //we're not building/writing train results
    protected long buildTime = -1; 
    
    //data info
    protected int numTrainInsts;
    protected int numAttributes;
    protected int numClasses;
    protected int testInstCounter = 0;
    protected int numTestInsts = -1;
    protected Instance prevTestInstance;

    //results file handling
    protected boolean writeEnsembleTrainingFile = false;
    protected boolean readIndividualsResults = false;
    protected boolean writeIndividualsResults = false;
    protected boolean resultsFilesParametersInitialised;
    
    //MultiThreadable
    protected int numThreads = 1;
    protected boolean multiThread = false;
    
    //TrainTimeContractable
    protected boolean contractingTrainTime = false;
    protected long contractTrainTime = TimeUnit.DAYS.toNanos(7); // if contracting with no time limit given, default to 7 days.
    protected TimeUnit contractTrainTimeUnit = TimeUnit.NANOSECONDS;
    
    /**
     * An annoying compromise to deal with base classfiers that dont produce dists 
     * while getting their train estimate. Off by default, shouldnt be turned on for 
     * mass-experiments, intended for cases where user knows that dists are missing
     * (for BOSS, in this case) but still just wants to get ensemble results anyway... 
     */
    protected boolean fillMissingDistsWithOneHotVectors; 
    
    /**
     * if readResultsFilesDirectories.length == 1, all classifier's results read from that one path
     * else, resultsPaths.length must equal classifiers.length, with each index aligning
     * to the path to read the classifier's results from.
     *
     * e.g to read 2 classifiers from one directory, and another 2 from 2 different directories:
     *
     *     Index |  Paths  | Classifier
     *     --------------------------
     *       0   |  pathA  |   c1
     *       1   |  pathA  |   c2
     *       2   |  pathB  |   c3
     *       3   |  pathC  |   c4
     *
     */
    protected String readResultsFilesDirectories[];

    /**
     * if resultsWritePath is not set, will default to resultsPaths[0]
     * i.e, if only reading from one directory, will write back the chosen results
     * under the same directory. if reading from multiple directories but a particular
     * write path not set, will simply pick the first one given.
     */
    protected String writeResultsFilesDirectory;
    protected int resampleIdentifier;
    protected String datasetName;

    public AbstractEnsemble() {
        setupDefaultSettings();
    }
    
    
    /**
     * Defines the default setup of any particular instantiation of this class, called in the
     * constructor.
     * 
     * Minimum requirements for implementations of this method: 
     *      - A default name for this ensemble, as a String (e.g. "CAWPE", "HIVE-COTE")
     *      - A ModuleWeightingScheme. If classifiers do not need to be weighted,
     *          use EqualWeighting.
     *      - A ModuleVotingScheme, to define how the base classifier outputs should be 
     *          combined. 
     *      - An Evaluator, to define the method of performance estimation on the train 
     *          set. If not required, either 
     *              TODO: set a dummy evaluator that performs no real work
     *              TODO: or leave as null, so long as the ModuleWeightingScheme does not require train estimates
     *      - Classifiers and their names, passed to setClassifiers(...)
     * 
     * See CAWPE and HIVE-COTE for examples of particular instantiations of this method.
     */
    public abstract void setupDefaultSettings();
    
    
    
    
    
    
    
    
    
    
    
    
    
    public Classifier[] getClassifiers(){
        Classifier[] classifiers = new Classifier[modules.length];
        for (int i = 0; i < modules.length; i++)
            classifiers[i] = modules[i].getClassifier();
        return classifiers;
    }

    public void setClassifiersNamesForFileRead(String[] classifierNames) {
        setClassifiers(null,classifierNames,null);

    }

    /**
     * If building CAWPE from scratch, the minimum requirement for running is the
 classifiers array, the others could be left null.
     *
     * If building CAWPE from the results files of individuals, the minimum requirement for
 running is the classifierNames list.
     *
     * @param classifiers array of classifiers to use
     * @param classifierNames if null, will use the classifiers' class names by default
     * @param classifierParameters  if null, parameters of each classifier empty by default
     */
    public void setClassifiers(Classifier[] classifiers, String[] classifierNames, String[] classifierParameters) {
        if (classifiers == null) {
            classifiers = new Classifier[classifierNames.length];
            for (int i = 0; i < classifiers.length; i++)
                classifiers[i] = null;
        }

        if (classifierNames == null) {
            classifierNames = new String[classifiers.length];
            for (int i = 0; i < classifiers.length; i++)
                classifierNames[i] = classifiers[i].getClass().getSimpleName();
        }

        if (classifierParameters == null) {
            classifierParameters = new String[classifiers.length];
            for (int i = 0; i < classifiers.length; i++)
                classifierParameters[i] = "";
        }

        this.modules = new EnsembleModule[classifiers.length];
        for (int m = 0; m < modules.length; m++)
            modules[m] = new EnsembleModule(classifierNames[m], classifiers[m], classifierParameters[m]);
    }
    
    public void setEstimateEnsemblePerformance(boolean b) {
        estimateEnsemblePerformance = b;
    }

    public void setRandSeed(int seed){
        this.seed = seed;
    }
    
    
    protected void initialiseModules() throws Exception {
        //currently will only have file reading ON or OFF (not load some files, train the rest)
        //having that creates many, many, many annoying issues, especially when classifying test cases
        if (readIndividualsResults) {
            if (!resultsFilesParametersInitialised)
                throw new Exception("Trying to load CAWPE modules from file, but parameters for results file reading have not been initialised");
            loadModules(); //will throw exception if a module cannot be loaded (rather than e.g training that individual instead)
        }
        else
            trainModules();

        for (EnsembleModule module : modules) {
            //in case train results didnt have probability distributions, hack for old hive cote results tony todo clean
            module.trainResults.setNumClasses(trainInsts.numClasses());
            if (fillMissingDistsWithOneHotVectors)
                module.trainResults.populateMissingDists();
                        
            module.trainResults.findAllStatsOnce();
        }
    }

//    protected void trainModules_unThreaded() throws Exception {
//
//        for (EnsembleModule module : modules) {
//            if (module.getClassifier() instanceof TrainAccuracyEstimator) {
//                module.getClassifier().buildClassifier(trainInsts);
//
//                //these train results should also include the buildtime
//                module.trainResults = ((TrainAccuracyEstimator)module.getClassifier()).getTrainResults();
//                module.trainResults.finaliseResults();
//                
//                // TODO: should errorEstimateTime be forced to zero? by the intention of the interface,
//                // the estimate should have been produced during the normal process of building
//                // the classifier, but depending on how it was programmatically produced, 
//                // the reported estimate time may have already been accounted for in the 
//                // build time. Investigate when use cases arise
//                
//                if (writeIndividualsResults) { //if we're doing trainFold# file writing
//                    String params = module.getParameters();
//                    if (module.getClassifier() instanceof SaveParameterInfo)
//                        params = ((SaveParameterInfo)module.getClassifier()).getParameters();
//                    writeResultsFile(module.getModuleName(), params, module.trainResults, "train"); //write results out
//                    printlnDebug(module.getModuleName() + " writing train file data gotten through TrainAccuracyEstimate...");
//                }
//            }
//            else {
//                printlnDebug(module.getModuleName() + " estimateing performance...");
//                module.trainResults = trainEstimator.evaluate(module.getClassifier(), trainInsts);
//                module.trainResults.finaliseResults();
//                
//                //assumption: classifiers that maintain a classifierResults object, which may be the same object that module.trainResults refers to,
//                //and which this subsequent building of the final classifier would tamper with, would have been handled as an instanceof TrainAccuracyEstimate above
//                long startTime = System.nanoTime();
//                module.getClassifier().buildClassifier(trainInsts);
//                module.trainResults.setBuildTime(System.nanoTime() - startTime);
//                module.trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
//
//                if (writeIndividualsResults) { //if we're doing trainFold# file writing
//                    writeResultsFile(module.getModuleName(), module.getParameters(), module.trainResults, "train"); //write results out
//                    printlnDebug(module.getModuleName() + " writing train file with full preds from scratch...");
//                }
//            }
//        }
//    }

    protected synchronized void trainModule_unThreaded(EnsembleModule module) throws Exception {
        //todo give numThreads to module's classifier if it implements Threadable and numthreads > 1 
        
        if (module.getClassifier() instanceof TrainAccuracyEstimator) {
            module.getClassifier().buildClassifier(trainInsts);

            //these train results should also include the buildtime
            module.trainResults = ((TrainAccuracyEstimator)module.getClassifier()).getTrainResults();
            module.trainResults.finaliseResults();

            // TODO: should errorEstimateTime be forced to zero? by the intention of the interface,
            // the estimate should have been produced during the normal process of building
            // the classifier, but depending on how it was programmatically produced, 
            // the reported estimate time may have already been accounted for in the 
            // build time. Investigate when use cases arise
        }
        else {
            printlnDebug(module.getModuleName() + " estimateing performance...");
            module.trainResults = trainEstimator.evaluate(module.getClassifier(), trainInsts);
            module.trainResults.finaliseResults();

            //assumption: classifiers that maintain a classifierResults object, which may be the same object that module.trainResults refers to,
            //and which this subsequent building of the final classifier would tamper with, would have been handled as an instanceof TrainAccuracyEstimate above
            long startTime = System.nanoTime();
            module.getClassifier().buildClassifier(trainInsts);
            module.trainResults.setBuildTime(System.nanoTime() - startTime);
            module.trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        }
    }
        
    protected boolean hasThreadableBaseClassifiers() { 
        for (EnsembleModule module : modules)
            if (module.getClassifier() instanceof MultiThreadable)
                return true;
        return false;
    }
    
    protected void trainModules() throws Exception {
        if (!multiThread) {
            for (EnsembleModule module : modules)
                trainModule_unThreaded(module); //use the much clearer old definition of training modules
        }
        else {
            //TODO can be optimised a lot, but still giving big speedup over single thread
            //so leaving for now.
            //In future, set off trainaccestimators first to run in background, then group 
            //non-trainaccestimators and pass to evaluator in one go, then build 
            //non-trainaccestimators in their own threads, then collect back all estimates from 
            //trainaccestimators
            
            //Give all the threads to the estimator, gather non-self-generated estimates first
            if (trainEstimator instanceof MultiThreadable)
                ((MultiThreadable)trainEstimator).setThreadAllowance(numThreads);
            
            for (int i = 0; i < modules.length; i++) {
                Classifier clf = modules[i].getClassifier();
                if (!(clf instanceof TrainAccuracyEstimator))
                    modules[i].trainResults = trainEstimator.evaluate(clf, trainInsts);
            }
            
            //Have easily threadable estimates
            //Now build all classifiers, individually threaded
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            List<Future<Long>> buildTimes = new ArrayList<>();
            for (int i = 0; i < modules.length; i++) {
                final Classifier clf = modules[i].getClassifier();
                
                Callable<Long> call = () -> {
                    Long buildTime = System.nanoTime();
                    clf.buildClassifier(trainInsts);
                    return System.nanoTime() - buildTime;
                };
                
                buildTimes.add(executor.submit(call));
            }
   
            
            //All built/building, now to wait and get the train estimates of the modules that 
            //didnt have them built separately earlier
            for (int i = 0; i < modules.length; i++) {
                Classifier clf = modules[i].getClassifier();
                if ((clf instanceof TrainAccuracyEstimator)) { 
                    // The trainResults should have the classifiers'own recorded 
                    // build time in it, just need to get the results. This is 
                    // however the easiest way to wait for completion of the thread/build process
                    buildTimes.get(i).get();
                    modules[i].trainResults = ((TrainAccuracyEstimator)clf).getTrainResults();
                }
                else {
                    // Have the estimate results, just need to get the build time
                    modules[i].trainResults.setBuildTime(buildTimes.get(i).get());
                    modules[i].trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
                }
                modules[i].trainResults.finaliseResults();
            }
            
                    
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
        }
        
        for (EnsembleModule module : modules) {
            if (writeIndividualsResults) { //if we're doing trainFold# file writing
                String params = module.getParameters();
                if (module.getClassifier() instanceof SaveParameterInfo)
                    params = ((SaveParameterInfo)module.getClassifier()).getParameters();
                writeResultsFile(module.getModuleName(), params, module.trainResults, "train"); //write results out
                printlnDebug(module.getModuleName() + " writing train file data gotten through TrainAccuracyEstimate...");
            }
        }
    }
    
    
    
    protected void loadModules() throws Exception {
        //will look for all files and report all that are missing, instead of bailing on the first file not found
        //just helps debugging/running experiments a little
        ErrorReport errors = new ErrorReport("Errors while loading modules from file. Directories given: " + Arrays.toString(readResultsFilesDirectories));

        //for each module
        for(int m = 0; m < this.modules.length; m++){
            String readResultsFilesDirectory = readResultsFilesDirectories.length == 1 ? readResultsFilesDirectories[0] : readResultsFilesDirectories[m];

            boolean trainResultsLoaded = false;
            boolean testResultsLoaded = false;

            //try and load in the train/test results for this module
            File moduleTrainResultsFile = findResultsFile(readResultsFilesDirectory, modules[m].getModuleName(), "train");
            if (moduleTrainResultsFile != null) {
                printlnDebug(modules[m].getModuleName() + " train loading... " + moduleTrainResultsFile.getAbsolutePath());

                modules[m].trainResults = new ClassifierResults(moduleTrainResultsFile.getAbsolutePath());
                trainResultsLoaded = true;
            }

            File moduleTestResultsFile = findResultsFile(readResultsFilesDirectory, modules[m].getModuleName(), "test");
            if (moduleTestResultsFile != null) {
                //of course these results not actually used at all during training,
                //only loaded for future use when classifying with ensemble
                printlnDebug(modules[m].getModuleName() + " test loading..." + moduleTestResultsFile.getAbsolutePath());

                modules[m].testResults = new ClassifierResults(moduleTestResultsFile.getAbsolutePath());

                numTestInsts = modules[m].testResults.numInstances();
                testResultsLoaded = true;
            }

            if (!trainResultsLoaded)
                errors.log("\nTRAIN results files for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. ");
            else if (needIndividualTrainPreds() && modules[m].trainResults.getProbabilityDistributions().isEmpty())
                errors.log("\nNo pred/distribution for instance data found in TRAIN results file for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + resampleIdentifier + "'. ");

            if (!testResultsLoaded)
                errors.log("\nTEST results files for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. ");
            else if (modules[m].testResults.numInstances()==0)
                errors.log("\nNo prediction data found in TEST results file for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + resampleIdentifier + "'. ");
        }

        errors.throwIfErrors();
    }

    protected boolean needIndividualTrainPreds() {
        return estimateEnsemblePerformance || weightingScheme.needTrainPreds || votingScheme.needTrainPreds;
    }

    protected File findResultsFile(String readResultsFilesDirectory, String classifierName, String trainOrTest) {
        File file = new File(readResultsFilesDirectory+classifierName+"/Predictions/"+datasetName+"/"+trainOrTest+"Fold"+resampleIdentifier+".csv");
        if(!file.exists() || file.length() == 0)
            return null;
        else
            return file;
    }

    //hack for handling train accuracy estimate. experiments is giving us the full path and filename
    //to write to, instead of just the folder and expecting us to fill in the +classifierName+"/Predictions/"+datasetName+filename;
    //when doing the interface overhaul, sort this stuff out.
    protected void writeEnsembleTrainAccuracyEstimateResultsFile() throws Exception {
        ensembleTrainResults.writeFullResultsToFile(writeResultsFilesDirectory);
    }
    
    protected void writeResultsFile(String classifierName, String parameters, ClassifierResults results, String trainOrTest) throws Exception {
        String fullPath = writeResultsFilesDirectory+classifierName+"/Predictions/"+datasetName;
        new File(fullPath).mkdirs();
        fullPath += "/" + trainOrTest + "Fold" + seed + ".csv";
        
        results.setClassifierName(classifierName);
        results.setDatasetName(datasetName);
        results.setFoldID(seed);
        results.setSplit(trainOrTest);
        
        results.setParas(parameters);
        results.writeFullResultsToFile(fullPath);
    }

    /**
     * must be called (this or the directory ARRAY overload) in order to build ensemble from results files or to write individual's
     * results files
     *
     * exitOnFilesNotFound defines whether the ensemble will simply throw exception/exit if results files
     * arnt found, or will try to carry on (e.g train the classifiers normally)
     */
    public void setResultsFileLocationParameters(String individualResultsFilesDirectory, String datasetName, int resampleIdentifier) {
        resultsFilesParametersInitialised = true;

        this.readResultsFilesDirectories = new String [] {individualResultsFilesDirectory};
        this.datasetName = datasetName;
        this.resampleIdentifier = resampleIdentifier;
    }

    /**
     * must be called (this or the single directory string overload) in order to build ensemble from results files or to write individual's
     * results files
     *
     * exitOnFilesNotFound defines whether the ensemble will simply throw exception/exit if results files
     * arnt found, or will try to carry on (e.g train the classifiers normally)
     */
    public void setResultsFileLocationParameters(String[] individualResultsFilesDirectories, String datasetName, int resampleIdentifier) {
        resultsFilesParametersInitialised = true;

        this.readResultsFilesDirectories = individualResultsFilesDirectories;
        this.datasetName = datasetName;
        this.resampleIdentifier = resampleIdentifier;
    }

    /**
     * if writing results of individuals/ensemble, but want to define a specific folder to write to as opposed to defaulting to the (only or first)
     * reading location
     */
    public void setResultsFileWritingLocation(String writeResultsFilesDirectory) {
        this.writeResultsFilesDirectory = writeResultsFilesDirectory;
    }

    public void setBuildIndividualsFromResultsFiles(boolean b) {
        readIndividualsResults = b;
        if (b)
            writeIndividualsResults = false;
    }

    public void setWriteIndividualsTrainResultsFiles(boolean b) {
        writeIndividualsResults = b;
        if (b)
            readIndividualsResults = false;
    }
    
    protected ClassifierResults estimateEnsemblePerformance(Instances data) throws Exception {
        double actual, pred;
        double[] dist;

        ClassifierResults trainResults = new ClassifierResults(data.numClasses());
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        
        long estimateTimeStart = System.nanoTime();
        
        //for each train inst
        for (int i = 0; i < numTrainInsts; i++) {
            long startTime = System.nanoTime();
            dist = votingScheme.distributionForTrainInstance(modules, i);
            long predTime = System.nanoTime()- startTime; //time for ensemble to form vote
            for (EnsembleModule module : modules) //                 +time for each member's predictions
                predTime += module.trainResults.getPredictionTime(i);

            pred = utilities.GenericTools.indexOfMax(dist);
            actual = data.instance(i).classValue();

            trainResults.turnOffZeroTimingsErrors();
            trainResults.addPrediction(actual, dist, pred, predTime, "");
            trainResults.turnOnZeroTimingsErrors();
        }
        
        long estimateTime = System.nanoTime() - estimateTimeStart;
        for (EnsembleModule module : modules)
            estimateTime += module.trainResults.getErrorEstimateTime();

        trainResults.setClassifierName(ensembleName);
        if (datasetName == null || datasetName.equals(""))
            datasetName = data.relationName();
        trainResults.setDatasetName(datasetName);
        trainResults.setFoldID(seed);
        trainResults.setSplit("train");
        trainResults.setParas(getParameters());
        
        trainResults.setErrorEstimateTime(estimateTime);
        trainResults.setErrorEstimateMethod(modules[0].trainResults.getErrorEstimateMethod());
        
        trainResults.finaliseResults();
        
        return trainResults;
    }

    /**
     * If building individuals from scratch, i.e not read results from files, call this
     * after testing is complete to build each module's testResults (accessible by module.testResults)
     *
     * This will be done internally anyway if writeIndividualTestFiles(...) is called, this method
     * is made public only so that results can be accessed from memory during the same run if wanted
     */
    public void finaliseIndividualModuleTestResults(double[] testSetClassVals) throws Exception {
        for (EnsembleModule module : modules)
            module.testResults.finaliseResults(testSetClassVals); //converts arraylists to double[]s and preps for writing
    }

    /**
     * If building individuals from scratch, i.e not read results from files, call this
     * after testing is complete to build each module's testResults (accessible by module.testResults)
     *
     * This will be done internally anyway if writeIndividualTestFiles(...) is called, this method
     * is made public only so that results can be accessed from memory during the same run if wanted
     */
    public void finaliseEnsembleTestResults(double[] testSetClassVals) throws Exception {
        this.ensembleTestResults.finaliseResults(testSetClassVals);
    }

    /**
     * @param throwExceptionOnFileParamsNotSetProperly added to make experimental code smoother,
     *  i.e if false, can leave the call to writeIndividualTestFiles(...) in even if building from file, and this
     *  function will just do nothing. else if actually intending to write test results files, pass true
     *  for exceptions to be thrown in case of genuine missing parameter settings
     * @throws Exception
     */
    public void writeIndividualTestFiles(double[] testSetClassVals, boolean throwExceptionOnFileParamsNotSetProperly) throws Exception {
        if (!writeIndividualsResults || !resultsFilesParametersInitialised) {
            if (throwExceptionOnFileParamsNotSetProperly)
                throw new Exception("to call writeIndividualTestFiles(), must have called setResultsFileLocationParameters(...) and setWriteIndividualsResultsFiles()");
            else
                return; //do nothing
        }

        finaliseIndividualModuleTestResults(testSetClassVals);

        for (EnsembleModule module : modules)
            writeResultsFile(module.getModuleName(), module.getParameters(), module.testResults, "test");
    }

    /**
     * @param throwExceptionOnFileParamsNotSetProperly added to make experimental code smoother,
     *  i.e if false, can leave the call to writeIndividualTestFiles(...) in even if building from file, and this
     *  function will just do nothing. else if actually intending to write test results files, pass true
     *  for exceptions to be thrown in case of genuine missing parameter settings
     * @throws Exception
     */
    public void writeEnsembleTrainTestFiles(double[] testSetClassVals, boolean throwExceptionOnFileParamsNotSetProperly) throws Exception {
        if (!resultsFilesParametersInitialised) {
            if (throwExceptionOnFileParamsNotSetProperly)
                throw new Exception("to call writeEnsembleTrainTestFiles(), must have called setResultsFileLocationParameters(...)");
            else
                return; //do nothing
        }

        if (ensembleTrainResults != null) //performed trainEstimator
            writeResultsFile(ensembleName, getParameters(), ensembleTrainResults, "train");

        this.ensembleTestResults.finaliseResults(testSetClassVals);
        writeResultsFile(ensembleName, getParameters(), ensembleTestResults, "test");
    }

    public EnsembleModule[] getModules() {
        return modules;
    }

    public SamplingEvaluator getTrainEstimator() {
        return trainEstimator;
    }
    
    public void setTrainEstimator(SamplingEvaluator trainEstimator) {
        this.trainEstimator = trainEstimator;
    }

    public String[] getClassifierNames() {
        String[] classifierNames = new String[modules.length];
        for (int m = 0; m < modules.length; m++)
            classifierNames[m] = modules[m].getModuleName();
        return classifierNames;
    }

    @Override
    public double[] getTrainPreds() {
        return ensembleTrainResults.getPredClassValsAsArray();
    }

    @Override
    public double getTrainAcc() {
        return ensembleTrainResults.getAcc();
    }

    public String getEnsembleName() {
        return ensembleName;
    }

    public void setEnsembleName(String ensembleName) {
        this.ensembleName = ensembleName;
    }

    public boolean getFillMissingDistsWithOneHotVectors() {
        return fillMissingDistsWithOneHotVectors;
    }

    public void setFillMissingDistsWithOneHotVectors(boolean fillMissingDistsWithOneHotVectors) {
        this.fillMissingDistsWithOneHotVectors = fillMissingDistsWithOneHotVectors;
    }
    
    public double[][] getPosteriorIndividualWeights() {
        double[][] weights = new double[modules.length][];
        for (int m = 0; m < modules.length; ++m)
            weights[m] = modules[m].posteriorWeights;

        return weights;
    }

    public ModuleVotingScheme getVotingScheme() {
        return votingScheme;
    }

    public void setVotingScheme(ModuleVotingScheme votingScheme) {
        this.votingScheme = votingScheme;
    }

    public ModuleWeightingScheme getWeightingScheme() {
        return weightingScheme;
    }

    public void setWeightingScheme(ModuleWeightingScheme weightingScheme) {
        this.weightingScheme = weightingScheme;
    }

    public double[] getIndividualAccEstimates() {
        double [] accs = new double[modules.length];
        for (int i = 0; i < modules.length; i++)
            accs[i] = modules[i].trainResults.getAcc();

        return accs;
    }

    public double[] getPriorIndividualWeights() {
        double[] priors = new double[modules.length];
        for (int i = 0; i < modules.length; i++)
            priors[i] = modules[i].priorWeight;
        
        return priors;
    }

    public void setPriorIndividualWeights(double[] priorWeights) throws Exception {
        if (priorWeights.length != modules.length) 
            throw new Exception("Number of prior weights being set (" + priorWeights.length 
                    + ") not equal to the number of modules (" + modules.length + ")");
        
        for (int i = 0; i < modules.length; i++)
            modules[i].priorWeight = priorWeights[i];
    }

    private void setDefaultPriorWeights() {
        for (int i = 0; i < modules.length; i++)
           modules[i].priorWeight = 1.0;
    }

    public double[][] getIndividualEstimatePredictions() {
        double [][] preds = new double[modules.length][];
        for (int i = 0; i < modules.length; i++)
            preds[i] = modules[i].trainResults.getPredClassValsAsArray();
        return preds;
    }

    public SimpleBatchFilter getTransform(){
        return this.transform;
    }

    public void setTransform(SimpleBatchFilter transform){
        this.transform = transform;
    }
    
    @Override //TrainAccuracyEstimate
    public void writeTrainEstimatesToFile(String path) {
        estimateEnsemblePerformance=true;
        writeEnsembleTrainingFile=true;
        
        setResultsFileWritingLocation(path);
    }
    @Override
    public void setFindTrainAccuracyEstimate(boolean estimatePerformance){
        estimateEnsemblePerformance=estimatePerformance;
    }

    @Override
    public boolean findsTrainAccuracyEstimate(){ return estimateEnsemblePerformance;}

    @Override
    public ClassifierResults getTrainResults(){
        return ensembleTrainResults;
    }

    public ClassifierResults getTestResults(){
        return ensembleTestResults;
    }

    @Override
    public String getParameters(){
        StringBuilder out = new StringBuilder();
        out.append(weightingScheme.toString()).append(",").append(votingScheme.toString()).append(",");

        for(int m = 0; m < modules.length; m++){
            out.append(modules[m].getModuleName()).append("(").append(modules[m].priorWeight);
            for (int j = 0; j < modules[m].posteriorWeights.length; ++j)
                out.append("/").append(modules[m].posteriorWeights[j]);
            out.append("),");
        }

        return out.toString();
    }

//    public void readParameters(String paramLine) {
//        String[] classifiers = paramLine.split(",");
//
//        String[] classifierNames = new String[classifiers.length];
//        double[] priorWeights = new double[classifiers.length];
//        double[] postWeights = new double[classifiers.length];
//
//        for (int i = 0; i < classifiers.length; ++i) {
//            String[] parts = classifiers[i].split("(");
//            classifierNames[i] = parts[0];
//            String[] weights = parts[1].split("/");
//            priorWeights[i] = Integer.parseInt(weights[0]);
//            for (int j = 1; j < weights.length; ++j)
//                postWeights[j-1] = Integer.parseInt(weights[j]);
//        }
//
//    }
    
    
    

    @Override
    public void buildClassifier(Instances data) throws Exception {        
        printlnDebug("**ENSEMBLE TRAIN: " + ensembleName + "**");
        
        long startTime = System.nanoTime();

        //housekeeping
        if (resultsFilesParametersInitialised) {
            if (readResultsFilesDirectories.length > 1)
                if (readResultsFilesDirectories.length != modules.length)
                    throw new Exception("Ensemble, " + this.getClass().getSimpleName() + ".buildClassifier: "
                            + "more than one results path given, but number given does not align with the number of classifiers/modules.");

            if (writeResultsFilesDirectory == null)
                writeResultsFilesDirectory = readResultsFilesDirectories[0];
        }
        
        //transform data if specified
        if(this.transform==null){
            this.trainInsts = data;
//            this.trainInsts = new Instances(data);
        }else{
           transform.setInputFormat(data);
           this.trainInsts = Filter.useFilter(data,transform);
        }
          
        //init
        this.numTrainInsts = trainInsts.numInstances();
        this.numClasses = trainInsts.numClasses();
        this.numAttributes = trainInsts.numAttributes();

        //set up modules
        initialiseModules();
        
        //if modules' results are being read in from file, ignore the i/o overhead 
        //of loading the results, we'll sum the actual buildtimes of each module as 
        //reported in the files
        if (readIndividualsResults)
            startTime = System.nanoTime();
        
        //set up ensemble
        weightingScheme.defineWeightings(modules, numClasses);
        votingScheme.trainVotingScheme(modules, numClasses);

        buildTime = System.nanoTime() - startTime;
        if (readIndividualsResults) {
            //we need to sum the modules' reported build time as well as the weight
            //and voting definition time
            for (EnsembleModule module : modules) {
                buildTime += module.trainResults.getBuildTimeInNanos();
                
                //TODO see other todo in trainModules also. Currently working under 
                //assumption that the estimate time is already accounted for in the build
                //time of TrainAccuracyEstimators, i.e. those classifiers that will 
                //estimate their own accuracy during the normal course of training
                if (!(module.getClassifier() instanceof TrainAccuracyEstimator))
                    buildTime += module.trainResults.getErrorEstimateTime();
            }
        }
        
        ensembleTrainResults = new ClassifierResults();
        ensembleTrainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        
        if(estimateEnsemblePerformance)
            ensembleTrainResults = estimateEnsemblePerformance(data); //combine modules to find overall ensemble trainpreds
        
        //HACK FOR CAWPE_EXTENSION PAPER: 
        //since experiments expects us to make a train results object 
        //and for us to record our build time, going to record it here instead of 
        //editting experiments to record the buildtime at that level
        
        //buildTime does not include the ensemble's trainEstimator in any case, only the work required to be ready for testing
        //time unit has been set in estimateEnsemblePerformance(data);
        ensembleTrainResults.turnOffZeroTimingsErrors();
        ensembleTrainResults.setBuildTime(buildTime);
        ensembleTrainResults.turnOnZeroTimingsErrors();
        
        if (writeEnsembleTrainingFile)
            writeEnsembleTrainAccuracyEstimateResultsFile();
        
        this.testInstCounter = 0; //prep for start of testing
        this.prevTestInstance = null;
    }

    

    
    
    
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            transform.setInputFormat(rawContainer);
            Instances converted = Filter.useFilter(rawContainer,transform);
            ins = converted.instance(0);
        }

        if (ensembleTestResults == null || (testInstCounter == 0 && prevTestInstance == null)) {//definitely the first call, not e.g the first inst being classified for the second time
            printlnDebug("\n**TEST**");

            ensembleTestResults = new ClassifierResults(numClasses);
            ensembleTestResults.setTimeUnit(TimeUnit.NANOSECONDS);
            ensembleTestResults.setBuildTime(buildTime);
        }

        if (readIndividualsResults && testInstCounter >= numTestInsts) //if no test files loaded, numTestInsts == -1
            throw new Exception("Received more test instances than expected, when loading test results files, found " + numTestInsts + " test cases");

        double[] dist;
        long startTime = System.nanoTime();
        long predTime;
        if (readIndividualsResults) { //have results loaded from file
            dist = votingScheme.distributionForTestInstance(modules, testInstCounter);
            predTime = System.nanoTime() - startTime; //time for ensemble to form vote
            for (EnsembleModule module : modules) //            +time for each member's predictions
                predTime += module.testResults.getPredictionTime(testInstCounter);
        }
        else {//need to classify them normally
            dist = votingScheme.distributionForInstance(modules, ins);
            predTime = System.nanoTime() - startTime;
        }
        
        ensembleTestResults.turnOffZeroTimingsErrors();
        ensembleTestResults.addPrediction(dist, indexOfMax(dist), predTime, "");
        ensembleTestResults.turnOnZeroTimingsErrors();
        
        if (prevTestInstance != instance)
            ++testInstCounter;
        prevTestInstance = instance;

        return dist;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        return utilities.GenericTools.indexOfMax(dist);
    }

    /**
     * @return the predictions of each individual module, i.e [0] = first module's vote, [1] = second...
     */
    public double[] classifyInstanceByConstituents(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            transform.setInputFormat(rawContainer);
            Instances converted = Filter.useFilter(rawContainer,transform);
            ins = converted.instance(0);
        }

        double[] predsByClassifier = new double[modules.length];

        for(int i=0;i<modules.length;i++)
            predsByClassifier[i] = modules[i].getClassifier().classifyInstance(ins);

        return predsByClassifier;
    }

    /**
     * @return the distributions of each individual module, i.e [0] = first module's dist, [1] = second...
     */
    public double[][] distributionForInstanceByConstituents(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            transform.setInputFormat(rawContainer);
            Instances converted = Filter.useFilter(rawContainer,transform);
            ins = converted.instance(0);
        }

        double[][] distsByClassifier = new double[this.modules.length][];

        for(int i=0;i<modules.length;i++){
            distsByClassifier[i] = modules[i].getClassifier().distributionForInstance(ins);
        }

        return distsByClassifier;
    }
    
    @Override //MultiThreadable
    public void setThreadAllowance(int numThreads) {
        if (numThreads > 1) {
            this.numThreads = numThreads;
            this.multiThread = true;
        }
        else{
            this.numThreads = 1;
            this.multiThread = false;
        }
    }

    @Override //MultiThreadable
    public int getNumUtilisableThreads() {
        int nThreads = modules.length;
        for (EnsembleModule module : modules)
            if (module.getClassifier() instanceof MultiThreadable)
                nThreads += ((MultiThreadable) module.getClassifier()).getNumUtilisableThreads();
            
        //todo update for evaluator threads if really wanted
        return nThreads;
    }
    
    /**
     * Will split time given evenly among the contractable base classifiers. 
     * 
     * This is currently very naive, and likely innaccurate. Consider these TODO s
     * 
     *  1) If there are any non-contractable base classifiers, these are ignored in 
     *      the contract setting. The full time is allocated among the contractable 
     *      base classifiers, instead of trying to any wonky guessing of how long the 
     *      non-contractable ones might take
     *  2) Currently, generating accuracy estimates (if needed) is not considered in the contract.
     *      If there are any non-TrainAccuracyEstimating classifiers, the estimation procedure (e.g.
     *      a 10fold cv) will very likely overshoot the contract, since the classifier would be
     *      trying to keep to contract on each fold and the full build individually, not in total. 
     *  3) The contract currently does not consider whether the ensemble is being threaded,
     *      i.e. even if it can run the building of two or more classifiers in parallel, 
     *      this will still naively set the contract per classifier as amount/numClassifiers
     */
    @Override //TrainTimeContractable
    public void setTrainTimeLimit(TimeUnit time, long amount) {
        contractingTrainTime=true;
        contractTrainTime = amount;
        contractTrainTimeUnit = time;
        
        int numContractableBaseClassifiers = 0;
        
        for (EnsembleModule module : modules) {
            if(module.getClassifier() instanceof TrainTimeContractable)
                numContractableBaseClassifiers++;
            else 
                System.out.println("WARNING: trying to contract " + ensembleName + ", but base classifier " + module.getModuleName() + " is not contractable");
        }
        
        long timePerClassifier = amount / numContractableBaseClassifiers;
        
        for (EnsembleModule module : modules)
            if(module.getClassifier() instanceof TrainTimeContractable)
                ((TrainTimeContractable) module.getClassifier()).setTrainTimeLimit(time, timePerClassifier);
    }
    
    public String produceEnsembleReport(boolean printPreds, boolean builtFromFile) {
        StringBuilder sb = new StringBuilder();

        sb.append(ensembleName).append(" REPORT");
        sb.append("\nname: ").append(ensembleName);
        sb.append("\nmodules: ").append(modules[0].getModuleName());
        for (int i = 1; i < modules.length; i++)
            sb.append(",").append(modules[i].getModuleName());
        sb.append("\nweight scheme: ").append(weightingScheme);
        sb.append("\nvote scheme: ").append(votingScheme);
        sb.append("\ndataset: ").append(datasetName);
        sb.append("\nfold: ").append(resampleIdentifier);
        sb.append("\ntrain acc: ").append(ensembleTrainResults.getAcc());
        sb.append("\ntest acc: ").append(builtFromFile ? ensembleTestResults.getAcc() : "NA");

        int precision = 4;
        int numWidth = precision+2;
        int trainAccColWidth = 8;
        int priorWeightColWidth = 12;
        int postWeightColWidth = 12;

        String moduleHeaderFormatString = "\n\n%20s | %"+(Math.max(trainAccColWidth, numWidth))+"s | %"+(Math.max(priorWeightColWidth, numWidth))+"s | %"+(Math.max(postWeightColWidth, this.numClasses*(numWidth+2)))+"s";
        String moduleRowHeaderFormatString = "\n%20s | %"+trainAccColWidth+"."+precision+"f | %"+priorWeightColWidth+"."+precision+"f | %"+(Math.max(postWeightColWidth, this.numClasses*(precision+2)))+"s";

        sb.append(String.format(moduleHeaderFormatString, "modules", "trainacc", "priorweights", "postweights"));
        for (EnsembleModule module : modules) {
            String postweights = String.format("  %."+precision+"f", module.posteriorWeights[0]);
            for (int c = 1; c < this.numClasses; c++)
                postweights += String.format(", %."+precision+"f", module.posteriorWeights[c]);

            sb.append(String.format(moduleRowHeaderFormatString, module.getModuleName(), module.trainResults.getAcc(), module.priorWeight, postweights));
        }


        if (printPreds) {
            sb.append("\n\nensemble train preds: ");
            sb.append("\ntrain acc: ").append(ensembleTrainResults.getAcc());
            sb.append("\n");
            for(int i = 0; i < ensembleTrainResults.numInstances();i++)
                sb.append(produceEnsemblePredsLine(true, i)).append("\n");

            sb.append("\n\nensemble test preds: ");
            sb.append("\ntest acc: ").append(builtFromFile ? ensembleTestResults.getAcc() : "NA");
            sb.append("\n");
            for(int i = 0; i < ensembleTestResults.numInstances();i++)
                sb.append(produceEnsemblePredsLine(false, i)).append("\n");
        }

        return sb.toString();
    }

    /**
     * trueClassVal,predClassVal,[empty],dist1,...,distC,#indpreddist1,...,indpreddistC,#module1pred,...,moduleMpred
     * split on "#"
     * [0] = normal results file format (true class, pred class, distforinst)
     * [1] = number of individual unweighted votes per class
     * [2] = the unweighted prediction of each module
     */
    private String produceEnsemblePredsLine(boolean train, int index) {
        StringBuilder sb = new StringBuilder();

        if (train) //pred
            sb.append(modules[0].trainResults.getTrueClassValue(index)).append(",").append(ensembleTrainResults.getPredClassValue(index)).append(",");
        else
            sb.append(modules[0].testResults.getTrueClassValue(index)).append(",").append(ensembleTestResults.getPredClassValue(index)).append(",");

        if (train){ //dist
            double[] pred=ensembleTrainResults.getProbabilityDistribution(index);
            for (int j = 0; j < pred.length; j++)
                sb.append(",").append(pred[j]);
        }
        else{
            double[] pred=ensembleTestResults.getProbabilityDistribution(index);
            for (int j = 0; j < pred.length; j++)
                sb.append(",").append(pred[j]);
        }
        sb.append(",");


        double[] predDist = new double[numClasses]; //indpreddist
        for (int m = 0; m < modules.length; m++) {
            if (train)
                ++predDist[(int)modules[m].trainResults.getPredClassValue(index)];
            else
                ++predDist[(int)modules[m].testResults.getPredClassValue(index)];
        }
        for (int c = 0; c < numClasses; c++)
            sb.append(",").append(predDist[c]);
        sb.append(",");

        for (int m = 0; m < modules.length; m++) {
            if (train)
                sb.append(",").append(modules[m].trainResults.getPredClassValue(index));
            else
                sb.append(",").append(modules[m].testResults.getPredClassValue(index));
        }

        return sb.toString();
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    protected static void testBuildingInds(int testID) throws Exception {
        System.out.println("testBuildingInds()");

        (new File("C:/Temp/EnsembleTests"+testID+"/")).mkdirs();

        int numFolds = 5;

        for (int fold = 0; fold < numFolds; fold++) {
            String dataset = "breast-cancer-wisc-prog";
    //        String dataset = "ItalyPowerDemand";

            Instances all = DatasetLoading.loadDataNullable("C:/UCI Problems/"+dataset+"/"+dataset);
    //        Instances train = ClassifierTools.loadDataThrowable("C:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
    //        Instances test = ClassifierTools.loadDataThrowable("C:/tsc problems/"+dataset+"/"+dataset+"_TEST");

            Instances[] insts = InstanceTools.resampleInstances(all, fold, 0.5);
            Instances train = insts[0];
            Instances test = insts[1];

            CAWPE cawpe = new CAWPE();
            cawpe.setResultsFileLocationParameters("C:/Temp/EnsembleTests"+testID+"/", dataset, fold);
            cawpe.setWriteIndividualsTrainResultsFiles(true);
            cawpe.setEstimateEnsemblePerformance(true); //now defaults to true
            cawpe.setRandSeed(fold);

            cawpe.buildClassifier(train);

            double acc = .0;
            for (Instance instance : test) {
                if (instance.classValue() == cawpe.classifyInstance(instance))
                    acc++;
            }
            acc/=test.numInstances();

            cawpe.writeIndividualTestFiles(test.attributeToDoubleArray(test.classIndex()), true);
            cawpe.writeEnsembleTrainTestFiles(test.attributeToDoubleArray(test.classIndex()), true);

            System.out.println("TrainAcc="+cawpe.getTrainResults().getAcc());
            System.out.println("TestAcc="+cawpe.getTestResults().getAcc());
        }
    }

    protected static void testLoadingInds(int testID) throws Exception {
        System.out.println("testBuildingInds()");

        (new File("C:/Temp/EnsembleTests"+testID+"/")).mkdirs();

        int numFolds = 5;

        for (int fold = 0; fold < numFolds; fold++) {
            String dataset = "breast-cancer-wisc-prog";
    //        String dataset = "ItalyPowerDemand";

            Instances all = DatasetLoading.loadDataNullable("C:/UCI Problems/"+dataset+"/"+dataset);
    //        Instances train = ClassifierTools.loadDataThrowable("C:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
    //        Instances test = ClassifierTools.loadDataThrowable("C:/tsc problems/"+dataset+"/"+dataset+"_TEST");

            Instances[] insts = InstanceTools.resampleInstances(all, fold, 0.5);
            Instances train = insts[0];
            Instances test = insts[1];

            CAWPE cawpe = new CAWPE();
            cawpe.setResultsFileLocationParameters("C:/Temp/EnsembleTests"+testID+"/", dataset, fold);
            cawpe.setBuildIndividualsFromResultsFiles(true);
            cawpe.setEstimateEnsemblePerformance(true); //now defaults to true
            cawpe.setRandSeed(fold);

            cawpe.buildClassifier(train);

            double acc = .0;
            for (Instance instance : test) {
                if (instance.classValue() == cawpe.classifyInstance(instance))
                    acc++;
            }
            acc/=test.numInstances();
            cawpe.finaliseEnsembleTestResults(test.attributeToDoubleArray(test.classIndex()));

            System.out.println("TrainAcc="+cawpe.getTrainResults().getAcc());
            System.out.println("TestAcc="+cawpe.getTestResults().getAcc());
        }
    }


}
