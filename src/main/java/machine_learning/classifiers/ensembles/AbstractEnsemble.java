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

package machine_learning.classifiers.ensembles;

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
import java.util.concurrent.TimeUnit;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.MultiThreadable;
import tsml.classifiers.TestTimeContractable;
import tsml.classifiers.TrainTimeContractable;
import tsml.transformers.Transformer;
import utilities.DebugPrinting;
import utilities.ErrorReport;
import utilities.InstanceTools;
import utilities.ThreadingUtilities;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import machine_learning.classifiers.ensembles.voting.ModuleVotingScheme;
import machine_learning.classifiers.ensembles.weightings.ModuleWeightingScheme;

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
 *          - Can estimate own performance on train data
 *          - Optional filewriting for individuals' and ensemble's results
 *          - Can train from scratch, or build on results saved to file in ClassifierResults format
 *          - Can thread the component evaluation/building, current just assigning one thread per base classifier
 * 
 * TODO Expand javadoc
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public abstract class AbstractEnsemble extends EnhancedAbstractClassifier implements DebugPrinting, MultiThreadable {

    //Main ensemble design decisions/variables
    protected String ensembleName;
    protected ModuleWeightingScheme weightingScheme;
    protected ModuleVotingScheme votingScheme;
    protected EnsembleModule[] modules;
    protected SamplingEvaluator trainEstimator;
    protected Transformer transform;

    protected Instances trainInsts;

    //protected ClassifierResults trainResults; inherited from EnhancedAbstractClassifier data generated during buildclassifier if above = true
    protected ClassifierResults testResults;//data generated during testing

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
    protected boolean readIndividualsResults = false;
    protected boolean writeIndividualsResults = false;
    protected boolean resultsFilesParametersInitialised;
    
    //MultiThreadable
    protected int numThreads = 1;
    protected boolean multiThread = false;
        
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
    protected String datasetName;

    /**
     * resampleIdentifier is now deprecated, using ONLY the seed for both fold-file naming purposes and any internal
     * seeding required, e.g tie resolution
     */
    //protected int resampleIdentifier;

    public AbstractEnsemble() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        setupDefaultEnsembleSettings();
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
    public abstract void setupDefaultEnsembleSettings();
    
    
    
    
    
    
    
    
    /**
     * Simple data type to hold a classifier and it's related information and results.
     */
    public static class EnsembleModule implements DebugPrinting {
        private Classifier classifier;

        private String moduleName;
        private String parameters;

        public ClassifierResults trainResults;
        public ClassifierResults testResults;

        //by default (and i imagine in the vast majority of cases) all prior weights are equal (i.e 1)
        //however may be circumstances where certain classifiers are themselves part of 
        //a subensemble or something
        public double priorWeight = 1.0; 

        //each module makes a vote, with a weight defined for this classifier when predicting this class 
        //many weighting schemes will have weights for each class set to a single classifier equal, but some 
        //will have e.g certain members being experts at classifying certain classes etc
        public double[] posteriorWeights;

        public EnsembleModule() {
            this.moduleName = "ensembleModule";
            this.classifier = null;

            trainResults = null;
            testResults = null;
        }

        public EnsembleModule(String moduleName, Classifier classifier, String parameters) {
            this.classifier = classifier;
            this.moduleName = moduleName;
            this.parameters = parameters;

            trainResults = null;
            testResults = null;
        }

        public boolean isAbleToEstimateOwnPerformance() {
            return classifierAbleToEstimateOwnPerformance(classifier);
        }
        
        public boolean isEstimatingOwnPerformance() {
            return classifierIsEstimatingOwnPerformance(classifier);
        }
        
        public boolean isTrainTimeContractable() {
            return classifier instanceof TrainTimeContractable;
        }
        
        public boolean isTestTimeContractable() {
            return classifier instanceof TestTimeContractable;
        }
        
        public boolean isMultiThreadable() {
            return classifier instanceof MultiThreadable;
        }
        
        public boolean isCheckpointable() {
            return classifier instanceof Checkpointable;
        }
        
        public String getModuleName() {
            return moduleName;
        }

        public void setModuleName(String moduleName) {
            this.moduleName = moduleName;
        }

        public String getParameters() {
            return parameters;
        }

        public void setParameters(String parameters) {
            this.parameters = parameters;
        }

        public Classifier getClassifier() {
            return classifier;
        }

        public void setClassifier(Classifier classifier) {
            this.classifier = classifier;
        }

        @Override
        public String toString() {
            return moduleName;
        }
    }

    
    
    
    
    
    public Classifier[] getClassifiers(){
        Classifier[] classifiers = new Classifier[modules.length];
        for (int i = 0; i < modules.length; i++)
            classifiers[i] = modules[i].getClassifier();
        return classifiers;
    }

    public void setClassifiersNamesForFileRead(String[] classifierNames) {
        setClassifiers(null, classifierNames, null);
    }

    public void setClassifiersForBuildingInMemory(Classifier[] classifiers) {
        setClassifiers(classifiers, null ,null);
    }
    
    /**
     * If building the ensemble from scratch, the minimum requirement for running is the
     * classifiers array, the others could be left null.
     *
     * If building the ensemble from the results files of individuals (i.e. setBuildIndividualsFromResultsFiles(true)), 
     * the minimum requirement for running is the classifierNames list.
     *
     * @param classifiers array of classifiers to use
     * @param classifierNames if null, will use the classifiers' class names by default
     * @param classifierParameters  if null, parameters of each classifier empty by default
     */
    public void setClassifiers(Classifier[] classifiers, String[] classifierNames, String[] classifierParameters) {
        if (classifiers == null && classifierNames == null) {
            System.out.println("setClassifiers() was passed null for both the classifiers and classifiernames."
                    + "If building the ensemble from scratch in memory (default), the classifiers are needed at minimum."
                    + "Otherwise if building the ensemble from the saved results of base classifiers on disk, the "
                    + "classifier names are needed at minimum. ");
            
            //ClassifierLists does not want to throw exceptions, killing here for now todo review
            System.exit(1);
        }
        
        if (classifiers == null) {
            classifiers = new Classifier[classifierNames.length];
            for (int i = 0; i < classifiers.length; i++)
                classifiers[i] = null;
        }
        else {
            //If they are able to, make the classifiers estimate their own performance. This helps with contracting
            for (Classifier c : classifiers) {
                if (c instanceof EnhancedAbstractClassifier)
                    if (((EnhancedAbstractClassifier) c).ableToEstimateOwnPerformance())
                        ((EnhancedAbstractClassifier) c).setEstimateOwnPerformance(true);
            }
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
    
    protected void initialiseModules() throws Exception {
        //currently will only have file reading ON or OFF (not load some files, train the rest)
        //having that creates many, many, many annoying issues, especially when classifying test cases
        if (readIndividualsResults) {
            if (!resultsFilesParametersInitialised)
                throw new Exception("Trying to load "+ensembleName+" modules from file, but parameters for results file reading have not been initialised");
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
    
    protected synchronized void trainModules() throws Exception {
        
        //define the operations to build and evaluate each module, as a function
        //that will build the classifier and return train results for it, either 
        //generated by the classifier itself or the trainEstimator
        List<Callable<ClassifierResults>> moduleBuilds = new ArrayList<>();
        for (EnsembleModule module : modules) {
            final Classifier classifier = module.getClassifier();
            final Evaluator eval = trainEstimator.cloneEvaluator();
            
            Callable<ClassifierResults> moduleBuild = () -> {
                ClassifierResults trainResults = null;
                
                if (EnhancedAbstractClassifier.classifierIsEstimatingOwnPerformance(classifier)) { 
                    classifier.buildClassifier(trainInsts);
                    trainResults = ((EnhancedAbstractClassifier)classifier).getTrainResults();
                }
                else { 
                    trainResults = eval.evaluate(classifier, trainInsts);
                    classifier.buildClassifier(trainInsts);
                }
                
                return trainResults;
            };
            
            moduleBuilds.add(moduleBuild);
        }
        
        
        //complete the operations, either threaded via the executor service or 
        //locally/sequentially
        List<ClassifierResults> results = new ArrayList<>();
        if (multiThread) {
            ExecutorService executor = ThreadingUtilities.buildExecutorService(numThreads);
            boolean shutdownAfter = true;
            
            results = ThreadingUtilities.computeAll(executor, moduleBuilds, shutdownAfter);  
        }
        else { 
            for (Callable<ClassifierResults> moduleBuild : moduleBuilds)
                results.add(moduleBuild.call());
        }
        
        
        //gather back the train results, write them if needed 
        for (int i = 0; i < modules.length; i++) {
            modules[i].trainResults = results.get(i);
            
            if (writeIndividualsResults) { //if we're doing trainFold# file writing
                String params = modules[i].getParameters();
                if (modules[i].getClassifier() instanceof EnhancedAbstractClassifier)
                    params = ((EnhancedAbstractClassifier)modules[i].getClassifier()).getParameters();
                writeResultsFile(modules[i].getModuleName(), params, modules[i].trainResults, "train"); //write results out
            }
        }
    }
    
//    protected void trainModules_unThreaded() throws Exception {
//        for (EnsembleModule module : modules) {
//            Classifier clf = module.getClassifier();
//            if (clf instanceof TrainAccuracyEstimator) {
//                clf.buildClassifier(trainInsts);
//
//                //these train results should also include the buildtime
//                module.trainResults = ((TrainAccuracyEstimator)clf).getTrainResults();
//                module.trainResults.finaliseResults();
//
//                // TODO: should errorEstimateTime be forced to zero? by the intention of the interface,
//                // the estimate should have been produced during the normal process of building
//                // the classifier, but depending on how it was programmatically produced, 
//                // the reported estimate time may have already been accounted for in the 
//                // build time. Investigate when use cases arise
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
//            }
//        }
//    }
        

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
                errors.log("\nTRAIN results files for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + seed + "' not found. ");
            else if (needIndividualTrainPreds() && modules[m].trainResults.getProbabilityDistributions().isEmpty())
                errors.log("\nNo pred/distribution for instance data found in TRAIN results file for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + seed + "'. ");

            if (!testResultsLoaded)
                errors.log("\nTEST results files for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + seed + "' not found. ");
            else if (modules[m].testResults.numInstances()==0)
                errors.log("\nNo prediction data found in TEST results file for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + seed + "'. ");
        }

        errors.throwIfErrors();
    }

    protected boolean needIndividualTrainPreds() {
        return getEstimateOwnPerformance() || weightingScheme.needTrainPreds || votingScheme.needTrainPreds;
    }

    protected File findResultsFile(String readResultsFilesDirectory, String classifierName, String trainOrTest) {
        File file = new File(readResultsFilesDirectory+classifierName+"/Predictions/"+datasetName+"/"+trainOrTest+"Fold"+seed+".csv");
        if(!file.exists() || file.length() == 0)
            return null;
        else
            return file;
    }

    //hack for handling train accuracy estimate. experiments is giving us the full path and filename
    //to write to, instead of just the folder and expecting us to fill in the +classifierName+"/Predictions/"+datasetName+filename;
    //when doing the interface overhaul, sort this stuff out.
    protected void writeEnsembleTrainAccuracyEstimateResultsFile() throws Exception {
        trainResults.writeFullResultsToFile(writeResultsFilesDirectory);
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
        setResultsFileLocationParameters(new String[] { individualResultsFilesDirectory }, datasetName, resampleIdentifier);
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

        if (this.seedClassifier && this.seed != resampleIdentifier)
            System.out.println("**************WARNING: have set the seed via setSeed() already, but now setting up to build the ensemble" +
                    " from file with a different fold id identifier. Using the new value for future seeding operations");
        setSeed(resampleIdentifier);
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

            pred = findIndexOfMax(dist, rand);
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
        this.testResults.finaliseResults(testSetClassVals);
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

        if (trainResults != null) //performed trainEstimator
            writeResultsFile(ensembleName, getParameters(), trainResults, "train");

        this.testResults.finaliseResults(testSetClassVals);
        writeResultsFile(ensembleName, getParameters(), testResults, "test");
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

    public Transformer getTransform(){
        return this.transform;
    }

    public void setTransform(Transformer transform){
        this.transform = transform;
    }

    @Override
    public ClassifierResults getTrainResults(){
        return trainResults;
    }

    public ClassifierResults getTestResults(){
        return testResults;
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
        
        //housekeeping
        if (resultsFilesParametersInitialised) {
            if (readResultsFilesDirectories.length > 1)
                if (readResultsFilesDirectories.length != modules.length)
                    throw new Exception("Ensemble, " + this.getClass().getSimpleName() + ".buildClassifier: "
                            + "more than one results path given, but number given does not align with the number of classifiers/modules.");

            if (writeResultsFilesDirectory == null)
                writeResultsFilesDirectory = readResultsFilesDirectories[0];
        }
        
        // can classifier handle the data?
        getCapabilities().testWithFail(data);
        
        long startTime = System.nanoTime();
        
        //transform data if specified
        if(this.transform==null){
            this.trainInsts = data;
        }else{
           printlnDebug(" Transform is being used: Transform = "+transform.getClass().getSimpleName());

           this.trainInsts = transform.transform(data);           
           printlnDebug(" Transform "+transform.getClass().getSimpleName()+" complete");
           printlnDebug(" Transform "+transform.toString());
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
                if (weightingScheme.needTrainPreds || votingScheme.needTrainPreds) {
                    if (module.trainResults.getBuildPlusEstimateTime() == -1){
                        //assumes estimate time is not included in the total build time
                        buildTime += module.trainResults.getBuildTime() + module.trainResults.getErrorEstimateTime();
                    }
                    else{   
                        buildTime += module.trainResults.getBuildPlusEstimateTime();
                    }
                }
                else{
                    buildTime += module.trainResults.getBuildTime();
                }
            }
        }
        
        trainResults = new ClassifierResults();
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        
        if(getEstimateOwnPerformance())
            trainResults = estimateEnsemblePerformance(data); //combine modules to find overall ensemble trainpreds
        
        //HACK FOR CAWPE_EXTENSION PAPER: 
        //since experiments expects us to make a train results object 
        //and for us to record our build time, going to record it here instead of 
        //editing experiments to record the buildTime at that level
        
        //buildTime does not include the ensemble's trainEstimator in any case, only the work required to be ready for testing
        //time unit has been set in estimateEnsemblePerformance(data);
        trainResults.turnOffZeroTimingsErrors();
        trainResults.setBuildTime(buildTime);
        trainResults.turnOnZeroTimingsErrors();
                
        this.testInstCounter = 0; //prep for start of testing
        this.prevTestInstance = null;
    }

    

    
    
    
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
//            transform.setInputFormat(rawContainer);
//            Instances converted = Filter.useFilter(rawContainer,transform);
            Instances converted = transform.transform(rawContainer);            
            ins = converted.instance(0);
            
        }

        if (testResults == null || (testInstCounter == 0 && prevTestInstance == null)) {//definitely the first call, not e.g the first inst being classified for the second time
            printlnDebug("\n**TEST**");

            testResults = new ClassifierResults(numClasses);
            testResults.setTimeUnit(TimeUnit.NANOSECONDS);
            testResults.setBuildTime(buildTime);
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
        
        testResults.turnOffZeroTimingsErrors();
        testResults.addPrediction(dist, findIndexOfMax(dist, rand), predTime, "");
        testResults.turnOnZeroTimingsErrors();
        
        if (prevTestInstance != instance)
            ++testInstCounter;
        prevTestInstance = instance;

        return dist;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        return findIndexOfMax(dist, rand);
    }

    /**
     * @return the predictions of each individual module, i.e [0] = first module's vote, [1] = second...
     */
    public double[] classifyInstanceByConstituents(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
//            transform.setInputFormat(rawContainer);
//            Instances converted = Filter.useFilter(rawContainer,transform);


            Instances converted = transform.transform(rawContainer);
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
            Instances converted = transform.transform(rawContainer);
            ins = converted.instance(0);
        }

        double[][] distsByClassifier = new double[this.modules.length][];

        for(int i=0;i<modules.length;i++){
            distsByClassifier[i] = modules[i].getClassifier().distributionForInstance(ins);
        }

        return distsByClassifier;
    }
    
    @Override //MultiThreadable
    public void enableMultiThreading(int numThreads) {
        if (numThreads > 1) {
            this.numThreads = numThreads;
            this.multiThread = true;
        }
        else{
            this.numThreads = 1;
            this.multiThread = false;
        }
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
        sb.append("\nfold: ").append(seed);
        sb.append("\ntrain acc: ").append(trainResults.getAcc());
        sb.append("\ntest acc: ").append(builtFromFile ? testResults.getAcc() : "NA");

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
            sb.append("\ntrain acc: ").append(trainResults.getAcc());
            sb.append("\n");
            for(int i = 0; i < trainResults.numInstances();i++)
                sb.append(produceEnsemblePredsLine(true, i)).append("\n");

            sb.append("\n\nensemble test preds: ");
            sb.append("\ntest acc: ").append(builtFromFile ? testResults.getAcc() : "NA");
            sb.append("\n");
            for(int i = 0; i < testResults.numInstances();i++)
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
            sb.append(modules[0].trainResults.getTrueClassValue(index)).append(",").append(trainResults.getPredClassValue(index)).append(",");
        else
            sb.append(modules[0].testResults.getTrueClassValue(index)).append(",").append(testResults.getPredClassValue(index)).append(",");

        if (train){ //dist
            double[] pred=trainResults.getProbabilityDistribution(index);
            for (int j = 0; j < pred.length; j++)
                sb.append(",").append(pred[j]);
        }
        else{
            double[] pred=testResults.getProbabilityDistribution(index);
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
            cawpe.setEstimateOwnPerformance(true); //now defaults to true
            cawpe.setSeed(fold);

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
            cawpe.setEstimateOwnPerformance(true); //now defaults to true
            cawpe.setSeed(fold);

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
