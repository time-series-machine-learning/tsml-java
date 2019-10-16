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
package experiments;

import weka_extras.classifiers.SaveEachParameter;
import weka_extras.classifiers.tuned.TunedRandomForest;
import experiments.data.DatasetLists;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.JCommander.Builder;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.ParameterSplittable;
import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.SingleSampleEvaluator;
import weka.classifiers.Classifier;
import evaluation.storage.ClassifierResults;
import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.evaluators.StratifiedResamplesEvaluator;
import experiments.data.DatasetLoading;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import timeseriesweka.classifiers.EnhancedAbstractClassifier;
import weka_extras.classifiers.ensembles.SaveableEnsemble;
import weka.core.Instances;

/**
 * The main experimental class of the timeseriesclassification codebase. The 'main' method to run is 
 setupAndRunExperiment(ExperimentalArguments expSettings)
 
 An execution of this will evaluate a single classifier on a single resample of a single dataset. 
 
 Given an ExperimentalArguments object, which may be parsed from command line arguments
 or constructed in code, (and in the future, perhaps other methods such as JSON files etc),
 will load the classifier and dataset specified, prep the location to write results to, 
 train the classifier - potentially generating an error estimate via cross validation on the train set 
 as well - and then predict the cases of the test set. 
 
 The primary outputs are the train and/or 'testFoldX.csv' files, in the so-called ClassifierResults format,
 (see the class of the same name under utilities). 
 
 main(String[] args) info: 
      Parses args into an ExperimentalArguments object, then calls setupAndRunExperiment(ExperimentalArguments expSettings).
      Calling with the --help argument, or calling with un-parsable parameters, will print a summary of the possible parameters.
 
      Argument key-value pairs are separated by '='. The 5 basic, always required, arguments are: 
          Para name (short/long)  |    Example                           
          -dp --dataPath          |    --dataPath=C:/Datasets/             
          -rp --resultsPath       |    --resultsPath=C:/Results/
          -cn --classifierName    |    --classifierName=RandF
          -dn --datasetName       |    --datasetName=ItalyPowerDemand
          -f  --fold              |    --fold=1 
 
      Use --help to see all the optional parameters, and more information about each of them.
 
      If running locally, it may be easier to build the ExperimentalArguments object yourself and call setupAndRunExperiment(...)
      directly, instead of building the String[] args and calling main like a lot of legacy code does.   
 * 
 * @author James Large (james.large@uea.ac.uk), Tony Bagnall (anthony.bagnall@uea.ac.uk)
 */
public class Experiments  {

    private final static Logger LOGGER = Logger.getLogger(Experiments.class.getName());

    public static boolean debug = false;
    
    /**
     * If true, experiments will not print or log to stdout/err anything other that exceptions (SEVERE)
     */
    public static boolean beQuiet = false; 
    
    //A few 'should be final but leaving them not final just in case' public static settings 
    public static int numCVFolds = 10;

    @Parameters(separators = "=")
    public static class ExperimentalArguments implements Runnable {
        
    //REQUIRED PARAMETERS
        @Parameter(names={"-dp","--dataPath"}, required=true, order=0, description = "(String) The directory that contains the dataset to be evaluated on, in the form "
                + "[--dataPath]/[--datasetName]/[--datasetname].arff (the actual arff file(s) may be in different forms, see Experiments.sampleDataset(...).")
        public String dataReadLocation = null;
        
        @Parameter(names={"-rp","--resultsPath"}, required=true, order=1, description = "(String) The parent directory to write the results of the evaluation to, in the form "
                + "[--resultsPath]/[--classifierName]/Predictions/[--datasetName]/...")
        public String resultsWriteLocation = null;
        
        @Parameter(names={"-cn","--classifierName"}, required=true, order=2, description = "(String) The name of the classifier to evaluate. A case matching this value should exist within the ClassifierLists")
        public String classifierName = null;
        
        @Parameter(names={"-dn","--datasetName"}, required=true, order=3, description = "(String) The name of the dataset to be evaluated on, which resides within the dataPath in the form "
                + "[--dataPath]/[--datasetName]/[--datasetname].arff (the actual arff file(s) may be of different forms, see Experiments.sampleDataset(...).")
        public String datasetName = null;
        
        @Parameter(names={"-f","--fold"}, required=true, order=4, description = "(int) The fold index for dataset resampling, also used as the rng seed. *Indexed from 1* to conform with cluster array "
                + "job indices. The fold id pass will be automatically decremented to be zero-indexed internally.")
        public int foldId = 0;
        
    //OPTIONAL PARAMETERS
        @Parameter(names={"--help"}, hidden=true) //hidden from usage() printout
        private boolean help = false;
        
        //todo separate verbosity into it own thing
        @Parameter(names={"-d","--debug"}, arity=1, description = "(boolean) Increases verbosity and turns on the printing of debug statements")
        public boolean debug = false;
        
        @Parameter(names={"-gtf","--genTrainFiles"}, arity=1, description = "(boolean) Turns on the production of trainFold[fold].csv files, the results of which are calculate either via a cross validation of "
                + "the train data, or if a classifier implements the TrainAccuracyEstimate interface, the classifier will write its own estimate via its own means of evaluation.")
        public boolean generateErrorEstimateOnTrainSet = false;
        
        @Parameter(names={"-cp","--checkpointing"}, arity=1, description = "(boolean) Turns on the usage of checkpointing, if the classifier implements the SaveParameterInfo and/or CheckpointClassifier interfaces. The "
                + "classifier by default will write its checkpointing files to the same location as the --resultsPath, unless another path is optionally supplied to --checkpointPath.")
        public boolean checkpointing = false;
        
        @Parameter(names={"-sp","--supportingFilePath"}, description = "(String) Specifies the directory to write any files that may be produced by the classifier if it is a FileProducer. This includes but may not be "
                + "limited to: parameter evaluations, checkpoints, and logs. By default, these files are written to a generated subdirectory in the same location that the train and testFold[fold] files are written, relative"
                + "the --resultsPath. If a path is supplied via this parameter however, the files shall be written to that precisely that directory, as opposed to e.g. [-sp]/[--classifierName]/Predictions... "
                + "THIS IS A PLACEHOLDER PARAMETER. TO BE FULLY IMPLEMENTED WHEN INTERFACES AND SETCLASSIFIER ARE UPDATED.")
        public String supportingFilePath = null;
        
        @Parameter(names={"-pid","--parameterSplitIndex"}, description = "(Integer) If supplied and the classifier implements the ParameterSplittable interface, this execution of experiments will be set up to evaluate "
                + "the parameter set -pid within the parameter space used by the classifier (whether that be a supplied space or default). How the integer -pid maps onto the parameter space is up to the classifier.")
        public Integer singleParameterID = null;

        @Parameter(names={"-tb","--timingBenchmark"}, arity=1, description = "(boolean) Turns on the computation of a standard operation to act as a simple benchmark for the speed of computation on this hardware, which may "
                + "optionally be used to normalise build/test/predictions times across hardware in later analysis. Expected time on Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz is ~0.8 seconds. For experiments that are likely to be very "
                + "short, it is recommended to leave this off, as it will proportionally increase the total time to perform all your experiments by a great deal, and for short evaluation time the proportional affect of "
                + "any processing noise may make any benchmark normalisation process unreliable anyway.")
        public boolean performTimingBenchmark = false;
        
        //todo expose the filetype enum in some way, currently just using an unconnected if statement, if e.g the order of the enum values changes in the classifierresults, which we have no knowledge 
        //of here, the ifs will call the wrong things. decide on the design of this 
        @Parameter(names={"-ff","--fileFormat"}, description = "(int) Specifies the format for the classifier results file to be written in, accepted values = { 0, 1, 2 }, default = 0. 0 writes the first 3 lines of meta information "
                + "as well as the full prediction information, and requires the most disk space. 1 writes the first three lines and a list of the performance metrics calculated from the prediction info. 2 writes the first three lines only, and "
                + "requires the least space. Use options other than 0 if generating too many files with too much prediction information for the disk space available, however be aware that there is of course a loss of information.")
        public int classifierResultsFileFormat = 0;

        @Parameter(names={"-ctrs","--contractTrainSecs"}, description = "(long) Defines a time limit, in seconds, for the training of the classifier if it implements the TrainTimeContractClassifier interface. Defaults to 0, which sets "
                + "no contract time. Only one of --contractTrainSecs, and --contractTrainHours should be supplied. If both are supplied, seconds takes preference over hours. "
                + "THIS IS A PLACEHOLDER PARAMETER. TO BE FULLY IMPLEMENTED WHEN INTERFACES AND SETCLASSIFIER ARE UPDATED.")
        public long contractTrainTimeSeconds = 0;
        
        @Parameter(names={"-ctrh","--contractTrainHours"}, description = "(long) Defines a time limit, in hours, for the training of the classifier if it implements the TrainTimeContractClassifier interface. Defaults to 0, which sets "
                + "no contract time. Only one of --contractTimeNanos, --contractTimeMinutes, or --contractTimeHours should be supplied. If both are supplied, seconds hours takes preference over hours."
                + "\n\n THIS IS A PLACEHOLDER PARAMETER. TO BE FULLY IMPLEMENTED WHEN INTERFACES AND SETCLASSIFIER ARE UPDATED.")
        public long contractTrainTimeHours = 0;
        
                @Parameter(names={"-ctem","--contractTestMillis"}, description = "(long) Defines a time limit, in miliseconds, for the time given to the classifier to make each test prediction if it implements the ContractablePredictions interface. "
                + "Defaults to 0, which sets no contract time. Only one of --contractTestMillis and --contractTestSecs should be supplied. If both are supplied, milis takes preference over seconds. "
                + "THIS IS A PLACEHOLDER PARAMETER. TO BE FULLY IMPLEMENTED WHEN INTERFACES AND SETCLASSIFIER ARE UPDATED.")
        public long contractPredTimeMillis = 0;
        
        @Parameter(names={"-ctes","--contractTestSecs"}, description = "(long) Defines a time limit, in seconds, for the time given to the classifier to make each test prediction if it implements the ContractablePredictions interface. "
                + "Defaults to 0, which sets no contract time. Only one of --contractTestMillis and --contractTestSecs should be supplied. If both are supplied, milis takes preference over seconds. "
                + "THIS IS A PLACEHOLDER PARAMETER. TO BE FULLY IMPLEMENTED WHEN INTERFACES AND SETCLASSIFIER ARE UPDATED.")
        public long contractPredTimeSeconds= 0;
        
        @Parameter(names={"-sc","--serialiseClassifier"}, arity=1, description = "(boolean) If true, and the classifier is serialisable, the classifier will be serialised to the --supportingFilesPath after training, but before testing.  "
                + "THIS IS A PLACEHOLDER PARAMETER. TO BE FULLY IMPLEMENTED")
        public boolean serialiseTrainedClassifier = false;
        
        @Parameter(names={"--force"}, arity=1, description = "(boolean) If true, the evaluation will occur even if what would be the resulting file already exists. The old file will be overwritten with the new evaluation results.")
        public boolean forceEvaluation = false;
        
        @Parameter(names={"-tem --trainEstimateMethod"}, arity=1, description = "(String) Defines the method and parameters of the evaluation method used to estimate error on the train set, if --genTrainFiles == true. Current implementation is a hack to get the option in for"
                + " experiment running in the short term. Give one of 'cv' and 'hov' for cross validation and hold-out validation set respectively, and a number of folds (e.g. cv_10) or train set proportion (e.g. hov_0.7) respectively. Default is a 10 fold cv, i.e. cv_10.")
        public String trainEstimateMethod = "cv_10";
        
        public ExperimentalArguments() {
            
        }
        
        public ExperimentalArguments(String[] args) throws Exception {
            parseArguments(args);
        }
        
        @Override //Runnable
        public void run() {
            try {
                setupAndRunExperiment(this);
            } catch (Exception ex) {
                System.out.println("Threaded Experiment Failed: " + ex);
            }
        }
        
        /**
         * This is a bit of a bolt-on method for now. It assumes that the object on which 
         * this method is being called has all the other parameters not passed to it set already 
         * (e.g data location, results location) and these will be replicated across all experiments.
         * The current value of this.classifierName, this.datasetName, and this.foldId are ignored within 
         * this method. 
         * 
         * 
         * @param minFold inclusive
         * @param maxFold exclusive, i.e will make folds [ for (int f = minFold; f < maxFold; ++f) ]
         * @return a list of unique experimental arguments, covering all combinations of classifier, datasets, and folds passed, with the same meta info as 'this' currently stores
         */
        public List<ExperimentalArguments> generateExperiments(String[] classifierNames, String[] datasetNames, int minFold, int maxFold) { 
            
            if (minFold > maxFold) {
                int t = minFold;
                minFold = maxFold;
                maxFold = t;
            }
            
            ArrayList<ExperimentalArguments> exps = new ArrayList<>(classifierNames.length * datasetNames.length * (maxFold - minFold));
            
            for (String classifier : classifierNames) {
                for (String dataset : datasetNames) {
                    for (int fold = minFold; fold < maxFold; fold++) {
                        ExperimentalArguments exp = new ExperimentalArguments();
                        exp.classifierName = classifier;
                        exp.datasetName = dataset;
                        exp.foldId = fold;
                        
                        exp.dataReadLocation = this.dataReadLocation;
                        exp.resultsWriteLocation = this.resultsWriteLocation;
                        exp.generateErrorEstimateOnTrainSet = this.generateErrorEstimateOnTrainSet;
                        exp.checkpointing = this.checkpointing;
                        exp.singleParameterID = this.singleParameterID;
                        exp.contractTrainTimeSeconds = this.contractTrainTimeSeconds;
                        exp.contractTrainTimeHours = this.contractTrainTimeHours;
                        exp.contractPredTimeMillis = this.contractPredTimeMillis;
                        exp.contractPredTimeSeconds = this.contractPredTimeSeconds;
                        exp.performTimingBenchmark = this.performTimingBenchmark;
                        exp.supportingFilePath = this.supportingFilePath;
                        exp.debug = this.debug;
                        exp.classifierResultsFileFormat = this.classifierResultsFileFormat;
                        exp.serialiseTrainedClassifier = this.serialiseTrainedClassifier;
                        
                        exps.add(exp);
                    }
                }
            }
            
            return exps;
        }
        
        private void parseArguments(String[] args) throws Exception {
            Builder b = JCommander.newBuilder();
            b.addObject(this);
            JCommander jc = b.build();
            jc.setProgramName("Experiments.java");  //todo maybe add copyright etcetc
            try {
                jc.parse(args);
            } catch (Exception e) {
                if (!help) {
                    //we actually errored, instead of the program simply being called with the --help flag
                    System.err.println("Parsing of arguments failed, parameter information follows after the error. Parameters that require values should have the flag and value separated by '='.");
                    System.err.println("For example: java -jar TimeSeriesClassification.jar -dp=data/path/ -rp=results/path/ -cn=someClassifier -dn=someDataset -f=0");
                    System.err.println("Parameters prefixed by a * are REQUIRED. These are the first five parameters, which are needed to run a basic experiment.");
                    System.err.println("Error: \n\t"+e+"\n\n");
                }
                jc.usage();
//                Thread.sleep(1000); //usage can take a second to print for some reason?... no idea what it's actually doing
//                System.exit(1);
            }
            
            foldId -= 1; //go from one-indexed to zero-indexed
            Experiments.debug = this.debug;
            
            //populating the contract times if present
            //todo refactor to timeunits
            if (contractTrainTimeSeconds > 0)
                contractTrainTimeHours = contractTrainTimeSeconds / 60 / 60;
            else if (contractTrainTimeHours > 0)
                contractTrainTimeSeconds = contractTrainTimeHours * 60 * 60;
                    
            if (contractPredTimeMillis > 0)
                contractPredTimeSeconds = contractPredTimeMillis / 1000; 
            else if (contractPredTimeSeconds > 0)
                contractPredTimeMillis = contractPredTimeSeconds * 1000;
            
            //supporting file path generated in setupAndRunExperiment(...), if not explicitly passed
        }
        
        public String toShortString() { 
            return "["+classifierName+","+datasetName+","+foldId+"]";
        }
        
        @Override
        public String toString() { 
            StringBuilder sb = new StringBuilder();
            
            sb.append("EXPERIMENT SETTINGS "+ this.toShortString());
            sb.append("\ndataReadLocation: ").append(dataReadLocation);
            sb.append("\nresultsWriteLocation: ").append(resultsWriteLocation);
            sb.append("\nclassifierName: ").append(classifierName);
            sb.append("\ndatasetName: ").append(datasetName);
            sb.append("\nfoldId: ").append(foldId);
            sb.append("\ngenerateErrorEstimateOnTrainSet: ").append(generateErrorEstimateOnTrainSet);
            sb.append("\ncheckpoint: ").append(checkpointing);
            sb.append("\nsingleParameterID: ").append(singleParameterID);
            sb.append("\ncheckpointPath: ").append(supportingFilePath);
            sb.append("\ncontractTrainTimeSeconds: ").append(contractTrainTimeSeconds);
            sb.append("\ncontractPredTimeMillis: ").append(contractPredTimeMillis);
            sb.append("\nclassifierResultsFileFormat: ").append(classifierResultsFileFormat);
            sb.append("\nperformTimingBenchmark: ").append(performTimingBenchmark);
            sb.append("\nserialiseTrainedClassifier: ").append(serialiseTrainedClassifier);
            sb.append("\ndebug: ").append(debug);
            
            return sb.toString();
        }        
    }
    
    /** 
     * Parses args into an ExperimentalArguments object, then calls setupAndRunExperiment(ExperimentalArguments expSettings).
     * Calling with the --help argument, or calling with un-parsable parameters, will print a summary of the possible parameters.
 
 Argument key-value pairs are separated by '='. The 5 basic, always required, arguments are: 
     Para name (short/long)  |    Example                           
     -dp --dataPath          |    --dataPath=C:/Datasets/             
     -rp --resultsPath       |    --resultsPath=C:/Results/
     -cn --classifierName    |    --classifierName=RandF
     -dn --datasetName       |    --datasetName=ItalyPowerDemand
     -f  --fold              |    --fold=1 
 
 Use --help to see all the optional parameters, and more information about each of them.
 
 If running locally, it may be easier to build the ExperimentalArguments object yourself and call setupAndRunExperiment(...)
 directly, instead of building the String[] args and calling main like a lot of legacy code does.
     */
    public static void main(String[] args) throws Exception {        
        //even if all else fails, print the args as a sanity check for cluster.
        if (!beQuiet) {
            System.out.println("Raw args:");
            for (String str : args)
                System.out.println("\t"+str);
            System.out.println("");
        }
        
        if (args.length > 0) {
            ExperimentalArguments expSettings = new ExperimentalArguments(args);
            setupAndRunExperiment(expSettings);
        }else{
            int folds=1;
            boolean threaded=false;
            if(threaded){
                String[] settings=new String[6];
                settings[0]="-dp=E:/Data/TSCProblems2018/";//Where to get data                
                settings[1]="-rp=E:/Results/";//Where to write results                
                settings[2]="-gtf=true"; //Whether to generate train files or not               
                settings[3]="-cn=TSF"; //Classifier name
                settings[5]="1";
                settings[4]="-dn="+"ItalyPowerDemand"; //Problem file   
                settings[5]="-f=1";//Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)               
                ExperimentalArguments expSettings = new ExperimentalArguments(settings);
                setupAndRunMultipleExperimentsThreaded(expSettings, new String[]{settings[3]},DatasetLists.tscProblems78,0,folds);
            }else{//Local run without args, mainly for debugging
                String[] settings=new String[6];
//Location of data set
                settings[0]="-dp=Z:\\ArchiveData\\Univariate_arff\\";//Where to get data   
                settings[1]="-rp=E:\\Temp\\";
                settings[2]="-gtf=true"; //Whether to generate train files or not               
                settings[3]="-cn=RISE"; //Classifier name
//                for(String str:DatasetLists.tscProblems78){
                    settings[4]="-dn="+"Chinatown"; //Problem file   
                    settings[5]="-f=2";//Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)  
                System.out.println("Manually set args:");
                for (String str : settings)
                    System.out.println("\t"+str);
                System.out.println("");
                ExperimentalArguments expSettings = new ExperimentalArguments(settings);
                setupAndRunExperiment(expSettings);
            }
        }
    }

    /**
     * Runs an experiment with the given settings. For the more direct method in case e.g 
     * you have a bespoke classifier not handled by ClassifierList or dataset that 
     * is sampled in a bespoke way, use runExperiment
     * 
     * 1) Sets up the logger. 
     * 2) Sets up the results write path
     * 3) Checks whether this experiments results already exist. If so, exit
     * 4) Constructs the classifier
     * 5) Samples the dataset.
     * 6) If we're good to go, runs the experiment.
     */
    public static void setupAndRunExperiment(ExperimentalArguments expSettings) throws Exception {
        //todo: when we convert to e.g argparse4j for parameter passing, add a para 
        //for location to log to file as well. for now, assuming console output is good enough
        //for local running, and cluster output files are good enough on there. 
//        LOGGER.addHandler(new FileHandler()); 

        if (beQuiet) {
            LOGGER.setLevel(Level.SEVERE);
        }
        else {
            if (debug)
                LOGGER.setLevel(Level.FINEST);
            else
                LOGGER.setLevel(Level.INFO);
            
            DatasetLoading.setDebug(debug); //TODO when we got full enterprise and figure out how to properly do logging, clean this up
        }
        LOGGER.log(Level.FINE, expSettings.toString());
        
        
        //2019_06_03: cases in the classifier can now change the classifier name to reflect 
        //paritcular parameters wanting to be represented as different classifiers
        //e.g. a case ShapletsContracted might take a contract time (e.g. 1 day) from the args and set up the 
        //shapelet transform, but also change the classifier name stored in the experimentalargs to e.g. Shapelets_1day
        //such that if the experimenter is looping over contract times, they need only create one case 
        //in the setclassifier switch and pass one classifier name, but loop over contract time directly 
        //
        //so, the setClassifier has been moved to up here, previously only done after the check for 
        //whether we abort due to the results file already existing. the instantiation of a classifier
        //shouldn't be too much work, so despite it looking a little ugly, the call is 
        //moved to here before the first proper usage of classifiername, such that it can 
        //be updated first if need be
        Classifier classifier = ClassifierLists.setClassifier(expSettings);
        
        //Build/make the directory to write the train and/or testFold files to
        String fullWriteLocation = expSettings.resultsWriteLocation + expSettings.classifierName + "/Predictions/" + expSettings.datasetName + "/";
        File f = new File(fullWriteLocation);
        if (!f.exists())
            f.mkdirs();
        
        String targetFileName = fullWriteLocation + "testFold" + expSettings.foldId + ".csv";
        
        //Check whether fold already exists, if so, dont do it, just quit
        if (!expSettings.forceEvaluation && experiments.CollateResults.validateSingleFoldFile(targetFileName)) {
            LOGGER.log(Level.INFO, expSettings.toShortString() + " already exists at "+targetFileName+", exiting.");
            return;
        }
        else {     
            Instances[] data = DatasetLoading.sampleDataset(expSettings.dataReadLocation, expSettings.datasetName, expSettings.foldId);
        
            //If needed, build/make the directory to write the train and/or testFold files to
            if (expSettings.supportingFilePath == null || expSettings.supportingFilePath.equals(""))
                expSettings.supportingFilePath = fullWriteLocation;
            
            ///////////// 02/04/2019 jamesl to be put back in in place of above when interface redesign finished. 
            // default builds a foldx/ dir in normal write dir
//            if (expSettings.supportingFilePath == null || expSettings.supportingFilePath.equals(""))
//                expSettings.supportingFilePath = fullWriteLocation + "fold" + expSettings.foldId + "/";
//            if (classifier instanceof FileProducer) {
//                f = new File(expSettings.supportingFilePath);
//                if (!f.exists())
//                    f.mkdirs();
//            }
            
            //If this is to be a single _parameter_ evaluation of a fold, check whether this exists, and again quit if it does.
            if (expSettings.singleParameterID != null && classifier instanceof ParameterSplittable) {
                expSettings.checkpointing = false; //Just to tie up loose ends in case user defines both checkpointing AND para splitting
                
                targetFileName = fullWriteLocation + "fold" + expSettings.foldId + "_" + expSettings.singleParameterID + ".csv";
                if (experiments.CollateResults.validateSingleFoldFile(targetFileName)) {
                    LOGGER.log(Level.INFO, expSettings.toShortString() + ", parameter " + expSettings.singleParameterID +", already exists at "+targetFileName+", exiting.");
                    return;
                }
            }

            double acc = runExperiment(expSettings, data[0], data[1], classifier, fullWriteLocation);
            LOGGER.log(Level.INFO, "Experiment finished " + expSettings.toShortString() + ", Test Acc:" + acc);
        }
    }
    
    /**
     * Perform an actual experiment, using the loaded classifier and resampled dataset given, writing to the specified results location. 
     * 
     * 1) If needed, set up file paths and flags related to a single parameter evaluation and/or the classifier's internal parameter saving things
     * 2) If we want to be performing cv to find an estimate of the error on the train set, either do that here or set up the classifier to do it internally 
     *          during buildClassifier()
     * 3) Do the actual training, i.e buildClassifier()
     * 4) Save any train cv results
     * 5) Evaluate on the test set
     * 6) Save test results
     * 7) Done
     * 
     * NOTES: 1. If the classifier is a SaveableEnsemble, then we save the
     * internal cross validation accuracy and the internal test predictions 2.
     * The output of the file testFold+fold+.csv is Line 1:
     * ProblemName,ClassifierName, train/test Line 2: parameter information for
     * final classifierName, if it is available Line 3: test accuracy then each line
     * is Actual Class, Predicted Class, Class probabilities
     * 
     * @param resultsPath The exact folder in which to write the train and/or testFoldX.csv files
     * @return the accuracy of c on fold for problem given in train/test, or -1 on an error 
     */


    public static double runExperiment(ExperimentalArguments expSettings, Instances trainSet, Instances testSet, Classifier classifier, String resultsPath) {
        
        //if this is a parameter split run, train file name is defined by this
        //otherwise generally if the classifier wants to save parameter info itnerally, set that up here too
        String trainFoldFilename = setupParameterSavingInfo(expSettings, classifier, trainSet, resultsPath);
        if (trainFoldFilename == null) 
            //otherwise, defined by this as default
            trainFoldFilename = "trainFold" + expSettings.foldId + ".csv";
        String testFoldFilename = "testFold" + expSettings.foldId + ".csv";       
        
        //these train results shall now be initialised regardless of whether
        //we're going to write them out, to store at minimum the build times
        ClassifierResults trainResults = new ClassifierResults();
        ClassifierResults testResults = new ClassifierResults(); 
        
        LOGGER.log(Level.FINE, "Preamble complete, real experiment starting.");
                
        try {
            if (expSettings.generateErrorEstimateOnTrainSet) {
                //Tell the classifier to generate train results if it can do it internally, 
                //otherwise perform the evaluation externally here (e.g. cross validation on the
                //train data
                if (EnhancedAbstractClassifier.classifierAbleToEstimateOwnPerformance(classifier))
                    ((EnhancedAbstractClassifier) classifier).setEstimateOwnPerformance(true);
                else 
                    trainResults = findExternalTrainEstimate(expSettings, classifier, trainSet, expSettings.foldId);
            }
            LOGGER.log(Level.FINE, "Train estimate ready.");


            //Build on the full train data here
            long buildTime = System.nanoTime();
            classifier.buildClassifier(trainSet);
            buildTime = System.nanoTime() - buildTime;
            LOGGER.log(Level.FINE, "Training complete");

            trainResults = finaliseTrainResults(expSettings, classifier, trainResults, buildTime);
            //At this stage, regardless of whether the classifier is able to estimate it's 
            //own accuracy or not, train results should contain either
            //    a) timings, if expSettings.generateErrorEstimateOnTrainSet == false 
            //    b) full predictions, if expSettings.generateErrorEstimateOnTrainSet == true 
            
            if (expSettings.generateErrorEstimateOnTrainSet)
                writeResults(expSettings, trainResults, resultsPath + trainFoldFilename, "train");
            LOGGER.log(Level.FINE, "Train estimate written");

            if (expSettings.serialiseTrainedClassifier && classifier instanceof Serializable)
                serialiseClassifier(expSettings, classifier);
            
            //And now evaluate on the test set, if this wasn't a single parameter fold
            if (expSettings.singleParameterID == null) {
                //This is checked before the buildClassifier also, but 
                //a) another process may have been doing the same experiment 
                //b) we have a special case for the file builder that copies the results over in buildClassifier (apparently?)
                //no reason not to check again
                if (expSettings.forceEvaluation || !CollateResults.validateSingleFoldFile(resultsPath + testFoldFilename)) {
                    long testBenchmark = findBenchmarkTime(expSettings);
                    
                    testResults = evaluateClassifier(expSettings, classifier, testSet);
                    assert(testResults.getTimeUnit().equals(TimeUnit.NANOSECONDS)); //should have been set as nanos in the evaluation
                    
                    testResults.setParas(trainResults.getParas());
                    
                    testResults.turnOffZeroTimingsErrors();
                    testResults.setBenchmarkTime(testBenchmark);
                    testResults.setBuildTime(trainResults.getBuildTime());
                    testResults.turnOnZeroTimingsErrors();
                    
                    LOGGER.log(Level.FINE, "Testing complete");
                  
                    writeResults(expSettings, testResults, resultsPath + testFoldFilename, "test");
                    LOGGER.log(Level.FINE, "Testing written");
                } 
                else {
                    LOGGER.log(Level.INFO, "Test file already found, written by another process.");
                    testResults = new ClassifierResults(resultsPath + testFoldFilename);
                }
                return testResults.getAcc();
            } 
            else {
                return 0; //not error, but we dont have a test acc. just returning 0 for now
            }
        } 
        catch (Exception e) {
            //todo expand..
            LOGGER.log(Level.SEVERE, "Experiment failed. Settings: " + expSettings + "\n\nERROR: " + e.toString(), e);
            return -1; //error state
        }
    }

    /**
     * This method cleans up and consolidates the information we have about the 
     * build process into a ClassifierResults object, based on the capabilities of the 
     * classifier and whether we want to be writing a train predictions file
     * 
     * Regardless of whether the classifier is able to estimate it's own accuracy 
     * or not, the returned train results should contain either
     *    a) timings, if expSettings.generateErrorEstimateOnTrainSet == false 
     *    b) full predictions, if expSettings.generateErrorEstimateOnTrainSet == true
     * 
     * @param exp
     * @param classifier 
     * @param trainResults the results object so far which may be empty, contain the recorded 
     *      timings of the particular classifier, or contain the results of a previous executed
     *      external estimation process, 
     * @param buildTime as recorded by experiments.java, but which may not be used if the classifier 
     *      records it's own build time more accurately
     * @return the finalised train results object
     * @throws Exception 
     */
    private static ClassifierResults finaliseTrainResults(ExperimentalArguments exp, Classifier classifier, ClassifierResults trainResults, long buildTime) throws Exception { 
        
        /*
        if estimateacc { //want full predictions
            timingToUpdateWith = buildTime (the one passed to this func)
            if is EnhancedAbstractClassifier {
                if able to estimate own acc
                    just return getTrainResults()
                else 
                    timingToUpdateWith = getTrainResults().getBuildTime()
            }
            trainResults.setBuildTime(timingToUpdateWith)
            return trainResults
        }
        else not estimating acc { //just want timings
            if is EnhancedAbstractClassifier
                just return getTrainResults(), contains the timings and other maybe useful metainfo 
            else 
                trainResults passed are empty
                trainResults.setBuildTime(buildTime)
                return trainResults
        */
        
        if (exp.generateErrorEstimateOnTrainSet) { //want timings and full predictions
            long timingToUpdateWith = buildTime; //the timing that experiments measured by default
            String paras = "No parameter info";
            
            if (classifier instanceof EnhancedAbstractClassifier) {
                EnhancedAbstractClassifier eac = ((EnhancedAbstractClassifier)classifier);
                if (eac.getEstimateOwnPerformance())
                    return eac.getTrainResults(); //classifier internally estimateed/recorded itself, just return that directly
                else {
                    timingToUpdateWith = eac.getTrainResults().getBuildTime(); //update with classifier's own timings instead
                    paras = eac.getParameters();
                }
            }
            
            //update the externally produced results with the appropriate timing
            trainResults.setBuildTime(timingToUpdateWith);
            trainResults.setBuildPlusEstimateTime(timingToUpdateWith + trainResults.getErrorEstimateTime());
            trainResults.setParas(paras);
        }
        else { // just want the timings
            if (classifier instanceof EnhancedAbstractClassifier)
                return ((EnhancedAbstractClassifier)classifier).getTrainResults(); //classifier recorded timings/paras internally, just return that directly
            else 
                trainResults.setBuildTime(buildTime);
        }
        
        return trainResults;
    }

    private static String setupParameterSavingInfo(ExperimentalArguments expSettings, Classifier classifier, Instances train, String resultsPath) { 
        String parameterFileName = null;
        if (expSettings.singleParameterID != null && classifier instanceof ParameterSplittable)//Single parameter fold
        {
            if (classifier instanceof TunedRandomForest)
                ((TunedRandomForest) classifier).setNumFeaturesInProblem(train.numAttributes() - 1);
            
            expSettings.checkpointing = false;
            ((ParameterSplittable) classifier).setParametersFromIndex(expSettings.singleParameterID);
            parameterFileName = "fold" + expSettings.foldId + "_" + expSettings.singleParameterID + ".csv";
            expSettings.generateErrorEstimateOnTrainSet = true;
        } 
        else {
            //Only do all this if not an internal _single parameter_ experiment
            // Save internal info for ensembles
            if (classifier instanceof SaveableEnsemble) {
                ((SaveableEnsemble) classifier).saveResults(resultsPath + "internalCV_" + expSettings.foldId + ".csv", resultsPath + "internalTestPreds_" + expSettings.foldId + ".csv");
            }
            if (expSettings.checkpointing && classifier instanceof SaveEachParameter) {
                ((SaveEachParameter) classifier).setPathToSaveParameters(resultsPath + "fold" + expSettings.foldId + "_");
            }
        }
        
        return parameterFileName;
    }
    
    private static ClassifierResults findExternalTrainEstimate(ExperimentalArguments exp, Classifier classifier, Instances train, int fold) throws Exception { 
        ClassifierResults trainResults = null;
        long trainBenchmark = findBenchmarkTime(exp);

        //todo clean up this hack. default is cv_10, as with all old trainFold results pre 2019/07/19
        String[] parts = exp.trainEstimateMethod.split("_");
        String method = parts[0];

        String para1 = null;
        if (parts.length > 1) 
            para1 = parts[1];

        String para2 = null;
        if (parts.length > 2) 
            para2 = parts[2];

        switch (method) {
            case "cv":
            case "CV": 
            case "CrossValidationEvaluator":
                int numCVFolds = Experiments.numCVFolds;
                if (para1 != null)
                    numCVFolds = Integer.parseInt(para1);
                numCVFolds = Math.min(train.numInstances(), numCVFolds);

                CrossValidationEvaluator cv = new CrossValidationEvaluator();
                cv.setSeed(fold);
                cv.setNumFolds(numCVFolds);
                trainResults = cv.crossValidateWithStats(classifier, train);
                break;

            case "hov":
            case "HOV":
            case "SingleTestSetEvaluator":
                double trainPropHov = DatasetLoading.getProportionKeptForTraining();
                if (para1 != null)
                    trainPropHov = Double.parseDouble(para1);

                SingleSampleEvaluator hov = new SingleSampleEvaluator();
                hov.setSeed(fold);
                hov.setPropInstancesInTrain(trainPropHov);
                trainResults = hov.evaluate(classifier, train);
                break;

            case "sr":
            case "SR":
            case "StratifiedResamplesEvaluator":
                int numSRFolds = 30;
                if (para1 != null)
                    numSRFolds = Integer.parseInt(para1);

                double trainPropSRR = DatasetLoading.getProportionKeptForTraining();
                if (para2 != null)
                    trainPropSRR = Double.parseDouble(para2);

                StratifiedResamplesEvaluator srr = new StratifiedResamplesEvaluator();
                srr.setSeed(fold);
                srr.setNumFolds(numSRFolds);
                srr.setUseEachResampleIdAsSeed(true);
                srr.setPropInstancesInTrain(trainPropSRR);
                trainResults = srr.evaluate(classifier, train);
                break;

            default:
                throw new Exception("Unrecognised method to estimate error on the train given: " + exp.trainEstimateMethod);
        }

        trainResults.setErrorEstimateMethod(exp.trainEstimateMethod);
        trainResults.setBenchmarkTime(trainBenchmark);    
        
        return trainResults;
    }
    
    public static void serialiseClassifier(ExperimentalArguments expSettings, Classifier classifier) throws FileNotFoundException, IOException {
        String filename = expSettings.supportingFilePath + expSettings.classifierName + "_" + expSettings.datasetName + "_" + expSettings.foldId + ".ser";
        
        LOGGER.log(Level.FINE, "Attempting classifier serialisation, to " + filename);
        
        FileOutputStream fos = new FileOutputStream(filename);
        try (ObjectOutputStream out = new ObjectOutputStream(fos)) {
            out.writeObject(classifier);
            fos.close();
            out.close();
        }
        
        LOGGER.log(Level.FINE, "Classifier serialised successfully");
    }
    
    /**
     * Meta info shall be set by writeResults(...), just generating the prediction info and 
     * any info directly calculable from that here
     */
    public static ClassifierResults evaluateClassifier(ExperimentalArguments exp, Classifier classifier, Instances testSet) throws Exception {
        SingleTestSetEvaluator eval = new SingleTestSetEvaluator(exp.foldId, false, true); //DONT clone data, DO set the class to be missing for each inst
        
        return eval.evaluate(classifier, testSet);
    }
    
    /**
     * If exp.performTimingBenchmark = true, this will return the total time to 
     * sort 1,000 arrays of size 10,000
     * 
     * Expected time on Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz is ~0.8 seconds
     * 
     * This can still anecdotally vary between 0.75 to 1.05 on my windows machine, however. 
     */
    public static long findBenchmarkTime(ExperimentalArguments exp) {
        if (!exp.performTimingBenchmark)
            return -1; //the default in classifierresults, i.e no benchmark
        
        // else calc benchmark
        
        int arrSize = 10000;
        int repeats = 1000;
        long[] times = new long[repeats];
        long total = 0L;
        for (int i = 0; i < repeats; i++) {
            times[i] = atomicBenchmark(arrSize);
            total+=times[i];
        }

        if (debug) { 
            long mean = 0L, max = Long.MIN_VALUE, min = Long.MAX_VALUE;
            for (long time : times) { 
                mean += time;
                if (time < min)
                    min = time;
                if (time > max)
                    max = time;
            }
            mean/=repeats;
            
            int halfR = repeats/2;
            long median = repeats % 2 == 0 ? 
                (times[halfR] + times[halfR+1]) / 2 :
                times[halfR];

            double d = 1000000000;
            StringBuilder sb = new StringBuilder("BENCHMARK TIMINGS, summary of times to "
                    + "sort "+repeats+" random int arrays of size "+arrSize+" - in seconds\n");
            sb.append("total = ").append(total/d).append("\n");
            sb.append("min = ").append(min/d).append("\n");
            sb.append("max = ").append(max/d).append("\n");
            sb.append("mean = ").append(mean/d).append("\n");
            sb.append("median = ").append(median/d).append("\n");
            
            LOGGER.log(Level.FINE, sb.toString());
        }

        return total;
    }
    
    private static long atomicBenchmark(int arrSize) {
        long startTime = System.nanoTime();
        int[] arr = new int[arrSize];
        Random rng = new Random(0);
        for (int j = 0; j < arrSize; j++)
            arr[j] = rng.nextInt();

        Arrays.sort(arr);
        return System.nanoTime() - startTime;
    }
    
    public static void writeResults(ExperimentalArguments exp, ClassifierResults results, String fullTestWritingPath, String split) throws Exception {
        results.setClassifierName(exp.classifierName);
        results.setDatasetName(exp.datasetName);
        results.setFoldID(exp.foldId);
        results.setSplit(split);
        
        //TODO get system information, e.g. cpu clock-speed. generic across os too
        Date date = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");    
        results.setDescription("Generated by Experiments.java on " + formatter.format(date));
        

        //todo, need to make design decisions with the classifierresults enum to clean this switch up
        switch (exp.classifierResultsFileFormat) {
            case 0: //PREDICTIONS
                results.writeFullResultsToFile(fullTestWritingPath);
                break;
            case 1: //METRICS
                results.writeSummaryResultsToFile(fullTestWritingPath);
                break;
            case 2: //COMPACT
                results.writeCompactResultsToFile(fullTestWritingPath);
                break;
            default: {
                System.err.println("Classifier Results file writing format not recognised, "+exp.classifierResultsFileFormat+", just writing the full predictions.");
                results.writeFullResultsToFile(fullTestWritingPath);
                break;
            }
                
        }
        
        File f = new File(fullTestWritingPath);
        if (f.exists()) {
            f.setWritable(true, false);
        }
    }
    
    /**
     * Will run through all combinations of classifiers*datasets*folds provided, using the meta experimental info stored in the 
     * standardArgs. Will by default set numThreads = numCores
     */
    public static void setupAndRunMultipleExperimentsThreaded(ExperimentalArguments standardArgs, String[] classifierNames, String[] datasetNames, int minFolds, int maxFolds) throws Exception{
        setupAndRunMultipleExperimentsThreaded(standardArgs, classifierNames, datasetNames, minFolds, maxFolds, 0);
    }
    
    /**
     * Will run through all combinations of classifiers*datasets*folds provided, using the meta experimental info stored in the 
     * standardArgs. If numThreads > 0, will spawn that many threads. If numThreads == 0, will use as many threads as there are cores, 
     * else if numThreads == -1, will spawn as many threads as there are cores minus 1, to aid usability of the machine. 
     */
    public static void setupAndRunMultipleExperimentsThreaded(ExperimentalArguments standardArgs, String[] classifierNames, String[] datasetNames, int minFolds, int maxFolds, int numThreads) throws Exception{
        int numCores = Runtime.getRuntime().availableProcessors();        
        if (numThreads == 0)
            numThreads = numCores;
        else if (numThreads < 0)
            numThreads = Math.max(1, numCores-1);
        
        System.out.println("# cores ="+numCores);
        System.out.println("# threads ="+numThreads);
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
        List<ExperimentalArguments> exps = standardArgs.generateExperiments(classifierNames, datasetNames, minFolds, maxFolds);
        for (ExperimentalArguments exp : exps)
            executor.execute(exp);
        
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all threads");            
    }
}
