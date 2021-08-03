package experiments;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import weka.classifiers.Classifier;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.logging.Level;

@Parameters(separators = "=")
public class ExperimentalArguments implements Runnable {

    //REQUIRED PARAMETERS
    @Parameter(names = {"-dp", "--dataPath"}, required = true, order = 0, description = "(String) The directory that contains the dataset to be evaluated on, in the form "
            + "[--dataPath]/[--datasetName]/[--datasetname].arff (the actual arff file(s) may be in different forms, see ClassifierExperiments.sampleDataset(...).")
    public String dataReadLocation = null;

    @Parameter(names = {"-rp", "--resultsPath"}, required = true, order = 1, description = "(String) The parent directory to write the results of the evaluation to, in the form "
            + "[--resultsPath]/[--classifierName]/Predictions/[--datasetName]/...   This defaults to current working directory + 'results/' ")
    public String resultsWriteLocation = "results/";

    @Parameter(names = {"-cn", "--classifierName"}, required = true, order = 2, description = "(String) The name of the classifier to evaluate. A case matching this value should exist within the ClassifierLists")
    public String classifierName = null;

    @Parameter(names = {"-dn", "--datasetName"}, required = true, order = 3, description = "(String) The name of the dataset to be evaluated on, which resides within the dataPath in the form "
            + "[--dataPath]/[--datasetName]/[--datasetname].arff (the actual arff file(s) may be of different forms, see ClassifierExperiments.sampleDataset(...).")
    public String datasetName = null;

    @Parameter(names = {"-f", "--fold"}, required = true, order = 4, description = "(int) The fold index for dataset resampling, also used as the rng seed. *Indexed from 1* to conform with cluster array "
            + "job indices. The fold id pass will be automatically decremented to be zero-indexed internally.")
    public int foldId = 0;

    //OPTIONAL PARAMETERS
    @Parameter(names = {"--help"}, hidden = true) //hidden from usage() printout
    private boolean help = false;

    //todo separate verbosity into it own thing
    @Parameter(names = {"-d", "--debug"}, arity = 1, description = "(boolean) Increases verbosity and turns on the printing of debug statements")
    public boolean debug = false;

    @Parameter(names = {"-s", "--seed"}, arity = 1, description = "(int) seed for the classifier. If not set the foldId is used as the seed unless --useSeed is set to false.")
    public int seed = Integer.MIN_VALUE;

    @Parameter(names = {"-us", "--useSeed"}, arity = 1, description = "(boolean) Whether to use the foldId or seed (if set) for the classifier, defaults to true. If false "
            + "prevents the classifiers setSeed() from being called.")
    public boolean useSeed = true;

    @Parameter(names = {"-gtf", "--genTrainFiles"}, arity = 1, description = "(boolean) Turns on the production of trainFold[fold].csv files, the results of which are calculate either via a cross validation of "
            + "the train data, or if a classifier implements the TrainAccuracyEstimate interface, the classifier will write its own estimate via its own means of evaluation.")
    public boolean generateErrorEstimateOnTrainSet = false;

    @Parameter(names = {"-cp", "--checkpointing"}, arity = 1, description = "(boolean or String) Turns on the usage of checkpointing, if the classifier implements the SaveParameterInfo and/or CheckpointClassifier interfaces. "
            + "Default is false/0, for no checkpointing. if -cp = true, checkpointing is turned on and checkpointing frequency is determined by the classifier. if -cp is a timing of the form [int][char], e.g. 1h, "
            + "checkpoints shall be made at that frequency (as close as possible according to the atomic unit of learning for the classifier). Possible units, in order: n (nanoseconds), u, m, s, M, h, d (days)."
            + "Lastly, if -cp is of the the [int] only, it is assumed to be a timing in hours."
            + "The classifier by default will write its checkpointing files to workspace path parallel to the --resultsPath, unless another path is optionally supplied to --supportingFilePath.")
    private String checkpointingStr = null;
    public boolean checkpointing = false;
    public long checkpointInterval = 0;

    @Parameter(names = {"-vis", "--visualisation"}, arity = 1, description = "(boolean) Turns on the production of visualisation files, if the classifier implements the Visualisable interface. "
            + "Figures are created using Python. Exact requirements are to be determined, but a a Python 3.7 installation is the current recommendation with the numpy and matplotlib packages installed on the global environment. "
            + "The classifier by default will write its visualisation files to workspace path parallel to the --resultsPath, unless another path is optionally supplied to --supportingFilePath.")
    public boolean visualise = false;

    @Parameter(names = {"-int", "--interpretability"}, arity = 1, description = "(boolean) Turns on the production of interpretability files, if the classifier implements the Interpretable interface. "
            + "The classifier by default will write its interpretability files to workspace path parallel to the --resultsPath, unless another path is optionally supplied to --supportingFilePath.")
    public boolean interpret = false;

    @Parameter(names = {"-sp", "--supportingFilePath"}, description = "(String) Specifies the directory to write any files that may be produced by the classifier if it is a FileProducer. This includes but may not be "
            + "limited to: parameter evaluations, checkpoints, and logs. By default, these files are written to a generated subdirectory in the same location that the train and testFold[fold] files are written, relative"
            + "the --resultsPath. If a path is supplied via this parameter however, the files shall be written to that precisely that directory, as opposed to e.g. [-sp]/[--classifierName]/Predictions... "
            + "THIS IS A PLACEHOLDER PARAMETER. TO BE FULLY IMPLEMENTED WHEN INTERFACES AND SETCLASSIFIER ARE UPDATED.")
    public String supportingFilePath = null;

    @Parameter(names = {"-pid", "--parameterSplitIndex"}, description = "(Integer) If supplied and the classifier implements the ParameterSplittable interface, this execution of experiments will be set up to evaluate "
            + "the parameter set -pid within the parameter space used by the classifier (whether that be a supplied space or default). How the integer -pid maps onto the parameter space is up to the classifier.")
    public Integer singleParameterID = null;

    @Parameter(names = {"-tb", "--timingBenchmark"}, arity = 1, description = "(boolean) Turns on the computation of a standard operation to act as a simple benchmark for the speed of computation on this hardware, which may "
            + "optionally be used to normalise build/test/predictions times across hardware in later analysis. Expected time on Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz is ~0.8 seconds. For experiments that are likely to be very "
            + "short, it is recommended to leave this off, as it will proportionally increase the total time to perform all your experiments by a great deal, and for short evaluation time the proportional affect of "
            + "any processing noise may make any benchmark normalisation process unreliable anyway.")
    public boolean performTimingBenchmark = false;

    //todo expose the filetype enum in some way, currently just using an unconnected if statement, if e.g the order of the enum values changes in the classifierresults, which we have no knowledge
    //of here, the ifs will call the wrong things. decide on the design of this
    @Parameter(names = {"-ff", "--fileFormat"}, description = "(int) Specifies the format for the classifier results file to be written in, accepted values = { 0, 1, 2 }, default = 0. 0 writes the first 3 lines of meta information "
            + "as well as the full prediction information, and requires the most disk space. 1 writes the first three lines and a list of the performance metrics calculated from the prediction info. 2 writes the first three lines only, and "
            + "requires the least space. Use options other than 0 if generating too many files with too much prediction information for the disk space available, however be aware that there is of course a loss of information.")
    public int classifierResultsFileFormat = 0;

    @Parameter(names = {"-nt", "--numberOfThreads"}, arity = 1, description = "(int) Number of threads to be set for MultiThreadable classifiers, defaults to 1. If set to"
            + " < 1, Runtime.getRuntime().availableProcessors()-1 threads are used.")
    public int numberOfThreads = 1;

    @Parameter(names = {"-co", "--classifierOptions"}, arity = 1, description = "(String) Classifier specific comma delimited options string to be split and passed to a"
            + " classifiers setOptions() method. Each option should have the parameter name/tag, a comma and then the parameter value for each options i.e. T,500,I,0.5")
    private String classifierOptionsStr = null;
    public String[] classifierOptions = null;

    @Parameter(names = {"-ctr", "--contractTrain"}, description = "(String) Defines a time limit for the training of the classifier if it implements the TrainTimeContractClassifier interface. Defaults to "
            + "no contract time. If an integral value is given, it is assumed to be in HOURS. Otherwise, a string of the form [int][char] can be supplied, with the [char] defining the time unit. "
            + "e.g.1 10s = 10 seconds,   e.g.2 1h = 60M = 3600s. Possible units, in order: n (nanoseconds), u, m, s, M, h, d (days).")
    private String contractTrainTimeString = null;
    public long contractTrainTimeNanos = 0;

    @Parameter(names = {"-cte", "--contractTest"}, description = "(String) Defines a time limit for the testing of the classifier if it implements the TestTimeContractable interface. Defaults to "
            + "no contract time. If an integral value is given, it is assumed to be in HOURS. Otherwise, a string of the form [int][char] can be supplied, with the [char] defining the time unit. "
            + "e.g.1 10s = 10 seconds,   e.g.2 1h = 60M = 3600s. Possible units, in order: n (nanoseconds), u, m, s, M, h, d (days).")
    private String contractTestTimeString = null;
    public long contractTestTimeNanos = 0;

    @Parameter(names = {"-sc", "--serialiseClassifier"}, arity = 1, description = "(boolean) If true, and the classifier is serialisable, the classifier will be serialised to the --supportingFilesPath after training, but before testing.")
    public boolean serialiseTrainedClassifier = false;

    @Parameter(names = {"--force"}, arity = 1, description = "(boolean) If true, the evaluation will occur even if what would be the resulting files already exists. The old files will be overwritten with the new evaluation results.")
    public boolean forceEvaluation = false;

    @Parameter(names = {"--forceTest"}, arity = 1, description = "(boolean) If true, the evaluation will occur even if what would be the resulting test file already exists. The old test file will be overwritten with the new evaluation results.")
    public boolean forceEvaluationTestFold = false;

    @Parameter(names = {"--forceTrain"}, arity = 1, description = "(boolean) If true, the evaluation will occur even if what would be the resulting train file already exists. The old train file will be overwritten with the new evaluation results.")
    public boolean forceEvaluationTrainFold = false;

    @Parameter(names = {"-tem", "--trainEstimateMethod"}, arity = 1, description = "(String) Defines the method and parameters of the evaluation method used to estimate error on the train set, if --genTrainFiles == true. Current implementation is a hack to get the option in for"
            + " experiment running in the short term. Give one of 'cv' and 'hov' for cross validation and hold-out validation set respectively, and a number of folds (e.g. cv_10) or train set proportion (e.g. hov_0.7) respectively. Default is a 10 fold cv, i.e. cv_10.")
    public String trainEstimateMethod = "cv_10";

    @Parameter(names = {"--conTrain"}, arity = 2, description = "todo")
    private List<String> trainContracts = new ArrayList<>();

    @Parameter(names = {"--contractInName"}, arity = 1, description = "todo")
    private boolean appendTrainContractToClassifierName = true;

    @Parameter(names = {"-l", "--logLevel"}, description = "log level")
    private String logLevelStr = null;

    private Level logLevel = null;

    public boolean hasTrainContracts() {
        return trainContracts.size() > 0;
    }


    // calculated/set during experiment setup, indirectly using the parameters passed
    public String trainFoldFileName = null;
    public String testFoldFileName = null;

    // a function that returns a classifier instance, mainly for generating multiple instances for different
    // threaded exps. If not supplied (default), the classifier is instantiated via setClassifier(classifierName)
    public Supplier<Classifier> classifierGenerator = null;
    public Classifier classifier = null;

    public ExperimentalArguments() {

    }

    public ExperimentalArguments(String[] args) throws Exception {
        parseArguments(args);
    }

    @Override //Runnable
    public void run() {
        try {
            ClassifierExperiments.setupAndRunExperiment(this);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    /**
     * This is a bit of a bolt-on method for now. It assumes that the object on which
     * this method is being called has all the other parameters not passed to it set already
     * (e.g data location, results location) and these will be replicated across all experiments.
     * The current value of this.classifierName, this.datasetName, and this.foldId are ignored within
     * this method.
     *
     * @param minFold inclusive
     * @param maxFold exclusive, i.e will make folds [ for (int f = minFold; f < maxFold; ++f) ]
     * @return a list of unique experimental arguments, covering all combinations of classifier, datasets, and folds passed, with the same meta info as 'this' currently stores
     */
    public List<ExperimentalArguments> generateExperiments(String[] classifierNames, List<Supplier<Classifier>> classifierGenerators, String[] datasetNames, int minFold, int maxFold) {

        if (minFold > maxFold) {
            int t = minFold;
            minFold = maxFold;
            maxFold = t;
        }

        ArrayList<ExperimentalArguments> exps = new ArrayList<>(classifierNames.length * datasetNames.length * (maxFold - minFold));


        for (int i = 0; i < classifierNames.length; i++) {
            String classifier = classifierNames[i];

            for (String dataset : datasetNames) {
                for (int fold = minFold; fold < maxFold; fold++) {
                    ExperimentalArguments exp = new ExperimentalArguments();
                    exp.classifierName = classifier;
                    exp.datasetName = dataset;
                    exp.foldId = fold;

                    // enforce that if a classifier instance has been provided, it's nulled to avoid
                    // the same instance being accessed across multiple threads
                    exp.classifier = null;

                    if (classifierGenerators != null && classifierGenerators.get(i) != null)
                        exp.classifierGenerator = classifierGenerators.get(i);
                    else
                        exp.classifierGenerator = null;


                    // copying fields via reflection now to avoid cases of forgetting to account for newly added paras
                    for (Field field : ExperimentalArguments.class.getFields()) {

                        // these are the ones being set individually per exp, skip the copying over
                        if (field.getName().equals("classifierName") ||
                                field.getName().equals("datasetName") ||
                                field.getName().equals("foldId") ||
                                field.getName().equals("classifier") ||
                                field.getName().equals("classifierGenerator")
                        )
                            continue;

                        try {
                            field.set(exp, field.get(this));
                        } catch (IllegalAccessException ex) {
                            System.out.println("Fatal, should-be-unreachable exception thrown while copying across exp args");
                            System.out.println(ex);
                            ex.printStackTrace();
                            System.exit(0);
                        }
                    }

                    exps.add(exp);
                }
            }
        }

        return exps;
    }

    private void parseArguments(String[] args) throws Exception {
        JCommander.Builder b = JCommander.newBuilder();
        b.addObject(this);
        JCommander jc = b.build();
        jc.setProgramName("ClassifierExperiments.java");  //todo maybe add copyright etcetc
        try {
            jc.parse(args);
        } catch (Exception e) {
            if (!help) {
                //we actually errored, instead of the program simply being called with the --help flag
                System.err.println("Parsing of arguments failed, parameter information follows after the error. Parameters that require values should have the flag and value separated by '='.");
                System.err.println("For example: java -jar TimeSeriesClassification.jar -dp=data/path/ -rp=results/path/ -cn=someClassifier -dn=someDataset -f=0");
                System.err.println("Parameters prefixed by a * are REQUIRED. These are the first five parameters, which are needed to run a basic experiment.");
                System.err.println("Error: \n\t" + e + "\n\n");
            }
            jc.usage();
//                Thread.sleep(1000); //usage can take a second to print for some reason?... no idea what it's actually doing
//                System.exit(1);
        }

        foldId -= 1; //go from one-indexed to zero-indexed
        ClassifierExperiments.debug = this.debug;

        resultsWriteLocation = StrUtils.asDirPath(resultsWriteLocation);
        dataReadLocation = StrUtils.asDirPath(dataReadLocation);
        if (checkpointingStr != null) {
            //some kind of checkpointing is wanted

            // is it simply "true"?

            checkpointing = Boolean.parseBoolean(checkpointingStr.toLowerCase());
            if (!checkpointing) {
                //it's not. must be a timing string
                checkpointing = true;
                checkpointInterval = parseTiming(checkpointingStr);

            }
        }

        if (classifierOptionsStr != null)
            classifierOptions = classifierOptionsStr.split(",");

        //populating the contract times if present
        if (contractTrainTimeString != null)
            contractTrainTimeNanos = parseTiming(contractTrainTimeString);
        if (contractTestTimeString != null)
            contractTestTimeNanos = parseTiming(contractTestTimeString);

        if (contractTrainTimeNanos > 0) {
            trainContracts.add(String.valueOf(contractTrainTimeNanos));
            trainContracts.add(TimeUnit.NANOSECONDS.toString());
        }

        // check the contracts are in ascending order // todo sort them
        for (int i = 1; i < trainContracts.size(); i += 2) {
            trainContracts.set(i, trainContracts.get(i).toUpperCase());
        }
        long prev = -1;
        for (int i = 0; i < trainContracts.size(); i += 2) {
            long nanos = TimeUnit.NANOSECONDS.convert(Long.parseLong(trainContracts.get(i)),
                    TimeUnit.valueOf(trainContracts.get(i + 1)));
            if (prev > nanos) {
                throw new IllegalArgumentException("contracts not in asc order");
            }
            prev = nanos;
        }

        if (trainContracts.size() % 2 != 0) {
            throw new IllegalStateException("illegal number of args for time");
        }

        if (logLevelStr != null) {
            logLevel = Level.parse(logLevelStr);
        }
    }

    /**
     * Helper func to parse a timing string of the form [int][char], e.g. 10s = 10 seconds = 10,000,000,000 nanosecs.
     * 1h = 60M = 3600s = 3600,000,000,000n
     * <p>
     * todo Alternatively, string can be of form [int][TimeUnit.toString()], e.g. 10SECONDS
     * <p>
     * If just a number is given without a time unit character, HOURS is assumed to be the time unit
     * <p>
     * Possible time unit chars:
     * n - nanoseconds
     * u - microseconds
     * m - milliseconds
     * s - seconds
     * M - minutes
     * h - hours
     * d - days
     * w - weeks
     * <p>
     * todo learn/use java built in timing things if really wanted, e.g. TemporalAmount
     *
     * @return long number of nanoseconds the input string represents
     */
    private long parseTiming(String timeStr) throws IllegalArgumentException {
        try {
            // check if it's just a number, in which case return it under assumption that it's in hours
            int val = Integer.parseInt(timeStr);
            return TimeUnit.NANOSECONDS.convert(val, TimeUnit.HOURS);
        } catch (Exception e) {
            //pass
        }

        // convert it
        char unit = timeStr.charAt(timeStr.length() - 1);
        int amount = Integer.parseInt(timeStr.substring(0, timeStr.length() - 1));

        long nanoAmount = 0;

        switch (unit) {
            case 'n':
                nanoAmount = amount;
                break;
            case 'u':
                nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.MICROSECONDS);
                break;
            case 'm':
                nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.MILLISECONDS);
                break;
            case 's':
                nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.SECONDS);
                break;
            case 'M':
                nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.MINUTES);
                break;
            case 'h':
                nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.HOURS);
                break;
            case 'd':
                nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.DAYS);
                break;
            default:
                throw new IllegalArgumentException("Unrecognised time unit string conversion requested, was given " + timeStr);
        }

        return nanoAmount;
    }

    public String toShortString() {
        return "[" + classifierName + "," + datasetName + "," + foldId + "]";
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("EXPERIMENT SETTINGS " + this.toShortString());

        // printing fields via reflection now to avoid cases of forgetting to account for newly added  paras
        for (Field field : ExperimentalArguments.class.getFields()) {
            try {
                sb.append("\n").append(field.getName()).append(": ").append(field.get(this));
            } catch (IllegalAccessException ex) {
                System.out.println("Fatal, should-be-unreachable exception thrown while printing exp args");
                System.out.println(ex);
                ex.printStackTrace();
                System.exit(0);
            }
        }

        return sb.toString();
    }
}
