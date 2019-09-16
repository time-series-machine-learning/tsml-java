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
package weka_extras.classifiers.ensembles;

import experiments.CollateResults;
import experiments.Experiments;
import evaluation.MultipleClassifierEvaluation;
import weka_extras.classifiers.ensembles.weightings.TrainAcc;
import weka_extras.classifiers.ensembles.weightings.TrainAccByClass;
import weka_extras.classifiers.ensembles.voting.MajorityVote;

import java.io.File;

import utilities.ClassifierTools;
import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.evaluators.StratifiedResamplesEvaluator;
import evaluation.storage.ClassifierResults;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import experiments.data.DatasetLoading;
import weka_extras.classifiers.ensembles.voting.MajorityConfidence;
import timeseriesweka.filters.SAX;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka_extras.classifiers.kNN;

/**
 * Can be constructed and will be ready for use from the default constructor like any other classifier.
 * Default settings are equivalent to the CAWPE in the paper.
 See exampleCAWPEUsage() for more detailed options on defining different component sets, ensemble schemes, and file handling


 For examples of file creation and results analysis for reproduction purposes, see
 buildCAWPEPaper_AllResultsForFigure3()


 CLASSIFICATION SETTINGS:
 Default setup is defined by setupDefaultEnsembleSettings(), i.e:
   Comps: SVML, MLP, NN, Logistic, C4.5
   Weight: TrainAcc(4) (train accuracies to the power 4)
   Vote: MajorityConfidence (summing probability distributions)

 For the original settings used in an older version of cote, call setupOriginalHESCASettings(), i.e:
   Comps: NN, SVML, SVMQ, C4.5, NB, bayesNet, RotF, RandF
   Weight: TrainAcc
   Vote: MajorityVote

 EXPERIMENTAL USAGE:
 By default will build/trainEstimator members normally, and perform no file reading/writing.
 To turn on file handling of any kind, call
          setResultsFileLocationParameters(...)
 1) Can build ensemble and classify from results files of its members, call
          setBuildIndividualsFromResultsFiles(true)
 2) If members built from scratch, can write the results files of the individuals with
          setWriteIndividualsTrainResultsFiles(true)
          and
          writeIndividualTestFiles(...) after testing is complete
 3) And can write the ensemble train/testing files with
         writeEnsembleTrainTestFiles(...) after testing is complete

 There are a bunch of little intricacies if you want to do stuff other than a bog standard run
 Best bet will be to email me for any specific usage questions.
 *
 * @author James Large (james.large@uea.ac.uk)
 *
 */

public class CAWPE extends AbstractEnsemble implements TechnicalInformationHandler {

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "J. Large, J. Lines and A. Bagnall");
        result.setValue(TechnicalInformation.Field.YEAR, "2019");
        result.setValue(TechnicalInformation.Field.MONTH, "June");
        result.setValue(TechnicalInformation.Field.TITLE, "A probabilistic classifier ensemble weighting scheme based on cross-validated accuracy estimates");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.URL, "https://link.springer.com/article/10.1007/s10618-019-00638-y");
        result.setValue(TechnicalInformation.Field.ISSN, "1573-756X");

        return result;
    }
    
    
    public CAWPE() {
        super();
    }


    /**
     * Uses the 'basic UCI' set up:
     * Comps: SVML, MLP, NN, Logistic, C4.5
     * Weight: TrainAcc(4) (train accuracies to the power 4)
     * Vote: MajorityConfidence (summing probability distributions)
     */
    public final void setupDefaultEnsembleSettings() {
        this.ensembleName = "CAWPE";
        
        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();
        this.transform = null;
        
        CrossValidationEvaluator cv = new CrossValidationEvaluator(seed, false, false, false, false); 
        cv.setNumFolds(10);
        this.trainEstimator = cv; 

        Classifier[] classifiers = new Classifier[5];
        String[] classifierNames = new String[5];

        SMO smo = new SMO();
        smo.turnChecksOff();
        smo.setBuildLogisticModels(true);
        PolyKernel kl = new PolyKernel();
        kl.setExponent(1);
        smo.setKernel(kl);
        smo.setRandomSeed(seed);
        classifiers[0] = smo;
        classifierNames[0] = "SVML";

        kNN k=new kNN(100);
        k.setCrossValidate(true);
        k.normalise(false);
        k.setDistanceFunction(new EuclideanDistance());
        classifiers[1] = k;
        classifierNames[1] = "NN";

        classifiers[2] = new J48();
        classifierNames[2] = "C4.5";

        classifiers[3] = new Logistic();
        classifierNames[3] = "Logistic";

        classifiers[4] = new MultilayerPerceptron();
        classifierNames[4] = "MLP";
        
        setClassifiers(classifiers, classifierNames, null);
    }

    /**
     * Uses the 'basic UCI' set up:
     * Comps: SVML, MLP, NN, Logistic, C4.5
     * Weight: TrainAcc(4) (train accuracies to the power 4)
     * Vote: MajorityConfidence (summing probability distributions)
     */
    public final void setupDefaultSettings_NoLogistic() {
        this.ensembleName = "CAWPE-NoLogistic";
        
        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();
        
        CrossValidationEvaluator cv = new CrossValidationEvaluator(seed, false, false, false, false); 
        cv.setNumFolds(10);
        this.trainEstimator = cv; 
        
        Classifier[] classifiers = new Classifier[4];
        String[] classifierNames = new String[4];

        SMO smo = new SMO();
        smo.turnChecksOff();
        smo.setBuildLogisticModels(true);
        PolyKernel kl = new PolyKernel();
        kl.setExponent(1);
        smo.setKernel(kl);
        smo.setRandomSeed(seed);
        classifiers[0] = smo;
        classifierNames[0] = "SVML";

        kNN k=new kNN(100);
        k.setCrossValidate(true);
        k.normalise(false);
        k.setDistanceFunction(new EuclideanDistance());
        classifiers[1] = k;
        classifierNames[1] = "NN";

        classifiers[2] = new J48();
        classifierNames[2] = "C4.5";

        classifiers[3] = new MultilayerPerceptron();
        classifierNames[3] = "MLP";

        setClassifiers(classifiers, classifierNames, null);
    }


    public final void setupAdvancedSettings() {
        this.ensembleName = "CAWPE-A";
        
        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();
        
        CrossValidationEvaluator cv = new CrossValidationEvaluator(seed, false, false, false, false); 
        cv.setNumFolds(10);
        this.trainEstimator = cv; 

        Classifier[] classifiers = new Classifier[3];
        String[] classifierNames = new String[3];

        SMO smo = new SMO();
        smo.turnChecksOff();
        smo.setBuildLogisticModels(true);
        PolyKernel kl = new PolyKernel();
        kl.setExponent(2);
        smo.setKernel(kl);
        smo.setRandomSeed(seed);
        classifiers[0] = smo;
        classifierNames[0] = "SVMQ";
        RandomForest rf= new RandomForest();
        rf.setNumTrees(500);
        classifiers[1] = rf;
        classifierNames[1] = "RandF";
        RotationForest rotf=new RotationForest();
        rotf.setNumIterations(200);
        classifiers[2] = rotf;
        classifierNames[2] = "RotF";

        setClassifiers(classifiers, classifierNames, null);
    }


    /**
     * Comps: NN, SVML, SVMQ, C4.5, NB, BN, RotF, RandF
     * Weight: TrainAcc
     * Vote: MajorityVote
     *
     * As used originally in ST_HESCA, COTE.
     */
    public final void setupOriginalHESCASettings() {
        this.ensembleName = "HESCA";
        
        this.weightingScheme = new TrainAcc();
        this.votingScheme = new MajorityVote();
        
        CrossValidationEvaluator cv = new CrossValidationEvaluator(seed, false, false, false, false); 
        cv.setNumFolds(10);
        this.trainEstimator = cv; 

        Classifier[] classifiers = new Classifier[8];
        String[] classifierNames = new String[8];

        kNN k=new kNN(100);
        k.setCrossValidate(true);
        k.normalise(false);
        k.setDistanceFunction(new EuclideanDistance());
        classifiers[0] = k;
        classifierNames[0] = "NN";

        classifiers[1] = new NaiveBayes();
        classifierNames[1] = "NB";

        classifiers[2] = new J48();
        classifierNames[2] = "C45";

        SMO svml = new SMO();
        svml.turnChecksOff();
        PolyKernel kl = new PolyKernel();
        kl.setExponent(1);
        svml.setKernel(kl);
        svml.setRandomSeed(seed);
        classifiers[3] = svml;
        classifierNames[3] = "SVML";

        SMO svmq =new SMO();
//Assumes no missing, all real valued and a discrete class variable
        svmq.turnChecksOff();
        PolyKernel kq = new PolyKernel();
        kq.setExponent(2);
        svmq.setKernel(kq);
        svmq.setRandomSeed(seed);
        classifiers[4] =svmq;
        classifierNames[4] = "SVMQ";

        RandomForest r=new RandomForest();
        r.setNumTrees(500);
        r.setSeed(seed);
        classifiers[5] = r;
        classifierNames[5] = "RandF";


        RotationForest rf=new RotationForest();
        rf.setNumIterations(50);
        rf.setSeed(seed);
        classifiers[6] = rf;
        classifierNames[6] = "RotF";

        classifiers[7] = new BayesNet();
        classifierNames[7] = "bayesNet";

        setClassifiers(classifiers, classifierNames, null);
    }


    
    
   
    public static void exampleCAWPEUsage() throws Exception {
        String datasetName = "ItalyPowerDemand";

        Instances train = DatasetLoading.loadDataNullable("c:/tsc problems/"+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = DatasetLoading.loadDataNullable("c:/tsc problems/"+datasetName+"/"+datasetName+"_TEST");

        //Uses predefined default settings. This is the CAWPE classifier built on 'simple' components in the paper, equivalent to setupDefaultEnsembleSettings()
        CAWPE cawpe = new CAWPE();

        //Setting a transform (not used in CAWPE paper, mostly for COTE/HiveCOTE or particular applications)
        SimpleBatchFilter transform = new SAX();
        cawpe.setTransform(transform);
        cawpe.setTransform(null); //back to null for this example

        //Setting member classifiers
        Classifier[] classifiers = new Classifier[] { new kNN() };
        String [] names = new String[] { "NN" };
        String [] params = new String[] { "k=1" };
        cawpe.setClassifiers(classifiers, names, params); //see setClassifiers(...) javadoc

        //Setting ensemble schemes
        cawpe.setWeightingScheme(new TrainAccByClass()); //or set new methods
        cawpe.setVotingScheme(new MajorityVote()); //some voting schemes require dist for inst to be defined

        //Using predefined default settings. This is the CAWPE classifier in the paper, equivalent to default constructor
        cawpe.setupDefaultEnsembleSettings();

        int resampleID = 0;
        cawpe.setSeed(resampleID);

        //File handling
        cawpe.setResultsFileLocationParameters("CAWPETest/", datasetName, resampleID); //use this to set the location for any results file reading/writing

        cawpe.setBuildIndividualsFromResultsFiles(true); //turns on file reading, will read from location provided in setResultsFileLocationParameters(...)
        cawpe.setWriteIndividualsTrainResultsFiles(true); //include this to turn on file writing for individuals trainFold# files
        //can only have one of these (or neither) set to true at any one time (internally, setting one to true
        //will automatically set the other to false)

        //Then build/test as normal
        cawpe.buildClassifier(train);
        System.out.println(ClassifierTools.accuracy(test, cawpe));

        //Call these after testing is complete for fill writing of the individuals test files, and ensemble train AND test files.
        boolean throwExceptionOnFileParamsNotSetProperly = false;
        cawpe.writeIndividualTestFiles(test.attributeToDoubleArray(test.classIndex()), throwExceptionOnFileParamsNotSetProperly);
        cawpe.writeEnsembleTrainTestFiles(test.attributeToDoubleArray(test.classIndex()), throwExceptionOnFileParamsNotSetProperly);
    }

    

    
    
    
    
    
    
    
    
    
    
    
    


    /**
     * This will build all the base classifier results
     *
     * @param dataHeaders e.g { "UCI", "UCR" }
     * @param dataPaths e.g { "C:/Data/UCI/", "C:/Data/UCR/" }
     * @param datasetNames for each datapath, a list of the dataset names located there to be used [archive][dsetnames]
     * @param classifiers the names of classifiers that can all be found in Experiments.setClassifier(...)
     * @param baseWritePath e.g { "C:/Results/" }
     */
    protected static void buildCAWPEPaper_BuildClassifierResultsFiles(String baseWritePath, String[] dataHeaders, String[] dataPaths,
                                                            String[][] datasetNames, String[] classifiers, int numFolds) throws Exception {
        for (int archive = 0; archive < dataHeaders.length; archive++) {
            for (String classifier : classifiers) {
                System.out.println("\t" + classifier);

                for (String dset : datasetNames[archive]) {
                    System.out.println(dset);
                    for (int fold = 0; fold < numFolds; fold++) {
                          /*1: Problem path args[0]
                            2. Results path args[1]
                            3. booleanw Whether to generate train files (true/false)
                            4. Classifier =args[3];
                            5. String problem=args[4];
                            6. int fold=Integer.parseInt(args[5])-1;
                        Optional:
                            7. boolean whether to checkpoint parameter search for applicable tuned classifiers (true/false)
                            8. integer for specific parameter search (0 indicates ignore this)
                            */
                        Experiments.main(new String[] { "-dp="+dataPaths[archive], "-rp="+baseWritePath+dataHeaders[archive]+"/", "-cn="+classifier, "-dn="+dset, "-f="+(fold+1), "-gtf=true"});
                    }
                }
            }
        }
    }

    /**
     * This method would build all the results files leading up to figure 3 of 
     * https://link.springer.com/article/10.1007/s10618-019-00638-y,
     * the heterogeneous ensemble comparison on the basic classifiers.
     *
     * It would take a long time to run, almost all of which is comprised of
     * building the base classifiers.
     *
     * The experiments and results presented in the paper were distributed on the HPC cluster at UEA,
     * this method is to demonstrate the experimental procedure and to provide a base to copy/edit for
     * full results reproduction of everything in the paper.
     *
     * There are also cases that can't be entirely captured neatly in a method like this, despite
     * my best efforts. For example, while we can call matlab code from here to build diagrams for
     * the analysis, the implementation of the DNN requires that to be run separately. Likewise, while
     * a lot of the legwork of analysis is done programmatically, the deeper exploratory analysis
     * cannot really be done automatically.
     *
     * Still, the idea of getting as close a possible to being able to reproduce the entirety
     * of a paper's results and statistics in a single function call is nice, especially for a
     * paper as extensive and empirically-driven as CAWPE's.
     *
     * For inquiries into specific details of reproduction, best bet is to email us
     * james.large@uea.ac.uk
     * anthony.bagnall@uea.ac.uk
     */
    protected static void buildCAWPEPaper_AllResultsForFigure3() throws Exception {
        //init, edit the paths for local running ofc
        String[] dataHeaders = { "UCI", };
        String[] dataPaths = { "C:/UCI Problems/", };
        String[][] datasets = { { "hayes-roth", "pittsburg-bridges-T-OR-D", "teaching", "wine" } };
        String writePathBase = "C:/Temp/MCEUpdateTests/CAWPEReprod05/";
        String writePathResults =  writePathBase + "Results/";
        String writePathAnalysis =  writePathBase + "Analysis/";
        int numFolds = 5;
//        String[] dataHeaders = { "UCI", };
//        String[] dataPaths = { "Z:/Data/UCIDelgado/", };
//        String[][] datasets = { DataSets.UCIContinuousFileNames, };
//        String writePathBase = "Z:/Results_7_2_19/CAWPEReproducabiltyTest2/";
//        String writePathResults =  writePathBase + "Results/";
//        String writePathAnalysis =  writePathBase + "Analysis/";
//        int numFolds = 30;

        //build the base classifiers
        String[] baseClassifiers = { "NN", "C45", "MLP", "Logistic", "SVML" };
        buildCAWPEPaper_BuildClassifierResultsFiles(writePathResults, dataHeaders, dataPaths, datasets, baseClassifiers, numFolds);

        //build the ensembles
        String[] ensembleIDsInStorage = {
            "CAWPE_BasicClassifiers",
            "EnsembleSelection_BasicClassifiers",
            "SMLR_BasicClassifiers",
            "SMLRE_BasicClassifiers",
            "SMM5_BasicClassifiers",
            "PickBest_BasicClassifiers",
            "MajorityVote_BasicClassifiers",
            "WeightMajorityVote_BasicClassifiers",
            "RecallCombiner_BasicClassifiers",
            "NaiveBayesCombiner_BasicClassifiers"
        };

        String[] ensembleIDsOnFigures = {
            "CAWPE", "ES", "SMLR", "SMLRE", "SMM5",
            "PB", "MV", "WMV", "RC", "NBC"
        };

        String pkg = "weka_extras.classifiers.ensembles.";
        Class[] ensembleClasses = {
            Class.forName(pkg + "CAWPE"),
            Class.forName(pkg + "EnsembleSelection"),
            Class.forName(pkg + "stackers.SMLR"),
            Class.forName(pkg + "stackers.SMLRE"),
            Class.forName(pkg + "stackers.SMM5"),
            Class.forName(pkg + "weightedvoters.CAWPE_PickBest"),
            Class.forName(pkg + "weightedvoters.CAWPE_MajorityVote"),
            Class.forName(pkg + "weightedvoters.CAWPE_WeightedMajorityVote"),
            Class.forName(pkg + "weightedvoters.CAWPE_RecallCombiner"),
            Class.forName(pkg + "weightedvoters.CAWPE_NaiveBayesCombiner"),
        };

        for (int ensemble = 0; ensemble < ensembleIDsInStorage.length; ensemble++)
            buildCAWPEPaper_BuildEnsembleFromResultsFiles(writePathResults, dataHeaders, dataPaths, datasets, baseClassifiers, numFolds, ensembleIDsInStorage[ensemble], ensembleClasses[ensemble]);



        //build the results analysis sheets and figures
        for (int archive = 0; archive < dataHeaders.length; archive++) {
            String analysisName = dataHeaders[archive] + "CAWPEvsHeteroEnsembles_BasicClassifiers";
            buildCAWPEPaper_BuildResultsAnalysis(writePathResults+dataHeaders[archive]+"/", writePathAnalysis,
                                       analysisName, ensembleIDsInStorage, ensembleIDsOnFigures, datasets[archive], numFolds);
        }

        //done!
    }
 
    protected static void buildCAWPEPaper_BuildResultsAnalysis(String resultsReadPath, String analysisWritePath,
                                       String analysisName, String[] classifiersInStorage, String[] classifiersOnFigs, String[] datasets, int numFolds) throws Exception {
        System.out.println("buildCAWPEPaper_BuildResultsAnalysis");

        new MultipleClassifierEvaluation(analysisWritePath, analysisName, numFolds).
            setTestResultsOnly(false).
//            setBuildMatlabDiagrams(true).
            setBuildMatlabDiagrams(false).
            setDatasets(datasets).
            readInClassifiers(classifiersInStorage, classifiersOnFigs, resultsReadPath).
            runComparison();
    }

    protected static void buildCAWPEPaper_BuildEnsembleFromResultsFiles(String baseWritePath, String[] dataHeaders, String[] dataPaths, String[][] datasetNames,
                                                                String[] baseClassifiers, int numFolds, String ensembleID, Class ensembleClass) throws Exception {

        Instances train = null, test = null, all = null; //UCR has predefined train/test splits, UCI data just comes as a whole, so are loaded/resampled differently
        Instances[] data = null; //however it's loaded/resampled, will eventually end up here, { train, test }

        for (int archive = 0; archive < dataHeaders.length; archive++) {
            String writePath = baseWritePath + dataHeaders[archive] + "/";

            for (String dset : datasetNames[archive]) {
                System.out.println(dset);

                if (dataHeaders[archive].equals("UCI"))
                    all = DatasetLoading.loadDataNullable(dataPaths[archive] + dset + "/" + dset + ".arff");
                else if ((dataHeaders[archive].contains("UCR"))) {
                    train = DatasetLoading.loadDataNullable(dataPaths[archive] + dset + "/" + dset + "_TRAIN.arff");
                    test = DatasetLoading.loadDataNullable(dataPaths[archive] + dset + "/" + dset + "_TEST.arff");
                }

                for (int fold = 0; fold < numFolds; fold++) {
                    //building particular ensembles with different parameters is a bit
                    //more involved so we skip some of the automated stages (especically setClassifier(...) in the
                    //experiments class to build the particular format wanted.
                    //in this example code, i've jsut assumed that default parameters
                    //(aside from the base classifiers) are being used.
                    //this code could ofc be editted to build whatever particular classifiers
                    //you want, instead of using the janky reflection

                    String predictions = writePath+ensembleID+"/Predictions/"+dset+"/";
                    File f=new File(predictions);
                    if(!f.exists())
                        f.mkdirs();

                    //Check whether fold already exists, if so, dont do it, just quit
                    if(!CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv")){
                        if (dataHeaders[archive].equals("UCI"))
                            data = InstanceTools.resampleInstances(all, fold, .5);
                        else if ((dataHeaders[archive].contains("UCR")))
                            data = InstanceTools.resampleTrainAndTestInstances(train, test, fold);

                        //cawpe is the base class from which all the heterogeneous ensembles are implemented, since this
                        //already has the base classifier file reading/writing built in etcetc.
                        CAWPE c = (CAWPE) ensembleClass.getConstructor().newInstance();

                        c.setEnsembleName(ensembleID);
                        c.setClassifiers(null, baseClassifiers, null);
                        c.setBuildIndividualsFromResultsFiles(true);
                        c.setResultsFileLocationParameters(writePath, dset, fold);
                        c.setSeed(fold);
                        c.setEstimateEnsemblePerformance(true);

                        //'custom' classifier built, now put it back in the normal experiments pipeline
                        Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments();
                        exp.classifierName = ensembleID;
                        exp.datasetName = dset;
                        exp.foldId = fold;
                        exp.generateErrorEstimateOnTrainSet = true;
                        Experiments.runExperiment(exp,data[0],data[1],c,predictions);
                    }
                }
            }
        }
    }

    public static void test_basic() throws Exception {
        System.out.println("test_basic()");
        
        int seed = 0;
        Instances[] data = DatasetLoading.sampleItalyPowerDemand(seed);
//        Instances[] data = DatasetLoading.sampleBeef(seed);
        
        StratifiedResamplesEvaluator trainEval = new StratifiedResamplesEvaluator();
        trainEval.setNumFolds(30);
        trainEval.setPropInstancesInTrain(0.5);
        trainEval.setSeed(seed);
        
        CAWPE c = new CAWPE();
        c.setSeed(seed);
//        c.setTrainEstimator(trainEval);
        
        long t1 = System.currentTimeMillis();
        c.buildClassifier(data[0]);
        t1 = System.currentTimeMillis() - t1;
        
        SingleTestSetEvaluator eval = new SingleTestSetEvaluator();
        eval.setSeed(seed);
        ClassifierResults res = eval.evaluate(c, data[1]);
        
        System.out.println("acc="+res.getAcc() 
                + " buildtime="+t1+"ms");
        System.out.print("BaseClassifier train accs: ");
        for (EnsembleModule module : c.getModules())
            System.out.print(module.getModuleName() + ":" +module.trainResults.getAcc() + ", ");
        System.out.println("");
        System.out.println("IPD_CrossValidation: " + 0.9650145772594753);
        System.out.println("IPD_StratifiedResample: " + 0.9630709426627794);
    }
    
    public static void test_threaded() throws Exception {
        System.out.println("test_threaded()");
        
        int seed = 0;
        Instances[] data = DatasetLoading.sampleItalyPowerDemand(seed);
//        Instances[] data = DatasetLoading.sampleBeef(seed);
        
        StratifiedResamplesEvaluator trainEval = new StratifiedResamplesEvaluator();
        trainEval.setNumFolds(30);
        trainEval.setPropInstancesInTrain(0.5);
        trainEval.setSeed(seed);
        
        CAWPE c = new CAWPE();
        c.setSeed(seed);
//        c.setTrainEstimator(trainEval);
        c.enableMultiThreading();
        
        long t1 = System.currentTimeMillis();
        c.buildClassifier(data[0]);
        t1 = System.currentTimeMillis() - t1;
        
        SingleTestSetEvaluator eval = new SingleTestSetEvaluator();
        eval.setSeed(seed);
        ClassifierResults res = eval.evaluate(c, data[1]);
        System.out.println("acc="+res.getAcc() 
                + " buildtime="+t1+"ms");
        System.out.print("BaseClassifier train accs: ");
        for (EnsembleModule module : c.getModules())
            System.out.print(module.getModuleName() + ":" +module.trainResults.getAcc() + ", ");
        System.out.println("");
        System.out.println("IPD_CrossValidation: " + 0.9650145772594753);
        System.out.println("IPD_StratifiedResample: " + 0.9630709426627794);
    }

    public static void main(String[] args) throws Exception {
//        exampleCAWPEUsage();

        buildCAWPEPaper_AllResultsForFigure3();
//        test_basic();
//        System.out.println("");
//        test_threaded();
        
        //run:
        //test_basic()
        //acc=0.9650145772594753 buildtime=1646ms
        //BaseClassifier train accs: SVML:0.9701492537313433, NN:0.9552238805970149, C4.5:0.9552238805970149, Logistic:0.9402985074626866, MLP:0.9701492537313433, 
        //IPD_CrossValidation: 0.9650145772594753
        //IPD_StratifiedResample: 0.9630709426627794
        //
        //test_threaded()
        //acc=0.9650145772594753 buildtime=532ms
        //BaseClassifier train accs: SVML:0.9701492537313433, NN:0.9552238805970149, C4.5:0.9552238805970149, Logistic:0.9402985074626866, MLP:0.9701492537313433, 
        //IPD_CrossValidation: 0.9650145772594753
        //IPD_StratifiedResample: 0.9630709426627794
        //BUILD SUCCESSFUL (total time: 2 seconds)
        
//        testBuildingInds(3);
//        testLoadingInds(2);
    }
}
