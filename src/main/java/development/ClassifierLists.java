
package development;

import static development.ExperimentsClean.debug;
import multivariate_timeseriesweka.classifiers.MultivariateShapeletTransformClassifier;
import multivariate_timeseriesweka.classifiers.NN_DTW_A;
import multivariate_timeseriesweka.classifiers.NN_DTW_D;
import multivariate_timeseriesweka.classifiers.NN_DTW_I;
import multivariate_timeseriesweka.classifiers.NN_ED_I;
import timeseriesweka.classifiers.BOSS;
import timeseriesweka.classifiers.BagOfPatterns;
import timeseriesweka.classifiers.DD_DTW;
import timeseriesweka.classifiers.DTD_C;
import timeseriesweka.classifiers.ElasticEnsemble;
import timeseriesweka.classifiers.FastShapelets;
import timeseriesweka.classifiers.FastWWS.FastDTWWrapper;
import timeseriesweka.classifiers.FlatCote;
import timeseriesweka.classifiers.HiveCote;
import timeseriesweka.classifiers.LPS;
import timeseriesweka.classifiers.LearnShapelets;
import timeseriesweka.classifiers.NN_CID;
import timeseriesweka.classifiers.RISE;
import timeseriesweka.classifiers.SAXVSM;
import timeseriesweka.classifiers.ShapeletTransformClassifier;
import timeseriesweka.classifiers.SlowDTW_1NN;
import timeseriesweka.classifiers.TSBF;
import timeseriesweka.classifiers.TSF;
import timeseriesweka.classifiers.WEASEL;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ED1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.MSM1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.WDTW1NN;
import vector_classifiers.CAWPE;
import vector_classifiers.ContractRotationForest;
import vector_classifiers.RotationForestBootstrap;
import vector_classifiers.TunedMultiLayerPerceptron;
import vector_classifiers.TunedRandomForest;
import vector_classifiers.TunedRotationForest;
import vector_classifiers.TunedSVM;
import vector_classifiers.TunedSingleLayerMLP;
import vector_classifiers.TunedTwoLayerMLP;
import vector_classifiers.TunedXGBoost;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.EuclideanDistance;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class ClassifierLists {
    public static String horribleGlobalPath="";
    public static String nastyGlobalDatasetName=""; //leaving in for now

    public String[] bakeOffClassifierList = { };    //todo, as an example of the kind of thing we could do with this class
    public String[] CAWPE_fig1Ensembles = { };      //todo, as an example of the kind of thing we could do with this class
    
    /**
     * This is the method exactly as it was in experiments. 
     * 
     * @param classifier
     * @param fold
     * @return 
     */
    public static Classifier setClassifierClassic(String classifier, int fold){
        Classifier c=null;
        TunedSVM svm=null;
        switch(classifier){
            case "ContractRotationForest":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setDayLimit(5);
                ((ContractRotationForest)c).setSeed(fold);
                
                break;
            case "ContractRotationForest1Day":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(24);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest5Minutes":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setMinuteLimit(5);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest30Minutes":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setMinuteLimit(30);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest1Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(1);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest2Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(2);
                break;
            case "ContractRotationForest3Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(3);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest12Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(12);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            
            case "ShapeletI": case "Shapelet_I": case "ShapeletD": case "Shapelet_D": case  "Shapelet_Indep"://Multivariate version 1
                c=new MultivariateShapeletTransformClassifier();
//Default to 1 day max run: could do this better
                ((MultivariateShapeletTransformClassifier)c).setOneDayLimit();
                ((MultivariateShapeletTransformClassifier)c).setSeed(fold);
                ((MultivariateShapeletTransformClassifier)c).setTransformType(classifier);
                break;
            case "ED_I":
                c=new NN_ED_I();
                break;
            case "DTW_I":
                c=new NN_DTW_I();
                break;
            case "DTW_D":
                c=new NN_DTW_D();
                break;
            case "DTW_A":
                c=new NN_DTW_A();
                break;
//TIME DOMAIN CLASSIFIERS   
            
            case "ED":
                c=new ED1NN();
                break;
            case "C45":
                c=new J48();
                break;
            case "NB":
                c=new NaiveBayes();
                break;
            case "SVML":
                c=new SMO();
                PolyKernel p=new PolyKernel();
                p.setExponent(1);
                ((SMO)c).setKernel(p);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                break;
            case "SVMQ": case "SVMQuad":
                c=new SMO();
                PolyKernel poly=new PolyKernel();
                poly.setExponent(2);
                ((SMO)c).setKernel(poly);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);

                
                /*                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.QUADRATIC);
                svm.optimiseParas(false);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
 */               break;
            case "SVMRBF": 
                c=new SMO();
                RBFKernel rbf=new RBFKernel();
                rbf.setGamma(0.5);
                ((SMO)c).setC(5);
                ((SMO)c).setKernel(rbf);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                
                break;
            case "BN":
                c=new BayesNet();
                break;
            case "MLP":
                c=new MultilayerPerceptron();
                break;
            case "TwoLayerMLP":
                TunedTwoLayerMLP twolayer=new TunedTwoLayerMLP();
                twolayer.setParamSearch(false);
                twolayer.setSeed(fold);
                c= twolayer;
                break;
                
            case "RandFOOB":
                c= new TunedRandomForest();
                ((RandomForest)c).setNumTrees(500);
                ((TunedRandomForest)c).tuneParameters(false);
                ((TunedRandomForest)c).setCrossValidate(false);
                ((TunedRandomForest)c).setEstimateAcc(true);
                ((TunedRandomForest)c).setSeed(fold);
                ((TunedRandomForest)c).setDebug(debug);
                
                break;
            case "RandF": case "RandomForest": case "RandF500": case "RandomForest500":
                c= new TunedRandomForest();
                ((RandomForest)c).setNumTrees(500);
                ((TunedRandomForest)c).tuneParameters(false);
                ((TunedRandomForest)c).setCrossValidate(true);
                ((TunedRandomForest)c).setSeed(fold);
                break;
            case "RandF10000":
                c= new TunedRandomForest();
                ((RandomForest)c).setNumTrees(10000);
                ((TunedRandomForest)c).tuneParameters(false);
                ((TunedRandomForest)c).setCrossValidate(false);
                ((TunedRandomForest)c).setSeed(fold);
                break;


            case "RotF": case "RotationForest": case "RotF200": case "RotationForest200":
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(200);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFRandomTree":
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(200);
                ((RotationForest)c).setClassifier(new RandomTree());
                
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;


            case "RotFBootstrap":
                c= new RotationForestBootstrap();
                ((RotationForestBootstrap)c).setNumIterations(200);
                ((RotationForestBootstrap)c).setSeed(fold);
                ((RotationForestBootstrap)c).tuneParameters(false);
                ((RotationForestBootstrap)c).setSeed(fold);
                ((RotationForestBootstrap)c).estimateAccFromTrain(false);
                break;
            case "RotFLimited":
                c= new RotationForestLimitedAttributes();
                ((RotationForestLimitedAttributes)c).setNumIterations(200);
                ((RotationForestLimitedAttributes)c).tuneParameters(false);
                ((RotationForestLimitedAttributes)c).setSeed(fold);
                ((RotationForestLimitedAttributes)c).estimateAccFromTrain(false);
                break;
            case "TunedRandF":
                c= new TunedRandomForest();
                ((TunedRandomForest)c).tuneParameters(true);
                ((TunedRandomForest)c).setCrossValidate(true);
                ((TunedRandomForest)c).setSeed(fold);             
                ((TunedRandomForest)c).setDebug(debug);
                break;
            case "TunedRandFOOB":
                c= new TunedRandomForest();
                ((TunedRandomForest)c).tuneParameters(true);
                ((TunedRandomForest)c).setCrossValidate(false);
                ((TunedRotationForest)c).setSeed(fold);
                break;
            case "TunedRotF":
                c= new TunedRotationForest();
                ((TunedRotationForest)c).tuneParameters(true);
                ((TunedRotationForest)c).setSeed(fold);
                break;
            case "TunedSVMRBF":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.RBF);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
                break;
            case "TunedSVMQuad":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.QUADRATIC);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                svm.setLargePolynomialParameterSpace(1089);                
                c= svm;
                break;
            case "TunedSVMQuad17": // C in {-16 -14 -12....12 14 16} 
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.QUADRATIC);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                svm.setLargePolynomialParameterSpace(17);                
                c= svm;
                break;
            case "TunedSVMLinear":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.LINEAR);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                svm.setLargePolynomialParameterSpace(1089);
                c= svm;
                break;
            case "TunedSVMLinear17": // C in {-16 -14 -12....12 14 16} 
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.LINEAR);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                svm.setLargePolynomialParameterSpace(17);
                c= svm;
                break;
            case "TunedSVMPolynomial":
                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.POLYNOMIAL);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
                break;
            case "TunedSVMKernel":
                svm=new TunedSVM();
                svm.optimiseParas(true);
                svm.optimiseKernel(true);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
                break;
            case "TunedSingleLayerMLP":
                TunedSingleLayerMLP mlp=new TunedSingleLayerMLP();
                mlp.setParamSearch(true);
                mlp.setTrainingTime(200);
                mlp.setSeed(fold);
                c= mlp;
                break;
            case "TunedTwoLayerMLP":
                TunedTwoLayerMLP mlp2=new TunedTwoLayerMLP();
                mlp2.setParamSearch(true);
                mlp2.setSeed(fold);
                c= mlp2;
                break;
            case "TunedMultiLayerPerceptron":
                TunedMultiLayerPerceptron mlp3=new TunedMultiLayerPerceptron();
               
                mlp3.setParamSearch(true);
                mlp3.setSeed(fold);
                mlp3.setTrainingTime(200);
                c= mlp3;
                break;
            case "RandomRotationForest1":
                c= new RotationForestLimitedAttributes();
                ((RotationForestLimitedAttributes)c).setNumIterations(200);
                ((RotationForestLimitedAttributes)c).setMaxNumAttributes(100);
                break;
            case "Logistic":
                c= new Logistic();
                break;
            case "NN":
                kNN k=new kNN(100);
                k.setCrossValidate(true);
                k.normalise(false);
                k.setDistanceFunction(new EuclideanDistance());
                return k;
            case "HESCA":
            case "CAWPE":
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);
                break;
            case "CAWPEPLUS":
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);                
                ((CAWPE)c).setAdvancedCAWPESettings();
                break;
            case "CAWPEFROMFILE":
                String[] classifiers={"XGBoost","RandF","RotF"};
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);  
                ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                ((CAWPE)c).setResultsFileLocationParameters(horribleGlobalPath, nastyGlobalDatasetName, fold);
                
                ((CAWPE)c).setClassifiersNamesForFileRead(classifiers);
                
                
                break;
            case "CAWPE_AS_COTE":
                String[] cls={"TSF","ST","SLOWDTWCV","BOSS"};
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);  
                ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                ((CAWPE)c).setResultsFileLocationParameters(horribleGlobalPath, nastyGlobalDatasetName, fold);
                ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                break;
            case "XGBoost":
                c=new TunedXGBoost();
                ((TunedXGBoost)c).setTuneParameters(false);
                ((TunedXGBoost)c).setSeed(fold);
                break;

//ELASTIC CLASSIFIERS     
            case "EE": case "ElasticEnsemble":
                c=new ElasticEnsemble();
                break;
            case "DTW":
                c=new DTW1NN();
                ((DTW1NN )c).setWindow(1);
                break;
            case "SLOWDTWCV":
//                c=new DTW1NN();
                c=new SlowDTW_1NN();
                ((SlowDTW_1NN)c).optimiseWindow(true);
                break;
            case "DTWCV":
//                c=new DTW1NN();
//                c=new FastDTW_1NN();
//                ((FastDTW_1NN)c).optimiseWindow(true);
//                break;
//            case "FastDTWWrapper":
                c= new FastDTWWrapper();
                break;
            case "DD_DTW":
                c=new DD_DTW();
                break;
            case "DTD_C":
                c=new DTD_C();
                break;
            case "CID_DTW":
                c=new NN_CID();
                ((NN_CID)c).useDTW();
                break;
            case "MSM":
                c=new MSM1NN();
                break;
            case "TWE":
                c=new MSM1NN();
                break;
            case "WDTW":    
                c=new WDTW1NN();
                break;
                
            case "LearnShapelets": case "LS":
                c=new LearnShapelets();
                break;
            case "FastShapelets": case "FS":
                c=new FastShapelets();
                break;
            case "ShapeletTransform": case "ST": case "ST_Ensemble": case "ShapeletTransformClassifier":
                c=new ShapeletTransformClassifier();
//Default to 1 day max run: could do this better
                ((ShapeletTransformClassifier)c).setOneDayLimit();
                ((ShapeletTransformClassifier)c).setSeed(fold);
                break;
            case "RISE":
                c=new RISE();
                break;
            case "RISEV2":
                c=new RiseV2();
                ((RiseV2)c).buildFromSavedData(true);
                break;
            case "TSBF":
                c=new TSBF();
                break;
            case "BOP": case "BoP": case "BagOfPatterns":
                c=new BagOfPatterns();
                break;
            case "BOSS": case "BOSSEnsemble": 
                c=new BOSS();
                break;
            case "WEASEL":
                c = new WEASEL();
                ((WEASEL)c).setSeed(fold);
                break;
            case "TSF":
                c=new TSF();
                break;
             case "SAXVSM": case "SAX": 
                c=new SAXVSM();
                break;
             case "LPS":
                c=new LPS();
                break; 
             case "FlatCOTE":
                c=new FlatCote();
                break; 
             case "HiveCOTE": case "HIVECOTE": case "HiveCote": case "Hive-COTE":
                c=new HiveCote();
                ((HiveCote)c).setContract(24);
                break; 
            case "TunedXGBoost":
                 c=new TunedXGBoost();
                ((TunedXGBoost)c).setSeed(fold);
                ((TunedXGBoost)c).setDebug(false);
                ((TunedXGBoost)c).setDebugPrinting(false);
                ((TunedXGBoost)c).setTuneParameters(true);
                 break;
            case "RotFDefault":
                c = new RotationForest();
                ((RotationForest)c).setSeed(fold);
                return c;
//Hacky bit for paper
            case "RotF10": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(10);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF50": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(50);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF100": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(100);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF150": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(150);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF250": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(250);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF300": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(300);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF350": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(350);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF400": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(400);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF450": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(450);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotF500": 
                c= new TunedRotationForest();
                ((RotationForest)c).setNumIterations(450);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
//1000 attributes per group (10 values) 3; 4; : : : ; 12g
            case "RotFG3": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(3);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG4": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(4);
                ((RotationForest)c).setMaxGroup(4);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG5": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(5);
                ((RotationForest)c).setMaxGroup(5);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG6": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(6);
                ((RotationForest)c).setMaxGroup(6);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG7": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(7);
                ((RotationForest)c).setMaxGroup(7);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG8": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(8);
                ((RotationForest)c).setMaxGroup(8);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG9": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(9);
                ((RotationForest)c).setMaxGroup(9);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG10": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(10);
                ((RotationForest)c).setMaxGroup(10);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG11": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(11);
                ((RotationForest)c).setMaxGroup(11);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
            case "RotFG12": 
                c= new TunedRotationForest();
                ((RotationForest)c).setMinGroup(12);
                ((RotationForest)c).setMaxGroup(12);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;
                
                
//combinations sampling proportion (10 values) f0:1; 0:2; : : : ; 1:0g                
            case "RotRP1": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(0);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP2": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(10);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP3": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(20);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP4": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(30);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP5": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(40);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP6": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(50);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP7": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(60);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP8": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(70);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP9": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(80);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            case "RotRP10": 
                c= new TunedRotationForest();
                ((RotationForest)c).setRemovedPercentage(90);
                ((RotationForest)c).setMaxGroup(3);
                ((TunedRotationForest)c).tuneParameters(false);
                ((TunedRotationForest)c).setSeed(fold);
                ((TunedRotationForest)c).estimateAccFromTrain(false);
                break;                
            
           default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
    }
    
}
