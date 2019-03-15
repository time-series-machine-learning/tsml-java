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


import multivariate_timeseriesweka.classifiers.MultivariateShapeletTransformClassifier;
import multivariate_timeseriesweka.classifiers.NN_DTW_A;
import multivariate_timeseriesweka.classifiers.NN_DTW_D;
import multivariate_timeseriesweka.classifiers.NN_DTW_I;
import multivariate_timeseriesweka.classifiers.NN_ED_I;
import timeseriesweka.classifiers.*;
import timeseriesweka.classifiers.FastWWS.FastDTWWrapper;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ED1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.MSM1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.WDTW1NN;
import vector_classifiers.CAWPE;
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
    //leaving in for now, in case particular classifiers require it.
    //eventually should be removed in favour of using the info in the experimental settings passed 
    //in the newer setClassifier
    public static String horribleGlobalPath="";
    public static String nastyGlobalDatasetName="";  

    public static String[] bakeOffClassifierList = { };    //todo, as an example of the kind of thing we could do with this class
    public static String[] CAWPE_fig1Ensembles = { };      //todo, as an example of the kind of thing we could do with this class
    
    /**
     * This method is currently a placeholder that simply call setClassifierClassic(classifierName, fold),
     * exactly where to take this newer method is still up for debate
     * 
     * This shall be the start of the newer setClassifier, which take the experimental 
     * arguments themselves and therefore the classifiers can take from them whatever they 
     * need, e.g the dataset name, the fold id, separate checkpoint paths, etc. 
     * 
     * To take this idea further, to be honest each of the TSC-specific classifiers
     * could/should have a constructor and/or factory that builds the classifier
     * from the experimental args. 
     */
    public static Classifier setClassifier(Experiments.ExperimentalArguments exp){
        return setClassifierClassic(exp.classifierName, exp.foldId);
    }
    
    /**
     * This is the method exactly as it was in old experiments.java. 
     * 
     * @param classifier
     * @param fold
     * @return 
     */
    public static Classifier setClassifierClassic(String classifier, int fold){
        Classifier c=null;
        switch(classifier){
            
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
            case "RandF": 
                RandomForest r=new RandomForest();
                r.setNumTrees(500);
                r.setSeed(fold);
                c = r;
                break;
            case "RotF":
                RotationForest rf=new RotationForest();
                rf.setNumIterations(50);
                rf.setSeed(fold);
                c = rf;
                break;
            case "bayesNet": 
                c = new BayesNet();
                break;
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
               break;
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
                c=new RISE(fold);
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
            case "BOSSMV":
                c = new BOSS();
                ((BOSS) c).setSeed(fold);
                ((BOSS) c).setSavePath("/gpfs/scratch/pfm15hbu/checkpointfiles");
                break;
            case "RBOSSMV":
                c = new BOSS();
                ((BOSS) c).setRandomEnsembleSelection(true);
                ((BOSS) c).setEnsembleSize(100);
                ((BOSS) c).setSavePath("/gpfs/scratch/pfm15hbu/checkpointfiles");
                ((BOSS) c).setSeed(fold);
                break;
            case "RBOSSMV250":
                c = new BOSS();
                ((BOSS) c).setRandomEnsembleSelection(true);
                ((BOSS) c).setEnsembleSize(250);
                ((BOSS) c).setSavePath("/gpfs/scratch/pfm15hbu/checkpointfiles");
                ((BOSS) c).setSeed(fold);
                break;
            case "RCBOSSMV":
                c = new BOSS();
                ((BOSS) c).useCAWPE(true);
                ((BOSS) c).setEnsembleSize(100);
                ((BOSS) c).setNumCAWPEFolds(2);
                ((BOSS) c).setSavePath("/gpfs/scratch/pfm15hbu/checkpointfiles");
                ((BOSS) c).setSeed(fold);
                break;
            case "RCBOSSMV250":
                c = new BOSS();
                ((BOSS) c).useCAWPE(true);
                ((BOSS) c).setEnsembleSize(250);
                ((BOSS) c).setNumCAWPEFolds(2);
                ((BOSS) c).setSavePath("/gpfs/scratch/pfm15hbu/checkpointfiles");
                ((BOSS) c).setSeed(fold);
                break;
            case "RandomBOSSContracted1Hour":
                c = new BOSS();
                ((BOSS) c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 1);
                ((BOSS) c).setSavePath("/gpfs/scratch/pfm15hbu/checkpointfiles");
                ((BOSS) c).setSeed(fold);
                break;
            case "RandomBOSSContracted24Hour":
                c = new BOSS();
                ((BOSS) c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 24);
                ((BOSS) c).setSavePath("/gpfs/scratch/pfm15hbu/checkpointfiles");
                ((BOSS) c).setSeed(fold);
                break;
            case "RandomBOSSContracted1HourMV":
                c = new BOSS();
                ((BOSS) c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 1);
                ((BOSS) c).setSavePath("/gpfs/scratch/pfm15hbu/checkpointfiles");
                ((BOSS) c).setSeed(fold);
                break;
            case "RandomRTreeBOSS":
                c = new BOSS();
                ((BOSS) c).setAlternateIndividualClassifier(new RandomTree());
                ((BOSS) c).setSavePath("/gpfs/scratch/pfm15hbu/checkpointfiles");
                ((BOSS) c).setSeed(fold);
            case "WEASEL":
                c = new WEASEL();
                ((WEASEL)c).setSeed(fold);
                break;
            case "TSF":
                c=new TSF();
                ((TSF)c).setSeed(fold);
                break;
             case "TSFProb":
                c=new TSF();
                ((TSF)c).setSeed(fold);
                ((TSF)c).setProbabilityEnsemble(true);
                break;
             case "TSFC45":
                c=new TSF();
                ((TSF)c).setSeed(fold);
                ((TSF)c).setBaseClassifier(new J48());
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
            case "RotFDefault":
                c = new RotationForest();
                ((RotationForest)c).setSeed(fold);
                break;
            case "cRISE_BOTH_NOSTBLS":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.ACF_PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                //((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.MINUTE, 1);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_PS_NOSTBLS":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.MINUTE, 1);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_BOTH_NOSTBLS_24HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.ACF_PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 24);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_BOTH_NOSTBLS_48HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.ACF_PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 48);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_BOTH_NOSTBLS_72HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.ACF_PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 72);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_BOTH_NOSTBLS_96HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.ACF_PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 96);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_BOTH_NOSTBLS_120HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.ACF_PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 120);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_BOTH_NOSTBLS_144HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.ACF_PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 144);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_BOTH_NOSTBLS_168HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.ACF_PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 168);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_PS_NOSTBLS_24HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 24);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_PS_NOSTBLS_48HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 48);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_PS_NOSTBLS_72HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 72);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_PS_NOSTBLS_96HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 96);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_PS_NOSTBLS_120HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 120);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_PS_NOSTBLS_144HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 144);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
            case "cRISE_PS_NOSTBLS_168HR":
                c = new CRISE((long)fold);
                //Classieir settings.
                ((CRISE)c).setDownSample(false);
                ((CRISE)c).setTransformType(CRISE.TransformType.PS);
                ((CRISE)c).setMinNumTrees(200);
                //((CRISE)c).setStabilise(5);
                //Timer settings.
                ((CRISE)c).setTimeLimit(ContractClassifier.TimeLimit.HOUR, 168);
                ((CRISE)c).setModelOutPath(DataSets.resultsPath+classifier+"/Predictions/");
                ((CRISE)c).setSavePath(DataSets.resultsPath+classifier+"/Predictions/" + nastyGlobalDatasetName);
                break;
           default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
    }

    public static void main(String[] args) throws Exception {
        System.out.println(setClassifierClassic("RCBOSSMV", 0));
        System.out.println(setClassifierClassic("RBOSSMV", 0));
    }
}
