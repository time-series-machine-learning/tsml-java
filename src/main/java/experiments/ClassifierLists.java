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


import experiments.Experiments.ExperimentalArguments;
import timeseriesweka.classifiers.dictionary_based.*;
import timeseriesweka.classifiers.hybrids.FlatCote;
import timeseriesweka.classifiers.hybrids.HiveCote;
import timeseriesweka.classifiers.shapelet_based.ShapeletTransformClassifier;
import timeseriesweka.classifiers.shapelet_based.FastShapelets;
import timeseriesweka.classifiers.shapelet_based.LearnShapelets;
import timeseriesweka.classifiers.interval_based.TSF;
import timeseriesweka.classifiers.interval_based.TSBF;
import timeseriesweka.classifiers.interval_based.LPS;
import timeseriesweka.classifiers.frequency_based.RISE;
import timeseriesweka.classifiers.distance_based.SlowDTW_1NN;
import timeseriesweka.classifiers.distance_based.NN_CID;
import timeseriesweka.classifiers.distance_based.ElasticEnsemble;
import timeseriesweka.classifiers.distance_based.DTD_C;
import timeseriesweka.classifiers.distance_based.DD_DTW;
import multivariate_timeseriesweka.classifiers.MultivariateShapeletTransformClassifier;
import multivariate_timeseriesweka.classifiers.NN_DTW_A;
import multivariate_timeseriesweka.classifiers.NN_DTW_D;
import multivariate_timeseriesweka.classifiers.NN_DTW_I;
import multivariate_timeseriesweka.classifiers.NN_ED_I;
import timeseriesweka.classifiers.distance_based.FastDTW;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.ED1NN;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.MSM1NN;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.WDTW1NN;
import timeseriesweka.classifiers.distance_based.ProximityForestWrapper;
import weka_extras.classifiers.ensembles.CAWPE;
import weka_extras.classifiers.PLSNominalClassifier;
import weka_extras.classifiers.tuned.TunedXGBoost;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka_extras.classifiers.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.EuclideanDistance;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class ClassifierLists {
    public static String[] bakeOffClassifierList = { };    //todo, as an example of the kind of thing we could do with this class
    public static String[] CAWPE_fig1Ensembles = { };      //todo, as an example of the kind of thing we could do with this class
    
    /**
     * 
     * setClassifier, which takes the experimental 
     * arguments themselves and therefore the classifiers can take from them whatever they 
     * need, e.g the dataset name, the fold id, separate checkpoint paths, etc. 
     * 
     * To take this idea further, to be honest each of the TSC-specific classifiers
     * could/should have a constructor and/or factory that builds the classifier
     * from the experimental args.
     * 
     * previous usage was setClassifier(String classifier name, int fold).
     * this can be reproduced with setClassifierClassic below. 
     * 
     */
    public static Classifier setClassifier(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        int fold=exp.foldId;
        String resultsPath="", dataset="";
        boolean canLoadFromFile=true;
        if(exp.resultsWriteLocation==null || exp.datasetName==null)
            canLoadFromFile=false;
        else{
            resultsPath=exp.resultsWriteLocation;
            dataset=exp.datasetName;
        }
        Classifier c=null;
        switch(classifier){
            case "XGBoostMultiThreaded":
                c = new TunedXGBoost(); 
                break;
            case "XGBoost":
                c = new TunedXGBoost(); 
                ((TunedXGBoost)c).setRunSingleThreaded(true);
                break;
            case "SmallTunedXGBoost":
                c = new TunedXGBoost(); 
                ((TunedXGBoost)c).setRunSingleThreaded(true);
                ((TunedXGBoost)c).setSmallParaSearchSpace_64paras();
                break;
            case "ProximityForest":
                c = new ProximityForestWrapper();
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
            case "PLSNominalClassifier":
                c = new PLSNominalClassifier();
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
                ((CAWPE)c).setSeed(fold);
                break;
            case "CAWPEPLUS":
                c=new CAWPE();
                ((CAWPE)c).setSeed(fold);                
                ((CAWPE)c).setupAdvancedSettings();
                break;
            case "CAWPEFROMFILE":
                if(canLoadFromFile){
                    String[] classifiers={"TSF","BOSS","RISE","ST"};
                    c=new CAWPE();
                    ((CAWPE)c).setSeed(fold);  
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(classifiers);
                }
                else
                    throw new UnsupportedOperationException("ERROR: Cannot load CAWPE from file since no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "CAWPE_AS_COTE":
                if(canLoadFromFile){
                    String[] cls={"TSF","BOSS","RISE","ST","ElasticEnsemble"};//RotF for ST
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);  
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "CAWPE_AS_COTE_NO_EE":
                if(canLoadFromFile){
                    String[] cls2={"TSF","BOSS","RISE","ST"};
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);  
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls2);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
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
                c= new FastDTW();
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
                ((ShapeletTransformClassifier)c).setSeed(fold);
                break;
            case "TSBF":
                c=new TSBF();
                break;
            case "BOP": case "BoP": case "BagOfPatterns":
                c=new BagOfPatterns();
                break;
            case "BOSS": case "BOSSEnsemble": 
                c=new BOSS();
                ((BOSS) c).setSeed(fold);
                break;
            case "RBOSS": case "cBOSS":
                c = new cBOSS();
                ((cBOSS) c).setSeed(fold);
                break;
            case "WEASEL":
                c = new WEASEL();
                ((WEASEL)c).setSeed(fold);
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
            case "RISE":
                c=new RISE();
                ((RISE) c).setSeed(fold);
                ((RISE) c).setTransforms("PS","ACF");
                break;
            case "TSF":
                c=new TSF();
                ((TSF)c).setSeed(fold);
                break;

           default:
                System.out.println("UNKNOWN CLASSIFIER IN LocalClassifierLists"+classifier);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;    
    }    
    /**
     * This method redproduces the old usage exactly as it was in old experiments.java. 
     * If you try build any classifier that uses any experimental info other than  
     * exp.classifierName or exp.foldID, an exception will be thrown.
     * In particular, any classifier that needs access to the results from others
     * e.g. CAWPEFROMFILE, will throw an UnsupportedOperationException if you try use it like this.
     *      * @param classifier
     * @param fold
     * @return 
     */
    public static Classifier setClassifierClassic(String classifier, int fold){
        Experiments.ExperimentalArguments exp=new ExperimentalArguments();
        exp.classifierName=classifier;
        exp.foldId=fold;
        return setClassifier(exp);
    }

    
    
//All implemented classifiers in tsml  
   //<editor-fold defaultstate="collapsed" desc="All univariate time series classifiers">    
    public static String[] allClassifiers={
//Dictionary Based
        "BOSS","BagOfPatterns","SAXVSM","SAX_1NN","WEASEL","cBOSS","BOSSC45","BOSSSpatialPyramids","BOSSSpatialPyramids_BD","BoTSWEnsemble",
//Frequency Based
        "RISE","cRISE",
//Interval Based
        "LPS","TSBF","TSF","cTSF",
//Shapelet Based
        "FastShapelets","LearnShapelets","ShapeletTransformClassifier",
//Distance Based
        "ApproxElasticEnsemble","DD_DTW","DTD_C","DTW_kNN","ElasticEnsemble","FastDTW","FastDTW_1NN","FastElasticEnsemble","NN_CID","ProximityForestWrapper","SlowDTW_1NN"
};    
       //</editor-fold>       



    
    public static void main(String[] args) throws Exception {
        System.out.println(setClassifierClassic("cBOSS", 0));
    }
}
