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

package AlcoholPaper;

import experiments.Experiments;
import experiments.Experiments.ExperimentalArguments;
import timeseriesweka.classifiers.BOSS;
import timeseriesweka.classifiers.CRISE;
import timeseriesweka.classifiers.RISE;
import timeseriesweka.classifiers.ShapeletTransformClassifier;
import timeseriesweka.classifiers.TSF;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ED1NN;
import vector_classifiers.CAWPE;
import vector_classifiers.PLSNominalClassifier;
import vector_classifiers.TunedXGBoost;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.RandomForest;

/**
 * Classifier specifications for the alcohol paper. 
 * 
 *  CLASSICAL: 
 *           ED/1NN 
 *           PLS 
 *           SVMQ
 *
 *   MODERN ENSEMBLES: 
 *           RandF
 *           XGBoost
 *           CAWPE
 * 
 *   DL: 
 *           ResNet (handled in python/keras separately)
 * 
 *   TSC: 
 *           ST
 *           TSF
 *           BOSS
 *           RISE
 *           HIVECOTE(with ed)
 * 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class AlcoholClassifierList {
    
    public static String[] classifiers_all = { 
        "ED",
        "PLS",
        "SVMQ",
        "RandF",
        "XGBoost",
        "CAWPE",
        "ResNet",
        "RotF_ST12Hour",
        "BOSS",
        "TSF",
        "RISE",
        "HIVE-COTE",  
    };
    
    public static String[] classifiers_hiveCoteMembers = { 
        "ED", //using ed > ee, we KNOW there's no gain from any kind of warping, and ee is a bit of a pig
              //derivative measures are definitely an option worth considering, however skipping for the 
              //feasibility analysis 
        
        "RotF_ST12Hour",
        "BOSS",
        "TSF",
        "RISE",
    };
    
    public static String[] classifiers_tsc = { 
        "ResNet",
        "ST",
        "BOSS",
        "TSF",
        "RISE",
        "HIVE-COTE",  
    };
    
    public static String[] classifiers_nonTsc = { 
        "ED",
        "PLS",
        "SVMQ",
        "RandF",
        "XGBoost",
        "CAWPE",
    };
    
    public static String[] replaceLabelsForImages(String[] a) {
        final String find = "RotF_ST12Hour";
        final String replace = "ST";
        
        String[] copy = new String[a.length];
        
        for (int i = 0; i < a.length; i++)
            if (a[i].equals(find))
                copy[i] = replace;
            else 
                copy[i] = a[i];
        
        return copy;
    }
    
    public static String[] removeClassifier(String[] a, String toremove) {
        String[] copy = new String[a.length-1];
        
        for (int i = 0, j = 0; j < a.length; i++, j++)
            if (!a[j].equals(toremove))
                copy[i] = a[j];
            else 
                i--;
        
        return copy;
    }
    
    public static Classifier setClassifier(Experiments.ExperimentalArguments exp){
        
        String classifier = exp.classifierName;
        int fold = exp.foldId;
        
        switch(classifier){
            
            ////////// CLASSICAL/SIMPLE BENCHMARKS
            case "ED":
                return new ED1NN();
            
            case "PLS":
                return new PLSNominalClassifier();
                
            case "SVMQ":
                SMO svmq = new SMO();
                PolyKernel k=new PolyKernel();
                k.setExponent(2);
                svmq.setKernel(k);
                svmq.setRandomSeed(fold);
                svmq.setBuildLogisticModels(true); //for probabilistic output
                return svmq;
                 
            ////////// MODERN ENSEMBLES
            case "RandF": 
                RandomForest randf=new RandomForest();
                randf.setNumTrees(500);
                randf.setSeed(fold);
                return randf;
            
            case "XGBoost":
                TunedXGBoost xg = new TunedXGBoost(); 
                xg.setTuneParameters(false);
                xg.setRunSingleThreaded(true);
                xg.setSeed(fold);
                return xg;
            
            case "CAWPE": 
                //default base classifiers: C4.5, Logistic, MLP, NN, SVML
                CAWPE cawpe = new CAWPE();
                cawpe.setRandSeed(fold);
                return cawpe;
                
            
                
            ////////// DEEP LEARNING 
                
            // done in python, resnet from  https://github.com/hfawaz/dl-4-tsc
            
            
            
            ////////// TIME SERIES CLASSIFIERS
            case "ST": 
//                ShapeletTransformClassifier st = new ShapeletTransformClassifier();
//                st.setOneDayLimit();
//                st.setSeed(fold);
//                return st;
                
                //something weird happening with shapelettransformclassifier, either outdatedwith current experiments
                //class or parameters set incorrectly - jobs jsut are not finishing.
                //thesefore using the transformExperiments class to generate the transforms for each fold 
                //by themselves, then building Rotation Forest on the transform data (10 fold cv on train set, and build on full train/evaluate on test) as a 
                //secondary step
                
                //therefore, not handled here. called TransformExperiments with same args, ST
                return null;
                
            case "RotF": 
                //see previous case, for building on the shapelet transform data
                RotationForest rotf = new RotationForest();
                rotf.setNumIterations(50);
                rotf.setSeed(fold);
                return rotf;
                
            case "BOSS":
                BOSS boss = new BOSS();
                boss.setSeed(fold);
                return boss;
                
            case "TSF":
                TSF tsf = new TSF();
                tsf.setSeed(fold);
                return tsf;
                
            case "RISE":
                CRISE crise = new CRISE(fold);
                crise.setTransformType(CRISE.TransformType.ACF_PS);
                return crise; 
                
            case "HIVE-COTE": {
                //jamesl: i'm just going to build hive-cote via cawpe, it's what i know 
                //postprocessing from the saved results of the above to save a lot of time
                //1nn-ed is used instead of elastic ensemble since we know from the data domain 
                //the it specifically is NOT elastic - the wavelength are always aligned because
                //physics, plus ee is a pig to run, esp with long series length
                
                CAWPE hive = new CAWPE();

                hive.setEnsembleIdentifier("HIVE-COTE");
                hive.setClassifiers(null, classifiers_hiveCoteMembers, null);
                hive.setBuildIndividualsFromResultsFiles(true);
                hive.setResultsFileLocationParameters(exp.resultsWriteLocation, exp.datasetName, fold);
                hive.setRandSeed(fold);
                hive.setPerformCV(exp.generateErrorEstimateOnTrainSet);
                hive.setFillMissingDistsWithOneHotVectors(true); //for boss, missing train probabilities, but is 1nn in the end anyway
                
                return hive;
            }
               
            default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
                
        }
        
        return null;
    }
    
    public static void buildHIVECOTEResults() { 
        for (String dset  : new String[] { "JWRorJWB_RedBottle", "JWRorJWB_BlackBottle", "RandomBottlesEthanol" }) {
            for (int fold = 0; fold < 30; fold++) {
                ExperimentalArguments args = new Experiments.ExperimentalArguments();
                args.resultsWriteLocation = AlcoholAnalysis.resultsPath;
                args.dataReadLocation = AlcoholAnalysis.datasetPath;
                args.classifierName = "HIVE-COTE";
                args.generateErrorEstimateOnTrainSet = true;
                
                args.datasetName = dset;
                args.foldId = fold;
                
                args.run();
            }
        }
        for (String dset  : new String[] { "AlcoholForgeryEthanol", "AlcoholForgeryMethanol" }) {
            for (int fold = 0; fold < 44; fold++) {
                ExperimentalArguments args = new Experiments.ExperimentalArguments();
                args.resultsWriteLocation = AlcoholAnalysis.resultsPath;
                args.dataReadLocation = AlcoholAnalysis.datasetPath;
                args.classifierName = "HIVE-COTE";
                args.generateErrorEstimateOnTrainSet = true;
                
                args.datasetName = dset;
                args.foldId = fold;
                
                args.run();
            }
        }
    }
    
    public static void main(String[] args) {
        buildHIVECOTEResults();
    }
}
