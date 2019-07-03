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

package CawpeExtensionPaper;

import experiments.Experiments;
import vector_classifiers.CAWPE;
import vector_classifiers.TunedXGBoost;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.RandomForest;

/**
 * Datasets, classifiers, and their setups listed as used in the CAWPE paper 
 * investigating the effects of retaining CV-fold models
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPEClassifierList {
    
    //copied from experiments.DataSets.ReducedUCI to have as a permanantly fixed list here
    //this is the 39 datasets left of when removing toy, heavily related etc datasets from the 
    //delgado 121. first decided upon in tony rotf paper
    public static final String[] datasetList = {
        "bank","blood","breast-cancer-wisc-diag",
        "breast-tissue","cardiotocography-10clases", 
        "conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding",
        "ecoli","glass","hill-valley",
        "image-segmentation","ionosphere","iris","libras","magic",
        "miniboone",
        "oocytes_merluccius_nucleus_4d","oocytes_trisopterus_states_5b",
        "optical","ozone","page-blocks","parkinsons","pendigits",
        "planning","post-operative","ringnorm","seeds","spambase",
        "statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates",
        "synthetic-control","twonorm","vertebral-column-3clases",
        "wall-following","waveform-noise","wine-quality-white","yeast"
    };
    
    
    //replaceLabelsForImages() for final labels
    public static final String[] cawpeConfigs = {  
        "CAWPE", 
        "CAWPE_retrain_noWeight", 
        "CAWPE_retrain_foldWeight", 
        "CAWPE_noRetrain" 
    };
    
    public static final String[] homoEnsembles = { 
        "XGBoost", 
        "RandF", 
    };
    
    public static final String[] allTopLevelClassifiers = arrConcat(cawpeConfigs, homoEnsembles);
    
    public static final String[] coreModules = { 
        "C4.5",
        "Logistic",
        "MLP",
        "NN",
        "SVML",
    };
    
    public static final String[] foldModules = { 
        "C4.5_cvFold0","C4.5_cvFold1","C4.5_cvFold2","C4.5_cvFold3","C4.5_cvFold4",	
        "C4.5_cvFold5","C4.5_cvFold6","C4.5_cvFold7","C4.5_cvFold8","C4.5_cvFold9",	
        
        "Logistic_cvFold0","Logistic_cvFold1","Logistic_cvFold2","Logistic_cvFold3","Logistic_cvFold4",	
        "Logistic_cvFold5","Logistic_cvFold6","Logistic_cvFold7","Logistic_cvFold8","Logistic_cvFold9",	
        
        "MLP_cvFold0","MLP_cvFold1","MLP_cvFold2","MLP_cvFold3","MLP_cvFold4",
        "MLP_cvFold5","MLP_cvFold6","MLP_cvFold7","MLP_cvFold8","MLP_cvFold9",
        
        "NN_cvFold0","NN_cvFold1","NN_cvFold2","NN_cvFold3","NN_cvFold4",
        "NN_cvFold5","NN_cvFold6","NN_cvFold7","NN_cvFold8","NN_cvFold9",
        
        "SVML_cvFold0","SVML_cvFold1","SVML_cvFold2","SVML_cvFold3","SVML_cvFold4",
        "SVML_cvFold5","SVML_cvFold6","SVML_cvFold7","SVML_cvFold8","SVML_cvFold9",
    };
    
    public static final String[] allBaseClassifiers = arrConcat(coreModules, foldModules);
    
    public static Classifier setClassifier(Experiments.ExperimentalArguments exp) {        
        return setClassifier(exp.classifierName, exp.foldId);
    }
    
    public static Classifier setClassifier(String classifier, int fold) {                
        switch(classifier){
            
            // COMPARED HOMOGENEOUS ENSEMBLES
            //
            //
            //
            case "XGBoost":
                TunedXGBoost xg = new TunedXGBoost(); 
                xg.setRunSingleThreaded(true);
                xg.setTuneParameters(false);
                xg.setSeed(fold);
                
                // defaults as of 26/06/2019, but setting permanently here anyway
                xg.setNumIterations(500); 
                xg.setMaxTreeDepth(4);
                xg.setLearningRate(0.1f);
                return xg;
                
            case "RandF": 
                RandomForest randf=new RandomForest();
                randf.setNumTrees(500);
                randf.setSeed(fold);
                return randf;
                
            
            // CAWPE CONFIGURATIONS
            //
            // For base classifier specifications, see CAWPE.setDefaultCAWPESettings()
            //
            case "CAWPE": 
                CAWPE cawpe_base = new CAWPE();
                cawpe_base.setRandSeed(fold);
                cawpe_base.setPerformCV(false);
                return cawpe_base;
            
            case "CAWPE_retrain_noWeight": 
                CAWPE_Extended cawpe_retrain_noWeight = new CAWPE_Extended();
                cawpe_retrain_noWeight.setRandSeed(fold);
                cawpe_retrain_noWeight.setPerformCV(false);
                cawpe_retrain_noWeight.setSubModulePriorWeightingScheme(CAWPE_Extended.priorScheme_none);
                cawpe_retrain_noWeight.setRetrainOnFullTrainSet(true);
                return cawpe_retrain_noWeight;
            
            case "CAWPE_retrain_foldWeight": 
                CAWPE_Extended cawpe_retrain_foldWeight = new CAWPE_Extended();
                cawpe_retrain_foldWeight.setRandSeed(fold);
                cawpe_retrain_foldWeight.setPerformCV(false);
                cawpe_retrain_foldWeight.setSubModulePriorWeightingScheme(CAWPE_Extended.priorScheme_oneOverNumFolds);
                cawpe_retrain_foldWeight.setRetrainOnFullTrainSet(true);
                return cawpe_retrain_foldWeight;
            
            case "CAWPE_noRetrain": 
                CAWPE_Extended cawpe_noRetrain = new CAWPE_Extended();
                cawpe_noRetrain.setRandSeed(fold);
                cawpe_noRetrain.setPerformCV(false);
                cawpe_noRetrain.setSubModulePriorWeightingScheme(CAWPE_Extended.priorScheme_none); //irrelevant, all submodules will have same prior
                cawpe_noRetrain.setRetrainOnFullTrainSet(false);
                return cawpe_noRetrain;
                
                
                
                
            default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
                
        }
        
        return null;
    }
    
    public static String[] replaceLabelsForImages(String[] a) {
        final String[] find = { 
            "CAWPE_retrain_noWeight", 
            "CAWPE_retrain_foldWeight", 
            "CAWPE_noRetrain" 
        };
        
        final String[] replace = { 
            "CAWPE\\_M", 
            "CAWPE\\_M\\_DW", 
            "CAWPE\\_R" 
        };
        
        String[] copy = new String[a.length];
        
        for (int i = 0; i < a.length; i++) {
            copy[i] = a[i];
        
            for (int j = 0; j < find.length; j++) {
                if (a[i].equals(find[j])) {
                    copy[i] = replace[j];
                    break;
                }
            }
        }
        
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
    
    public static String[] arrConcat(String[] a, String[] b) {
        String[] res = new String[a.length + b.length];
        
        int i = 0;
        for (int j = 0; j < a.length; j++, i++)
            res[i] = a[j];
        for (int j = 0; j < b.length; j++, i++)
            res[i] = b[j];
        
        return res;
    }
    
    public static void main(String[] args) {
        System.out.println(datasetList.length);
    }
}
