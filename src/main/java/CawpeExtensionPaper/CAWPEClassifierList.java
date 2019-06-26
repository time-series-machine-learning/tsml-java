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
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPEClassifierList {
    
    //copied from experiments.DataSets.ReducedUCI to have as a permanantly fixed list here
    //this is the 39 datasets left of when removing toy, heavily related etc datasets from the 
    //delgado 121. first decided upon in tony rotf paper
    public static final String[] datasetList = {"bank","blood","breast-cancer-wisc-diag",
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
            "wall-following","waveform-noise","wine-quality-white","yeast"};
    
    
    public static final String[] cawpeConfigs = { 
        "CAWPE", 
        "CAWPE_retrain_noWeight", 
        "CAWPE_retrain_foldWeight", 
        "CAWPE_noRetrain" 
    };
    
    public static final String[] homoEnsembles = { 
        "XGBoost", 
        "RandF", 
        "RotF",
    };
    
    public static final String[] allClassifiers = { 
        "CAWPE", 
        "CAWPE_retrain_noWeight", 
        "CAWPE_retrain_foldWeight", 
        "CAWPE_noRetrain",
        "XGBoost", 
        "RandF", 
        "RotF",
    };
    
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
                break;
                
            case "RandF": 
                RandomForest randf=new RandomForest();
                randf.setNumTrees(500);
                randf.setSeed(fold);
                return randf;
                
            case "RotF":
                RotationForest rotf=new RotationForest();
                rotf.setNumIterations(50);
                rotf.setSeed(fold);
                return rotf;
            
            
            // CAWPE CONFIGURATIONS
            //
            //
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
    
    public static void main(String[] args) {
        System.out.println(datasetList.length);
    }
}
