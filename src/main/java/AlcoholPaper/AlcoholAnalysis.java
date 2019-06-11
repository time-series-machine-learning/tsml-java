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

import static AlcoholPaper.AlcoholClassifierList.classifiers_all;
import static AlcoholPaper.AlcoholClassifierList.classifiers_nonTsc;
import static AlcoholPaper.AlcoholClassifierList.removeClassifier;
import static AlcoholPaper.AlcoholClassifierList.replaceLabelsForImages;
import evaluation.MultipleClassifierEvaluation;
import static evaluation.ROCDiagramMaker.concatenateClassifierResults;
import static evaluation.ROCDiagramMaker.matlab_buildROCDiagrams;
import evaluation.storage.ClassifierResults;

/**
 * Analysis setups to create/collate the results reported in the alcohol paper
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class AlcoholAnalysis {
    
    static String datasetPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Datasets/";
    static String analysisPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Analysis/";
    static String resultsPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/";
            
    
    public static void main(String[] args) throws Exception {
//        ana_alcConc_LOBO();
//        ana_alcConc_LOBOPCA();
//        ana_alcConc_RandomBottle();
//        ana_jw_30RandFold();
//        ana_jw_1FoldUserSplit();
//        ana_jw_30RandFoldPCA();
//        ana_jw_30RandFoldPCA_top3noPLS();


//        rocDiaAlcoholConc();
//        rocDiaJW();
//        rocDiaJW_top5only();
        rocDiaAlcoholConc_top5only();
    }
    
    public static void ana_alcConc_LOBO() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "alcConc_LOBO", 44);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "AlcoholForgeryEthanol", "AlcoholForgeryMethanol" });
        mce.setUseAllStatistics();
        mce.readInClassifiers(classifiers_all, replaceLabelsForImages(classifiers_all), resultsPath);
        
        mce.runComparison();
    }
    
    public static void ana_alcConc_LOBOPCA() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "alcConc_LOBOPCA", 44);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "PCAAlcoholForgeryEthanol", "PCAAlcoholForgeryMethanol" });
        mce.setUseAllStatistics();
        mce.readInClassifiers(classifiers_nonTsc, replaceLabelsForImages(classifiers_nonTsc), resultsPath);
        
        mce.runComparison();
    }
    
    public static void ana_alcConc_RandomBottle() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "alcConc_RandomBottle", 30);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "RandomBottlesEthanol", });
        mce.setUseAllStatistics();
        mce.readInClassifiers(classifiers_all, replaceLabelsForImages(classifiers_all), resultsPath);
        
        mce.runComparison();
    }
    
    
    public static void ana_jw_30RandFold() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "jw_30RandFold", 30);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "JWRorJWB_RedBottle", "JWRorJWB_BlackBottle", });
        mce.setUseAllStatistics();
        mce.readInClassifiers(classifiers_all, replaceLabelsForImages(classifiers_all), resultsPath);
        
        mce.runComparison();
    }
    
    
    public static void ana_jw_1FoldUserSplit() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "jw_1FoldUserSplit", 1);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "JWRorJWB_RedBottle", "JWRorJWB_BlackBottle", });
        mce.setUseAllStatistics();
        mce.readInClassifiers(classifiers_all, replaceLabelsForImages(classifiers_all), resultsPath);
        
        mce.runComparison();
    }
    
    
    public static void ana_jw_30RandFoldPCA() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "jw_30RandFoldPCA", 30);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "PCA90_JWRorJWB_RedBottle", "PCA90_JWRorJWB_BlackBottle",  });
//        mce.setDatasets(new String[] { "PCA90_JWRorJWB_RedBottle", "PCA90_JWRorJWB_BlackBottle", 
//                                        "PCAtop3_JWRorJWB_RedBottle", "PCAtop3_JWRorJWB_BlackBottle" });
        mce.setUseAllStatistics();
        mce.readInClassifiers(classifiers_nonTsc, replaceLabelsForImages(classifiers_nonTsc), resultsPath);
        
        mce.runComparison();
    }
    
    public static void ana_jw_30RandFoldPCA_top3noPLS() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "jw_30RandFoldPCA_top3noPLS", 30);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "PCAtop3_JWRorJWB_RedBottle", "PCAtop3_JWRorJWB_BlackBottle",  });
//        mce.setDatasets(new String[] { "PCA90_JWRorJWB_RedBottle", "PCA90_JWRorJWB_BlackBottle", 
//                                        "PCAtop3_JWRorJWB_RedBottle", "PCAtop3_JWRorJWB_BlackBottle" });
        mce.setUseAllStatistics();
        mce.readInClassifiers(removeClassifier(classifiers_nonTsc, "PLS"), replaceLabelsForImages(removeClassifier(classifiers_nonTsc, "PLS")), resultsPath);
        
        mce.runComparison();
    }
    
    
    
    
    public static void rocDiaAlcoholConc() throws Exception { 
        String baseReadPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/";
        String baseWritePath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Analysis/alcConc_LOBO/semi-manualROCDias/";
        int numFolds = 44;
        String[] cnames = classifiers_all; 
        
        for (String dset : new String[] { "AlcoholForgeryEthanol", "AlcoholForgeryMethanol" }) {
            ClassifierResults[][] res = new ClassifierResults[cnames.length][numFolds];
            for (int i = 0; i < res.length; i++) {
                for (int f = 0; f < numFolds; f++) {
                    res[i][f] = new ClassifierResults(baseReadPath + cnames[i] + "/Predictions/" + dset + "/testFold"+f+".csv");
                }
            }

            ClassifierResults[] concatenatedRes = concatenateClassifierResults(res);
            matlab_buildROCDiagrams(baseWritePath, "alcConc_LOBO", dset, concatenatedRes, cnames);
        }
    }
    
    public static void rocDiaAlcoholConc_top5only() throws Exception { 
        String baseReadPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/";
        String baseWritePath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Analysis/alcConc_LOBO/semi-manualROCDias/";
        int numFolds = 44;
        
        String[] dsets = { "AlcoholForgeryEthanol", "AlcoholForgeryMethanol" };
        String[][] cnames_perDset = { 
            { "CAWPE", "PLS", "ResNet", "HIVE-COTE", "BOSS", }, //ethanol
            { "SVMQ", "PLS", "RISE", "CAWPE", "RotF_ST12Hour", },     //methanol
        };
        
        for (int d = 0; d < dsets.length; d++) {
            String dset = dsets[d];
            String[] cnames = cnames_perDset[d];
            
            ClassifierResults[][] res = new ClassifierResults[cnames.length][numFolds];
            for (int i = 0; i < res.length; i++) {
                for (int f = 0; f < numFolds; f++) {
                    res[i][f] = new ClassifierResults(baseReadPath + cnames[i] + "/Predictions/" + dset + "/testFold"+f+".csv");
                }
            }

            ClassifierResults[] concatenatedRes = concatenateClassifierResults(res);
            matlab_buildROCDiagrams(baseWritePath, "v", dset, concatenatedRes, cnames);
        }
    }
    
    public static void rocDiaJW() throws Exception { 
        String baseReadPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/";
        String baseWritePath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Analysis/jw_30RandFold/semi-manualROCDias/";
        int numFolds = 30;
        
        String[] cnames = classifiers_all; 
        
        for (String dset : new String[] { "JWRorJWB_BlackBottle", "JWRorJWB_RedBottle" }) {
            ClassifierResults[][] res = new ClassifierResults[cnames.length][numFolds];
            for (int i = 0; i < res.length; i++) {
                for (int f = 0; f < numFolds; f++) {
                    res[i][f] = new ClassifierResults(baseReadPath + cnames[i] + "/Predictions/" + dset + "/testFold"+f+".csv");
                }
            }

            ClassifierResults[] concatenatedRes = concatenateClassifierResults(res);
            matlab_buildROCDiagrams(baseWritePath, "jw_30Fold", dset, concatenatedRes, cnames);
        }
    }
    
    public static void rocDiaJW_top5only() throws Exception { 
        String baseReadPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/";
        String baseWritePath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Analysis/jw_30RandFold/semi-manualROCDias/";
        int numFolds = 30;
        
        String[] dsets = { "JWRorJWB_BlackBottle", "JWRorJWB_RedBottle" };
        String[][] cnames_perDset = { 
            { "RandF", "HIVE-COTE", "XGBoost", "BOSS", "TSF" }, //black
            { "TSF", "RotF_ST12Hour", "HIVE-COTE", "RandF", "CAWPE" },     //red
        };
        
        for (int d = 0; d < dsets.length; d++) {
            String dset = dsets[d];
            String[] cnames = cnames_perDset[d];
            
            ClassifierResults[][] res = new ClassifierResults[cnames.length][numFolds];
            for (int i = 0; i < res.length; i++) {
                for (int f = 0; f < numFolds; f++) {
                    res[i][f] = new ClassifierResults(baseReadPath + cnames[i] + "/Predictions/" + dset + "/testFold"+f+".csv");
                }
            }

            ClassifierResults[] concatenatedRes = concatenateClassifierResults(res);
            matlab_buildROCDiagrams(baseWritePath, "jw_30Fold", dset, concatenatedRes, cnames);
        }
    }
    
    
}
