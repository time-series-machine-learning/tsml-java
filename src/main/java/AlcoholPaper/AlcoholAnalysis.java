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
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.util.ArrayList;
import utilities.ClassifierTools;
import weka.core.Attribute;

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
//        rocDiaAlcoholConc_top5only();

//        rocDiaAlcoholConc_top5only_particularBottleFoldsOnly();
//        locationOfErrors_ethanol();
        randombottle_svmqConfMat();
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
            matlab_buildROCDiagrams(baseWritePath, "alcConc_LOBO", dset, concatenatedRes, cnames);
        }
    }
    
    public static void rocDiaAlcoholConc_top5only_particularBottleFoldsOnly() throws Exception { 
        String baseReadPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/";
        String baseWritePath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Analysis/alcConc_LOBO/semi-manualROCDias/";
        
        
        // class values were set up such that first 28 classes were 'standard' bottles, last 16 'irregular' 
        //so just reading and concatenating predictions of first 28 lobo-sampled folds for standard bottles
//        int startFold = 0; int endFold = 28; String str = "standard";
        int startFold = 28; int endFold = 44; String str = "irregular";
        int numFolds = endFold-startFold;
        

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
                    res[i][f] = new ClassifierResults(baseReadPath + cnames[i] + "/Predictions/" + dset + "/testFold"+(f+startFold)+".csv");
                }
            }

            ClassifierResults[] concatenatedRes = concatenateClassifierResults(res);
            matlab_buildROCDiagrams(baseWritePath, "alcConc_LOBO_"+str+"BottlesOnly", dset, concatenatedRes, cnames);
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
    
    
    public static void locationOfErrors_ethanol() throws Exception { 
        String baseReadPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/";
        String[] classifiers = { "PLS", "CAWPE" };
//        int numFolds = 44;
        String dset = "AlcoholForgeryEthanol";
        
        for (String classifier : classifiers) {
            System.out.println(classifier + ": ");
            
            int numstdBottlesWithErrors = 0;
            int numstderror = 0;
            for (int f = 0; f < 28; f++) {
                ClassifierResults res = new ClassifierResults(baseReadPath + classifier + "/Predictions/" + dset + "/testFold" + f+ ".csv");
               
                int erronfold = 0;
                for (int i = 0; i < res.numInstances(); i++)
                    if ((int)res.getPredClassValue(i) != (int)res.getTrueClassValue(i))
                        erronfold++;
                
                if (erronfold > 0)
                    numstdBottlesWithErrors++;
                
                numstderror += erronfold;
                System.out.print(erronfold +", ");
//                System.out.print(erronfold+"("+res.getAcc()+"), ");
            }
            
            System.out.println("\n standardbottle errors: " + numstderror + " across " + numstdBottlesWithErrors + " bottles");
            
            int numirregBottlesWithErrors = 0;
            int numirregerror = 0;
            for (int f = 28; f < 44; f++) {
                ClassifierResults res = new ClassifierResults(baseReadPath + classifier + "/Predictions/" + dset + "/testFold" + f+ ".csv");
               
                int erronfold = 0;
                for (int i = 0; i < res.numInstances(); i++)
                    if ((int)res.getPredClassValue(i) != (int)res.getTrueClassValue(i))
                        erronfold++;
                
                if (erronfold > 0)
                    numirregBottlesWithErrors++;
                
                numirregerror += erronfold;
                System.out.print(erronfold +", ");
//                System.out.print(erronfold+"("+res.getAcc()+"), ");
            }
            System.out.println("\n irregbottle errors: " + numirregerror + " across " + numirregBottlesWithErrors + " bottles");
            System.out.println("\n total errs: " + (numstderror+numirregerror));
            System.out.println("");
        }
    }
    
    public static void randombottle_svmqConfMat() throws Exception {
        File[] resfiles = (new File("C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/SVMQ/Predictions/RandomBottlesEthanol/")).listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.contains("testFold");
            }
        });
        
        ArrayList<String> trueLabels = new ArrayList<>();
        ArrayList<String> predLabels = new ArrayList<>();
        Attribute classAtt = ClassifierTools.loadData("C:/JamesLPHD/Alcohol/JOURNALPAPER/Datasets/RandomBottlesEthanol/RandomBottlesEthanol0_TEST.arff").classAttribute();
        
        double[][] confmat = null;
        boolean first = true;
        
        for (File resfile : resfiles) {    
            ClassifierResults res = new ClassifierResults(resfile.getAbsolutePath());
            res.findAllStatsOnce();
            
            if (first) {
                confmat = res.confusionMatrix;
                first = false;
            } else {
                for (int i = 0; i < confmat.length; i++)
                    for (int j = 0; j < confmat.length; j++)
                        confmat[i][j] += res.confusionMatrix[i][j];
            }
            
            for (int i = 0; i < res.numInstances(); i++) {
                trueLabels.add(classAtt.value((int)res.getTrueClassValue(i)));
                predLabels.add(classAtt.value((int)res.getPredClassValue(i)));
            }
        }
        
        double sum = .0;
        for (int i = 0; i < confmat.length; i++)
            for (int j = 0; j < confmat.length; j++)
                sum += confmat[i][j];
        for (int i = 0; i < confmat.length; i++)
            for (int j = 0; j < confmat.length; j++)
                confmat[i][j] /= sum;
        
        double stdstdCor = .0;
        double stdstdIncor = .0;
        double stdirrCor = .0;
        double stdirrIncor = .0;
        double irrstdCor = .0;
        double irrstdIncor = .0;
        double irrirrCor = .0;
        double irrirrIncor = .0;
        
        for (int i = 0; i < 28; i++)
            for (int j = 0; j < 28; j++)
                if (i==j)
                    stdstdCor += confmat[i][j];
                else 
                    stdstdIncor += confmat[i][j];
        
        for (int i = 0; i < 28; i++)
            for (int j = 28; j < 44; j++)
                if (i==j)
                    stdirrCor += confmat[i][j];
                else 
                    stdirrIncor += confmat[i][j];
        
        for (int i = 28; i < 44; i++)
            for (int j = 0; j < 28; j++)
                if (i==j)
                    irrstdCor += confmat[i][j];
                else 
                    irrstdIncor += confmat[i][j];
        
        for (int i = 28; i < 44; i++)
            for (int j = 28; j < 44; j++)
                if (i==j)
                    irrirrCor += confmat[i][j];
                else 
                    irrirrIncor += confmat[i][j];
        
        System.out.println("stdstdCor   "+stdstdCor);
        System.out.println("stdstdIncor "+stdstdIncor);
        System.out.println("stdirrCor   "+stdirrCor);
        System.out.println("stdirrIncor "+stdirrIncor);
        System.out.println("irrstdCor   "+irrstdCor);
        System.out.println("irrstdIncor "+irrstdIncor);
        System.out.println("irrirrCor   "+irrirrCor);
        System.out.println("irrirrIncor "+irrirrIncor);
        
        double totalErr = stdstdIncor + stdirrIncor + irrstdIncor + irrirrIncor;
        double stdstdIncorProp = stdstdIncor / totalErr;
        double stdirrIncorProp = stdirrIncor / totalErr;
        double irrstdIncorProp = irrstdIncor / totalErr;
        double irrirrIncorProp = irrirrIncor / totalErr;
        
        System.out.println("");
        System.out.println("stdstdIncorProp = " + stdstdIncorProp);
        System.out.println("stdirrIncorProp = " + stdirrIncorProp);
        System.out.println("irrstdIncorProp = " + irrstdIncorProp);
        System.out.println("irrirrIncorProp = " + irrirrIncorProp);
        System.out.println("");
        
        double weightedStdstdIncorProp = stdstdIncorProp / (28*28 - 28);
        double weightedStdirrIncorProp = stdirrIncorProp / (28*16);
        double weightedIrrstdIncorProp = irrstdIncorProp / (16*28);
        double weightedIrrirrIncorProp = irrirrIncorProp / (16*16 - 16);
        
        System.out.println("");
        System.out.println("weightedStdstdIncorProp = " + weightedStdstdIncorProp);
        System.out.println("weightedStdirrIncorProp = " + weightedStdirrIncorProp);
        System.out.println("weightedIrrstdIncorProp = " + weightedIrrstdIncorProp);
        System.out.println("weightedIrrirrIncorProp = " + weightedIrrirrIncorProp);
        System.out.println("");
        
        System.out.print("confmat = [");
        for (int i = 0; i < confmat.length; i++) {
            for (int j = 0; j < confmat.length; j++) {
                System.out.print(" " + confmat[i][j]);
            }
            System.out.println(" ");
        }
        System.out.println("]");
        
        System.out.println("\n\n predLabels = " + predLabels.toString().replace(", ", "'; '").replace("[", "{'").replace("]", "'}"));
        
        System.out.println("\n\n trueLabels = " + trueLabels.toString().replace(", ", "'; '").replace("[", "{'").replace("]", "'}"));
    }
    
    public static void sigTestsUserSplitVs30Fold() {
        String baseReadPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/";
        
        String[] classifiers = classifiers_all;
        String[] dsets = { "JWRorJWB_BlackBottle", "JWRorJWB_RedBottle" };
        
    }
}
