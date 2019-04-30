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

package evaluation;

import experiments.Experiments;
import vector_classifiers.CAWPE;
import vector_classifiers.EnsembleSelection;
import vector_classifiers.TunedXGBoost;
import vector_classifiers.stackers.*;
import vector_classifiers.weightedvoters.*;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.EuclideanDistance;
import weka.core.Instances;

/**
 * Class with static methods for recreating large amounts of the analysis in the cawpe paper.
 * 
 * For examples of experimentation to generate the results, see vector_classifiers.CAWPE.buildCAWPEPaper_AllResultsForFigure3(),
 * or exampleExps() here
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPEResultsCollationCode {

//    public static String baseResultsReadPath = "C:/JamesLPHD/HESCA/";
    public static String baseResultsReadPath = "C:/JamesLPHD/HESCA/4TH_SUBMISSION_RESULTS/CAWPERerun/";
    
    
//    public static String analysisWritePath = "C:/JamesLPHD/HESCA/4TH_SUBMISSION_ANALYSIS/FinalisedAnalysis/";
    public static String analysisWritePath = "C:/JamesLPHD/HESCA/4TH_SUBMISSION_RESULTS/Analysis/";
    
    
    public static String[] UCI_FullSet_121 = { 
        "abalone","acute-inflammation","acute-nephritis","adult","annealing","arrhythmia","audiology-std","balance-scale","balloons","bank",
        "blood","breast-cancer","breast-cancer-wisc","breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car",
        "cardiotocography-3clases","cardiotocography-10clases","chess-krvk","chess-krvkp","congressional-voting","conn-bench-sonar-mines-rocks",
        "conn-bench-vowel-deterding","connect-4","contrac","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli",
        "energy-y1","energy-y2","fertility","flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian",
        "heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere",
        "iris","led-display","lenses","letter","libras","low-res-spect","lung-cancer","lymphography","magic","mammographic","miniboone",
        "molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom","musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d",
        "oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","optical","ozone","page-blocks",
        "parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D",
        "pittsburg-bridges-TYPE","planning","plant-margin","plant-shape","plant-texture","post-operative","primary-tumor",
        "ringnorm","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit","statlog-german-credit",
        "statlog-heart","statlog-image","statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates","synthetic-control",
        "teaching","thyroid","tic-tac-toe","titanic","trains","twonorm","vertebral-column-2clases","vertebral-column-3clases",
        "wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white","yeast","zoo",
    };
    
    public static String[] UCI_FullMinusPigsSet_117 = { 
        "abalone","acute-inflammation","acute-nephritis",/*"adult",*/"annealing","arrhythmia","audiology-std","balance-scale","balloons","bank",
        "blood","breast-cancer","breast-cancer-wisc","breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car",
        "cardiotocography-3clases","cardiotocography-10clases",/*"chess-krvk",*/"chess-krvkp","congressional-voting","conn-bench-sonar-mines-rocks",
        "conn-bench-vowel-deterding","connect-4","contrac","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli",
        "energy-y1","energy-y2","fertility","flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian",
        "heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere",
        "iris","led-display","lenses","letter","libras","low-res-spect","lung-cancer","lymphography",/*"magic",*/"mammographic",/*"miniboone",*/
        "molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom","musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d",
        "oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","optical","ozone","page-blocks",
        "parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D",
        "pittsburg-bridges-TYPE","planning","plant-margin","plant-shape","plant-texture","post-operative","primary-tumor",
        "ringnorm","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit","statlog-german-credit",
        "statlog-heart","statlog-image","statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates","synthetic-control",
        "teaching","thyroid","tic-tac-toe","titanic","trains","twonorm","vertebral-column-2clases","vertebral-column-3clases",
        "wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white","yeast","zoo",
    };
    
    public static String[] UCR_FullSet_85 = {
       "Adiac","ArrowHead","Beef","BeetleFly","BirdChicken","Car",
        "CBF","ChlorineConcentration","CinCECGtorso","Coffee","Computers","CricketX","CricketY","CricketZ","DiatomSizeReduction",
        "DistalPhalanxOutlineAgeGroup","DistalPhalanxOutlineCorrect","DistalPhalanxTW","Earthquakes","ECG200","ECG5000","ECGFiveDays",
        "ElectricDevices","FaceAll","FaceFour","FacesUCR","FiftyWords","Fish","FordA","FordB","GunPoint","Ham","HandOutlines","Haptics",
        "Herring","InlineSkate","InsectWingbeatSound","ItalyPowerDemand","LargeKitchenAppliances","Lightning2","Lightning7","Mallat","Meat",
        "MedicalImages","MiddlePhalanxOutlineAgeGroup","MiddlePhalanxOutlineCorrect","MiddlePhalanxTW","MoteStrain","NonInvasiveFetalECGThorax1",
        "NonInvasiveFetalECGThorax2","OliveOil","OSULeaf","PhalangesOutlinesCorrect","Phoneme","Plane","ProximalPhalanxOutlineAgeGroup",
        "ProximalPhalanxOutlineCorrect","ProximalPhalanxTW","RefrigerationDevices","ScreenType","ShapeletSim","ShapesAll","SmallKitchenAppliances",
        "SonyAIBORobotSurface1","SonyAIBORobotSurface2","StarlightCurves","Strawberry","SwedishLeaf","Symbols","SyntheticControl",
        "ToeSegmentation1","ToeSegmentation2","Trace","TwoLeadECG","TwoPatterns","UWaveGestureLibraryAll","UWaveGestureLibraryX",
        "UWaveGestureLibraryY","UWaveGestureLibraryZ","Wafer","Wine","WordSynonyms","Worms","WormsTwoClass","Yoga",
    };
    
    public static String[] UCIdsetList = UCI_FullSet_121;
    public static String[] UCRdsetList = UCR_FullSet_85;
    
    
    
    
    public static Classifier setClassifier_CAWPEPAPER(Experiments.ExperimentalArguments exp) { 
        Instances[] ins = null;
        try {
            ins = Experiments.sampleDataset(exp.dataReadLocation , exp.datasetName, 0);
        } catch (Exception e) {
            System.out.println("FAILED TO LOAD DATASET IN HACKY TRAIN INST NUMBER CHECK, LOCAL CAWPE SETCLASSIFIER");
            System.out.println(e);
            System.exit(0);
        }
        //tiny datasets like trains needs this 
        int cvfolds = Math.min(ins[0].numInstances()-1, 10);
        
        
        int fold = exp.foldId;
        switch (exp.classifierName) {
            
            case "CAWPE": 
                CAWPE cawpe = new CAWPE();
                cawpe.setRandSeed(fold);
                return cawpe;
                
            //CAWPE-A BASE CLASSIFIERS
            case "C45":
                return new J48();
            case "Logistic":
                return new Logistic();
            case "MLP":
                return new MultilayerPerceptron();
            case "NN":
                kNN k=new kNN(100);
                k.setCrossValidate(true);
                k.normalise(false);
                k.setDistanceFunction(new EuclideanDistance());
                return k;
            case "SVML":
                SMO svml=new SMO();
                PolyKernel p=new PolyKernel();
                p.setExponent(1);
                svml.setKernel(p);
                svml.setRandomSeed(fold);
                svml.setBuildLogisticModels(true);
                return svml;
                
            //CAWPE-S BASE CLASSIFIERS (+ 2 homogeneous ensembles, xgboost + randf+
            case "XGBoost":
                TunedXGBoost xg = new TunedXGBoost(); 
                xg.setRunSingleThreaded(true); 
                return xg;
            case "RotF":
                RotationForest rotf = new RotationForest();
                rotf.setSeed(fold);
                return rotf;
            case "RandF":
                RandomForest r=new RandomForest();
                r.setNumTrees(500);
                r.setSeed(fold);
                return r;
            case "SVMQ":
                SMO svmq=new SMO();
                PolyKernel poly=new PolyKernel();
                poly.setExponent(2);
                svmq.setKernel(poly);
                svmq.setRandomSeed(fold);
                svmq.setBuildLogisticModels(true);
                return svmq;
            case "MLP2":
                //still to be decided exactly what will happen with this in results remake
                //was initially a keras classifier
                break;
                
                
            //HETERO ENSEMBLES - SIMPLE COMPONENTS
            case "ES": 
                EnsembleSelection es = new EnsembleSelection();
                es.setRandSeed(fold);
                return es;
            case "SMLR":
                SMLR smlr = new SMLR();
                smlr.setRandSeed(fold);
                return smlr;
            case "SMLRE": 
                SMLRE smlre = new SMLRE();
                smlre.setRandSeed(fold);
                return smlre;
            case "SMM5":
                SMM5 smm5 = new SMM5();
                smm5.setRandSeed(fold);
                return smm5;
            case "PB":
                CAWPE_PickBest pb = new CAWPE_PickBest();
                pb.setRandSeed(fold);
                return pb;
            case "MV": 
                CAWPE_MajorityVote mv = new CAWPE_MajorityVote();
                mv.setRandSeed(fold);
                return mv;
            case "WMV":
                CAWPE_WeightedMajorityVote wmv = new CAWPE_WeightedMajorityVote();
                wmv.setRandSeed(fold);
                return wmv;
            case "RC":
                CAWPE_RecallCombiner rc = new CAWPE_RecallCombiner();
                rc.setRandSeed(fold);
                return rc;
            case "NBC":
                CAWPE_NaiveBayesCombiner nbc = new CAWPE_NaiveBayesCombiner();
                nbc.setRandSeed(fold);
                return nbc;
                
                
            //HOMOGENEOUS ENSEMBLES, xgboost randf above
            case "AdaBoost":
                //decision stump
                AdaBoostM1 ada = new AdaBoostM1();
                ada.setNumIterations(500);
                ada.setSeed(fold);
                return ada;
            case "Bagging":
                Bagging bag = new Bagging();
                bag.setNumIterations(500);
                bag.setBagSizePercent(70);
                bag.setSeed(fold);
                return bag;
            case "LogitBoost":
                LogitBoost lb = new LogitBoost();
                lb.setNumFolds(cvfolds);
                lb.setNumIterations(500);
                lb.setSeed(fold);
                return lb;   
                
                
            //TUNED CLASSIFIERS, TO BE DECIDED WHAT EXACTLY TO DO WITH THIS IN REMAKE
            //Bespoke-ly tuned classes removed in purge, exact reproduciblitiy with the 
            //generic tuned classifier to be confirmed
            //classes etc still available in old code snapshot 
        }
        return null; 
    }

    public static void examplesExps() throws Exception {
        int numFolds = 30;
        
//        String[] dsets = UCI_FullSet_121;

        String[] dsets = {
            "hayes-roth",
        };
         
        String[] classifiers = { "CAWPE", "C45", "NN", "Logistic", "MLP", "SVML" };
        
        Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments();
        exp.dataReadLocation = "C:/UCI Problems/";
        exp.resultsWriteLocation = "C:/JamesLPHD/HESCA/4TH_SUBMISSION_RESULTS/CAWPERerun/";
        exp.generateErrorEstimateOnTrainSet = true;
        
        Experiments.setupAndRunMultipleExperimentsThreaded(exp, classifiers, dsets, 0, numFolds, -1); 
        
//        for (String classifier : classifiers) {
//            for (String dset : dsets) {
//                for (int fold = 0; fold < numFolds; fold++) {
//                    Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments();
//                    exp.classifierName = classifier;
//                    exp.datasetName = dset;
//                    exp.foldId = fold;
//                    exp.dataReadLocation = "C:/UCI Problems/";
//                    exp.resultsWriteLocation = "C:/JamesLPHD/HESCA/4TH_SUBMISSION_RESULTS/CAWPERerun";
//
//                        Experiments.setupAndRunExperiment(exp);
//                }
//            }
//        }
    }
    
    
    
    
    public static void main(String[] args) throws Exception {
        generateALLStatsInPaper();

//        examplesExps();
    }

    public static void generateALLStatsInPaper() throws Exception {
        UCIdsetList = UCI_FullSet_121;
        UCRdsetList = UCR_FullSet_85;
        
        // SECTION 5.1 
//        ana_5_1_CAWPEVsComponents_UCI();
//        ana_5_1_CAWPEplusVsComponents_UCI(); 
//        
//        // SECTION 5.2
        ana_5_2_CAWPEVsHeteroEnsembles("UCI");
//        ana_5_2_CAWPEplusVsHeteroEnsembles("UCI"); 
//        
//        // SECTION 5.3
//        ana_5_3_CAWPEVsUntunedHomoEnsembles("UCI");
//        ana_5_3_CAWPEplusVsUntunedHomoEnsembles("UCI"); //not fully presented in paper, but results available online
//        
//        // SECTION 5.4 //5 folds only, missing 4 pig dsets
        UCIdsetList = UCI_FullMinusPigsSet_117;
//        ana_5_4_CAWPEVsTunedSingleClassifiers("UCI"); 
//        ana_5_4_CAWPEplusVsTunedSingleClassifiers("UCI"); 
//        ana_5_4_CAWPE_ON_TunedSingleClassifiers("UCI"); //not fully presented in paper, but results available online
        UCIdsetList = UCI_FullSet_121;
//        
//        // SECTION 5.5
//        ana_5_5_CAWPEVsComponents_UCR();
////        ana_5_5_CAWPEplusVsComponents_UCR();
//        
//        // SECTION 6.1 
//        //all the information this produces is actually already in the output of  ana_5_2_CAWPEVsHeteroEnsembles("UCI");
//        //but no harm recreating it here, without the noise of the other ensembles
//        ana_6_1_CAWPEvsPickBestTrainTestDiff("UCI");
//        ana_6_1_CAWPEplusvsPickBestTrainTestDiff("UCI");
//        
//        
//        // SECTION 6.2
//        ana_6_2_CAWPESchemeAblation("UCI");
//        ana_6_2_CAWPEvsWMCGroupings("UCI");
//        
//        // SECTION 6.3 
//        ana_6_3_CAWPE_alphaSensitivityComparison("UCI");
//        ana_6_3_CAWPEplus_alphaSensitivityComparison("UCI");
//        ana_6_3_CAWPE_isTuningAlphaWorthIt("UCI");
//        ana_6_3_CAWPEplus_isTuningAlphaWorthIt("UCI");
        




//        timingsRedo_ana_5_1_CAWPEVsComponents_UCI();
//        timingsRedo_ana_5_3_CAWPEVsUntunedHomoEnsembles("UCI");

//        ana_5_5_CAWPEVsComponents_UCR_FORTABLE();
//        ana_5_5_CAWPEplusVsComponents_UCR_FORTABLE();

        System.out.println("done");
        System.exit(0);
        //force disconnection of matlab instance if left open

    }
    
    public static void timingsRedo_ana_5_3_CAWPEVsUntunedHomoEnsembles(String dsetGroup) throws Exception {
        System.out.println("ana_5_3_CAWPEVsUntunedHomoEnsembles");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        //all homo ensembles have 500 iterations, otherwise default params
        //although, bagging has bagpercentsize set to 70, since default is 100 (thus making the making procedure useless...)
        new MultipleClassifierEvaluation(analysisWritePath, "5_3"+dsetGroup+"cawpeSVSuntunedHomoEnsembles", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] {  "CAWPE-S", "AdaBoost", "XGBoost", "RandF", "Bagging", "LogitBoost" }, 
                              new String[] {  "CAWPE-S", "AdaBoost",     "XGBoost",              "RandF", "Bagging",    "LogitBoost" },
                              basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    public static void  timingsRedo_ana_5_1_CAWPEVsComponents_UCI() throws Exception {
        System.out.println("ana_5_1_CAWPEVsComponents_UCI");
        String dsetGroup = "UCI";
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "5_1"+dsetGroup+"cawpeSVScomponents", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true, false).
            setDatasets(UCIdsetList).
            readInClassifiers(new String[] { "CAWPE-S", "SVML", "MLP", "NN", "C45", "Logistic" },
                              new String[] { "CAWPE-S", "SVML", "MLP1", "NN", "C4.5", "Logistic" },          
                              basePath+dsetGroup+"Results/").
            runComparison(); 
    }

    public static void ana_5_1_CAWPEVsComponents_UCI() throws Exception {
        System.out.println("ana_5_1_CAWPEVsComponents_UCI");
        String dsetGroup = "UCI";
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "5_1"+dsetGroup+"cawpeSVScomponents", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true, false).
            setDatasets(UCIdsetList).
            readInClassifiers(new String[] { "HESCA", "SVML", "MLP", "NN", "C4.5", "Logistic" },
                              new String[] { "CAWPE-S", "SVML", "MLP1", "NN", "C4.5", "Logistic" },          
                              basePath+dsetGroup+"Results/").
            runComparison(); 
    }
    
    public static void ana_5_5_CAWPEVsComponents_UCR() throws Exception {
        System.out.println("ana_5_5_CAWPEVsComponents_UCR");
        String dsetGroup = "UCR";
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        MultipleClassifierEvaluation mcc = new MultipleClassifierEvaluation(analysisWritePath, "5_5"+dsetGroup+"cawpeSVScomponents", 30);
        mcc.setTestResultsOnly(true);
        mcc.setBuildMatlabDiagrams(true,false);
        mcc.clearEvaluationStatistics();//dtw doesn't do preds, is 1nn. and the old results files don't have probabilities in them anyway
        mcc.addEvaluationStatistic(PerformanceMetric.acc); 
        mcc.addEvaluationStatistic(PerformanceMetric.balacc); 
        mcc.setDatasets(UCRdsetList);
//        mcc.readInClassifiers(new String[] { "DTW_Rn_1NN"}, 
//                            new String[] { "DTW", },          
//                            "Z:/Backups/Results_7_2_19/JayMovingInProgress/EEConstituentResults/");
        mcc.readInClassifiers(new String[] { "HESCA(-Logistic)", "SVML", "MLP", "NN", "C4.5", "DTW_Rn_1NN" }, //logistic regression cannot be reasonably run on some of the larger ucr datasets, even on the cluster
                            new String[] { "CAWPE-S",            "SVML", "MLP1", "NN", "C4.5", "DTW" },          
                            basePath+dsetGroup+"Results/");
        mcc.runComparison(); 
    }
    
    public static void ana_5_1_CAWPEplusVsComponents_UCI() throws Exception {
        System.out.println("ana_5_1_CAWPEplusVsComponents_UCI");
        String dsetGroup = "UCI";
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "5_1"+dsetGroup+"cawpeAVScomponents", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(UCIdsetList).
            readInClassifiers(new String[] { "HESCA+", "XGBoost500Iterations", "RotFDefault", "RandF", "SVMQ", "DNN" },
                              new String[] { "CAWPE-A",  "XGBoost",              "RotF",        "RandF", "SVMQ", "MLP2" },          
                              basePath+dsetGroup+"Results/").
            runComparison(); 
    }
    
    public static void ana_5_5_CAWPEplusVsComponents_UCR() throws Exception {
        System.out.println("ana_5_5_CAWPEplusVsComponents_UCR");
        String dsetGroup = "UCR";
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "5_5"+dsetGroup+"cawpeAVScomponents", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            clearEvaluationStatistics(). //dtw doesn't do preds, is 1nn. and the old results files don't have probabilities in them anyway
            addEvaluationStatistic(PerformanceMetric.acc). 
            addEvaluationStatistic(PerformanceMetric.balacc). 
            setDatasets(UCRdsetList).
//            readInClassifiers(new String[] { "DTW_Rn_1NN"}, 
//                              new String[] { "DTW", },          
//                              "Z:/Results/JayMovingInProgress/MovingSep_EE/").
            readInClassifiers(new String[] { "HESCA+", "XGBoost500Iterations", "RotFDefault", "RandF", "SVMQ", "DNN",  "DTW_Rn_1NN"},
                              new String[] { "CAWPE-A",  "XGBoost",              "RotF",        "RandF", "SVMQ", "MLP2", "DTW" },          
                              basePath+dsetGroup+"Results/").
            runComparison(); 
    }
    
    
    public static void ana_5_2_CAWPEVsHeteroEnsembles(String dsetGroup) throws Exception {
        System.out.println("ana_5_2_CAWPEVsHeteroEnsembles");

        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "5_2"+dsetGroup+"cawpeSVSheteroEnsembles", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
//            readInClassifiers(new String[] { "HESCA", "HESCA_MajorityVote", "HESCA_NaiveBayesCombiner", "HESCA_RecallCombiner", "HESCA_WeightedMajorityVote", "HESCA_PickBest", "EnsembleSelectionHESCAClassifiers", "SMLRE", "SMLR", "SMM5", },
//                              new String[] { "CAWPE", "MV",                 "NBC",                      "RC",                   "WMV",                        "PB",             "ES",                                "SMLRE", "SMLR", "SMM5", },
//                              basePath+dsetGroup+"Results/").
//            readInClassifiers(new String[] { "HESCA", "HESCA_MajorityVote", "HESCA_NaiveBayesCombiner", "HESCA_RecallCombiner", "HESCA_WeightedMajorityVote", "HESCA_PickBest", "EnsembleSelectionHESCAClassifiers_Preds", "SMLRE", "SMLR", "SMM5", },
//                              new String[] { "CAWPE-S", "MV-S",                 "NBC-S",                      "RC-S",                   "WMV-S",                        "PB-S",             "ES-S",                                "SMLRE-S", "SMLR-S", "SMM5-S", },
            readInClassifiers(new String[] { "CAWPE-S", "MV-S",   "NBC-S",    "RC-S",      "WMV-S",    "PB-S",     "ES-S",  "SMLRE-S", "SMLR-S", "SMM5-S", },
                              basePath+dsetGroup+"Results/").
            runComparison(); 
    }
    
    public static void ana_5_2_CAWPEplusVsHeteroEnsembles(String dsetGroup) throws Exception {
        System.out.println("ana_5_2_CAWPEplusVsHeteroEnsembles");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "5_2"+dsetGroup+"cawpeAVSheteroEnsembles", 30).
            setTestResultsOnly(false).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] { "HESCA+", "HESCA+_MajorityVote", "HESCA+_NaiveBayesCombiner", "HESCA+_RecallCombiner", "HESCA+_WeightedMajorityVote", "HESCA+_PickBest", "EnsembleSelectionHESCA+Classifiers_Preds", "SMLRE+", "SMLR+", "SMM5+", },
                              new String[] { "CAWPE-A", "MV-A",                 "NBC-A",                      "RC-A",                   "WMV-A",                        "PB-A",             "ES-A",                                "SMLRE-A", "SMLR-A", "SMM5-A", },
                              basePath+dsetGroup+"Results/").
//            readInClassifiers(new String[] { "HESCA+", "HESCA+_MajorityVote", "HESCA+_NaiveBayesCombiner", "HESCA+_RecallCombiner", "HESCA+_WeightedMajorityVote", "HESCA+_PickBest", "EnsembleSelectionHESCA+Classifiers", "SMLRE+", "SMLR+", "SMM5+", },
//                              new String[] { "CAWPE", "MV",                 "NBC",                      "RC",                   "WMV",                        "PB",             "ES",                                "SMLRE", "SMLR", "SMM5", },
//                              basePath+dsetGroup+"Results/").
            runComparison(); 
    }
    
    
    public static void ana_5_3_CAWPEVsUntunedHomoEnsembles(String dsetGroup) throws Exception {
        System.out.println("ana_5_3_CAWPEVsUntunedHomoEnsembles");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        //all homo ensembles have 500 iterations, otherwise default params
        //although, bagging has bagpercentsize set to 70, since default is 100 (thus making the making procedure useless...)
        new MultipleClassifierEvaluation(analysisWritePath, "5_3"+dsetGroup+"cawpeSVSuntunedHomoEnsembles", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] {  "HESCA", "AdaBoostM1DS", "XGBoost500Iterations", "RandF", "BaggingREP", "LogitBoost" }, 
                              new String[] {  "CAWPE-S", "AdaBoost",     "XGBoost",              "RandF", "Bagging",    "LogitBoost" },
                              basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    
    public static void ana_CAWPEandCAWPEplusVsUntunedHomoEnsembles(String dsetGroup) throws Exception {
        System.out.println("ana_CAWPEandCAWPEplusVsUntunedHomoEnsembles");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        //all homo ensembles have 500 iterations, otherwise default params
        //although, bagging has bagpercentsize set to 70, since default is 100 (thus making the making procedure useless...)
        new MultipleClassifierEvaluation(analysisWritePath, dsetGroup+"cawpeS-cawpeAVSuntunedHomoEnsembles", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] {  "HESCA+", "HESCA", "AdaBoostM1DS", "XGBoost500Iterations", "RandF", "BaggingREP", "LogitBoost" }, 
                              new String[] {  "CAWPE-A", "CAWPE-S", "AdaBoost",     "XGBoost",              "RandF", "Bagging",    "LogitBoost" },
                              basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    public static void ana_5_3_CAWPEplusVsUntunedHomoEnsembles(String dsetGroup) throws Exception {
        System.out.println("ana_5_3_CAWPEplusVsUntunedHomoEnsembles");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        //all homo ensembles have 500 iterations, otherwise default params
        //although, bagging has bagpercentsize set to 70, since default is 100 (thus making the making procedure useless...)
        new MultipleClassifierEvaluation(analysisWritePath, "5_3"+dsetGroup+"cawpeAVSuntunedHomoEnsembles", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] {  "HESCA+", "AdaBoostM1DS", "XGBoost500Iterations", "RandF", "BaggingREP", "LogitBoost" }, 
                              new String[] {  "CAWPE-A", "AdaBoost",     "XGBoost",              "RandF", "Bagging",    "LogitBoost" },
                              basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    
    public static void ana_5_4_CAWPEVsTunedSingleClassifiers(String dsetGroup) throws Exception {
        System.out.println("ana_5_4_CAWPEVsTunedSingleClassifiers");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "5_4"+dsetGroup+"cawpeSVStunedSingleClassifiers", 5).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
//            setDatasets(basePath + dsetGroup + "_4biggunsRemoved.txt").
            setDatasets(dsets).
            readInClassifiers(new String[] { "HESCA", }, 
                              new String[] { "CAWPE-S", }, 
                              basePath+dsetGroup+"Results/"). 
            readInClassifiers(new String[] { "TunedRandF", "TunedSVMRBF", },
                              new String[] { "TunedRandF", "TunedSVM", },
//                              "Z:/Results/FinalisedUCIContinuous"). 
//                              "Z:/Results and Code for Papers/CAWPE nee HESCA/RawResults/JamesCopyPasteDumpForCawpe/UCIResults/"). 
                                basePath+dsetGroup+"Results/"). 
            readInClassifiers(new String[] { "TunedXGBoost", "TunedTwoLayerMLP" },  
                              new String[] { "TunedXGBoost", "TunedMLP" },
//                              "Z:/Results/UCIContinuous"). 
//                              "Z:/Results and Code for Papers/CAWPE nee HESCA/RawResults/JamesCopyPasteDumpForCawpe/UCIResults/"). 
                                basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    
    public static void ana_5_4_CAWPEplusVsTunedSingleClassifiers(String dsetGroup) throws Exception {
        System.out.println("ana_5_4_CAWPEplusVsTunedSingleClassifiers");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "5_4"+dsetGroup+"cawpeAVStunedSingleClassifiers", 5).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
//            setDatasets(basePath + dsetGroup + "_4biggunsRemoved.txt").
            setDatasets(dsets).
            readInClassifiers(new String[] { "HESCA+", }, 
                              new String[] { "CAWPE-A", }, 
                              basePath+dsetGroup+"Results/"). 
            readInClassifiers(new String[] { "TunedRandF", "TunedSVMRBF", },
                              new String[] { "TunedRandF", "TunedSVM", },
//                              "Z:/Results/FinalisedUCIContinuous"). 
//                              "Z:/Results and Code for Papers/CAWPE nee HESCA/RawResults/JamesCopyPasteDumpForCawpe/UCIResults/"). 
                                basePath+dsetGroup+"Results/"). 
            readInClassifiers(new String[] { "TunedXGBoost", "TunedTwoLayerMLP" },  
                              new String[] { "TunedXGBoost", "TunedMLP" },
//                              "Z:/Results/UCIContinuous"). 
//                              "Z:/Results and Code for Papers/CAWPE nee HESCA/RawResults/JamesCopyPasteDumpForCawpe/UCIResults/"). 
                                basePath+dsetGroup+"Results/").    
            runComparison(); 
    }
    
    
    public static void ana_5_4_CAWPE_ON_TunedSingleClassifiers(String dsetGroup) throws Exception {
        System.out.println("ana_5_4_CAWPE_ON_TunedSingleClassifiers");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "5_4"+dsetGroup+"cawpeTVStunedSingleClassifiers", 5).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
//            setDatasets(basePath + dsetGroup + "_4biggunsRemoved.txt").
            setDatasets(dsets).
            readInClassifiers(new String[] { "HESCA_OverTunedClassifiers" }, 
                              new String[] { "CAWPE-T" }, 
                              basePath+dsetGroup+"Results/"). 
            readInClassifiers(new String[] { "TunedRandF", "TunedSVMRBF", },
                              new String[] { "TunedRandF", "TunedSVMRBF", },
//                              "Z:/Results/FinalisedUCIContinuous"). 
//                              "Z:/Results and Code for Papers/CAWPE nee HESCA/RawResults/JamesCopyPasteDumpForCawpe/UCIResults/"). 
                               basePath+dsetGroup+"Results/"). 
            readInClassifiers(new String[] { "TunedXGBoost", "TunedTwoLayerMLP" },  
                              new String[] { "TunedXGBoost", "TunedTwoLayerMLP" },
//                              "Z:/Results/UCIContinuous"). 
//                              "Z:/Results and Code for Papers/CAWPE nee HESCA/RawResults/JamesCopyPasteDumpForCawpe/UCIResults/"). 
                               basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    
    
    public static void ana_CAWPEBaseClassifiersInCompetition(String dsetGroup) throws Exception {
        System.out.println("ana_CAWPEBaseClassifiersInCompetition");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, dsetGroup+"CAWPEBaseClassifiersInCompetition", 5).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
//            setDatasets(basePath + dsetGroup + "_4biggunsRemoved.txt").
            setDatasets(dsets).
            readInClassifiers(new String[] { "HESCA_OverTunedClassifiers", "HESCA", "HESCA+", "HESCAks" }, 
                              new String[] { "CAWPEt", "CAWPE", "CAWPE+", "CAWPEks" }, 
                              basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    
    public static void ana_6_3_CAWPE_alphaSensitivityComparison(String dsetGroup) throws Exception {
        System.out.println("ana_6_3_CAWPE_alphaSensitivityComparison");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "6_3"+dsetGroup+"cawpeS_alphaSensitivityComparison", 30).
            setTestResultsOnly(false).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] { "HESCA_alpha=1","HESCA_alpha=2","HESCA_alpha=3","HESCA_alpha=4","HESCA_alpha=5",
                                            "HESCA_alpha=6","HESCA_alpha=7","HESCA_alpha=8","HESCA_alpha=9","HESCA_alpha=10",
                                            "HESCA_alpha=11","HESCA_alpha=12","HESCA_alpha=13","HESCA_alpha=14","HESCA_alpha=15",
                                            "HESCA_eq","HESCA_pb",
                            }, 
                              new String[] { "CAWPE_alpha=1","CAWPE_alpha=2","CAWPE_alpha=3","CAWPE_alpha=4","CAWPE_alpha=5",
                                            "CAWPE_alpha=6","CAWPE_alpha=7","CAWPE_alpha=8","CAWPE_alpha=9","CAWPE_alpha=10",
                                            "CAWPE_alpha=11","CAWPE_alpha=12","CAWPE_alpha=13","CAWPE_alpha=14","CAWPE_alpha=15",
                                            "CAWPE_eq","CAWPE_pb",
                            }, 
                            basePath+dsetGroup+"Results/AlphaSensitivities/"). 
            runComparison(); 
    }
    public static void ana_6_3_CAWPEplus_alphaSensitivityComparison(String dsetGroup) throws Exception {
        System.out.println("ana_6_3_CAWPEplus_alphaSensitivityComparison");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "6_3"+dsetGroup+"cawpeA_alphaSensitivityComparison", 30).
            setTestResultsOnly(false).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] { "HESCA+_alpha=1","HESCA+_alpha=2","HESCA+_alpha=3","HESCA+_alpha=4","HESCA+_alpha=5",
                                            "HESCA+_alpha=6","HESCA+_alpha=7","HESCA+_alpha=8","HESCA+_alpha=9","HESCA+_alpha=10",
                                            "HESCA+_alpha=11","HESCA+_alpha=12","HESCA+_alpha=13","HESCA+_alpha=14","HESCA+_alpha=15",
                                            "HESCA+_eq","HESCA+_pb",
                            }, 
                              new String[] { "CAWPE_alpha=1","CAWPE_alpha=2","CAWPE_alpha=3","CAWPE_alpha=4","CAWPE_alpha=5",
                                            "CAWPE_alpha=6","CAWPE_alpha=7","CAWPE_alpha=8","CAWPE_alpha=9","CAWPE_alpha=10",
                                            "CAWPE_alpha=11","CAWPE_alpha=12","CAWPE_alpha=13","CAWPE_alpha=14","CAWPE_alpha=15",
                                            "CAWPE_eq","CAWPE_pb",
                            }, 
                            basePath+dsetGroup+"Results/AlphaSensitivities/"). 
            runComparison(); 
    }
    
    public static void ana_6_3_CAWPE_isTuningAlphaWorthIt(String dsetGroup) throws Exception {
        System.out.println("ana_6_3_CAWPE_isTuningAlphaWorthIt");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "6_3"+dsetGroup+"cawpeS_isTuningAlphaWorthIt_noOracle", 30).
            setTestResultsOnly(false).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] { "HESCA_alpha=4","HESCA_bestAlphaTrain", },  //"HESCA_bestAlphaTest"
                              new String[] { "CAWPE-S(alpha=4)","CAWPE-S(RandTie)", },  //"CAWPE(Oracle)",
                            basePath+dsetGroup+"Results/AlphaSensitivities/"). 
            readInClassifiers(new String[] { "HESCA_TunedAlpha", }, 
                              new String[] { "CAWPE-S(ConTie)", }, 
                            basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    public static void ana_6_3_CAWPEplus_isTuningAlphaWorthIt(String dsetGroup) throws Exception {
        System.out.println("ana_6_3_CAWPEplus_isTuningAlphaWorthIt");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "6_3"+dsetGroup+"cawpeA_isTuningAlphaWorthIt_noOracle", 30).
            setTestResultsOnly(false).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] { "HESCA+_alpha=4","HESCA+_bestAlphaTrain", }, // "HESCA+_bestAlphaTest",
                              new String[] { "CAWPE-A(alpha=4)","CAWPE-A(RandTie)", },  // "CAWPE(Oracle)",
                            basePath+dsetGroup+"Results/AlphaSensitivities/"). 
            readInClassifiers(new String[] { "HESCA+_TunedAlpha", }, 
                              new String[] { "CAWPE-A(ConTie)", }, 
                            basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    public static void ana_CAWPEonTunedClassifiersVsHeteroEnsembles(String dsetGroup) throws Exception {
        System.out.println("ana_CAWPEonTunedClassifiersVsHeteroEnsembles");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, dsetGroup+"cawpeTVsHeteroEnsembles", 5).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
//            setDatasets(basePath + dsetGroup + "_4biggunsRemoved.txt").
            setDatasets(dsets).
            readInClassifiers(new String[] { 
                                    "EnsembleSelectionOverTunedClassifiers_Preds","HESCA_OverTunedClassifiers",
                                    "HESCA_OverTunedClassifiers_MajorityVote","HESCA_OverTunedClassifiers_NaiveBayesCombiner",
                                    "HESCA_OverTunedClassifiers_PickBest","HESCA_OverTunedClassifiers_RecallCombiner",
                                    "HESCA_OverTunedClassifiers_WeightedMajorityVote",
                              }, 
                              new String[] { 
                                  "ES","CAWPE-T",
                                  "MV","NBC",
                                  "PB","RC",
                                  "WMV",   
                              }, 
                            basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    
    public static void ana_6_2_CAWPESchemeAblation(String dsetGroup) throws Exception {
        System.out.println("ana_6_2_CAWPESchemeAblation");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "6_2"+dsetGroup+"cawpeS_SchemeAblation", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] { 
                                    "HESCA_MajorityVote",
                                    "HESCA_MajorityConfidence",
                                    "HESCA_WeightedMajorityVote",
                                    //"HESCA_alpha=1", below
                                    "HESCA_ExponentiallyWeightedVote",
                                    "HESCA",
                              }, 
                              new String[] { 
                                    "MV(a=0-preds)",
                                    "MC(a=0-probs)",
                                    "WMV(a=1-preds)",
                                    //"a=1,probs", below
                                    "EWMV(a=4-preds)",
                                    "CAWPE-S(a=4-probs)",
                              }, 
                            basePath+dsetGroup+"Results/"). 
            readInClassifiers(new String[] { "HESCA_alpha=1" }, 
                              new String[] { "WMC(a=1-probs)" }, 
                            basePath+dsetGroup+"Results/AlphaSensitivities/"). 
            runComparison(); 
    }
    public static void ana_6_2_CAWPEvsWMCGroupings(String dsetGroup) throws Exception {
        System.out.println("ana_6_2_CAWPEvsWMCGroupings");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "6_2"+dsetGroup+"CAWPESvsWMCGroupings", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            addDatasetGroupingFromDirectory("Z:/Data/DatasetGroupings/CAWPEvsWMCdsetGroupings/NumAttributes/").
            addDatasetGroupingFromDirectory("Z:/Data/DatasetGroupings/CAWPEvsWMCdsetGroupings/NumClasses/").
            addDatasetGroupingFromDirectory("Z:/Data/DatasetGroupings/CAWPEvsWMCdsetGroupings/NumTrainInstances/").
            readInClassifiers(new String[] { 
                                "HESCA",
                              }, 
                              new String[] { 
                                "CAWPE-S",
                              }, 
                            basePath+dsetGroup+"Results/"). 
            readInClassifiers(new String[] { "HESCA_alpha=1" }, 
                              new String[] { "WMC" }, 
                            basePath+dsetGroup+"Results/AlphaSensitivities/"). 
            runComparison(); 
    }
    public static void ana_6_1_CAWPEvsPickBestTrainTestDiff(String dsetGroup) throws Exception {
        System.out.println("ana_6_1_CAWPEvsPickBestTrainTestDiff");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "6_1"+dsetGroup+"CAWPESvsPickBestTrainTestDiff", 30).
            setTestResultsOnly(false).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] { 
                                "HESCA", "HESCA_PickBest"
                              }, 
                              new String[] { 
                                "CAWPE-S", "PB-S"
                              }, 
                            basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    public static void ana_6_1_CAWPEplusvsPickBestTrainTestDiff(String dsetGroup) throws Exception {
        System.out.println("ana_6_1_CAWPEplusvsPickBestTrainTestDiff");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "6_1"+dsetGroup+"CAWPEAvsPickBestTrainTestDiff", 30).
            setTestResultsOnly(false).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] { 
                                "HESCA+", "HESCA+_PickBest"
                              }, 
                              new String[] { 
                                "CAWPE-A", "PB-A"
                              }, 
                            basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    public static void ana_CAWPEvsCAWPEplus(String dsetGroup) throws Exception {
        System.out.println("ana_CAWPEvsCAWPEplus");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, dsetGroup+"CAWPESvsCAWPEA", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] { 
                                "HESCA", "HESCA+"
                              }, 
                              new String[] { 
                                "CAWPE-S", "CAWPE-A"
                              }, 
                            basePath+dsetGroup+"Results/"). 
            runComparison(); 
    }
    
        
    public static void ana_cawpeWeightingSchemes(String dsetGroup) throws Exception {
        System.out.println("ana_cawpeWeightingSchemes");
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, dsetGroup+"cawpeSWeightingSchemes", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
//            setDatasetGroupingFromDirectory("Z:/Data/DatasetGroupings/UCIGroupings_All121/GroupedByClassBalance/").
//            setUseAllStatistics().
            setDatasets(basePath + dsetGroup + ".txt").
            readInClassifiers(new String[] { 
//                                    "HESCA_AUROC",      "HESCA_BalancedAccuracy",      "HESCA_NLL",      "HESCA_TrainAcc",    //   "HESCA_FScore(1.0,1.0)", "HESCA_MCCWeighting", 
                                    "HESCA_AUROC(4.0)", "HESCA_BalancedAccuracy(4.0)", "HESCA_NLL(4.0)", "HESCA_TrainAcc(4.0)", // "HESCA_FScore(4.0,1.0)", "HESCA_MCCWeighting(4.0)",
//                                    "HESCA_AUROC(8.0)", "HESCA_BalancedAccuracy(8.0)", "HESCA_NLL(8.0)", "HESCA_TrainAcc(8.0)", // "HESCA_FScore(8.0,1.0)", "HESCA_MCCWeighting(8.0)",
                              },
                              new String[] { 
//                                    "AUC",   "BALACC",   "NLL",   "ACC",  //  "F1",   "MCC", 
                                    "AUC", "BALACC", "NLL", "ACC", // "F1^4", "MCC^4", 
//                                    "AUC^8", "BALACC^8", "NLL^8", "ACC^8", // "F1^8", "MCC^8", 
                              }, 
                              basePath+dsetGroup+"Results/WeightingSchemes/").
            runComparison(); 
    }
    
    
    public static void ana_CAWPEvsCAWPEWithTimingsCheck(String dsetGroup) throws Exception {
        System.out.println("ana_CAWPEvsCAWPEplus");
        String[] dsets = dsetGroup.equals("UCI") ? UCIdsetList : UCRdsetList;
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, dsetGroup+"cawpeMillisCheck", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(dsets).
            readInClassifiers(new String[] { 
                                "HESCA"
                              }, 
                              new String[] { 
                                "CAWPEbase", 
                              }, 
                            basePath+dsetGroup+"Results/"). 
            readInClassifiers(new String[] { 
                                "CAWPEInMillis"
                              }, 
                              new String[] { 
                                "CAWPEInMillis", 
                              }, 
                            "C:/JamesLPHD/HESCA/HESCATimingsRedone/results/"). 
            runComparison(); 
    }
    
    
    
    public static void ana_5_5_CAWPEVsComponents_UCR_FORTABLE() throws Exception {
        System.out.println("ana_5_5_CAWPEVsComponents_UCR_FORTABLE");
        String dsetGroup = "UCR";
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        MultipleClassifierEvaluation mcc = new MultipleClassifierEvaluation(analysisWritePath, "5_5"+dsetGroup+"cawpeSVScomponents_NoDTWallStats", 30);
        mcc.setTestResultsOnly(true);
        mcc.setBuildMatlabDiagrams(true,false);
        mcc.setDatasets(UCRdsetList);
//        mcc.readInClassifiers(new String[] { "DTW_Rn_1NN"}, 
//                            new String[] { "DTW", },          
//                            "Z:/Backups/Results_7_2_19/JayMovingInProgress/EEConstituentResults/");
        mcc.readInClassifiers(new String[] { "HESCA(-Logistic)", "SVML", "MLP", "NN", "C4.5" }, //logistic regression cannot be reasonably run on some of the larger ucr datasets, even on the cluster
                            new String[] { "CAWPE-S",            "SVML", "MLP1", "NN", "C4.5" },          
                            basePath+dsetGroup+"Results/");
        mcc.runComparison(); 
    }
    
    
    public static void ana_5_5_CAWPEplusVsComponents_UCR_FORTABLE() throws Exception {
        System.out.println("ana_5_5_CAWPEplusVsComponents_UCR_FORTABLE");
        String dsetGroup = "UCR";
        
        String basePath = baseResultsReadPath+dsetGroup+"/";
        
        new MultipleClassifierEvaluation(analysisWritePath, "5_5"+dsetGroup+"cawpeAVScomponents_NoDTWallStats", 30).
            setTestResultsOnly(true).
            setBuildMatlabDiagrams(true,false).
            setDatasets(UCRdsetList).
//            readInClassifiers(new String[] { "DTW_Rn_1NN"}, 
//                              new String[] { "DTW", },          
//                              "Z:/Results/JayMovingInProgress/MovingSep_EE/").
            readInClassifiers(new String[] { "HESCA+", "XGBoost500Iterations", "RotFDefault", "RandF", "SVMQ", "DNN"},
                              new String[] { "CAWPE-A",  "XGBoost",              "RotF",        "RandF", "SVMQ", "MLP2" },          
                              basePath+dsetGroup+"Results/").
            runComparison(); 
    }
}

