
package papers.smoothing;

import development.DataSets;
import development.Experiments;
import development.MultipleClassifierEvaluation;
import fileIO.OutFile;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import statistics.tests.TwoSampleTests;
import utilities.ClassifierResults;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.MultipleClassifierResultsCollection;
import utilities.StatisticalUtilities;
import vector_classifiers.ChooseClassifierFromFile;
import vector_classifiers.ChooseDatasetFromFile;
import vector_classifiers.CAWPE;
import weka.core.Instances;

/**
 * Some functions to compute generic evaluations on completed filtering results
 * 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class FilteringEvaluations {
    public static String[] UCRDsetsNoPigs = {	
        //Train Size, Test Size, Series Length, Nos Classes
        "Adiac",        // 390,391,176,37
        "ArrowHead",    // 36,175,251,3
        "Beef",         // 30,30,470,5
        "BeetleFly",    // 20,20,512,2
        "BirdChicken",  // 20,20,512,2
        "Car",          // 60,60,577,4
        "CBF",                      // 30,900,128,3
        "ChlorineConcentration",    // 467,3840,166,3
        "CinCECGtorso", // 40,1380,1639,4
        "Coffee", // 28,28,286,2
        "Computers", // 250,250,720,2
        "CricketX", // 390,390,300,12
        "CricketY", // 390,390,300,12
        "CricketZ", // 390,390,300,12
        "DiatomSizeReduction", // 16,306,345,4
        "DistalPhalanxOutlineCorrect", // 600,276,80,2
        "DistalPhalanxOutlineAgeGroup", // 400,139,80,3
        "DistalPhalanxTW", // 400,139,80,6
        "Earthquakes", // 322,139,512,2
        "ECG200",   //100, 100, 96
        "ECG5000",  //4500, 500,140
        "ECGFiveDays", // 23,861,136,2
        "FaceAll", // 560,1690,131,14
        "FaceFour", // 24,88,350,4
        "FacesUCR", // 200,2050,131,14
        "FiftyWords", // 450,455,270,50
        "Fish", // 175,175,463,7
        "GunPoint", // 50,150,150,2
        "Ham",      //105,109,431
        "Haptics", // 155,308,1092,5
        "Herring", // 64,64,512,2
        "InlineSkate", // 100,550,1882,7
        "InsectWingbeatSound",//1980,220,256
        "ItalyPowerDemand", // 67,1029,24,2
        "LargeKitchenAppliances", // 375,375,720,3
        "Lightning2", // 60,61,637,2
        "Lightning7", // 70,73,319,7
        "Mallat", // 55,2345,1024,8
        "Meat",//60,60,448
        "MedicalImages", // 381,760,99,10
        "MiddlePhalanxOutlineCorrect", // 600,291,80,2
        "MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
        "MiddlePhalanxTW", // 399,154,80,6
        "MoteStrain", // 20,1252,84,2
        "OliveOil", // 30,30,570,4
        "OSULeaf", // 200,242,427,6
//        "PhalangesOutlinesCorrect", // 1800,858,80,2      ED SIEVE DID NOT HAVE THIS INCLUDED, SO JUST DROPPING IT 
        "Phoneme",//1896,214, 1024
        "Plane", // 105,105,144,7
        "ProximalPhalanxOutlineCorrect", // 600,291,80,2
        "ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
        "ProximalPhalanxTW", // 400,205,80,6
        "RefrigerationDevices", // 375,375,720,3
        "ScreenType", // 375,375,720,3
        "ShapeletSim", // 20,180,500,2
        "ShapesAll", // 600,600,512,60
        "SmallKitchenAppliances", // 375,375,720,3
        "SonyAIBORobotSurface1", // 20,601,70,2
        "SonyAIBORobotSurface2", // 27,953,65,2
        "Strawberry",//370,613,235
        "SwedishLeaf", // 500,625,128,15
        "Symbols", // 25,995,398,6
        "SyntheticControl", // 300,300,60,6
        "ToeSegmentation1", // 40,228,277,2
        "ToeSegmentation2", // 36,130,343,2
        "Trace", // 100,100,275,4
        "TwoLeadECG", // 23,1139,82,2
        "TwoPatterns", // 1000,4000,128,4
        "UWaveGestureLibraryX", // 896,3582,315,8
        "UWaveGestureLibraryY", // 896,3582,315,8
        "UWaveGestureLibraryZ", // 896,3582,315,8
        "Wafer", // 1000,6164,152,2
        "Wine",//54	57	234
        "WordSynonyms", // 267,638,270,25
        "Worms", //77, 181,900,5
        "WormsTwoClass",//77, 181,900,5
        "Yoga" // 300,3000,426,2
    };   
    
    public static void main(String[] args) throws Exception {
//        extractDefaultParaResults();
//        defaultParameterComparisonPerClassifier();
//        tunedParameterComparisonPerClassifier();
//        extractFilterParameterSelectionsFromOptimisedFoldFiles();

        
        tunedParameterComparisonPerClassifier_WITHOUT_NO_SMOOTHING_OPTION();
                
//        selectFilterParametersAndWriteResults();
//        selectFilterAndWriteResults();
//        extractFilterParameterSelectionsFromOptimisedFoldFiles();
//        whatIfWeJustEnsembleOverOptimisedFilters();
//        performStandardClassifierComparisonWithFilteredAndUnfilteredDatasets();

        
//        performSomeSimpleStatsOnWhetherFilteringIsAtAllReasonableWithTestData();

//        //just wanted to double check none where missing
//        MultipleClassifierResultsCollection mcrc = new MultipleClassifierResultsCollection(
//                new String [] { "ED", "DTWCV", "RotF" }, 
//                UCRDsetsNoPigs, 
//                30, 
//                "C:/JamesLPHD/TSC_Smoothing/Results/TSCProblems_MovingAverage/", 
//                true, true, false);



    }
    
    
    
    
    
    /**
     * Assumes folder structure: 
     * some base path
     *      analysis
     *      results
     *          filtertype1
     *              classifier1
     *                  predictions
     *                      dataset1
     *                          foldfiles
     *                      dataset2
     *                      ...
     *              classifier2
     *                  ...
     *              ...
     *          filtertype2
     *              ...
     *          ...
     */
    public static  void performStandardClassifierComparisonWithFilteredAndUnfilteredDatasets() throws Exception {
        String expName = "RotFvsFilterEnesmble_MA,PCA,EXP,SG,DFT";
//        String expName = "RotFvsRotFwith(DFT_EXP_SG)vsRotFFilterANDParaSelected";
        
        String basePath = "C:/JamesLPHD/TSC_Smoothing/";
        String analysisPath = basePath + "Analysis/";
        String baseResultsPath = basePath + "Results/";
        String unfilteredResultsPath = baseResultsPath + "TSC_Unfiltered/";


        String baseClassifier = "RotF";        
//        String[] filters = { "_PCAFiltered", "_DFTFiltered", "_EXPFiltered", "_SGFiltered", "_MAFiltered" }; //
//        String[] filterResultsFolders = { "TSCProblems_PCA_smoothed", "TSC_FFT_zeroed", "TSC_Exponential", "TSC_SavitzkyGolay", "TSCProblems_MovingAverage" }; 
//        String[] filters = { "_SGFiltered" }; //
//        String[] filterResultsFolders = { "TSC_SavitzkyGolay" }; 
        
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, expName, 30);
        mce.setBuildMatlabDiagrams(true);
//            setUseAllStatistics().
        mce.setDatasets(UCRDsetsNoPigs);
        
        mce.setPerformPostHocDsetResultsClustering(true);
        mce.addAllDatasetGroupingsInDirectory("Z:/Data/DatasetGroupings/UCRUEAGroupings_77Dsets_Nonpigs/");

        mce.readInClassifier(baseClassifier, unfilteredResultsPath);
        mce.readInClassifier(baseClassifier+"_6FilterEnsemble", baseResultsPath + "ENSEMBLES");
//        mce.readInClassifier(baseClassifier+"_FILTERED2", baseResultsPath + "FilterSelected");
//        for (int i = 0; i < filters.length; i++)
//            mce.readInClassifier(baseClassifier + filters[i], baseResultsPath + filterResultsFolders[i]);
        
        mce.runComparison(); 
    }
    
    
    public static void defaultParameterComparisonPerClassifier() throws Exception { 
        //note atm the matlab proxy will disconnect itself after the first run in the loop
        //just run one at a time mb 
        for (String classifier : new String [] { 
            "ED", 
//            "DTWCV", 
//            "RotF" 
        }) { 
            String expName = classifier+"vs"+classifier+"WithDefaultFilters";
    //        String expName = "RotFvsRotFwith(DFT_EXP_SG)vsRotFFilterANDParaSelected";

            String basePath = "C:/JamesLPHD/TSC_Smoothing/";
            String analysisPath = basePath + "Analysis/";
            String baseResultsPath = basePath + "Results/";
            String unfilteredResultsPath = baseResultsPath + "TSC_Unfiltered/";     
            String[] filterSuffixes = { 
                //(unfiltered),
                "-DFT_1-", 
                "-EXP_5-", 
                "-SG_d0_n2_m5-", 
                "-MA_5-",
                "-Gauss_5-",
                "_Sieved_5th"
            }; 
            
            String[] cleanFilterNames = {
                //(unfiltered),
                "-DFT", 
                "-EXP", 
                "-SG", 
                "-MA",
                "-GF",
                "-SIV"
            };
            

            MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, expName, 10);
            mce.setBuildMatlabDiagrams(true);
    //            setUseAllStatistics().
            mce.setDatasets(UCRDsetsNoPigs);

//            mce.setPerformPostHocDsetResultsClustering(true);
//            mce.addAllDatasetGroupingsInDirectory("Z:/Data/DatasetGroupings/UCRUEAGroupings_77Dsets_Nonpigs/");

            mce.readInClassifier(classifier, unfilteredResultsPath);
            for (int i = 0; i < filterSuffixes.length; i++)
                mce.readInClassifier(classifier + filterSuffixes[i], classifier + cleanFilterNames[i], baseResultsPath + "DefaultParaResults/");
            mce.runComparison(); 
        }
    }    
    
    
    public static void tunedParameterComparisonPerClassifier() throws Exception { 
        //note atm the matlab proxy will disconnect itself after the first run in the loop
        //just run one at a time mb 
        for (String classifier : new String [] { 
                "ED", 
//                "DTWCV", 
//                "RotF" 
        }) { 
            String expName = classifier+"vs"+classifier+"WithTunedFilters";
    //        String expName = "RotFvsRotFwith(DFT_EXP_SG)vsRotFFilterANDParaSelected";

            String basePath = "C:/JamesLPHD/TSC_Smoothing/";
            String analysisPath = basePath + "Analysis/";
            String baseResultsPath = basePath + "Results/";
            String unfilteredResultsPath = baseResultsPath + "TSC_Unfiltered/";     
            String[] filterSuffixes = { 
                //(unfiltered),
                "_DFTFiltered", 
                "_EXPFiltered", 
                "_SGFiltered", 
                "_MAFiltered",
                "_GFiltered",
                "_Sieved"
            }; 
            String[] cleanFilterNames = {
                //(unfiltered),
                "-DFT", 
                "-EXP", 
                "-SG", 
                "-MA",
                "-GF",
                "-SIV"
            };
            

            MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, expName, 10);
            mce.setBuildMatlabDiagrams(true);
    //            setUseAllStatistics().
            mce.setDatasets(UCRDsetsNoPigs);

            mce.setPerformPostHocDsetResultsClustering(true);
            mce.addAllDatasetGroupingsInDirectory("Z:/Data/DatasetGroupings/UCRUEAGroupings_76Dsets_Nonpigs_NoPhalanges/");

            mce.setUseAccuracyOnly();

            mce.readInClassifier(classifier, unfilteredResultsPath);
            for (int i = 0; i < filterSuffixes.length; i++)
                mce.readInClassifier(classifier + filterSuffixes[i], classifier + cleanFilterNames[i], baseResultsPath + "FilterTuned/");

            mce.runComparison(); 
        }
    }    
    public static void tunedParameterComparisonPerClassifier_WITHOUT_NO_SMOOTHING_OPTION() throws Exception { 
        //note atm the matlab proxy will disconnect itself after the first run in the loop
        //just run one at a time mb 
        for (String classifier : new String [] { 
                "ED", 
//                "DTWCV", 
//                "RotF" 
        }) { 
            String expName = classifier+"vs"+classifier+"WithTunedFilters_No'None'Option";
    //        String expName = "RotFvsRotFwith(DFT_EXP_SG)vsRotFFilterANDParaSelected";

            String basePath = "C:/JamesLPHD/TSC_Smoothing/";
            String analysisPath = basePath + "Analysis/";
            String baseResultsPath = basePath + "Results/";
//            String unfilteredResultsPath = baseResultsPath + "TSC_Unfiltered/";     
            String[] filterSuffixes = { 
                "_DFTFiltered_No'None'Option",
                "_EXPFiltered_No'None'Option",
                "_SGFiltered_No'None'Option", 
                "_MAFiltered_No'None'Option", 
                "_GFiltered_No'None'Option", 
                "_Sieved_No'None'Option"  
            }; 
            String[] cleanFilterNames = {
                //(unfiltered),
                "-DFT", 
                "-EXP", 
                "-SG", 
                "-MA",
                "-GF",
                "-SIV"
            };
            

            MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, expName, 10);
            mce.setBuildMatlabDiagrams(true);
    //            setUseAllStatistics().
            mce.setDatasets(UCRDsetsNoPigs);

//            mce.setPerformPostHocDsetResultsClustering(true);
//            mce.addAllDatasetGroupingsInDirectory("Z:/Data/DatasetGroupings/UCRUEAGroupings_77Dsets_Nonpigs/");

            mce.setUseAccuracyOnly();

//            mce.readInClassifier(classifier, unfilteredResultsPath);
            for (int i = 0; i < filterSuffixes.length; i++)
                mce.readInClassifier(classifier + filterSuffixes[i], classifier + cleanFilterNames[i], baseResultsPath + "FilterTunedNo'None'Filter/");

            mce.runComparison(); 
        }
    }    
    
    public static void extractDefaultParaResults() throws Exception {
        String baseReadPath = "C:/JamesLPHD/TSC_Smoothing/Results/";
        String baseWritePath = "C:/JamesLPHD/TSC_Smoothing/Results/DefaultParaResults/";
//        String baseReadPath = "Z:/Results/SmoothingExperiments/";
        String[] baseDatasets = UCRDsetsNoPigs;
//        String[] baseDatasets = new String[] { "Worms" };
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 10;
        
//        String unfilteredResultsPath = "TSC_Unfiltered";
//        String unfilteredReadPath = baseReadPath + unfilteredResultsPath + "/";
        
    
        //TODO. THE PARA NUMBERS SUFFIXED TO THE DATASETS ARE UNEVEN, AND I DON'T KNOW THE FUNCTION 
        //THERE ARE 15 PARAS PER DATASET THOUGH, SO FOR NOW, JUST TAKING THE 5TH PARA FOR EACH DATASET
        //WHATEVER THAT IS
        final int SIEVE_PARA_NUMBER = 5;


        String[] filterSuffixes = { 
//            "-DFT_1-", 
//            "-EXP_5-", 
//            "-SG_d0_n2_m5-", 
//            "-MA_5-",
//            "-Gauss_5-",
            "_Sieved_" + SIEVE_PARA_NUMBER + "th"
        }; 
        String[] filterResultsPaths = { 
//            baseReadPath+"TSC_FFT_zeroed/",
//            baseReadPath+"TSC_Exponential/", 
//            baseReadPath+"TSC_SavitzkyGolay/",  
//            baseReadPath+"TSCProblems_MovingAverage/", 
//            baseReadPath+"TSCProblems_Gaussian/", 
            "Z:/Results/SmoothingExperiments/TSC_Sieved/"
        }; 
        
        for (String classifier : new String [] { "ED", "DTWCV", "RotF" }) { 
            for (int filterid = 0; filterid < filterResultsPaths.length; filterid++) {
                String classifierName = classifier + filterSuffixes[filterid];
                String classifierTargetDir = baseWritePath+classifierName+"/Predictions/";
                (new File(classifierTargetDir)).mkdirs();
                
                String classifierSourceDir = filterResultsPaths[filterid] + classifier + "/Predictions/";
                
                for (int dset = 0; dset < numBaseDatasets; dset++) {
                    String baseDset = baseDatasets[dset];

                    String datasetTargetDir = classifierTargetDir+baseDset+"/";
                    (new File(datasetTargetDir)).mkdirs();
                    
                    String datasetSourceDir = "";
                    if (filterSuffixes[filterid].contains("Sieved"))
                        try {
                            datasetSourceDir = classifierSourceDir + baseDset + getSieveDefaultPara(classifierSourceDir,baseDset,SIEVE_PARA_NUMBER) + "/";
                        }catch(Exception e) {
                            System.out.println(baseDset);
                            continue;
                        }
                    else
                        datasetSourceDir = classifierSourceDir + baseDset + filterSuffixes[filterid] + "/";
                    
                    for (int fold = 0; fold < numFolds; fold++) {
                        for (String split : new String[] { "train", "test" }) {
                            String fname = split+"Fold"+fold+".csv";
                            
                            File sourceFile = new File(datasetSourceDir + fname);
                            File targetFile = new File(datasetTargetDir + fname);
                            Files.copy(sourceFile.toPath(), targetFile.toPath());
                        }
                        
                    }
                }   
            }
        }
        
    }

    public static String getSieveDefaultPara(String folder, String baseDset, int paraNumberOutOf15){
        String[] datasets = (new File(folder)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                //e.g CricketZ-PCA_95-     =>    CricketZ
                //the "_" split is included jsut as future proofing. non of the datasetnames include _
                //...he said confidently
                return name.split("_")[0].equals(baseDset);
            }
        });
        
        ArrayList<Integer> paras = new ArrayList<>(datasets.length);
        for (String datasetPara : datasets)
            paras.add(Integer.parseInt(datasetPara.split("_")[2]));
        
        Collections.sort(paras);
        
        return "_Sieved_" + paras.get(paraNumberOutOf15);
    }
    
    /**
     * this makes the '_FILTERED'-suffixed results
     * given the base classifier results (with no filter), and results for each filtered version,
     * where the parameters of the filter have already been selected (the '_XXFiltered' versions),
     * selects the best from these. i.e from the train data picks the best filter to use (given optimal parameters on train set), 
     * or no filtering at all, and copies across the top trainfold files and corresponding test files 
     */
    public static void selectFilterAndWriteResults() throws Exception {
        String baseReadPath = "C:/JamesLPHD/TSC_Smoothing/Results/";
//        String baseReadPath = "Z:/Results/SmoothingExperiments/";
        String[] baseDatasets = UCRDsetsNoPigs;
//        String[] baseDatasets = new String[] { "Worms" };
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 30;
        
        String unfilteredResultsPath = "TSC_Unfiltered";
        String unfilteredReadPath = baseReadPath + unfilteredResultsPath + "/";
        
        String[] filterSuffixes = { 
            //(unfiltered),
            "_DFTFiltered", 
            "_EXPFiltered", 
            "_SGFiltered", 
            "_PCAFiltered", 
            "_MAFiltered",
            "_GFiltered",
//            "_Sieved"
        }; 
        String[] filterResultsPaths = { 
            unfilteredReadPath, 
            baseReadPath+"TSC_FFT_zeroed",
            baseReadPath+"TSC_Exponential", 
            baseReadPath+"TSC_SavitzkyGolay", 
            baseReadPath+"TSCProblems_PCA_smoothed", 
            baseReadPath+"TSCProblems_MovingAverage", 
            baseReadPath+"TSCProblems_Gaussian", 
//            baseReadPath+"Paul" 
        }; 
        
        for (String classifier : new String [] { "ED", "DTWCV", "RotF" }) { 

            String[] classifierNames = new String[filterResultsPaths.length];
            classifierNames[0] = classifier;
            for (int i = 0; i < filterSuffixes.length; i++)
                classifierNames[i+1] = classifier + filterSuffixes[i];
            
            for (int dset = 0; dset < numBaseDatasets; dset++) {
                String baseDset = baseDatasets[dset];

                for (int fold = 0; fold < numFolds; fold++) {
                    ChooseClassifierFromFile ccff = new ChooseClassifierFromFile();
                    ccff.setName(classifier + "_FILTERED3");
                    ccff.setClassifiers(classifierNames);
                    ccff.setRelationName(baseDset);
                    ccff.setResultsPath(filterResultsPaths);
                    ccff.setResultsWritePath(baseReadPath + "FilterSelected/");
                    ccff.setFold(fold);

                    ccff.buildClassifier(null);
                }
            }
        }
        
    }
    
    /**
     * this makes the '_XXFiltered'-suffixed results
     * given the classifier results for a filter with all it's different parameter settings, 
     * chooses the best 'dataset' (filter parameter) and organises the corresponding train/test fold files 
     * to mirror a single classifier on single dataset (the chosen filtered-version)
     */
    public static  void selectFilterParametersAndWriteResults() throws Exception {
        String baseReadPath = "Z:/Results/SmoothingExperiments/";
//        String[] baseDatasets = DataSets.tscProblems85;
        String[] baseDatasets = { "WordSynonyms" };//UCRDsetsNoPigs;
//        String[] baseDatasets = new String[] { "Worms" };
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 10;
        
        String unfilteredReadPath = baseReadPath + "TSC_Unfiltered/";
        boolean INCLUDE_NOSMOOTHING_OPTION = false;
        
        
        String[] filterSuffixes = { /*"_GFiltered_No'None'Option", "_DFTFiltered_No'None'Option", "_EXPFiltered_No'None'Option", "_SGFiltered_No'None'Option", "_MAFiltered_No'None'Option",*/ "_Sieved_No'None'Option"  }; //"_PCAFiltered_No'None'Option",
        String[] filterResultsPaths = { /*"TSCProblems_Gaussian", "TSC_FFT_zeroed", "TSC_Exponential", "TSC_SavitzkyGolay", "TSCProblems_MovingAverage",*/ "TSC_Sieved" }; //"TSCProblems_PCA_smoothed"
        
        for (int i = 0; i < filterResultsPaths.length; i++) {
            String filterReadPath = baseReadPath + filterResultsPaths[i] + "/";
            String filterSuffix = filterSuffixes[i];
            
            for (String classifier : new String [] { "ED", "DTWCV", "RotF" }) { // 
                for (int dset = 0; dset < numBaseDatasets; dset++) {
                    String baseDset = baseDatasets[dset];
                    
                    
                    //COPYING BASE DATASET OVER TO FILTERED RESULTS DIRECTORY
                    File sourceLocation = new File(unfilteredReadPath + classifier + "/Predictions/" + baseDset + "/");
                    File targetLocation = new File(filterReadPath + classifier + "/Predictions/" + baseDset + "/");
                    if (INCLUDE_NOSMOOTHING_OPTION) {
                        targetLocation.mkdirs();
                        for (File foldFile : sourceLocation.listFiles())
                            Files.copy(foldFile.toPath(), (new File(targetLocation.getAbsolutePath() + "/" + foldFile.getName())).toPath());    
                    }
                    //END

                    for (int fold = 0; fold < numFolds; fold++) {
                        ChooseDatasetFromFile cdff = new ChooseDatasetFromFile();
                        cdff.setName(classifier + filterSuffix);
                        cdff.setClassifier(classifier);
                        cdff.setFinalRelationName(baseDset);
                        cdff.setResultsPath(baseReadPath + filterResultsPaths[i] + "/");
//                        cdff.setResultsPath(filterReadPath);
                        cdff.setFold(fold);

                        String[] datasets = (new File(filterReadPath + classifier + "/Predictions/")).list(new FilenameFilter() {
                            @Override
                            public boolean accept(File dir, String name) {
                                //e.g CricketZ-PCA_95-     =>    CricketZ
                                //the "_" split is included jsut as future proofing. non of the datasetnames include _
                                //...he said confidently
                                return name.split("-")[0].split("_")[0].equals(baseDset);
                            }
                        });
                        Arrays.sort(datasets);
                        if (INCLUDE_NOSMOOTHING_OPTION) {
                            if (!datasets[0].equals(baseDset))
                                throw new Exception("hwut" + baseDset  +"/n" + Arrays.toString(datasets));
                        }
                        cdff.setRelationNames(datasets);

                        cdff.buildClassifier(null);
                    }
                    
                    //DELETING COPIED BASE DATASET FROM FILERED RESULTS DIRECTORY        
                    if (INCLUDE_NOSMOOTHING_OPTION) {
                        for (File foldFile : targetLocation.listFiles())
                            foldFile.delete();
                        targetLocation.delete();                        
                    }
                    //END
                }
            }
        }
    }
    
    public static void performSomeSimpleStatsOnWhetherFilteringIsAtAllReasonableWithTestData() throws Exception { 
        final double P_VAL = 0.05;
        
        String baseReadPath = "C:/JamesLPHD/TSC_Smoothing/Results/";
        String[] classifiers = { "ED" };
        String[] baseDatasets = DataSets.tscProblems85;
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 30;
        boolean testResultsOnly = false;
        boolean cleanResults = true;
        boolean allowMissing = false;
        
        MultipleClassifierResultsCollection[] mcrcs = new MultipleClassifierResultsCollection[numBaseDatasets];
        boolean [] aFilteredVersionIsSigBetter = new boolean[numBaseDatasets];
        boolean [] aFilteredVersionIsBetter = new boolean[numBaseDatasets];
        boolean [] unFilteredVersionIsSigBetterThanAllFiltered = new boolean[numBaseDatasets];
//        boolean [] unFilteredVersionIsBetterThanAllFiltered = new boolean[numBaseDatasets];
        
        for (int i = 0; i < numBaseDatasets; i++) {
            String datasetBase = baseDatasets[i];
            String[] datasets = (new File(baseReadPath + classifiers[0] + "/Predictions/")).list(new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    //e.g CricketZ-PCA_95-     =>    CricketZ
                    //the "_" split is included jsut as future proofing. non of the datasetnames include _
                    //...he said confidently
                    return name.split("-")[0].split("_")[0].equals(datasetBase);
                }
            });
            Arrays.sort(datasets);
            if (!datasets[0].equals(datasetBase))
                throw new Exception("hwut" + datasetBase  +"/n" + Arrays.toString(datasets));
            
            MultipleClassifierResultsCollection mcrc = new MultipleClassifierResultsCollection(classifiers, datasets, numFolds, baseReadPath, testResultsOnly, cleanResults, allowMissing);
            mcrcs[i] = mcrc;
            
            double[][] resFolds = mcrc.getAccuracies()[1][0]; // [test][firstclassifier]
            double[] resDsets = StatisticalUtilities.averageFinalDimension(resFolds); 
            
            double unfilteredAcc = resDsets[0];
            
            boolean allFilteredAreSigWorse = true;
            for (int j = 1; j < resDsets.length; j++) {
                double p = TwoSampleTests.studentT_PValue(resFolds[0], resFolds[j]);
                if (resDsets[j] > unfilteredAcc) {
                    aFilteredVersionIsBetter[i] = true;
                    if (p < P_VAL) 
                        aFilteredVersionIsSigBetter[i] = true;
                }
                else {
                    if (p > P_VAL)
                        allFilteredAreSigWorse = false;
                }
            }
            unFilteredVersionIsSigBetterThanAllFiltered[i] = allFilteredAreSigWorse;
        }    
        
        System.out.println("aFilteredVersionIsSigBetter: " + countNumTrue(aFilteredVersionIsSigBetter) );
        System.out.println("aFilteredVersionIsBetter: " + countNumTrue(aFilteredVersionIsBetter) );
        System.out.println("unFilteredVersionIsSigBetterThanAllFiltered: " + countNumTrue(unFilteredVersionIsSigBetterThanAllFiltered) );
    }
    
    public static int countNumTrue(boolean[] arr) { 
        int counter = 0;
        for (boolean b : arr)
            if (b) counter++;
        return counter;
    }
    
    public static void extractFilterParameterSelectionsFromOptimisedFoldFiles() throws IOException, Exception { 
        String baseReadPath = "C:/JamesLPHD/TSC_Smoothing/Results/";
        String[] baseDatasets = UCRDsetsNoPigs;
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 10;
        
        String unfilteredReadPath = baseReadPath + "TSC_Unfiltered/";
        
        String[] filterSuffixes = { "_DFTFiltered", "_EXPFiltered", "_SGFiltered", "_Sieved", "_MAFiltered", "_GFiltered"  }; //"_PCAFiltered"
//        String[] filterResultsPaths = { "TSC_FFT_zeroed", "TSC_Exponential", "TSC_SavitzkyGolay", "TSCProblems_PCA_smoothed", "TSCProblems_MovingAverage", "TSCProblems_Gaussian" }; //
        String[] filterResultsPaths = { "FilterTuned", "FilterTuned", "FilterTuned", "FilterTuned", "FilterTuned", "FilterTuned" }; //
        
        ArrayList<String> tssColumnHeaders = new ArrayList<>();
        ArrayList<String> tssRowHeaders = new ArrayList<>();
        ArrayList<ArrayList<Double>> tsstrainAccs = new ArrayList<>();
        ArrayList<ArrayList<Double>> tsstestAccs = new ArrayList<>();
        ArrayList<ArrayList<String>> tssparas = new ArrayList<>();
        int tssColumnIndex = -1;
        boolean rowHeadersFinished = false;
        
        for (int i = 0; i < filterResultsPaths.length; i++) {
            String filterReadPath = baseReadPath + filterResultsPaths[i] + "/";
            String filterSuffix = filterSuffixes[i];          
            
            for (String classifier : new String [] { "ED", "DTWCV", "RotF" }) { 
                String filteredClassifier = classifier + filterSuffix;
                
                tssColumnHeaders.add(filteredClassifier);
                tsstrainAccs.add(new ArrayList<>());
                tsstestAccs.add(new ArrayList<>());
                tssparas.add(new ArrayList<>());
                tssColumnIndex++;
                
                //columns = fold, rows = dataset
                OutFile clsfrTrainAccs = new OutFile(filterReadPath + filteredClassifier + "bestTrainACCS.csv");
                OutFile clsfrParas = new OutFile(filterReadPath + filteredClassifier + "parasSelected.csv");                
                
                clsfrTrainAccs.writeString("fold");
                clsfrParas.writeString("fold");
                for (int j = 0; j < numFolds; j++) {
                    clsfrTrainAccs.writeString("," + j);
                    clsfrParas.writeString("," + j);
                }
                clsfrTrainAccs.writeLine("");
                clsfrParas.writeLine("");
                
                for (int dset = 0; dset < numBaseDatasets; dset++) {
                    String baseDset = baseDatasets[dset];
                    
                    clsfrTrainAccs.writeString(baseDset);
                    clsfrParas.writeString(baseDset);
                    
                    for (int fold = 0; fold < numFolds; fold++) {
                        ClassifierResults crTrain = new ClassifierResults(filterReadPath + filteredClassifier + "/Predictions/" + baseDset + "/trainFold" + fold + ".csv");
                        ClassifierResults crTest = new ClassifierResults(filterReadPath + filteredClassifier + "/Predictions/" + baseDset + "/trainFold" + fold + ".csv");
                        
                        String selectedDataset ="";
                        try {
                            selectedDataset = parseDatasetChosenFromParasList(crTrain.getParas());
                        } catch (Exception ex) {
                            throw new Exception("Problem with: " + filterReadPath + filteredClassifier + "/Predictions/" + baseDset + "/trainFold" + fold + ".csv\n");
                        }
                         
                        String paras = parseFilterParasFromDatasetName(selectedDataset);
                        
                        clsfrTrainAccs.writeString(","+crTrain.acc);
                        clsfrTrainAccs.writeString(","+paras);
                        
                        tsstrainAccs.get(tssColumnIndex).add(crTrain.acc);
                        tsstestAccs.get(tssColumnIndex).add(crTest.acc);
                        tssparas.get(tssColumnIndex).add(paras);
                        if (!rowHeadersFinished)
                            tssRowHeaders.add(baseDset + "_" + fold);
                    }
                    
                    clsfrTrainAccs.writeLine("");
                    clsfrParas.writeLine("");
                }
                
                rowHeadersFinished = true; //first set of all dsets/folds done, have all the ehaders we need
                
                clsfrTrainAccs.closeFile();
                clsfrParas.closeFile();
            }
        }
        
        //texassharpshooterstyle columns = classifier, rows = dset/fold combo as list. group columns by filter method
        OutFile allTrainAccs = new OutFile(baseReadPath+"FilterTuned/AllFilters_BESTTRAINACCS_TSSStyle.csv");
        OutFile allTestAccs = new OutFile(baseReadPath+"FilterTuned/AllFilters_TESTACCS_TSSStyle.csv");
        OutFile allParas = new OutFile(baseReadPath+"FilterTuned/AllFilters_PARASSELECTED_TSSStyle.csv");
        
        allTrainAccs.writeString("dset_fold");
        allTestAccs.writeString("dset_fold");
        allParas.writeString("dset_fold");
        for (int i = 0; i < tssColumnHeaders.size(); i++) {
            allTrainAccs.writeString("," + tssColumnHeaders.get(i));
            allTestAccs.writeString("," + tssColumnHeaders.get(i));
            allParas.writeString("," + tssColumnHeaders.get(i));
        }
        allTrainAccs.writeLine("");
        allTestAccs.writeLine("");
        allParas.writeLine("");
        
        for (int i = 0; i < tssRowHeaders.size(); i++) {
            allTrainAccs.writeString(tssRowHeaders.get(i));
            allTestAccs.writeString(tssRowHeaders.get(i));
            allParas.writeString(tssRowHeaders.get(i));
            for (int j = 0; j < tssColumnHeaders.size(); j++) {
                allTrainAccs.writeString(","+tsstrainAccs.get(j).get(i));
                allTestAccs.writeString(","+tsstestAccs.get(j).get(i));
                allParas.writeString(","+tssparas.get(j).get(i));
            }
            allTrainAccs.writeLine("");
            allTestAccs.writeLine("");
            allParas.writeLine("");
        }
        allTrainAccs.closeFile();
        allTestAccs.closeFile();
        allParas.closeFile();
    }
    
    public static String parseDatasetChosenFromParasList(String parasLine) throws Exception {
        String[] parts = parasLine.split(",");
        
        for (int i = 0; i < parts.length; i++) {
            if (parts[i].contains("originalDataset"))
                return parts[i+1];
        }
        throw new Exception("Cant find originalDataset para");
    }
    
    /**
     * Takes dataset name, and produces a simple string representation of the paras for whatever filtered
     * version of the dataset (if there was any filtering at all)
     * 
     * If there was no filtering, returns "0", else in most cases where there is 1 para, will return 
     * that value as a reasonble string depending on its meaning in the context of the given filter, 
     * else if there is more than one para, will return as '_' delimited values
     * (',' and '/' will probs annoy excel)
     *
     * e.g for pca, the values are as a proportion, but because we don't want '.' in directory names, that will be added in here e.g "99" =>  "0.99"
     * e.g some cases may have a true string value, e.g "sqrt" or "log2", in those cases, simply those values are returned
     */
    public static String parseFilterParasFromDatasetName(String datasetName) { 
        
        String [] parts = null;
        
        //e.g CricketZ-PCA_95-  =>    PCA
        String[] temp =  datasetName.split("-");
        String filterTitle ="";
        
        if (temp.length == 1)
            if (datasetName.contains("Sieved"))
                filterTitle = "Sieved";
            else 
                return "0"; //no filtering
        else 
            filterTitle =temp[1].split("_")[0];
        
        if (filterTitle.equals("PCA")) {
            //e.g CricketZ-PCA_95-            =>          0.99
            parts = datasetName.split("_");
            return "0." + parts[1].replace("-", "");   
        } 
        else if (filterTitle.equals("EXP")) {
            //e.g CricketZ-EXP_50-             =>           50
            parts = datasetName.split("_");
            return parts[1].replace("-", "");            
        } 
        else if (filterTitle.equals("SG")) { 
            //BECAUSE DERIVATIVES ARE NOT CONSIDERED FOR THIS PAPER, ONLY RETURNING VALUES OF N AND M
            //e.g  CricketZ-SG_d0_n8_m65-       =>       8_65
            parts = datasetName.split("_");
            return parts[2].replace("n", "") + "_" + parts[3].replace("m", "").replace("-", "");
        } 
        else if (filterTitle.equals("MA")) {
            // e.g CricketZ-MA_11-        =>            11
            parts = datasetName.split("_");
            return parts[1].replace("-", "");
        } 
        else if (filterTitle.equals("DFT")) {
            //e.g CricketY-DFT_log2-           =>        log2
            //e.g CricketY-DFT_25-           =>        0.25
            
            if (datasetName.contains("log2"))
                return "log2";
            if (datasetName.contains("sqrt"))
                return "sqrt";
            
            parts = datasetName.split("_");
            return "0." + parts[1].replace("-", "");
        } 
        else if (filterTitle.equals("Sieved")) {
            //Acsf1_Sieved_1     =>     1
            return datasetName.split("_")[2];
        } 
        else if (filterTitle.equals("Gauss")) { 
            // e.g CricketZ-Gauss_11-        =>            11
            parts = datasetName.split("_");
            return parts[1].replace("-", "");
        } 
        else {
            System.out.println("Unrecognised filter");
            return "0"; //no filtering occured
        }       
    }
    
    
    
    /**
     * need to edit hesca to be able to handle resutls files in different locations
     */
    public static void whatIfWeJustEnsembleOverOptimisedFilters() throws Exception {
        String baseReadPath = "C:/JamesLPHD/TSC_Smoothing/Results/";
        String writePath = baseReadPath + "ENSEMBLES/";
        String[] baseDatasets = UCRDsetsNoPigs;
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 30;
        
        for (String classifier : new String [] { "ED", "DTWCV", "RotF" }) { 
            String[] filteredClassifiers = { 
                classifier + "",
                classifier + "_DFTFiltered", 
                classifier + "_EXPFiltered", 
                classifier + "_SGFiltered", 
                classifier + "_PCAFiltered", 
                classifier + "_MAFiltered",
                classifier + "_GFiltered",
            }; 
            String[] filterResultsPaths = { 
                baseReadPath + "TSC_Unfiltered/", 
                baseReadPath + "TSC_FFT_zeroed/",
                baseReadPath + "TSC_Exponential/", 
                baseReadPath + "TSC_SavitzkyGolay/", 
                baseReadPath + "TSCProblems_PCA_smoothed/", 
                baseReadPath + "TSCProblems_MovingAverage/", 
                baseReadPath + "TSCProblems_Gaussian/", 
            }; 

            for (int dset = 0; dset < numBaseDatasets; dset++) {
                String baseDset = baseDatasets[dset];

                Instances train = ClassifierTools.loadData("Z:/Data/TSCProblems/"+baseDset+"/"+baseDset+"_TRAIN");
                Instances test = ClassifierTools.loadData("Z:/Data/TSCProblems/"+baseDset+"/"+baseDset+"_TEST");
                
                for (int fold = 0; fold < numFolds; fold++) {
//                    ClassifierResults crTrain = new ClassifierResults(filterReadPath + filteredClassifier + "/Predictions/" + baseDset + "/trainFold" + fold + ".csv");
 
                    Instances[] data = InstanceTools.resampleTrainAndTestInstances(train, test, fold);

                    CAWPE hesca = new CAWPE();
                    hesca.setEnsembleIdentifier(classifier + "_" + filteredClassifiers.length + "FilterEnsemble");
                    hesca.setBuildIndividualsFromResultsFiles(true);
                    hesca.setClassifiers(null, filteredClassifiers, null);
                    hesca.setResultsFileLocationParameters(filterResultsPaths, baseDset, fold);
                    hesca.setResultsFileWritingLocation(writePath);
                    hesca.setRandSeed(fold);
                    hesca.setPerformCV(true);
                    
//                    hesca.setWeightingScheme(weight);
//                    hesca.setVotingScheme(votingScheme);
                    
                    hesca.buildClassifier(data[0]); //can pass null?
                    
                    //look back at ensembletests or something
                    for (int i = 0; i < data[1].numInstances(); i++)
                        hesca.classifyInstance(data[1].instance(i));
                    hesca.writeEnsembleTrainTestFiles(data[1].attributeToDoubleArray(data[1].classIndex()), true);
                }

            }

        }
    }
}
