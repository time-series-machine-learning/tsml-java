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

package experiments;

import evaluation.storage.ClassifierResults;
import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.junit.Test;
import utilities.ClassifierTools;
import utilities.FileHandlingTools;
import weka.classifiers.Classifier;
import weka.core.Randomizable;
import machine_learning.classifiers.ensembles.CAWPE;

/**
 *
 * Tests to compare test accuracies for important classifier on a quick italy power 
 * demand run to saved expected results, and to recreate results/analysis for a cawpe paper section.
 * 
 * Just confirms that old results are still reproducible
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class BasicReproductionTests {

    public static final int defaultSeed = 0;
    
    public static boolean failTestsOnTimingsDifference = false;
    public static double timingEqualityThreshold = 1.2;
    
    public static String reproductionDirectory = "src/main/java/experiments/reproductions/";
    
    static { 
        new File(reproductionDirectory).mkdirs();
    }
    
    private static final String tsClassifiers = "tsml.classifiers.";
    private static final String extraClassifiers = "machine_learning.classifiers.";
    
    public static final String[] classifierPaths = {
        
        tsClassifiers + "dictionary_based.BagOfPatternsClassifier",
        tsClassifiers + "dictionary_based.SAXVSM",
        tsClassifiers + "dictionary_based.WEASEL",
        tsClassifiers + "dictionary_based.cBOSS",
        tsClassifiers + "dictionary_based.TDE",
       
        tsClassifiers + "distance_based.DTWCV",
        tsClassifiers + "distance_based.ProximityForestWrapper",
        tsClassifiers + "distance_based.SlowDTW_1NN",
        
//        tsClassifiers + "frequency_based.cRISE",
//        tsClassifiers + "hybrids.HIVE_COTE",

//        tsClassifiers + "hybrids.FlatCote", 
//        tsClassifiers + "hybrids.HiveCote", //assumed to cover its consituents
        
        tsClassifiers + "interval_based.LPS",
        tsClassifiers + "interval_based.TSF",
        tsClassifiers + "interval_based.CIF",
        
        tsClassifiers + "shapelet_based.FastShapelets",
        tsClassifiers + "shapelet_based.LearnShapelets",        
        
        extraClassifiers + "PLSNominalClassifier",
        extraClassifiers + "kNN",
        
        extraClassifiers + "ensembles.CAWPE",
        extraClassifiers + "ensembles.stackers.SMLR",
        
    };
    
    ////////////////////////
    // ClassifierResults files will store the prob distributions to 6 decimal places
    // by default, while recreted results in memory will have arbitrary precision
    // Compare doubles based of the probability dists with these funcs
    //
    // Affects: prediction dists themselves, NLL
    // Ignores: ACC, AUROC, BALACC, these should be the same either way, despite being doubles.
    // possible mega edge case with AUROC where the higher precision resolves a tie differently
    // in the ordering of predictions. If this is a case, need to just blanket round to 6 places
    // before finding the values of stats
    public static final double eps = 10e-6;
    public static boolean doubleEqual(double v1, double v2) { 
        return Math.abs(v1 - v2) < eps;
    }
    public static boolean doubleArrayEquals(double[] a1, double[] a2) {
        for (int i = 0; i < a1.length; i++)
            if (!doubleEqual(a1[i], a2[i]))
                return false;
        return true;
    }
    
    public static class ExpectedClassifierResults { 
        
        public String simpleClassifierName; //simple unconditioned class name
        public String fullClassifierName; //includes package paths for construction
        public Classifier classifier = null; 
        public ClassifierResults results;
        public String dateTime;
        public long time;
        
        public ExpectedClassifierResults(File resFile) throws Exception { 
            simpleClassifierName = resFile.getName().split("\\.")[0]; //without any filetype extensions
            results = new ClassifierResults(resFile.getAbsolutePath());
            
            String[] p = results.getParas().split(",");
            fullClassifierName = p[0].trim();
            dateTime = p[1].trim();
            time = Long.parseLong(p[2].trim());
        }
        
        public ExpectedClassifierResults(String fullClassName) throws Exception { 
            fullClassifierName = fullClassName;
            
            String[] t = fullClassName.split("\\.");
            simpleClassifierName = t[t.length-1]; 
        }
        
        
        public void save(String directory) throws Exception {
            directory.replace("\\", "/");
            if (!directory.endsWith("/"))
                directory+="/";
            
            Date date = new Date();
            SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            
            dateTime = formatter.format(date);
            time = System.currentTimeMillis();
            
            results.setDescription("Generated by BasicReproductionTests at " + dateTime);
            
            //saving date in a couple formats for future-proofing
            results.setParas(fullClassifierName + ", " + dateTime + ", " + time); 
            results.writeFullResultsToFile(directory + simpleClassifierName + ".csv");
        }
        
        /**
         * todo should obviously go into classifierresults itself, and be spread around 
         * the sub-modules of it (predictions, etc) one that class is split up
         * 
         * Does not compare every tiny thing (for now), i.e. does not test for equality 
         * of characters in the constructed file, for example. Timings are very unlikely to 
         * be exact, meta info on the first line might change over time but that's not 
         * the target of these tests, etc. 
         * 
         * Tests for equality of summary performance metrics, equality of the 
         * last five predictions (prob distributions), and (to be tested on stability) 
         * pseudo-equality on the prediction times of the last 5 predictions and the 
         * build times.
         * 
         * Pseudo equality is defined as being within some proportional threshold of the 
         * expected value (timingEqualityThreshold, default 2). Because we are using the 
         * last five predictions, the JVM should have sorted out it's caching issues 
         * and allocated space with can make the timings of the first few instances of any
         * particular operation far larger than normal 
         */
        public boolean equal(ClassifierResults newResults) throws Exception { 
            
            newResults = ClassifierResults.util_roundAllPredictionDistsToDefaultPlaces(newResults);
            
            results.findAllStatsOnce();
            newResults.findAllStatsOnce();
            
            boolean res = true;
            
            ///////////////// SUMMARY PERFORMANCE METRICS
            if (results.getAcc() != newResults.getAcc()) {
                System.out.println("ACCURACY DIFFERS, exp="+results.getAcc()+" new="+newResults.getAcc());
                res = false;
            }
            if (results.balancedAcc!= newResults.balancedAcc) {
                System.out.println("BALANCED ACCURACY DIFFERS, exp="+results.balancedAcc+" new="+newResults.balancedAcc);
                res = false;
            }
            if (results.meanAUROC != newResults.meanAUROC) {
                System.out.println("AUROC DIFFERS, exp="+results.meanAUROC+" new="+newResults.meanAUROC);
                res = false;
            }
            if (!doubleEqual(results.nll, newResults.nll)) { //see comment at doubleEqual
                System.out.println("NLL DIFFERS, exp="+results.nll+" new="+newResults.nll);
                res = false;
            }
            
            ///////////////// BUILD TIMES 
            //assuming sub-millisecond timings are unreliable anyway
            long t1 = TimeUnit.MILLISECONDS.convert(results.getBuildTimeInNanos(), TimeUnit.NANOSECONDS); 
            long t2 = TimeUnit.MILLISECONDS.convert(newResults.getBuildTimeInNanos(), TimeUnit.NANOSECONDS);
            if (t1*timingEqualityThreshold < t2 || t1/timingEqualityThreshold > t2) {
                if (failTestsOnTimingsDifference) {
                    System.out.println("BUILD TIME OUTSIDE THRESHOLD, exp="+t1+" new="+t2);
                    res = false;
                }
            }
            
            ///////////////// FINAL FIVE PREDICTIONS
            for (int i = 0; i < 5; i++) {
                double[] expDist = results.getProbabilityDistribution(results.numInstances()-1-i);
                double[] newDist = newResults.getProbabilityDistribution(newResults.numInstances()-1-i);
                
                if (!doubleArrayEquals(expDist, newDist)) {
                    System.out.println("PREDICTION DIST 'NUMINSTS-"+i+"' DIFFERS, exp="+Arrays.toString(expDist)+" new="+Arrays.toString(newDist));
                    res = false;
                }
                
                long tt1 = results.getPredictionTimeInNanos(results.numInstances()-1-i);
                long tt2 = newResults.getPredictionTimeInNanos(newResults.numInstances()-1-i);
                t1 = TimeUnit.MILLISECONDS.convert(tt1, TimeUnit.NANOSECONDS);
                t2 = TimeUnit.MILLISECONDS.convert(tt2, TimeUnit.NANOSECONDS);
                
                if (t1*timingEqualityThreshold < t2 || t1/timingEqualityThreshold > t2) {   
                    if (failTestsOnTimingsDifference) {
                        System.out.println("PREDICTION TIME NUMINSTS-"+i+" OUTSIDE THRESHOLD, exp="+t1+" new="+t2);
                        res = false;
                    }
                }
            }
            
            return res;
        }
    }
    
    public static Classifier constructClassifier(String fullClassifierName) { 
        Classifier inst = null;

        try {
            Class c = Class.forName(fullClassifierName);
            inst = (Classifier) c.newInstance();

            if (inst instanceof Randomizable)
                ((Randomizable)inst).setSeed(defaultSeed);
            else {
                Method[] ms = c.getMethods();
                for (Method m : ms) {
                    if (m.getName().equals("setSeed") || m.getName().equals("setRandSeed")) {
                        m.invoke(inst, defaultSeed);
                        break;
                    }
                }
            }

        } catch (ClassNotFoundException ex) {
            Logger.getLogger(BasicReproductionTests.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            Logger.getLogger(BasicReproductionTests.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            Logger.getLogger(BasicReproductionTests.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IllegalArgumentException ex) {
            Logger.getLogger(BasicReproductionTests.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InvocationTargetException ex) {
            Logger.getLogger(BasicReproductionTests.class.getName()).log(Level.SEVERE, null, ex);
        }

        return inst;
    }

    public static void generateMissingExpectedResults() throws Exception {
        List<String> failedClassifiers = new ArrayList<>();
        
        List<String> existingFiles = Arrays.asList((new File(reproductionDirectory)).list());
        
        for (String classifierPath : classifierPaths) {
            String[] t = classifierPath.split("\\.");
            String simpleClassifierName = t[t.length-1]; 
            
            boolean exists = false;
            for (String existingFile : existingFiles) {
                if (simpleClassifierName.equals(existingFile.split("\\.")[0])) {
                    exists = true;
                    break;
                }
            }
            if (exists) 
                continue;
            else {
                System.out.println("Attempting to generate missing result for " + simpleClassifierName);
            }
            
            if (!generateExpectedResult(classifierPath))
                failedClassifiers.add(simpleClassifierName);
        }
        
        System.out.println("\n\n\n");
        System.out.println("Failing classifiers = " + failedClassifiers);
    }
    
    public static void generateAllExpectedResults() throws Exception {
        List<String> failedClassifiers = new ArrayList<>();
        
        for (String classifierPath : classifierPaths) {
            String[] t = classifierPath.split("\\.");
            String simpleClassifierName = t[t.length-1]; 
            
            if (!generateExpectedResult(classifierPath))
                failedClassifiers.add(simpleClassifierName);
        }
        
        System.out.println("\n\n\n");
        System.out.println("Failing classifiers = " + failedClassifiers);
    }
    
    public static boolean generateExpectedResult(String classifierPath) throws Exception {
        ExpectedClassifierResults expres = new ExpectedClassifierResults(classifierPath);
        
        boolean worked = true;
        
        try { 
            expres.classifier = constructClassifier(classifierPath);
        } catch (Exception e) { 
            System.err.println(expres.simpleClassifierName + " construction FAILED");
            System.err.println(e);
            e.printStackTrace();
            worked = false;
        }
        
        try {
            expres.results = ClassifierTools.testUtils_evalOnIPD(expres.classifier);
        } catch (Exception e) { 
            System.err.println(expres.simpleClassifierName + " evaluation on ItalyPowerDemand FAILED");
            System.err.println(e);
            e.printStackTrace();
            worked = false;
        }

        if (worked) {
            expres.save(reproductionDirectory);
            System.err.println(expres.simpleClassifierName + " evaluated and saved SUCCESFULLY, IPD acc = " + expres.results.getAcc());
        }
        
        return worked;
    }
   
    public static boolean confirmAllExpectedResultReproductions() throws Exception {
        System.out.println("--confirmAllExpectedResultReproductions()");

        File[] expectedResults = FileHandlingTools.listFiles(reproductionDirectory);
        if (expectedResults == null) 
            throw new Exception("No expected results saved to compare to, dir="+reproductionDirectory);
        
        List<String> failedClassifiers = new ArrayList<>();
        
        for (File expectedResultFile : expectedResults) {
            ExpectedClassifierResults expres = new ExpectedClassifierResults(expectedResultFile);
            
            Classifier c = constructClassifier(expres.fullClassifierName);
            ClassifierResults newres = ClassifierTools.testUtils_evalOnIPD(c);
            
            if (expres.equal(newres))
                System.out.println("\t" + expres.simpleClassifierName + " all good, parity with results created " + expres.dateTime);
            else {
                System.out.println("\t" + expres.simpleClassifierName + " was NOT recreated successfully! no parity with results created " + expres.dateTime);
                failedClassifiers.add(expres.simpleClassifierName);
            }
        }
        
        if (failedClassifiers.size() > 0) {
            System.out.println("\n\n\n");
            System.out.println("Failing classifiers = " + failedClassifiers);
            return false;
        }
        return true;
    }
    
    /**
     * Test of buildCAWPEPaper_AllResultsForFigure3 method, of class CAWPE.
     * 
     * Larger scale test (~19 secs locally), one that @jamesl used often before formulating into unit test
     * 
     * Implicitly provides tests for the 
     *      -cross validation evaluator
     *      -multiple classifier evaluation pipeline
     *      -basic experiments setup with soem built-in weka classifiers
     *      -slightly more bespoke ensemble experiments setup
     *      -datasets resampling
     */
    public static boolean testBuildCAWPEPaper_AllResultsForFigure3() throws Exception {
        System.out.println("--buildCAWPEPaper_AllResultsForFigure3()");
        
        Experiments.beQuiet = true;
        CAWPE.buildCAWPEPaper_AllResultsForFigure3("");
        
        File f = new File("Analysis/UCICAWPEvsHeteroEnsembles_BasicClassifiers/UCICAWPEvsHeteroEnsembles_BasicClassifiers_BIGglobalSummary.csv");
//        assertTrue(f.exists()); 
        
        //read in summary for later comparison
        Scanner scan = new Scanner(f);
        StringBuilder sb = new StringBuilder();
        while (scan.hasNext()) {
            String t = scan.nextLine();
            if (t.contains("AvgPredTimesBenchmarked:"))
                break;
            sb.append(t).append("\n");
        }
        scan.close();
        
        //confirm folder structure all there
//        assertTrue(new File("Analysis/UCICAWPEvsHeteroEnsembles_BasicClassifiers/Timings/TRAIN/TRAINTrainTimes_SUMMARY.csv").exists());
//        assertTrue(new File("Analysis/UCICAWPEvsHeteroEnsembles_BasicClassifiers/Timings/TEST/TESTAvgPredTimes_SUMMARY.csv").exists());
//        for (String set : new String[] { ClassifierResultsAnalysis.trainLabel, ClassifierResultsAnalysis.testLabel, ClassifierResultsAnalysis.trainTestDiffLabel }) {
//            for (PerformanceMetric metric : PerformanceMetric.getDefaultStatistics()) {
//                String name = metric.name;
//                assertTrue(new File("Analysis/UCICAWPEvsHeteroEnsembles_BasicClassifiers/"+name+"/"+set+"/"+set+name+"_SUMMARY.csv").exists());
//            }
//        }
        
        //clean up the generated files
        FileHandlingTools.recursiveDelete("Analysis/");
        FileHandlingTools.recursiveDelete("Results/");
//        assertTrue(!new File("Analysis").exists());
//        assertTrue(!new File("Results").exists());
        
        //confirm summary of results are the same (implying individual base classifier and ensemble results for folds are correct)
        //ignores timings, as no realistic way to make those equivalent
        
        /*
        String[] dataHeaders = { "UCI", };
        String[] dataPaths = { "src/main/java/experiments/data/uci/" };
        String[][] datasets = { { "hayes-roth", "iris", "teaching" } };
        String writePathResults =  writePathBase + "Results/";
        String writePathAnalysis =  writePathBase + "Analysis/";
        int numFolds = 3;
        */

        /*
        this was the expected output prior to setClassifier updates 2019_10_16, which revealed that mlp was not beingc
        correctly seeded (defaulted to 0).

        String expectedBigGlobalSummary = 
                "ACC:TESTACC,CAWPE,NBC,WMV,RC,MV,ES,SMLR,SMM5,PB,SMLRE\n" +
                "AvgTESTACCOverDsets:,0.7285445094217025,0.7318294707963324,0.7172998339470075,0.7145563497220419,0.6885834957764781,0.7117738791423003,0.7191336365605373,0.7116821890116237,0.6928972637354703,0.7012410656270306\n" +
                "AvgTESTACCRankOverDsets:,3.1666666666666665,3.1666666666666665,5.0,5.333333333333333,5.833333333333333,6.0,6.333333333333333,6.333333333333333,6.833333333333333,7.0\n" +
                "StddevOfTESTACCOverDsets:,0.22008900216444294,0.20946693560070861,0.22754669947650313,0.2282133513717843,0.2383855215392628,0.2202751366996902,0.22341886857881665,0.24839719907549082,0.25329683821988797,0.2409493775423368\n" +
                "AvgOfStddevsOfTESTACCOverDsetFolds:,0.03279891126799218,0.024754101494977986,0.04849787496303352,0.046816397436865304,0.04897749514576629,0.047987230632012545,0.024195798008593224,0.03431576869313286,0.0370264191912524,0.03595685865637037\n" +
                "StddevsOfTESTACCRanksOverDsets:,1.0408329997330663,2.0207259421636903,1.3228756555322954,1.755942292142123,3.6170890690351176,2.6457513110645907,3.0550504633038935,3.7859388972001824,2.9297326385411573,5.196152422706632\n" +
                "\n" +
                "BALACC:TESTBALACC,CAWPE,NBC,WMV,RC,MV,ES,SMLR,SMM5,PB,SMLRE\n" +
                "AvgTESTBALACCOverDsets:,0.7322852456185789,0.7398326210826212,0.7129216308382974,0.7106769619269618,0.6683040878874212,0.7025049641716308,0.7285234826901493,0.7235676206509541,0.6870069282569283,0.712745834412501\n" +
                "AvgTESTBALACCRankOverDsets:,3.1666666666666665,3.1666666666666665,5.0,5.333333333333333,5.833333333333333,6.333333333333333,6.333333333333333,6.333333333333333,6.5,7.0\n" +
                "StddevOfTESTBALACCOverDsets:,0.21979813009604562,0.2075778061006703,0.22885704565704787,0.22951237016096734,0.25279977059148956,0.2238033387573712,0.22506552355121331,0.25286563032293463,0.25503069275904094,0.2448594862094281\n" +
                "AvgOfStddevsOfTESTBALACCOverDsetFolds:,0.03455598374923909,0.024060473815657058,0.05177301734816558,0.050439169878735056,0.05127020812811125,0.05645152922577826,0.01776687586864427,0.03134730161003507,0.03765792057591982,0.030328734418572856\n" +
                "StddevsOfTESTBALACCRanksOverDsets:,1.0408329997330663,2.0207259421636903,1.3228756555322954,1.755942292142123,3.6170890690351176,3.055050463303893,3.0550504633038935,3.7859388972001824,2.598076211353316,5.196152422706632\n" +
                "\n" +
                "AUROC:TESTAUROC,CAWPE,SMM5,PB,SMLR,WMV,NBC,RC,MV,ES,SMLRE\n" +
                "AvgTESTAUROCOverDsets:,0.8500076854235693,0.8156226498687755,0.8214985405823803,0.8176827886374668,0.8228581929729067,0.819681196812097,0.8175782745531306,0.8108473566884643,0.8099753212757199,0.797600009436251\n" +
                "AvgTESTAUROCRankOverDsets:,1.3333333333333333,4.666666666666667,4.666666666666667,5.0,5.333333333333333,5.333333333333333,6.0,7.0,7.333333333333333,8.333333333333334\n" +
                "StddevOfTESTAUROCOverDsets:,0.152560502120108,0.18855774768880665,0.19229971562576004,0.18018940087531557,0.14586931174236345,0.16509673380553797,0.1491787090622888,0.15124348237259566,0.1528537298969881,0.1770574668705955\n" +
                "AvgOfStddevsOfTESTAUROCOverDsetFolds:,0.030274719407621924,0.04775468866122167,0.02190443701428843,0.03848402717616285,0.02989273753439181,0.02752977352082156,0.030072602448817185,0.029162226381979423,0.03277007476456647,0.04972644635842636\n" +
                "StddevsOfTESTAUROCRanksOverDsets:,0.5773502691896257,3.055050463303893,3.7859388972001824,2.0,3.7859388972001824,1.1547005383792517,2.0,3.605551275463989,2.0816659994661326,2.8867513459481287\n" +
                "\n" +
                "NLL:TESTNLL,CAWPE,SMM5,WMV,SMLR,NBC,MV,RC,SMLRE,PB,ES\n" +
                "AvgTESTNLLOverDsets:,0.8740473258902913,1.1672216448053325,1.2588112791030788,1.1748228744879337,1.2594131707464087,1.293933192179383,1.2679074849500689,1.2864376793720262,1.592847724655895,1.3038288622797296\n" +
                "AvgTESTNLLRankOverDsets:,1.0,4.666666666666667,5.333333333333333,5.666666666666667,5.666666666666667,6.0,6.0,6.666666666666667,6.666666666666667,7.333333333333333\n" +
                "StddevOfTESTNLLOverDsets:,0.691694787015834,0.9796905490544456,1.0223841041530592,0.9870722431231613,0.7746460079084273,1.0258722030842515,1.0360622981608925,1.0024121206377643,1.5111772632891516,1.0222936474880255\n" +
                "AvgOfStddevsOfTESTNLLOverDsetFolds:,0.13100093711191763,0.22899161342369623,0.1798812801999756,0.1874603882410609,0.20352196751423426,0.18541729066158644,0.16955322281243648,0.16941142924146801,0.2786626997782997,0.2000278240505596\n" +
                "StddevsOfTESTNLLRanksOverDsets:,0.0,2.0816659994661326,0.5773502691896258,2.516611478423583,4.041451884327381,3.4641016151377544,2.6457513110645907,4.041451884327381,4.163331998932266,1.5275252316519465";
        */

        String expectedBigGlobalSummary = "ACC:TESTACC,WMV,NBC,CAWPE,RC,MV,SMM5,ES,SMLR,SMLRE,PB\n" +
                "AvgTESTACCOverDsets:,0.7274629990614395,0.7153880586239261,0.7314684860298896,0.7217955382282869,0.6974651649700383,0.719172622915313,0.7124056024835751,0.7031225182297307,0.7075792361562342,0.6928972637354703\n" +
                "AvgTESTACCRankOverDsets:,4.0,4.333333333333333,4.666666666666667,4.833333333333333,5.166666666666667,5.333333333333333,5.5,7.0,7.0,7.166666666666667\n" +
                "StddevOfTESTACCOverDsets:,0.21965069702486056,0.22075061233906448,0.21599168566182694,0.22433813402114516,0.23250546652710627,0.23224263867524703,0.21152524227063668,0.24148559934127398,0.22786315741913124,0.25329683821988797\n" +
                "AvgOfStddevsOfTESTACCOverDsetFolds:,0.05097187092275754,0.03297117847198802,0.03642567280598806,0.043526350912095536,0.03818281787490734,0.03357343280121439,0.03932505708576283,0.03895688145388599,0.02497874944688039,0.0370264191912524\n" +
                "StddevsOfTESTACCRanksOverDsets:,1.3228756555322954,3.6170890690351176,0.7637626158259734,2.0207259421636903,4.193248541803041,3.7859388972001824,3.968626966596886,4.358898943540674,3.605551275463989,1.755942292142123\n" +
                "\n" +
                "BALACC:TESTBALACC,NBC,WMV,CAWPE,RC,MV,SMM5,ES,SMLRE,SMLR,PB\n" +
                "AvgTESTBALACCOverDsets:,0.7241418458085125,0.7228188940688942,0.7351912285245619,0.7176682422515755,0.6773329448329449,0.7306936890270225,0.7013012604679271,0.7188453984287317,0.7126520547353881,0.6870069282569283\n" +
                "AvgTESTBALACCRankOverDsets:,3.8333333333333335,4.0,4.666666666666667,4.833333333333333,5.166666666666667,5.333333333333333,5.666666666666667,6.666666666666667,7.333333333333333,7.5\n" +
                "StddevOfTESTBALACCOverDsets:,0.2167439508778403,0.2212685239000405,0.21563344429828277,0.22580992139595155,0.24870379259009287,0.23484868038157847,0.21817072502418683,0.22983957642097774,0.24379996657723263,0.25503069275904094\n" +
                "AvgOfStddevsOfTESTBALACCOverDsetFolds:,0.031271297432199995,0.05477493885465065,0.03814232820465679,0.04785776036381459,0.04280967408056264,0.0287796820308799,0.048019837791095754,0.029627808499106525,0.03138832405014855,0.03765792057591982\n" +
                "StddevsOfTESTBALACCRanksOverDsets:,2.753785273643051,1.3228756555322954,0.7637626158259734,2.0207259421636903,4.193248541803041,3.7859388972001824,4.163331998932266,4.163331998932266,3.7859388972001824,1.8027756377319946\n" +
                "\n" +
                "AUROC:TESTAUROC,CAWPE,SMM5,PB,WMV,NBC,RC,SMLR,ES,MV,SMLRE\n" +
                "AvgTESTAUROCOverDsets:,0.849770319016462,0.8229810950769959,0.8214985405823803,0.8213685981068424,0.8149374537219297,0.8173813462624793,0.8114881567997374,0.8111879235089292,0.810024282569528,0.7902312289775987\n" +
                "AvgTESTAUROCRankOverDsets:,2.3333333333333335,4.0,4.333333333333333,5.333333333333333,5.333333333333333,6.0,6.333333333333333,6.666666666666667,6.666666666666667,8.0\n" +
                "StddevOfTESTAUROCOverDsets:,0.15503685228831474,0.19001901503910637,0.19229971562576004,0.14654126643776394,0.1658309636107022,0.14875742473472578,0.18600285896473676,0.1510921761373448,0.1514632434016978,0.1726614810233521\n" +
                "AvgOfStddevsOfTESTAUROCOverDsetFolds:,0.02537300115566142,0.02220030849074056,0.02190443701428843,0.030191445372247588,0.02163090653520087,0.03171927947721526,0.04126691345011047,0.03380834704366238,0.027866043551935855,0.03213563345632965\n" +
                "StddevsOfTESTAUROCRanksOverDsets:,2.3094010767585034,2.6457513110645907,3.214550253664318,3.7859388972001824,1.1547005383792517,2.0,3.214550253664318,3.214550253664318,4.163331998932266,2.6457513110645907\n" +
                "\n" +
                "NLL:TESTNLL,CAWPE,RC,WMV,SMM5,MV,NBC,SMLR,ES,SMLRE,PB\n" +
                "AvgTESTNLLOverDsets:,0.8844348810641535,1.2620831740276812,1.2542313397786657,1.1878670618373748,1.2883865067790932,1.228698888158254,1.1797077162509737,1.2904134315630473,1.3122877664502341,1.592847724655895\n" +
                "AvgTESTNLLRankOverDsets:,1.3333333333333333,4.666666666666667,5.0,5.333333333333333,5.333333333333333,5.666666666666667,6.0,6.666666666666667,7.333333333333333,7.666666666666667\n" +
                "StddevOfTESTNLLOverDsets:,0.6905190819957883,1.0277175319674012,1.014957453111849,1.1166436736959424,1.0192580874883377,0.7436645726101814,0.9900691677923797,1.0091099400693595,1.0235332594700968,1.5111772632891516\n" +
                "AvgOfStddevsOfTESTNLLOverDsetFolds:,0.13267895502027546,0.15689429192128332,0.16443411413096812,0.1588387064257464,0.17331314838440773,0.20424033600313699,0.19044258606104228,0.1860761289798324,0.09880785325499852,0.2786626997782997\n" +
                "StddevsOfTESTNLLRanksOverDsets:,0.5773502691896257,2.309401076758503,1.7320508075688772,3.7859388972001824,4.041451884327381,4.041451884327381,2.6457513110645907,2.0816659994661326,2.8867513459481287,2.516611478423583";

//        assertEquals(sb.toString().trim(), expectedBigGlobalSummary.trim());
        boolean res = sb.toString().trim().equals(expectedBigGlobalSummary.trim());
        
        if (!res) {
            System.out.println("CAWPE not recreated sucessfully, expected: ");
            System.out.println(expectedBigGlobalSummary.trim());
            System.out.println("\n\n\n\n");
            System.out.println("Made just now: ");
            System.out.println(sb.toString().trim());
        }
        
        return res;
    }

    @Test
    public void test() throws Exception {
        main(new String[0]);
    }

    public static void main(String[] args) throws Exception {
//        generateAllExpectedResults();
//        generateMissingExpectedResults();

        boolean classifiersComplete = confirmAllExpectedResultReproductions();
        boolean analysisReproduced = testBuildCAWPEPaper_AllResultsForFigure3();

        if (!classifiersComplete) {
            System.out.println("Classifiers simple eval recreation failed!");
        }

        if (!analysisReproduced) {
            System.out.println("CAWPE analysis recreation failed!");
        }
        
        if (!classifiersComplete || !analysisReproduced) {
            System.out.println("\n\n*********************Integration tests failed");
            System.exit(1); //fail
        }
        
        System.out.println("\n\n*********************All tests passed");
    }
}
