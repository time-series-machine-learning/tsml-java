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
package weka_extras.classifiers.ensembles;

import evaluation.ClassifierResultsAnalysis;
import evaluation.PerformanceMetric;
import experiments.Experiments;
import java.io.File;
import java.util.Scanner;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import utilities.ClassifierTools;
import utilities.FileHandlingTools;

/**
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPETest {
    
    public CAWPETest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
        System.out.println("-----Start CAWPE tests-----");
    }
    
    @AfterClass
    public static void tearDownClass() {
        System.out.println("-----End CAWPE tests-----\n");
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
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
    @Test
    public void testBuildCAWPEPaper_AllResultsForFigure3() throws Exception {
        System.out.println("--buildCAWPEPaper_AllResultsForFigure3()");
        
        CAWPE.buildCAWPEPaper_AllResultsForFigure3("");
        
        File f = new File("Analysis/UCICAWPEvsHeteroEnsembles_BasicClassifiers/UCICAWPEvsHeteroEnsembles_BasicClassifiers_BIGglobalSummary.csv");
        assertTrue(f.exists()); 
        
        //read in summary for later comparison
        Scanner scan = new Scanner(f);
        StringBuilder sb = new StringBuilder();
        while (scan.hasNext()) {
            String t = scan.nextLine();
            if (t.contains("AvgPredTimes:"))
                break;
            sb.append(t).append("\n");
        }
        scan.close();
        
        //confirm folder structure all there
        assertTrue(new File("Analysis/UCICAWPEvsHeteroEnsembles_BasicClassifiers/Timings/TRAIN/TRAINTrainTimes_SUMMARY.csv").exists());
        assertTrue(new File("Analysis/UCICAWPEvsHeteroEnsembles_BasicClassifiers/Timings/TEST/TESTAvgPredTimes_SUMMARY.csv").exists());
        for (String set : new String[] { ClassifierResultsAnalysis.trainLabel, ClassifierResultsAnalysis.testLabel, ClassifierResultsAnalysis.trainTestDiffLabel }) {
            for (PerformanceMetric metric : PerformanceMetric.getDefaultStatistics()) {
                String name = metric.name;
                assertTrue(new File("Analysis/UCICAWPEvsHeteroEnsembles_BasicClassifiers/"+name+"/"+set+"/"+set+name+"_SUMMARY.csv").exists());
            }
        }
        
        //clean up the generated files
        FileHandlingTools.recursiveDelete("Analysis/");
        FileHandlingTools.recursiveDelete("Results/");
        assertTrue(!new File("Analysis").exists());
        assertTrue(!new File("Results").exists());
        
        //confirm summary of results are the same (implying individual base classifier and ensemble results for folds are correct
        //ingores timings, as no realistic way to make those equivalent
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

//        System.out.println(expectedBigGlobalSummary.trim());
//        System.out.println("\n\n\n\n");
//        System.out.println(sb.toString().trim());
        
        assertEquals(sb.toString().trim(), expectedBigGlobalSummary.trim());
    }

    /**
     * Test of simple results reproduction, of class CAWPE.
     */
    @Test
    public void testReproduction() throws Exception {
        System.out.println("--testReproduction()");
        double expectedAcc = 0.9650145772594753;
        System.out.println("CAWPE expected accuracy generated 2019_09_25");
        assertTrue(ClassifierTools.testUtils_confirmIPDReproduction(new CAWPE(), expectedAcc));
    }
    
}
