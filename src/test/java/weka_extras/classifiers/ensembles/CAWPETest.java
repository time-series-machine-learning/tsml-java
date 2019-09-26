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
                "ACC:TESTACC,CAWPE,NBC,WMV,MV,ES,SMLR,SMM5,RC,SMLRE,PB\n" +
                "AvgTESTACCOverDsets:,0.7663544891640865,0.7711099644536177,0.7539836792416772,0.7385921912624699,0.7525784504835072,0.7634726522187822,0.7373559033749952,0.7503846080342468,0.7450985169896419,0.7403221152008561\n" +
                "AvgTESTACCRankOverDsets:,3.75,4.125,4.375,4.5,4.625,6.125,6.5,6.5,7.25,7.25\n" +
                "StddevOfTESTACCOverDsets:,0.193139365129792,0.18649616456702356,0.2070760504462057,0.21763622011893996,0.20678817446685405,0.19968553338970385,0.17290145810134894,0.20639657828372002,0.2235732695347628,0.21110249148928095\n" +
                "AvgOfStddevsOfTESTACCOverDsetFolds:,0.03469526855694392,0.02826878976052585,0.03544759737613996,0.03457295250999298,0.0343811917661837,0.02538455985552823,0.09620283435803087,0.032235484880143085,0.03209326175055343,0.03942863082248407\n" +
                "StddevsOfTESTACCRanksOverDsets:,1.707825127659933,2.5289984842489197,2.7195281453467866,3.763863263545405,2.868652413009751,2.839454172900137,3.1091263510296048,2.273030282830976,4.272001872658765,2.0615528128088303\n" +
                "\n" +
                "BALACC:TESTBALACC,NBC,SMLR,CAWPE,WMV,ES,SMLRE,SMM5,MV,RC,PB\n" +
                "AvgTESTBALACCOverDsets:,0.717166490916491,0.7156700382950383,0.7050219109594109,0.689676356051356,0.68843083999334,0.6973141950641951,0.6893777333777334,0.6654317950567952,0.6753395054020055,0.6630781417656417\n" +
                "AvgTESTBALACCRankOverDsets:,3.125,4.5,5.0,5.0,5.5,5.5,5.5,5.5,6.375,9.0\n" +
                "StddevOfTESTBALACCOverDsets:,0.1974623795575887,0.20524434113700077,0.20277251946977584,0.20934503519172737,0.21111773833454342,0.23027922173187224,0.16343806624156665,0.21816184723542106,0.22049764602732044,0.22051340783053142\n" +
                "AvgOfStddevsOfTESTBALACCOverDsetFolds:,0.04173090827869327,0.044967363077189675,0.0542279194555557,0.05230817961646981,0.054318944790195556,0.04412477775148784,0.12820171003145453,0.05570792078274066,0.0456684183731932,0.04566526880952991\n" +
                "StddevsOfTESTBALACCRanksOverDsets:,1.4361406616345072,2.886751345948129,2.160246899469287,1.5811388300841898,3.3166247903554,4.203173404306164,4.203173404306164,3.1885210782848317,2.688710967483613,0.0\n" +
                "\n" +
                "AUROC:TESTAUROC,CAWPE,ES,WMV,RC,MV,PB,NBC,SMM5,SMLR,SMLRE\n" +
                "AvgTESTAUROCOverDsets:,0.8262914636498966,0.7193997153980937,0.7220092206275495,0.7190774301664393,0.7121700348212276,0.7284682578879959,0.7241957730733346,0.698369338370444,0.6929579203951286,0.6788054628549984\n" +
                "AvgTESTAUROCRankOverDsets:,1.25,4.5,4.625,4.875,5.5,5.75,5.75,7.0,7.5,8.25\n" +
                "StddevOfTESTAUROCOverDsets:,0.13482199658689864,0.2403190866043961,0.24374144164131392,0.24138238449945337,0.24249059219610417,0.26153680164801874,0.24839454205892592,0.2947896004777111,0.3074283953348226,0.32015620465706657\n" +
                "AvgOfStddevsOfTESTAUROCOverDsetFolds:,0.031071630539307594,0.03420301267828394,0.033884893321808114,0.03258354834179206,0.035126585267339375,0.06495326032682511,0.032480144453436416,0.09440746353201794,0.08114074689349532,0.07735431230312256\n" +
                "StddevsOfTESTAUROCRanksOverDsets:,0.5,2.6457513110645907,2.056493779875511,2.839454172900137,3.415650255319866,3.4034296427770228,1.2583057392117916,1.4142135623730951,3.1091263510296048,2.8722813232690143\n" +
                "\n" +
                "NLL:TESTNLL,CAWPE,WMV,RC,SMLR,NBC,ES,SMM5,MV,SMLRE,PB\n" +
                "AvgTESTNLLOverDsets:,0.7963719585782506,1.1338811540590144,1.1385311519494496,1.0899768746823426,1.0712993126746229,1.1498738678034464,1.1587740441940102,1.1623253775688167,1.152920053648824,1.275911506583001\n" +
                "AvgTESTNLLRankOverDsets:,2.5,4.75,5.0,5.25,5.25,5.5,5.75,5.75,6.75,8.5\n" +
                "StddevOfTESTNLLOverDsets:,0.5726732374761851,0.946307003393406,0.9582013847142676,0.8537297216696296,0.7625938818299605,0.9692650150361382,0.7499586325376641,0.9553630759951848,0.8940581604527054,0.9048757848873458\n" +
                "AvgOfStddevsOfTESTNLLOverDsetFolds:,0.12326114785227828,0.15749568214475473,0.1522647474378176,0.1733560671515387,0.25230947732172054,0.16729541094310804,0.35605359534657044,0.1607477770434029,0.1940380760608198,0.46567157407583115\n" +
                "StddevsOfTESTNLLRanksOverDsets:,3.0,1.8929694486000912,2.943920288775949,2.217355782608345,2.8722813232690143,4.123105625617661,2.753785273643051,2.8722813232690143,3.4034296427770228,1.9148542155126762\n" +
                "\n";
        
        System.out.println(expectedBigGlobalSummary);
        System.out.println("\n\n\n\n");
        System.out.println(sb.toString());
        
        assertEquals(sb.toString(), expectedBigGlobalSummary);
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
