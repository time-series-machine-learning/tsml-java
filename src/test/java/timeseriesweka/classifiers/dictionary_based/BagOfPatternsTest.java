package timeseriesweka.classifiers.dictionary_based;

import java.lang.invoke.MethodHandles;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import timeseriesweka.classifiers.dictionary_based.BagOfPatterns;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class BagOfPatternsTest {
    static String cleanClassNameString = MethodHandles.lookup().lookupClass().getSimpleName().replace("Test", "");
    
    public BagOfPatternsTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
        System.out.println("-----Start " + cleanClassNameString + " tests-----");
    }
    
    @AfterClass
    public static void tearDownClass() {
        System.out.println("-----End " + cleanClassNameString + " tests-----\n");
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    /**
     * Test of simple results reproduction.
     */
    @Test
    public void testReproduction() throws Exception {
        System.out.println("--testReproduction()");
        assertTrue(ClassifierTools.testUtils_confirmIPDReproduction(new BagOfPatterns(), 0.8425655976676385, "2019_09_26"));
    }
    
}